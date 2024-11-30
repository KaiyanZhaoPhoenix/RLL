import numpy as np
import torch as th
import threading
from typing import Optional, Dict, Any, List, Tuple
from gymnasium import spaces

class DBERReplayBufferSamples:
    """存储采样的经验数据"""
    def __init__(
        self,
        observations: th.Tensor,
        actions: th.Tensor,
        next_observations: th.Tensor,
        dones: th.Tensor,
        rewards: th.Tensor,
        weights: th.Tensor,
        indices: np.ndarray,
    ):
        self.observations = observations
        self.actions = actions
        self.next_observations = next_observations
        self.dones = dones
        self.rewards = rewards
        self.weights = weights
        self.indices = indices

class DBERReplayBuffer:
    """基于多样性的经验回放缓冲区"""
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str = "auto",
        n_envs: int = 1,
        segment_length: int = 2,
        clip_diversity: float = 1.0,
        optimize_memory_usage: bool = True,
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.segment_length = segment_length
        self.clip_diversity = clip_diversity
        self.device = device
        self.optimize_memory_usage = optimize_memory_usage
        
        # 初始化缓冲区
        self.obs_shape = observation_space.shape
        self.action_shape = action_space.shape if len(action_space.shape) > 0 else (1,)
        
        # 创建缓冲区字典
        self.buffers = {
            'observations': np.zeros((self.buffer_size, *self.obs_shape), dtype=np.float32),
            'next_observations': np.zeros((self.buffer_size, *self.obs_shape), dtype=np.float32),
            'actions': np.zeros((self.buffer_size, *self.action_shape), dtype=np.float32),
            'rewards': np.zeros((self.buffer_size, 1), dtype=np.float32),
            'dones': np.zeros((self.buffer_size, 1), dtype=np.float32),
            'diversity_scores': np.zeros((self.buffer_size, 1), dtype=np.float32)
        }
        
        # 内存管理
        self.current_size = 0
        self.n_transitions_stored = 0
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 轨迹管理
        self.current_trajectory = []
        self.trajectory_segments = []
        self.max_segments = buffer_size // segment_length
        self._eps = 1e-6

    def _compute_kernel_matrix(self, states: np.ndarray) -> np.ndarray:
        """计算核矩阵 L = M^T M"""
        # 将状态展平并归一化
        flat_states = states.reshape(states.shape[0], -1)
        norms = np.linalg.norm(flat_states, axis=1, keepdims=True)
        M = flat_states / (norms + self._eps)
        # 计算核矩阵
        L = M @ M.T
        return L

    def _compute_diversity_score(self, states: np.ndarray) -> float:
        """计算状态序列的多样性得分 d_τj = det(L_τj)"""
        try:
            # 计算核矩阵 L_τj = M^T M
            L = self._compute_kernel_matrix(states)
            # 添加小的对角项以确保数值稳定性
            L += self._eps * np.eye(L.shape[0])
            # Cholesky分解 L_τj = L_C L_C^T
            chol = np.linalg.cholesky(L)
            # 计算行列式 det(L_τj) = ∏(l_ii^2)
            diversity = np.prod(np.diagonal(chol)**2)
            return np.clip(float(diversity), 0, self.clip_diversity)
        except np.linalg.LinAlgError:
            return self._eps

    def _segment_trajectory(self, trajectory: List) -> List[np.ndarray]:
        """将轨迹分段为子轨迹"""
        segments = []
        for i in range(0, len(trajectory), self.segment_length):
            if i + self.segment_length <= len(trajectory):
                segment = trajectory[i:i + self.segment_length]
                segments.append(np.array([t[0] for t in segment]))  # 只使用状态进行多样性计算
        return segments

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, 
            reward: np.ndarray, done: np.ndarray, infos: Dict[str, Any]) -> None:
        """添加转换到缓冲区"""
        with self.lock:
            for i in range(self.n_envs):
                self.current_trajectory.append((
                    obs[i], next_obs[i], action[i], reward[i], done[i]
                ))
                
                if done[i] or len(self.current_trajectory) >= self.segment_length:
                    if len(self.current_trajectory) >= self.segment_length:
                        # 分段并计算多样性
                        segments = self._segment_trajectory(self.current_trajectory)
                        
                        for segment_states in segments:
                            # 计算段的多样性得分
                            diversity = self._compute_diversity_score(segment_states)
                            
                            # 存储轨迹段和对应的多样性得分
                            start_idx = len(segment_states)
                            for idx, transition in enumerate(self.current_trajectory[:start_idx]):
                                pos = self._get_storage_idx()
                                self.buffers['observations'][pos] = transition[0]
                                self.buffers['next_observations'][pos] = transition[1]
                                self.buffers['actions'][pos] = transition[2]
                                self.buffers['rewards'][pos] = transition[3]
                                self.buffers['dones'][pos] = transition[4]
                                self.buffers['diversity_scores'][pos] = diversity
                            
                            self.n_transitions_stored += start_idx
                    
                    # 重置当前轨迹
                    self.current_trajectory = []

    def sample(self, batch_size: int) -> DBERReplayBufferSamples:
        """基于多样性采样经验"""
        with self.lock:
            assert self.current_size > 0, "Cannot sample from an empty buffer"
            
            # 计算采样概率 p(τi) = d_τi / Σd_τn
            div_scores = self.buffers['diversity_scores'][:self.current_size]
            probs = div_scores / (np.sum(div_scores) + self._eps)
            
            # 采样索引
            indices = np.random.choice(
                self.current_size, 
                size=batch_size, 
                p=probs.flatten(),
                replace=True
            )
            
            # 构建批次
            samples = {
                key: self.buffers[key][indices] 
                for key in self.buffers.keys()
                if key != 'diversity_scores'
            }
            
            # 计算重要性权重
            weights = 1.0 / (probs[indices] + self._eps)
            weights = weights / np.max(weights)
            
            return DBERReplayBufferSamples(
                observations=th.as_tensor(samples['observations']).to(self.device),
                actions=th.as_tensor(samples['actions']).to(self.device),
                next_observations=th.as_tensor(samples['next_observations']).to(self.device),
                dones=th.as_tensor(samples['dones']).to(self.device),
                rewards=th.as_tensor(samples['rewards']).to(self.device),
                weights=th.as_tensor(weights).to(self.device),
                indices=indices
            )

    def _get_storage_idx(self) -> int:
        """获取存储索引"""
        if self.current_size < self.buffer_size:
            idx = self.current_size
            self.current_size += 1
        else:
            # 使用均匀分布替换旧的经验
            idx = np.random.randint(0, self.buffer_size)
        return idx

    def get_statistics(self) -> Dict[str, float]:
        """获取缓冲区统计信息"""
        with self.lock:
            if self.current_size == 0:
                return {
                    'size': 0,
                    'transitions_stored': 0,
                    'mean_diversity': 0.0,
                    'max_diversity': 0.0,
                    'min_diversity': 0.0
                }
                
            div_scores = self.buffers['diversity_scores'][:self.current_size]
            return {
                'size': self.current_size,
                'transitions_stored': self.n_transitions_stored,
                'mean_diversity': float(np.mean(div_scores)),
                'max_diversity': float(np.max(div_scores)),
                'min_diversity': float(np.min(div_scores))
            }