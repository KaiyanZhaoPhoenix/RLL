import numpy as np
import torch as th
from typing import Optional, Dict, Any, List, Tuple
from gymnasium import spaces

from algo.common.buffers import ReplayBuffer
from algo.common.vec_env import VecNormalize

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

class DBERBuffer(ReplayBuffer):
    """基于多样性的经验回放缓冲区"""
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str = "auto",
        n_envs: int = 1,
        segment_length: int = 2,
        max_trajectories: int = 1000,  # 最大轨迹数量
        optimize_memory_usage: bool = False,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
        )
        
        # 初始化参数
        self.segment_length = segment_length
        self.max_trajectories = max_trajectories
        self._eps = 1e-6
        
        # 轨迹相关存储
        self.trajectories = []  # 存储完整轨迹
        self.trajectory_segments = []  # 存储分段后的轨迹
        self.segment_diversity_scores = []  # 每个分段的多样性得分
        self.current_trajectory = []  # 当前正在收集的轨迹
        
        # 多样性统计
        self.diversity_metrics = {
            "avg_diversity": 0.0,
            "max_diversity": 0.0,
            "min_diversity": 0.0,
            "diversity_std": 0.0
        }
        
    def _compute_kernel_matrix(self, states: np.ndarray) -> np.ndarray:
        """计算核矩阵 L = M^T M"""
        flat_states = states.reshape(states.shape[0], -1)
        norms = np.linalg.norm(flat_states, axis=1, keepdims=True)
        M = flat_states / (norms + self._eps)
        L = M @ M.T
        return L
        
    def _compute_diversity_score(self, states: np.ndarray) -> float:
        """使用Cholesky分解计算多样性得分"""
        try:
            L = self._compute_kernel_matrix(states)
            L += self._eps * np.eye(L.shape[0])
            chol = np.linalg.cholesky(L)
            diversity = np.prod(np.diagonal(chol)**2)
            return float(diversity)
        except np.linalg.LinAlgError:
            return self._eps
            
    def _segment_trajectory(self, trajectory: List) -> List[np.ndarray]:
        """将轨迹分段为子轨迹"""
        segments = []
        for i in range(0, len(trajectory) - self.segment_length + 1):
            segment = trajectory[i:i + self.segment_length]
            states = np.array([s[0] for s in segment])
            segments.append(states)
        return segments
        
    def update_trajectories(self):
        """更新轨迹池和多样性得分"""
        if len(self.trajectories) > self.max_trajectories:
            # 根据多样性得分保留最多样的轨迹
            indices = np.argsort(self.segment_diversity_scores)[-self.max_trajectories:]
            self.trajectories = [self.trajectories[i] for i in indices]
            self.trajectory_segments = [self.trajectory_segments[i] for i in indices]
            self.segment_diversity_scores = [self.segment_diversity_scores[i] for i in indices]
            
        # 更新多样性统计指标
        self._update_diversity_metrics()
            
    def _update_diversity_metrics(self):
        """更新多样性统计指标"""
        if self.segment_diversity_scores:
            scores = np.array(self.segment_diversity_scores)
            self.diversity_metrics.update({
                "avg_diversity": float(np.mean(scores)),
                "max_diversity": float(np.max(scores)),
                "min_diversity": float(np.min(scores)),
                "diversity_std": float(np.std(scores))
            })

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: Dict[str, Any],
    ) -> None:
        """添加经验到缓冲区"""
        super().add(obs, next_obs, action, reward, done, infos)
        
        for i in range(self.n_envs):
            self.current_trajectory.append((
                obs[i],
                action[i],
                reward[i],
                next_obs[i],
                done[i]
            ))
            
            if done[i]:
                if len(self.current_trajectory) >= self.segment_length:
                    # 分段并计算多样性
                    segments = self._segment_trajectory(self.current_trajectory)
                    for segment in segments:
                        diversity = self._compute_diversity_score(segment)
                        self.segment_diversity_scores.append(diversity)
                        self.trajectory_segments.append(segment)
                    
                    # 存储完整轨迹
                    self.trajectories.append(self.current_trajectory)
                    
                    # 更新轨迹池
                    self.update_trajectories()
                
                self.current_trajectory = []

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DBERReplayBufferSamples:
        """基于多样性采样经验"""
        if not self.segment_diversity_scores:
            return self._uniform_sample(batch_size, env)
            
        # 计算接受概率
        max_diversity = max(self.segment_diversity_scores)
        acceptance_probs = np.array(self.segment_diversity_scores) / max_diversity
        
        # 接受-拒绝采样
        accepted_indices = []
        attempts = 0
        max_attempts = batch_size * 10  # 防止无限循环
        
        while len(accepted_indices) < batch_size and attempts < max_attempts:
            idx = np.random.randint(0, len(self.trajectories))
            if np.random.random() <= acceptance_probs[idx]:
                accepted_indices.append(idx)
            attempts += 1
            
        # 如果没有采样到足够的样本，补充随机样本
        if len(accepted_indices) < batch_size:
            remaining = batch_size - len(accepted_indices)
            random_indices = np.random.randint(0, len(self.trajectories), size=remaining)
            accepted_indices.extend(random_indices)
        
        # 获取采样的经验
        batch_inds = np.array(accepted_indices)
        weights = 1.0 / (acceptance_probs[batch_inds] + self._eps)
        weights = weights / np.max(weights)
        
        data = self._get_samples(batch_inds, env)
        
        return DBERReplayBufferSamples(
            observations=data.observations,
            actions=data.actions,
            next_observations=data.next_observations,
            dones=data.dones,
            rewards=data.rewards,
            weights=th.FloatTensor(weights).to(self.device),
            indices=batch_inds,
        )

    def get_diversity_metrics(self) -> Dict[str, float]:
        """获取多样性统计指标"""
        return self.diversity_metrics.copy()