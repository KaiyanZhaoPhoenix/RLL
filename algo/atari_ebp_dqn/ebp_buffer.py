import numpy as np
import torch as th
from typing import Optional, Union, Any, Dict
from gymnasium import spaces

from algo.common.buffers import ReplayBuffer
from algo.common.vec_env import VecNormalize
from algo.atari_ebp_dqn.segment_tree import SumSegmentTree, MinSegmentTree

class EBPReplayBufferSamples:
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

class EnergyBasedPrioritizedBuffer(ReplayBuffer):
    """能量优先级经验回放缓冲区"""
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        w_state: float = 1.0,
        w_reward: float = 1.0,
        w_action: float = 0.5,
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
        
        # 初始化分段树
        it_capacity = 1
        while it_capacity < buffer_size * n_envs:
            it_capacity *= 2
            
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        
        # 参数设置
        self.alpha = alpha
        self.beta = beta
        self.w_state = w_state
        self.w_reward = w_reward
        self.w_action = w_action
        self._max_priority = 1.0
        self._eps = 1e-6

    def _calculate_energy(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
    ) -> np.ndarray:
        """计算能量值
        
        使用状态变化、奖励和动作计算能量
        添加物理模型约束和数值保护
        """
        # 状态变化能量 (基于动能)
        delta_t = 1.0/30.0  # Atari典型帧率
        state_diff = next_obs - obs
        velocity = state_diff / delta_t
        kinetic_energy = 0.5 * np.sum(np.square(velocity), axis=tuple(range(1, velocity.ndim)))
        kinetic_energy = np.clip(kinetic_energy, 0, 10)
        
        # 奖励能量
        reward_energy = np.abs(reward)
        
        # 动作能量 (稀疏性)
        action_energy = np.mean(action != 0, axis=tuple(range(1, action.ndim)))
        
        # 总能量
        total_energy = (
            self.w_state * kinetic_energy + 
            self.w_reward * reward_energy + 
            self.w_action * action_energy
        )
        return np.clip(total_energy, 0, 10)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: Dict[str, Any],
    ) -> None:
        """添加新的经验到缓冲区"""
        # 计算能量和优先级
        energy = self._calculate_energy(obs, next_obs, action, reward)
        priority = np.clip((np.abs(energy) + self._eps) ** self.alpha, 0, 10)
        
        # 存储数据
        super().add(obs, next_obs, action, reward, done, infos)
        
        # 更新优先级树
        idx = (self.pos - 1) * self.n_envs
        if idx < 0:
            idx = self.buffer_size * self.n_envs - self.n_envs
            
        for env_idx in range(self.n_envs):
            self._it_sum[idx + env_idx] = priority[env_idx]
            self._it_min[idx + env_idx] = priority[env_idx]
            self._max_priority = max(self._max_priority, priority[env_idx])

    def sample(self, batch_size: int, beta: float = 0.4, env: Optional[VecNormalize] = None) -> EBPReplayBufferSamples:
        """采样一个batch的经验"""
        # 检查buffer是否为空
        if self.full:
            current_size = self.buffer_size
        else:
            current_size = self.pos
        current_size *= self.n_envs
        
        assert current_size > 0, "Cannot sample from an empty buffer"
        
        # 采样索引
        total = self._it_sum.sum()
        mass = np.random.random(batch_size) * total
        batch_idx = [self._it_sum.find_prefixsum_idx(m) for m in mass]
        
        # 计算IS权重
        p_min = self._it_min.min() / total
        max_weight = (p_min * current_size) ** (-beta)
        
        weights = []
        for idx in batch_idx:
            p_sample = self._it_sum[idx] / total
            weight = (p_sample * current_size) ** (-beta)
            weights.append(weight / max_weight)
        
        # 获取实际的buffer索引
        buffer_indices = np.array(batch_idx) // self.n_envs
        buffer_indices = buffer_indices % self.buffer_size
        
        # 获取经验数据
        samples = self._get_samples(buffer_indices, env)
        
        return EBPReplayBufferSamples(
            observations=samples.observations,
            actions=samples.actions,
            next_observations=samples.next_observations,
            dones=samples.dones,
            rewards=samples.rewards,
            weights=th.FloatTensor(weights).to(self.device),
            indices=np.array(batch_idx),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """更新优先级"""
        priorities = np.clip(priorities + self._eps, 0, 10) ** self.alpha
        
        for idx, priority in zip(indices, priorities.flatten()):
            self._it_sum[idx] = priority
            self._it_min[idx] = priority
            self._max_priority = max(self._max_priority, priority)