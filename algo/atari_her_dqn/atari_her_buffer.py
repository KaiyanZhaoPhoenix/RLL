import numpy as np
import torch as th
from typing import Any, Optional, Union
from gymnasium import spaces

from algo.common.buffers import ReplayBuffer
from algo.common.type_aliases import ReplayBufferSamples

class AtariHerReplayBuffer(ReplayBuffer):
    """
    Atari游戏的HER风格replay buffer
    核心思想：从高分轨迹中采样状态来创建额外的学习样本
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        n_sampled_transitions: int = 4,  # 每个transition采样的数量
        score_percentile: float = 75,  # 用于选择高分轨迹的分位数
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
        )
        
        self.n_sampled_transitions = n_sampled_transitions
        self.score_percentile = score_percentile
        
        # 存储episode信息
        self.ep_start = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.ep_length = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.ep_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self._current_ep_start = np.zeros(self.n_envs, dtype=np.int64)
        self._current_ep_return = np.zeros(self.n_envs, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """添加transition到buffer"""
        # 更新episode起始位置
        self.ep_start[self.pos] = self._current_ep_start.copy()
        
        # 更新当前episode的累积奖励
        self._current_ep_return += reward
        
        # 存储transition
        super().add(obs, next_obs, action, reward, done, infos)
        
        # 当episode结束时，处理episode信息
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                self._handle_episode_end(env_idx)

    def _handle_episode_end(self, env_idx: int) -> None:
        """处理episode结束时的逻辑"""
        episode_start = self._current_ep_start[env_idx]
        episode_end = self.pos
        if episode_end < episode_start:
            episode_end += self.buffer_size
            
        # 计算episode长度
        episode_length = episode_end - episode_start
        episode_indices = np.arange(episode_start, episode_end) % self.buffer_size
        
        # 存储episode信息
        self.ep_length[episode_indices, env_idx] = episode_length
        self.ep_returns[episode_indices, env_idx] = self._current_ep_return[env_idx]
        
        # 重置当前episode的信息
        self._current_ep_start[env_idx] = self.pos
        self._current_ep_return[env_idx] = 0.0

    def sample(self, batch_size: int, env: Optional[Any] = None) -> ReplayBufferSamples:
        """采样transitions"""
        # 计算真实样本和虚拟样本的数量
        n_real = batch_size // (self.n_sampled_transitions + 1)
        n_virtual = batch_size - n_real
        
        # 采样真实transitions
        real_samples = self._sample_real(n_real)
        
        # 采样并创建虚拟transitions
        virtual_samples = self._sample_virtual(n_virtual)
        
        # 合并样本
        return self._concatenate_samples(real_samples, virtual_samples)

    def _sample_real(self, batch_size: int) -> ReplayBufferSamples:
        """采样真实transitions"""
        batch_inds = np.random.randint(0, self.buffer_size, size=batch_size)
        return self._get_samples(batch_inds)

    def _sample_virtual(self, batch_size: int) -> ReplayBufferSamples:
        """从高分轨迹中采样并创建虚拟transitions"""
        # 找出高分轨迹
        valid_episodes = self.ep_length > 0
        if not np.any(valid_episodes):
            return self._sample_real(batch_size)
            
        episode_returns = self.ep_returns[valid_episodes]
        high_score_threshold = np.percentile(episode_returns, self.score_percentile)
        high_score_episodes = valid_episodes & (self.ep_returns >= high_score_threshold)
        
        if not np.any(high_score_episodes):
            return self._sample_real(batch_size)
            
        # 从高分轨迹中采样目标状态
        high_score_indices = np.where(high_score_episodes)
        future_states_indices = np.random.choice(
            high_score_indices[0],
            size=batch_size,
            replace=True
        )
        
        # 获取当前状态样本
        current_samples = self._sample_real(batch_size)
        # 获取目标状态样本
        future_samples = self._get_samples(future_states_indices)
        
        # 使用未来状态作为目标，计算新的奖励
        current_obs = current_samples.observations.cpu().numpy()
        future_obs = future_samples.observations.cpu().numpy()
        
        # 将图像数据展平用于计算相似度
        current_flat = current_obs.reshape(current_obs.shape[0], -1)
        future_flat = future_obs.reshape(future_obs.shape[0], -1)
        
        # 计算到目标状态的距离作为奖励
        distance_to_goal = -np.mean(
            np.abs(future_flat - current_flat),
            axis=-1,
            keepdims=True
        )
        
        # 创建新的虚拟样本
        return ReplayBufferSamples(
            observations=current_samples.observations,
            actions=current_samples.actions,
            next_observations=current_samples.next_observations,
            dones=current_samples.dones,
            rewards=self.to_torch(distance_to_goal)
        )

    def _concatenate_samples(
        self,
        real_samples: ReplayBufferSamples,
        virtual_samples: ReplayBufferSamples,
    ) -> ReplayBufferSamples:
        """合并真实和虚拟样本"""
        return ReplayBufferSamples(
            observations=th.cat([real_samples.observations, virtual_samples.observations]),
            actions=th.cat([real_samples.actions, virtual_samples.actions]),
            next_observations=th.cat([real_samples.next_observations, virtual_samples.next_observations]),
            dones=th.cat([real_samples.dones, virtual_samples.dones]),
            rewards=th.cat([real_samples.rewards, virtual_samples.rewards])
        ) 