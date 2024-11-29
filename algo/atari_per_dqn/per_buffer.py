import numpy as np
import torch as th
from typing import Optional, Union, Any
from gymnasium import spaces

from algo.common.buffers import ReplayBuffer
from algo.common.vec_env import VecNormalize
from algo.atari_per_dqn.sum_tree import SumTree

class PERReplayBufferSamples:
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

class PERReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination
        )
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.beta_increment = 0.001
        self.tree_capacity = buffer_size * n_envs
        self.sum_tree = SumTree(self.tree_capacity)
        self.max_priority = 1.0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Add a new experience to memory."""
        super().add(obs, next_obs, action, reward, done, infos)
        
        # 计算在树中的实际位置
        tree_idx = (self.pos - 1) * self.n_envs
        if tree_idx < 0:
            tree_idx = self.tree_capacity - self.n_envs
            
        # 为每个环境添加优先级
        for env_idx in range(self.n_envs):
            self.sum_tree.add(self.max_priority, tree_idx + env_idx)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> PERReplayBufferSamples:
        """Sample a batch of experiences."""
        if not self.full and self.pos == 0:
            raise RuntimeError("Not enough samples in buffer")

        indices = []
        priorities = []
        
        # 获取总优先级，并确保不为0
        total_priority = max(self.sum_tree.total(), 1e-8)
        segment = total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            # 限制范围以避免溢出
            mass = np.random.uniform(min(a, 1e8), min(b, 1e8))
            
            try:
                idx, priority, _ = self.sum_tree.get(mass)
            except Exception as e:
                print(f"Error in get: mass={mass}, a={a}, b={b}, total_priority={total_priority}")
                raise e
            
            # 将树的索引转换为缓冲区索引
            buffer_idx = idx // self.n_envs
            env_idx = idx % self.n_envs
            
            # 确保索引在有效范围内
            if buffer_idx >= self.buffer_size:
                buffer_idx = buffer_idx % self.buffer_size
            
            indices.append(buffer_idx * self.n_envs + env_idx)
            priorities.append(priority)

        indices = np.array(indices)
        priorities = np.array(priorities)

        # 计算重要性采样权重，添加数值稳定性
        samples_probs = priorities / (total_priority + 1e-8)
        
        # 限制权重计算中的值范围
        inv_probs = np.minimum(self.buffer_size * samples_probs, 1e8)
        weights = np.power(inv_probs + 1e-8, -self.beta)
        
        # 安全地归一化权重
        max_weight = weights.max() + 1e-8
        weights = weights / max_weight
        
        # 处理任何可能的 NaN 值
        weights = np.nan_to_num(weights, nan=1.0)
        
        # 获取经验样本
        buffer_indices = indices // self.n_envs
        env_indices = indices % self.n_envs
        samples = super()._get_samples(buffer_indices, env)
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return PERReplayBufferSamples(
            observations=samples.observations,
            actions=samples.actions,
            next_observations=samples.next_observations,
            dones=samples.dones,
            rewards=samples.rewards,
            weights=self.to_torch(weights.reshape(-1, 1)),
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities of sampled transitions."""
        # 限制优先级范围以避免数值问题
        priorities = np.clip(np.abs(priorities) + self.eps, 0, 1e8)
        priorities = priorities ** self.alpha
        
        for idx, priority in zip(indices, priorities):
            # 确保优先级是标量
            priority = float(priority)
            self.max_priority = min(max(self.max_priority, priority), 1e8)
            
            # 更新树中的优先级
            self.sum_tree.update(idx, priority)