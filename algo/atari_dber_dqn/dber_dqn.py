from typing import Any, Optional, Union, Dict
import numpy as np
import torch as th
from torch.nn import functional as F

from algo.dqn.dqn import DQN
from algo.common.type_aliases import GymEnv, Schedule
from algo.atari_dber_dqn.diversity_buffer import DBERBuffer

class DBERDQN(DQN):
    """
    Diversity-Based Experience Replay DQN (DBER-DQN)
    
    Paper: Diversity-Based Experience Replay
    """
    def __init__(
        self,
        policy: str,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        segment_length: int = 2,
        max_trajectories: int = 1000,  # 新增：最大轨迹数量
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            replay_buffer_class=DBERBuffer,
            replay_buffer_kwargs=dict(
                segment_length=segment_length,
                max_trajectories=max_trajectories,
            ),
            _init_setup_model=_init_setup_model,
        )
        
        self.segment_length = segment_length
        self.max_trajectories = max_trajectories

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """训练方法"""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # 基于多样性采样
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # 计算目标Q值
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # 获取当前Q值
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # 计算加权损失
            loss = (replay_data.weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()
            losses.append(loss.item())

            # 优化策略
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # 更新目标网络
            if self._n_updates % self.target_update_interval == 0:
                self.policy.q_net_target.load_state_dict(self.policy.q_net.state_dict())

        self._n_updates += gradient_steps

        # 记录训练指标
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))