from typing import Any, Optional, Union, Dict, Type
import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces

from algo.dqn.dqn import DQN
from algo.common.type_aliases import GymEnv, Schedule
from algo.common.utils import get_linear_fn
from algo.atari_dber_dqn.diversity_buffer import DBERReplayBuffer

class DBERDQN(DQN):
    """基于多样性的DQN算法实现"""
    def __init__(
        self,
        policy: Union[str, Type[th.nn.Module]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 5000,
        learning_starts: int = 1000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        segment_length: int = 2,
        clip_diversity: float = 1.0,
        target_update_interval: int = 1000,
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
            replay_buffer_class=DBERReplayBuffer,
            replay_buffer_kwargs=dict(
                segment_length=segment_length,
                clip_diversity=clip_diversity,
            ),
            _init_setup_model=_init_setup_model,
        )
        
        self.segment_length = segment_length
        self.clip_diversity = clip_diversity

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """训练方法"""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        diversity_scores = []

        for _ in range(gradient_steps):
            # 采样经验
            replay_data = self.replay_buffer.sample(batch_size)
            
            with th.no_grad():
                # 计算目标Q值
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                # 确保维度正确 [batch_size]
                target_q_values = replay_data.rewards.flatten() + (1 - replay_data.dones.flatten()) * self.gamma * next_q_values
            
            # 获取当前Q值 [batch_size, 1]
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
            
            # 计算损失 - 确保维度匹配
            loss = F.smooth_l1_loss(current_q_values.flatten(), target_q_values)  # 都是 [batch_size]
            loss = (loss * replay_data.weights.flatten()).mean()
            
            # 更新网络
            self.policy.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
            self.policy.optimizer.step()
            
            losses.append(loss.item())
            
            # 获取多样性统计
            diversity_stats = self.replay_buffer.get_statistics()
            diversity_scores.append(diversity_stats['mean_diversity'])

        # 更新目标网络
        if self._n_updates % self.target_update_interval == 0:
            self._update_target()

        self._n_updates += gradient_steps

        # 记录训练信息
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/mean_diversity", np.mean(diversity_scores))

    def _update_target(self) -> None:
        """更新目标网络"""
        for param, target_param in zip(self.q_net.parameters(), self.q_net_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[callable] = None,
        log_interval: int = 4,
        tb_log_name: str = "DBERDQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _on_step(self) -> None:
        """每步更新后的回调"""
        if self.num_timesteps % self.target_update_interval == 0:
            self._update_target()