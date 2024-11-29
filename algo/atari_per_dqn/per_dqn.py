from typing import Any, Optional, Union
import numpy as np
import torch as th
from torch.nn import functional as F

from algo.dqn.dqn import DQN
from algo.atari_per_dqn.per_buffer import PERReplayBuffer
from algo.common.type_aliases import GymEnv, Schedule

class PERDQN(DQN):
    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_eps: float = 1e-6,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
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
            replay_buffer_class=PERReplayBuffer,
            replay_buffer_kwargs=dict(
                alpha=per_alpha,
                beta=per_beta,
                eps=per_eps,
            ),
            _init_setup_model=_init_setup_model,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size)
            
            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + \
                                (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, 
                                       index=replay_data.actions.long())

            td_errors = th.abs(target_q_values - current_q_values).detach()
            
            loss = F.smooth_l1_loss(current_q_values, 
                                   target_q_values, 
                                   reduction="none")
            weighted_loss = (replay_data.weights * loss).mean()
            
            self.policy.optimizer.zero_grad()
            weighted_loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            
            self.replay_buffer.update_priorities(
                replay_data.indices,
                td_errors.cpu().numpy()
            )
            
            losses.append(weighted_loss.item())
        
        self._n_updates += gradient_steps
        
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/loss", np.mean(losses))
