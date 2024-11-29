from typing import Any, Optional, Union
import torch as th
from algo.dqn.dqn import DQN
from algo.common.policies import BasePolicy
from algo.common.type_aliases import GymEnv, Schedule
from algo.atari_her_dqn.atari_her_buffer import AtariHerReplayBuffer

class AtariHerDQN(DQN):
    """
    Atari游戏的HER风格DQN
    
    1. 记录完整episodes的信息
    2. 识别高分轨迹
    3. 从高分轨迹中采样额外的学习样本
    4. 使用状态相似度作为额外的奖励信号
    """
    
    def __init__(
        self,
        policy: Union[str, type[BasePolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        n_sampled_transitions: int = 4,
        score_percentile: float = 75,
        **kwargs,
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
            replay_buffer_class=AtariHerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_transitions=n_sampled_transitions,
                score_percentile=score_percentile,
            ),
            **kwargs,
        ) 