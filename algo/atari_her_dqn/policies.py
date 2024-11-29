from typing import Any, Dict, List, Optional, Type
import torch as th
from torch import nn
from gymnasium import spaces

from algo.common.policies import BasePolicy
from algo.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from algo.common.type_aliases import Schedule

class AtariCnnPolicy(BasePolicy):
    """
    Atari游戏的CNN策略网络
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [512]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.q_net, self.q_net_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """构建Q网络"""
        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def make_q_net(self) -> nn.Module:
        """创建Q网络"""
        q_net = QNetwork(
            features_extractor=self.features_extractor,
            **self.net_args,
        )
        return q_net

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self.q_net(obs)
        if deterministic:
            action = q_values.argmax(dim=1).reshape(-1)
        else:
            # 使用epsilon-greedy策略
            if self.training:
                eps = self.exploration_rate
                if random.random() < eps:
                    action = th.randint(0, self.action_space.n, (obs.shape[0],))
                else:
                    action = q_values.argmax(dim=1).reshape(-1)
            else:
                action = q_values.argmax(dim=1).reshape(-1)
        return action

class QNetwork(nn.Module):
    """Q网络"""
    def __init__(
        self,
        features_extractor: nn.Module,
        action_space: spaces.Space,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.activation_fn = activation_fn
        self.action_space = action_space
        
        # 构建Q网络
        modules = []
        last_layer_dim = self.features_extractor.features_dim
        
        for layer_size in net_arch:
            modules.append(nn.Linear(last_layer_dim, layer_size))
            modules.append(activation_fn())
            last_layer_dim = layer_size
            
        # 输出层
        self.q_net = nn.Sequential(
            *modules,
            nn.Linear(last_layer_dim, action_space.n)
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """前向传播"""
        features = self.features_extractor(obs)
        return self.q_net(features)