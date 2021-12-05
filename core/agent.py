import enum
from typing import Tuple, Dict, Union

import gym
import numpy as np
import torch
from torch import Tensor, nn

from config import DefaultConfig, NatureConfig, TestConfig
from core.deep_q_network import DeepQNetwork


class NetworkType(enum.Enum):
    Q_NETWORK = "q_network"
    TARGET_NETWORK = "target_network"


class Agent:
    def __init__(
        self,
        env: gym.Env,
        config: Union[DefaultConfig, NatureConfig, TestConfig],
        device: str,
    ):
        self.q_network = None
        self.target_network = None
        self.env = env
        self.device = device
        self.build_networks(config)

    def build_networks(self, config) -> None:
        state_shape = list(self.env.observation_space.shape)
        num_actions = self.env.action_space.n

        self.q_network = DeepQNetwork(config, state_shape, num_actions, self.device)
        self.target_network = DeepQNetwork(
            config, state_shape, num_actions, self.device
        )

        def init_weights(m):
            if hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight, gain=2 ** (1.0 / 2))
            if hasattr(m, "bias"):
                nn.init.zeros_(m.bias)

        self.q_network.apply(init_weights)

    def get_q_values(
        self, state: Tensor, network: NetworkType = NetworkType.Q_NETWORK
    ) -> Tensor:
        out = None

        if network == NetworkType.Q_NETWORK:
            out = self.q_network(state)
        elif network == NetworkType.TARGET_NETWORK:
            out = self.target_network(state)

        return out

    def get_action(self, state: Tensor, epsilon: float) -> int:
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_best_action(state)

    def get_best_action(self, state: Tensor) -> int:
        with torch.no_grad():
            state = torch.tensor(
                state, dtype=torch.uint8, device=self.device
            ).unsqueeze(0)
            action_values: np.ndarray = (
                self.q_network(state).squeeze().to("cpu").tolist()
            )
        best_action: int = np.argmax(action_values)
        return best_action

    def state_dict(self) -> Tensor:
        return self.q_network.state_dict()

    def parameters(self) -> Dict:
        return self.q_network.parameters()

    def synchronize_networks(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())
