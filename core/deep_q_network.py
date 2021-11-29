from collections import OrderedDict
from typing import Tuple, Type

import numpy as np
import torch
from gym import Env as GymEnv
from torch import Tensor, nn, optim, functional

from config import Config
from utils.replay_buffer import ReplayBuffer


class DeepQNetwork:
    def __init__(self, env: GymEnv, config: Type[Config]):
        self.q_network = None
        self.target_network = None
        self.optimizer = None

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Running model on device {self.device}")

        self.config = config
        self.env = env

        self.build_model()

    def build_model(self) -> None:
        self.initialize_models()

        print("Initializing parameters randomly")

        def init_weights(m):
            if hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight, gain=2 ** (1.0 / 2))
            if hasattr(m, "bias"):
                nn.init.zeros_(m.bias)

        self.q_network.apply(init_weights)

        self.q_network = self.q_network.to(self.device)
        self.target_network = self.target_network.to(self.device)
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
        )

    def initialize_models(self):
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n

        def get_padding(stride, kernel_size):
            return ((stride - 1) * img_height - stride + kernel_size) // 2

        layers = OrderedDict(
            [
                (
                    "conv1",
                    nn.Conv2d(
                        in_channels=n_channels * self.config.history_length,
                        out_channels=32,
                        kernel_size=8,
                        stride=4,
                        padding=get_padding(stride=4, kernel_size=8),
                    ),
                ),
                ("relu1", nn.ReLU()),
                (
                    "conv2",
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=4,
                        stride=2,
                        padding=get_padding(stride=2, kernel_size=4),
                    ),
                ),
                ("relu2", nn.ReLU()),
                (
                    "conv3",
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=get_padding(stride=1, kernel_size=3),
                    ),
                ),
                ("relu3", nn.ReLU()),
                ("flatten", nn.Flatten()),
                (
                    "linear1",
                    nn.Linear(
                        in_features=64 * img_height * img_width, out_features=512
                    ),
                ),
                ("relu4", nn.ReLU()),
                ("linear2", nn.Linear(in_features=512, out_features=num_actions)),
            ]
        )

        self.q_network = nn.Sequential(layers)
        self.target_network = nn.Sequential(layers)

    def calc_loss(
        self,
        q_values: Tensor,
        target_q_values: Tensor,
        actions: Tensor,
        rewards: Tensor,
        done_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            q_values: shape = (batch_size, num_actions)
                The Q-values that current network estimates (i.e. Q(s, a') for all a')
            target_q_values: shape = (batch_size, num_actions)
                The Target Q-values that target network estimates (i.e. Q_target(s', a') for all a')
            actions: shape = (batch_size,)
                The actions that agent actually took at each step (i.e. a)
            rewards: shape = (batch_size,)
                The rewards that agent actually got at each step (i.e. r)
            done_mask: shape = (batch_size,)
                A boolean mask of examples where agent reached the terminal state
        """
        q_target = (
            rewards
            + (~done_mask)
            * self.config.gamma
            * torch.max(target_q_values, dim=1).values
        )
        q_current = q_values[range(len(q_values)), actions.type(torch.LongTensor)]
        return functional.F.huber_loss(q_target, q_current)

    def process_state(self, state: Tensor) -> Tensor:
        """
        State placeholders are uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the graph.
        """
        state = state.float()
        state /= self.config.high

        return state

    def get_best_action(self, state: Tensor) -> Tuple[int, np.ndarray]:
        with torch.no_grad():
            state = torch.tensor(
                state, dtype=torch.uint8, device=self.device
            ).unsqueeze(0)
            state = self.process_state(state)
            action_values: np.ndarray = (
                self.q_network(state).squeeze().to("cpu").tolist()
            )
        best_action: int = np.argmax(action_values)
        return best_action, action_values

    def get_action(self, state: Tensor) -> int:
        if np.random.random() < self.config.soft_epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_best_action(state)[0]

    def update_params(self, replay_buffer: ReplayBuffer) -> Tuple[int, int]:
        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size
        )

        s_batch = torch.tensor(s_batch, dtype=torch.uint8, device=self.device)
        a_batch = torch.tensor(a_batch, dtype=torch.uint8, device=self.device)
        r_batch = torch.tensor(r_batch, dtype=torch.float, device=self.device)
        sp_batch = torch.tensor(sp_batch, dtype=torch.uint8, device=self.device)
        done_mask_batch = torch.tensor(
            done_mask_batch, dtype=torch.bool, device=self.device
        )

        self.optimizer.zero_grad()

        s = self.process_state(s_batch)
        q_values = self.q_network(s)

        with torch.no_grad():
            sp = self.process_state(sp_batch)
            target_q_values = self.target_network(sp)

        loss = self.calc_loss(
            q_values, target_q_values, a_batch, r_batch, done_mask_batch
        )
        loss.backward()

        total_norm = nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.config.clip_val
        )

        self.optimizer.step()
        return loss.item(), total_norm.item()

    def synchronize_networks(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_parameters(self) -> None:
        return self.q_network.state_dict()
