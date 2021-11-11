import enum
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.tensor import Tensor


class NetworkType(enum.Enum):
    Q_NETWORK = "q_network"
    TARGET_NETWORK = "target_network"


class DeepQNetwork:
    def __init__(self, env, config, timer):
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Running model on device {self.device}")

        self.config = config
        self.env = env
        self.timer = timer

        self.build_model()

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
                        in_channels=n_channels * self.config.state_history,
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

    def get_q_values(
        self, state: Tensor, network: NetworkType = NetworkType.Q_NETWORK
    ) -> Tensor:
        """
        Args:
            state:
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network:
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: shape = (batch_size, num_actions)
        """
        out = None
        state = state.permute(0, 3, 1, 2)

        if network == NetworkType.Q_NETWORK:
            out = self.q_network(state)
        elif network == NetworkType.TARGET_NETWORK:
            out = self.target_network(state)

        return out

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
        q_sa = q_values[range(len(q_values)), actions.type(torch.LongTensor)]
        return F.mse_loss(q_target, q_sa)

    def process_state(self, state: Tensor) -> Tensor:
        """
        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, height, width, nchannels)
                    of type tf.uint8.
                    if , values are between 0 and 255 -> 0 and 1
        """
        state = state.float()
        state /= self.config.high

        return state

    def build_model(self) -> None:
        self.initialize_models()

        if (
            hasattr(self.config, "model_weights_path")
            and os.path.exists(self.config.weights_path)
            and os.listdir(self.config.weights_path)
        ):
            weights_file_path = Path(self.config.model_weights_path)
            assert (
                weights_file_path.is_file()
            ), f"Provided weights file ({weights_file_path}) does not exist"
            self.q_network.load_state_dict(
                torch.load(weights_file_path, map_location="cpu")
            )
            print("Load successful!")
        else:
            print("Initializing parameters randomly")

        def init_weights(m):
            if hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight, gain=2 ** (1.0 / 2))
            if hasattr(m, "bias"):
                nn.init.zeros_(m.bias)

        self.q_network.apply(init_weights)

        self.q_network = self.q_network.to(self.device)
        self.target_network = self.target_network.to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters())

    def get_best_action(self, state: Tensor) -> Tuple[int, np.ndarray]:
        """
        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0)
            s = self.process_state(s)
            action_values = (
                self.get_q_values(s, NetworkType.Q_NETWORK).squeeze().to("cpu").tolist()
            )
        action = np.argmax(action_values)
        return action, action_values

    def get_action(self, state):
        """
        Args:
            state: observation from gym
        """
        if np.random.random() < self.config.soft_epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_best_action(state)[0]

    def update_step(self, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """
        self.timer.start("update_step/replay_buffer.sample")
        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size
        )
        self.timer.end("update_step/replay_buffer.sample")

        # Convert to Tensor and move to correct device
        self.timer.start("update_step/converting_tensors")
        s_batch = torch.tensor(s_batch, dtype=torch.uint8, device=self.device)
        a_batch = torch.tensor(a_batch, dtype=torch.uint8, device=self.device)
        r_batch = torch.tensor(r_batch, dtype=torch.float, device=self.device)
        sp_batch = torch.tensor(sp_batch, dtype=torch.uint8, device=self.device)
        done_mask_batch = torch.tensor(
            done_mask_batch, dtype=torch.bool, device=self.device
        )
        self.timer.end("update_step/converting_tensors")

        # Reset Optimizer
        self.timer.start("update_step/zero_grad")
        self.optimizer.zero_grad()
        self.timer.end("update_step/zero_grad")

        # Run a forward pass
        self.timer.start("update_step/forward_pass_q")
        s = self.process_state(s_batch)
        q_values = self.get_q_values(s, NetworkType.Q_NETWORK)
        self.timer.end("update_step/forward_pass_q")

        self.timer.start("update_step/forward_pass_target")
        with torch.no_grad():
            sp = self.process_state(sp_batch)
            target_q_values = self.get_q_values(sp, NetworkType.TARGET_NETWORK)
        self.timer.end("update_step/forward_pass_target")

        self.timer.start("update_step/loss_calc")
        loss = self.calc_loss(
            q_values, target_q_values, a_batch, r_batch, done_mask_batch
        )
        self.timer.end("update_step/loss_calc")
        self.timer.start("update_step/loss_backward")
        loss.backward()
        self.timer.end("update_step/loss_backward")

        # Clip norm
        self.timer.start("update_step/grad_clip")
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.config.clip_val
        )
        self.timer.end("update_step/grad_clip")

        # Update parameters with optimizer
        self.timer.start("update_step/optimizer")
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.timer.end("update_step/optimizer")
        return loss.item(), total_norm.item()

    def synchronize_networks(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_parameters(self):
        return self.q_network.state_dict()
