from typing import List, Union

import torch
from torch import nn

from config import DefaultConfig, NatureConfig, TestConfig


class DeepQNetwork(nn.Module):
    def __init__(
        self,
        config: Union[DefaultConfig, NatureConfig, TestConfig],
        state_shape: List,
        num_actions: int,
        device: str,
    ):
        super().__init__()

        img_height, img_width, n_channels = state_shape

        def get_padding(stride, kernel_size):
            return ((stride - 1) * img_height - stride + kernel_size) // 2

        self.conv1 = nn.Conv2d(
            in_channels=n_channels * config.history_length,
            out_channels=32,
            kernel_size=8,
            stride=4,
            padding=get_padding(
                stride=4, kernel_size=8
            ),  # todo: do i really need padding?
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=get_padding(stride=2, kernel_size=4),
        )
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=get_padding(stride=1, kernel_size=3),
        )

        self.fc1 = nn.Linear(
            in_features=64 * img_height * img_width, out_features=512
        )  # todo: why input is not simply 64?
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        self.to(device)

    def forward(self, x):
        # State placeholders are uint8 for fast transfer to GPU
        # Need to cast it to float32 for the rest of the graph.
        x = x.float() / 255.0
        # Reshape image from [batch size, height, width, channels]
        # to [batch size, channels, height, width]
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
