from collections import deque

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.progress_bar import ProgressBar


class Metrics:
    def __init__(self, num_steps_train: int, learning_start: int, output_path: str):
        self.summary_writer = SummaryWriter(output_path, max_queue=int(1e5))

        self.rewards = deque(maxlen=50)  # rewards for last 50 episodes
        self.max_q_values = deque(maxlen=1000)  # q values for last 1000 timesteps
        self.q_values_deque = deque(maxlen=1000)  # q values for last 1000 timesteps
        self.episode_length = deque(maxlen=50)
        self.episodes_counter = 0

        self.avg_reward = 0
        self.max_reward = 0
        self.std_reward = 0

        self.avg_q = 0
        self.avg_max_q = 0
        self.std_q = 0

        self.avg_eval_reward = 0

        self.loss = 0
        self.grad_norm = 0

        self.avg_episode_length = 0

        self.bar = ProgressBar(target=num_steps_train, base=learning_start)

    def update_metrics(self) -> None:
        self.avg_reward = np.mean(self.rewards)
        self.max_reward = np.max(self.rewards)
        self.std_reward = np.std(self.rewards)

        self.avg_max_q = np.mean(self.max_q_values)
        self.avg_q = np.mean(self.q_values_deque)
        self.std_q = np.std(self.q_values_deque)

        if len(self.episode_length) > 0:
            self.avg_episode_length = np.mean(self.episode_length)

    def add_summary(self, t: int) -> None:
        self.summary_writer.add_scalar("Loss", self.loss, t)
        self.summary_writer.add_scalar("Gradients Norm", self.grad_norm, t)
        self.summary_writer.add_scalar("Avg reward, 50 eps", self.avg_reward, t)
        self.summary_writer.add_scalar("Max reward, 50 eps", self.max_reward, t)
        self.summary_writer.add_scalar("Std reward", self.std_reward, t)
        self.summary_writer.add_scalar("Avg Q, 1000 ts", self.avg_q, t)
        self.summary_writer.add_scalar("Max Q, 1000 ts", self.avg_max_q, t)
        self.summary_writer.add_scalar("Std Q", self.std_q, t)
        self.summary_writer.add_scalar("Avg evaluated reward", self.avg_eval_reward, t)
        self.summary_writer.add_scalar("Episodes played", self.episodes_counter, t)
        self.summary_writer.add_scalar(
            "Avg episode length, 50 eps", self.avg_episode_length, t
        )

    def update_bar(self, epsilon, t):
        if len(self.rewards) > 0:
            self.bar.update(
                t + 1,
                exact=[
                    ("episodes", self.episodes_counter),
                    ("avg r", self.avg_reward),
                    ("max r", self.max_reward),
                    ("max q", self.avg_max_q),
                    ("eps", epsilon),
                    ("loss", self.loss),
                    ("grads", self.grad_norm),
                ],
            )

    def reset_bar(self):
        self.bar.reset_start()
