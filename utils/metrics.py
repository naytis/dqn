from collections import deque

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import config
from utils.progress_bar import ProgressBar


class Metrics:
    def __init__(self):
        self.summary_writer = SummaryWriter(config.output_path, max_queue=int(1e5))

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

        self.eval_reward = 0

        self.loss_eval = 0
        self.grad_eval = 0

        self.bar = ProgressBar(target=config.num_steps_train)

    def update_metrics(self) -> None:
        self.avg_reward = np.mean(self.rewards)
        self.max_reward = np.max(self.rewards)
        self.std_reward = np.std(self.rewards)

        self.avg_max_q = np.mean(self.max_q_values)
        self.avg_q = np.mean(self.q_values_deque)
        self.std_q = np.std(self.q_values_deque)

    def add_summary(self, t: int) -> None:
        self.summary_writer.add_scalar("Loss", self.loss_eval, t)
        self.summary_writer.add_scalar("Gradients Norm", self.grad_eval, t)
        self.summary_writer.add_scalar("Avg reward, 50 eps", self.avg_reward, t)
        self.summary_writer.add_scalar("Max reward, 50 eps", self.max_reward, t)
        self.summary_writer.add_scalar("Std reward", self.std_reward, t)
        self.summary_writer.add_scalar("Avg Q, 1000 ts", self.avg_q, t)
        self.summary_writer.add_scalar("Max Q, 1000 ts", self.avg_max_q, t)
        self.summary_writer.add_scalar("Std Q", self.std_q, t)
        self.summary_writer.add_scalar("Avg evaluated reward", self.eval_reward, t)
        self.summary_writer.add_scalar("Episodes played", self.episodes_counter, t)
        self.summary_writer.add_scalar(
            "Avg episodes length, 50 eps", np.mean(self.episode_length), t
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
                    ("loss", self.loss_eval),
                    ("grads", self.grad_eval),
                ],
                base=config.learning_start,
            )

    def reset_bar(self):
        self.bar.reset_start()
