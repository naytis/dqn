import csv
from collections import deque
from typing import List

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.progress_bar import ProgressBar


class Metrics:
    def __init__(self, num_steps_train: int, learning_start: int, output_path: str):
        self.summary_writer = SummaryWriter(output_path, max_queue=int(1e5))
        self.bar = ProgressBar(target=num_steps_train, base=learning_start)

        metrics_filename = output_path + "metrics.csv"
        self.file_handler = open(metrics_filename, "wt")
        self.logger = csv.DictWriter(
            self.file_handler,
            fieldnames=("avg_q", "reward", "ep_len"),
        )
        self.logger.writeheader()
        self.file_handler.flush()

        self.rewards_30_eps = deque(maxlen=30)
        self.episode_q_values = []
        self.episode_length_30_eps = deque(maxlen=30)
        self.episodes_counter = 0

        self.avg_episode_q_value = 0
        self.max_episode_q_value = 0
        self.std_episode_q_value = 0

        self.avg_reward_30_eps = 0
        self.max_reward_30_eps = 0
        self.std_reward_30_eps = 0

        self.avg_episode_length = 0

        self.avg_eval_reward = 0

        self.loss = 0
        self.grad_norm = 0

    def set_episode_results(self, reward: float, length: int) -> None:
        self.rewards_30_eps.append(reward)
        self.episode_length_30_eps.append(length)

        if len(self.episode_q_values) > 0:
            self.avg_episode_q_value = np.mean(self.episode_q_values)
            self.max_episode_q_value = np.max(self.episode_q_values)
            self.std_episode_q_value = np.std(self.episode_q_values)
            self.episode_q_values = []

        self.episodes_counter += 1

        episode_info_dict = {
            "avg_q": self.avg_episode_q_value,
            "reward": reward,
            "ep_len": length,
        }
        self.logger.writerow(episode_info_dict)
        self.file_handler.flush()

    def set_q_values(self, q_values: List) -> None:
        self.episode_q_values.append(q_values)

    def update_metrics(self) -> None:
        self.avg_reward_30_eps = np.mean(self.rewards_30_eps)
        self.max_reward_30_eps = np.max(self.rewards_30_eps)
        self.std_reward_30_eps = np.std(self.rewards_30_eps)

        if len(self.episode_length_30_eps) > 0:
            self.avg_episode_length = np.mean(self.episode_length_30_eps)

    def add_summary(self, t: int) -> None:
        self.summary_writer.add_scalar("Avg episode Q", self.avg_episode_q_value, t)
        self.summary_writer.add_scalar("Max episode Q", self.max_episode_q_value, t)
        self.summary_writer.add_scalar("Std episode Q", self.std_episode_q_value, t)

        self.summary_writer.add_scalar("Avg reward, 30 eps", self.avg_reward_30_eps, t)
        self.summary_writer.add_scalar("Max reward, 30 eps", self.max_reward_30_eps, t)
        self.summary_writer.add_scalar("Std reward, 30 eps", self.std_reward_30_eps, t)
        self.summary_writer.add_scalar("Episode reward", self.rewards_30_eps[-1], t)

        self.summary_writer.add_scalar("Avg evaluated reward", self.avg_eval_reward, t)

        self.summary_writer.add_scalar("Episodes played", self.episodes_counter, t)
        self.summary_writer.add_scalar(
            "Avg episode length, 30 eps", self.avg_episode_length, t
        )
        self.summary_writer.add_scalar(
            "Episode length", self.episode_length_30_eps[-1], t
        )

        self.summary_writer.add_scalar("Loss", self.loss, t)
        self.summary_writer.add_scalar("Gradients Norm", self.grad_norm, t)

    def update_bar(self, epsilon, t):
        if len(self.rewards_30_eps) > 0:
            self.bar.update(
                t + 1,
                exact=[
                    ("episodes", self.episodes_counter),
                    ("avg r", self.avg_reward_30_eps),
                    ("max r", self.max_reward_30_eps),
                    ("avg q", self.avg_episode_q_value),
                    ("eps", epsilon),
                    ("loss", self.loss),
                ],
            )

    def reset_bar(self):
        self.bar.reset_start()

    def close_file_handler(self):
        self.file_handler.close()
