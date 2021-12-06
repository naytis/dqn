import os
import sys
from typing import Tuple, Union

import numpy as np
import torch
from torch import optim, functional, Tensor

from config import DefaultConfig, NatureConfig, TestConfig
from core.agent import Agent, NetworkType
from utils.metrics import Metrics
from core.schedule import ExplorationSchedule
from utils.logger import get_logger
from utils.replay_buffer import ReplayBuffer
from utils.wrappers import make_evaluation_env, make_env


class Trainer:
    def __init__(
        self, env_name: str, config: Union[DefaultConfig, NatureConfig, TestConfig]
    ):
        output_path = "results/" + env_name + "/"
        log_path = output_path + "log.txt"
        self.model_output = output_path + "model.weights"
        self.record_path = output_path + "videos/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.env_name = env_name
        self.env = make_env(env_name)
        self.config = config
        self.logger = get_logger(log_path)
        self.metrics = Metrics(
            self.config.num_steps_train, self.config.learning_start, output_path
        )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Running model on device {self.device}")

        self.agent = Agent(self.env, self.config, self.device)

        self.replay_buffer = ReplayBuffer(
            self.config.buffer_size, self.config.history_length
        )

        self.exp_schedule = ExplorationSchedule(
            self.config.epsilon_init,
            self.config.epsilon_final,
            self.config.epsilon_interp_limit,
        )

        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=self.config.learning_rate,
        )

    def run(self) -> None:
        self.agent.synchronize_networks()

        if self.config.record:
            self.record()

        self.train()

        if self.config.record:
            self.record()

    def train(self) -> None:
        t = last_eval = last_record = 0
        self.metrics.avg_eval_reward = self.evaluate()

        while t < self.config.num_steps_train:
            state = self.env.reset()

            while True:
                t += 1
                last_eval += 1
                last_record += 1

                frame_index = self.replay_buffer.store_frame(state)
                dqn_input = self.replay_buffer.encode_recent_observation()

                action = self.agent.get_action(dqn_input, self.exp_schedule.epsilon)

                state, reward, done, info = self.env.step(action)
                self.replay_buffer.store_effect(frame_index, action, reward, done)

                self.metrics.loss, self.metrics.grad_norm = self.train_step(t)

                if (
                    t > self.config.learning_start
                    and t % self.config.learning_freq == 0
                ):
                    self.exp_schedule.update_epsilon(t)

                if done:  # or t >= config.num_steps_train: # todo
                    self.metrics.rewards.append(self.env.get_episode_rewards())
                    self.metrics.episode_length.append(self.env.get_episode_lengths())
                    self.metrics.episodes_counter += 1

                    if t > self.config.learning_start:
                        self.metrics.update_metrics()
                        self.metrics.add_summary(t)
                        self.metrics.update_bar(self.exp_schedule.epsilon, t)
                    elif t < self.config.learning_start:
                        sys.stdout.write(
                            "\rPopulating the memory {}/{}...".format(
                                t, self.config.learning_start
                            )
                        )
                        sys.stdout.flush()
                        self.metrics.reset_bar()
                    break

            if t > self.config.learning_start and last_eval > self.config.eval_freq:
                last_eval = 0
                self.metrics.avg_eval_reward = self.evaluate()

            if (
                t > self.config.learning_start
                and last_record > self.config.record_freq
                and self.config.record
            ):
                last_record = 0
                self.record()

        self.logger.info("- Training done.")

        if self.config.save_parameters:
            self.save_parameters()

    def train_step(self, t: int) -> Tuple[int, int]:
        loss_eval, grad_eval = 0, 0

        if t > self.config.learning_start and t % self.config.learning_freq == 0:
            loss_eval, grad_eval = self.update_params()

        if t % self.config.target_update_freq == 0:
            self.agent.synchronize_networks()

        if t % self.config.saving_freq == 0 and self.config.save_parameters:
            self.save_parameters()

        return loss_eval, grad_eval

    def update_params(self) -> Tuple[int, int]:
        (
            s_batch,
            a_batch,
            r_batch,
            sp_batch,
            done_mask_batch,
        ) = self.replay_buffer.sample(self.config.batch_size)

        s_batch = torch.tensor(s_batch, dtype=torch.uint8, device=self.device)
        a_batch = torch.tensor(a_batch, dtype=torch.uint8, device=self.device)
        r_batch = torch.tensor(r_batch, dtype=torch.float, device=self.device)
        sp_batch = torch.tensor(sp_batch, dtype=torch.uint8, device=self.device)
        done_mask_batch = torch.tensor(
            done_mask_batch, dtype=torch.bool, device=self.device
        )

        self.optimizer.zero_grad()

        q_values = self.agent.get_q_values(s_batch, NetworkType.Q_NETWORK)

        q_values_list = q_values.squeeze().to("cpu").tolist()
        self.metrics.max_q_values.append(max(q_values_list))
        self.metrics.q_values_deque += list(q_values_list)

        with torch.no_grad():
            target_q_values = self.agent.get_q_values(
                sp_batch, NetworkType.TARGET_NETWORK
            )

        loss = self.calculate_loss(
            q_values, target_q_values, a_batch, r_batch, done_mask_batch
        )
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.parameters(), self.config.clip_val
        )
        self.optimizer.step()

        return loss.item(), total_norm.item()

    def calculate_loss(
        self,
        q_values: Tensor,
        target_q_values: Tensor,
        actions: Tensor,
        rewards: Tensor,
        done_mask: Tensor,
    ) -> Tensor:
        q_target = (
            rewards
            + (~done_mask)
            * self.config.gamma
            * torch.max(target_q_values, dim=1).values
        )
        q_current = q_values[range(len(q_values)), actions.type(torch.LongTensor)]
        return functional.F.huber_loss(q_target, q_current)

    def evaluate(self, env=None, num_episodes=None) -> float:
        if num_episodes is None:
            self.logger.info("Evaluating...")
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env

        evaluation_replay_buffer = ReplayBuffer(
            self.config.buffer_size, self.config.history_length
        )
        rewards = []

        for i in range(num_episodes):
            state = env.reset()
            while True:
                index = evaluation_replay_buffer.store_frame(state)
                q_input = evaluation_replay_buffer.encode_recent_observation()

                action = self.agent.get_action(q_input, self.config.soft_epsilon)

                state, reward, done, info = env.step(action)
                evaluation_replay_buffer.store_effect(index, action, reward, done)

                if done:
                    episode_reward = self.env.get_episode_rewards()
                    break

            rewards.append(episode_reward)

        avg_reward = np.mean(rewards)
        sigma_reward = np.std(rewards)

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
                avg_reward, sigma_reward
            )
            self.logger.info(msg)

        return avg_reward

    def record(self) -> None:
        self.logger.info("Recording...")
        env = make_evaluation_env(self.env_name, record_path=self.record_path)
        self.evaluate(env, 1)

    def save_parameters(self) -> None:
        torch.save(
            self.agent.state_dict(),
            self.model_output,
        )
