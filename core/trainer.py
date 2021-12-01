import os
import sys
from typing import Tuple, Union

import gym
import numpy as np
import torch
from torch import optim, functional, Tensor

from config import config
from core.agent import Agent, NetworkType
from utils.benchmark_monitor import BenchmarkMonitor
from utils.metrics import Metrics
from core.schedule import ExplorationSchedule
from utils.logger import get_logger
from utils.replay_buffer import ReplayBuffer
from utils.wrappers import make_evaluation_env


class Trainer:
    def __init__(
        self,
        env: Union[gym.Env, BenchmarkMonitor],
    ):
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        self.env = env
        self.logger = get_logger(config.log_path)
        self.metrics = Metrics()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Running model on device {self.device}")

        self.agent = Agent(env, self.device)

        self.replay_buffer = ReplayBuffer(config.buffer_size, config.history_length)

        self.exp_schedule = ExplorationSchedule(
            config.epsilon_init, config.epsilon_final, config.epsilon_interp_limit
        )

        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=config.learning_rate,
        )

    def run(self) -> None:
        self.agent.synchronize_networks()
        self.record()
        self.train()
        self.record()

    def train(self) -> None:
        t = last_eval = last_record = 0
        self.metrics.eval_reward = self.evaluate()

        while t < config.num_steps_train:
            state = self.env.reset()

            while True:
                t += 1
                last_eval += 1
                last_record += 1

                frame_index = self.replay_buffer.store_frame(state)
                dqn_input = self.replay_buffer.encode_recent_observation()

                best_action, q_values = self.agent.get_best_action_and_q_values(
                    dqn_input
                )
                action = self.agent.get_action(dqn_input, self.exp_schedule.epsilon)

                self.metrics.max_q_values.append(max(q_values))
                self.metrics.q_values_deque += list(q_values)

                state, reward, done, info = self.env.step(action)
                self.replay_buffer.store_effect(frame_index, action, reward, done)

                self.metrics.loss_eval, self.metrics.grad_eval = self.train_step(t)

                if t > config.learning_start and t % config.learning_freq == 0:
                    self.exp_schedule.update_epsilon(t)

                if done:  # or t >= config.num_steps_train: # todo
                    episode_reward = self.env.get_episode_rewards()
                    self.metrics.episode_length.append(self.env.get_episode_lengths())
                    self.metrics.episodes_counter += 1

                    if t > config.learning_start:
                        self.metrics.update_metrics()
                        self.metrics.add_summary(t)
                        self.metrics.update_bar(self.exp_schedule.epsilon, t)
                    elif t < config.learning_start:
                        sys.stdout.write(
                            "\rPopulating the memory {}/{}...".format(
                                t, config.learning_start
                            )
                        )
                        sys.stdout.flush()
                        self.metrics.reset_bar()
                    break

            self.metrics.rewards.append(episode_reward)

            if t > config.learning_start and last_eval > config.eval_freq:
                last_eval = 0
                self.metrics.eval_reward = self.evaluate()

            if t > config.learning_start and last_record > config.record_freq:
                last_record = 0
                self.record()

        self.logger.info("- Training done.")
        self.save_parameters()

    def train_step(self, t: int) -> Tuple[int, int]:
        loss_eval, grad_eval = 0, 0

        if t > config.learning_start and t % config.learning_freq == 0:
            loss_eval, grad_eval = self.update_params()

        if t % config.target_update_freq == 0:
            self.agent.synchronize_networks()

        if t % config.saving_freq == 0:
            self.save_parameters()

        return loss_eval, grad_eval

    def update_params(self) -> Tuple[int, int]:
        (
            s_batch,
            a_batch,
            r_batch,
            sp_batch,
            done_mask_batch,
        ) = self.replay_buffer.sample(config.batch_size)

        s_batch = torch.tensor(s_batch, dtype=torch.uint8, device=self.device)
        a_batch = torch.tensor(a_batch, dtype=torch.uint8, device=self.device)
        r_batch = torch.tensor(r_batch, dtype=torch.float, device=self.device)
        sp_batch = torch.tensor(sp_batch, dtype=torch.uint8, device=self.device)
        done_mask_batch = torch.tensor(
            done_mask_batch, dtype=torch.bool, device=self.device
        )

        self.optimizer.zero_grad()

        q_values = self.agent.get_q_values(s_batch, NetworkType.Q_NETWORK)
        with torch.no_grad():
            target_q_values = self.agent.get_q_values(
                sp_batch, NetworkType.TARGET_NETWORK
            )

        loss = self.calculate_loss(
            q_values, target_q_values, a_batch, r_batch, done_mask_batch
        )
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.parameters(), config.clip_val
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
            * config.gamma
            * torch.max(target_q_values, dim=1).values  # todo done_mask -1?
        )
        q_current = q_values[range(len(q_values)), actions.type(torch.LongTensor)]
        return functional.F.huber_loss(q_target, q_current)

    def evaluate(self, env=None, num_episodes=None) -> float:
        if num_episodes is None:
            self.logger.info("Evaluating...")
            num_episodes = config.num_episodes_test

        if env is None:
            env = self.env

        evaluation_replay_buffer = ReplayBuffer(
            config.buffer_size, config.history_length
        )
        rewards = []

        for i in range(num_episodes):
            state = env.reset()
            while True:
                index = evaluation_replay_buffer.store_frame(state)
                q_input = evaluation_replay_buffer.encode_recent_observation()

                action = self.agent.get_action(q_input, config.soft_epsilon)

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
        env = make_evaluation_env(config.env_name)
        self.evaluate(env, 1)

    def save_parameters(self) -> None:
        torch.save(
            self.agent.state_dict(),
            config.model_output,
        )
