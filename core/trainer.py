import os
import sys
from collections import deque
from typing import Deque, List, Type, Tuple

import gym
import numpy as np
import torch
from gym import Env as GymEnv
from torch.utils.tensorboard import SummaryWriter

from config import Config
from core.deep_q_network import DeepQNetwork
from core.schedule import ExplorationSchedule, LearningRateSchedule
from utils.general import ProgressBar, get_logger, export_plot
from utils.preprocess import greyscale
from utils.replay_buffer import ReplayBuffer
from utils.wrappers import MaxAndSkipEnv, PreproWrapper


class Trainer:
    def __init__(
        self,
        env: GymEnv,
        config: Type[Config],
        logger=None,
    ):
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)

        self.env = env
        self.dqn = DeepQNetwork(env, config)

        self.summary_writer = SummaryWriter(self.config.output_path, max_queue=1e5)

        self.avg_reward = 0
        self.max_reward = 0
        self.std_reward = 0

        self.avg_q = 0
        self.max_q = 0
        self.std_q = 0

        self.eval_reward = 0

    def run(
        self, exp_schedule: ExplorationSchedule, lr_schedule: LearningRateSchedule
    ) -> None:
        self.dqn.synchronize_networks()

        if self.config.record:
            self.record()

        self.train(exp_schedule, lr_schedule)

        if self.config.record:
            self.record()

    def train(
        self, exp_schedule: ExplorationSchedule, lr_schedule: LearningRateSchedule
    ) -> None:
        replay_buffer = ReplayBuffer(
            self.config.buffer_size, self.config.history_length
        )
        rewards = deque(maxlen=1000)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)

        t = last_eval = last_record = 0
        scores_eval = []
        scores_eval += [self.evaluate()]

        bar = ProgressBar(target=self.config.num_steps_train)

        while t < self.config.num_steps_train:
            episode_reward = 0
            state = self.env.reset()

            while True:
                t += 1
                last_eval += 1
                last_record += 1
                if self.config.render_train:
                    self.env.render()

                frame_index = replay_buffer.store_frame(state)
                dqn_input = replay_buffer.encode_recent_observation()

                best_action, state_action_values = self.dqn.get_best_action(dqn_input)
                action = exp_schedule.get_action(best_action)

                max_q_values.append(max(state_action_values))
                q_values += list(state_action_values)

                state, reward, done, info = self.env.step(action)
                replay_buffer.store_effect(frame_index, action, reward, done)

                loss_eval, grad_eval = self.train_step(
                    t, replay_buffer, lr_schedule.alpha
                )

                if (
                    t > self.config.learning_start
                    and t % self.config.learning_freq == 0
                ):
                    exp_schedule.update_epsilon(t)
                    lr_schedule.update_alpha(t)

                if t > self.config.learning_start and t % self.config.log_freq == 0:
                    self.update_averages(rewards, max_q_values, q_values, scores_eval)
                    self.add_summary(loss_eval, grad_eval, t)
                    if len(rewards) > 0:
                        bar.update(
                            t + 1,
                            exact=[
                                ("Avg R", self.avg_reward),
                                ("Max R", np.max(rewards)),
                                ("Max Q", self.max_q),
                                ("Epsilon", exp_schedule.epsilon),
                                ("Alpha", lr_schedule.alpha),
                                ("Loss", loss_eval),
                                ("Grads", grad_eval),
                            ],
                            base=self.config.learning_start,
                        )
                elif t < self.config.learning_start and t % self.config.log_freq == 0:
                    sys.stdout.write(
                        "\rPopulating the memory {}/{}...".format(
                            t, self.config.learning_start
                        )
                    )
                    sys.stdout.flush()
                    bar.reset_start()

                episode_reward += reward
                if done or t >= self.config.num_steps_train:
                    break

            rewards.append(episode_reward)

            if t > self.config.learning_start and last_eval > self.config.eval_freq:
                last_eval = 0
                scores_eval += [self.evaluate()]

            if (
                t > self.config.learning_start
                and self.config.record
                and last_record > self.config.record_freq
            ):
                last_record = 0
                self.record()

        self.logger.info("- Training done.")
        self.save_parameters()
        scores_eval += [self.evaluate()]
        export_plot(scores_eval, "Scores", self.config.plot_output)

    def train_step(
        self, t: int, replay_buffer: ReplayBuffer, alpha: float
    ) -> Tuple[int, int]:
        loss_eval, grad_eval = 0, 0

        if t > self.config.learning_start and t % self.config.learning_freq == 0:
            loss_eval, grad_eval = self.dqn.update_params(replay_buffer, alpha)

        if t % self.config.target_update_freq == 0:
            self.dqn.synchronize_networks()

        if t % self.config.saving_freq == 0:
            self.save_parameters()

        return loss_eval, grad_eval

    def evaluate(self, env=None, num_episodes=None) -> float:
        if num_episodes is None:
            self.logger.info("\nEvaluating...")
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env

        replay_buffer = ReplayBuffer(
            self.config.buffer_size, self.config.history_length
        )
        rewards = []

        for i in range(num_episodes):
            total_reward = 0
            state = env.reset()
            while True:
                if self.config.render_test:
                    env.render()

                index = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                action = self.dqn.get_action(q_input)

                state, reward, done, info = env.step(action)
                replay_buffer.store_effect(index, action, reward, done)

                total_reward += reward
                if done:
                    break

            rewards.append(total_reward)

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
                avg_reward, sigma_reward
            )
            self.logger.info(msg)

        return avg_reward

    def record(self) -> None:
        self.logger.info("Recording...")
        env = gym.make(self.config.env_name)
        env = gym.wrappers.Monitor(
            env,
            self.config.record_path,
            resume=True,
        )
        env = MaxAndSkipEnv(env, skip=self.config.skip_frame)
        env = PreproWrapper(
            env,
            prepro=greyscale,
            shape=(80, 80, 1),
            overwrite_render=self.config.overwrite_render,
        )
        self.evaluate(env, 1)

    def update_averages(
        self, rewards: Deque, max_q_values: Deque, q_values: Deque, scores_eval: List
    ) -> None:
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self.max_q = np.mean(max_q_values)
        self.avg_q = np.mean(q_values)
        self.std_q = np.sqrt(np.var(q_values) / len(q_values))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def add_summary(self, latest_loss: int, latest_total_norm: int, t: int) -> None:
        self.summary_writer.add_scalar("Loss", latest_loss, t)
        self.summary_writer.add_scalar("Gradients Norm", latest_total_norm, t)
        self.summary_writer.add_scalar("Avg Reward", self.avg_reward, t)
        self.summary_writer.add_scalar("Max Reward", self.max_reward, t)
        self.summary_writer.add_scalar("Std Reward", self.std_reward, t)
        self.summary_writer.add_scalar("Avg Q", self.avg_q, t)
        self.summary_writer.add_scalar("Max Q", self.max_q, t)
        self.summary_writer.add_scalar("Std Q", self.std_q, t)
        self.summary_writer.add_scalar("Evaluated Reward", self.eval_reward, t)

    def save_parameters(self) -> None:
        torch.save(
            self.dqn.get_parameters(),
            self.config.model_output,
        )
