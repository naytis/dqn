import os
import sys
from collections import deque
from typing import Deque, Tuple, List

import gym
import numpy as np
import torch
from torch import optim, functional, Tensor
from torch.utils.tensorboard import SummaryWriter

from config import config
from core.agent import Agent, NetworkType
from core.schedule import ExplorationSchedule
from utils.logger import get_logger
from utils.progress_bar import ProgressBar
from utils.replay_buffer import ReplayBuffer
from utils.wrappers import make_evaluation_env


class Trainer:
    def __init__(
        self,
        env: gym.Env,
    ):
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        self.env = env
        self.logger = get_logger(config.log_path)

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

        self.summary_writer = SummaryWriter(config.output_path, max_queue=int(1e5))

        self.avg_reward = 0
        self.max_reward = 0
        self.std_reward = 0

        self.avg_q = 0
        self.max_q = 0
        self.std_q = 0

        self.eval_reward = 0

    def run(self) -> None:
        self.agent.synchronize_networks()
        self.record()
        self.train()
        self.record()

    def train(self) -> None:
        rewards = deque(maxlen=50)  # rewards for last 50 episodes
        max_q_values = deque(maxlen=1000)  # q values for last 1000 timesteps
        q_values_list = deque(maxlen=1000)  # q values for last 1000 timesteps
        episode_length = deque(maxlen=50)
        episodes_counter = 0

        t = last_eval = last_record = 0
        self.eval_reward = self.evaluate()

        bar = ProgressBar(target=config.num_steps_train)

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

                max_q_values.append(max(q_values))
                q_values_list += list(q_values)

                state, reward, done, info = self.env.step(action)
                self.replay_buffer.store_effect(frame_index, action, reward, done)

                loss_eval, grad_eval = self.train_step(t)

                if t > config.learning_start and t % config.learning_freq == 0:
                    self.exp_schedule.update_epsilon(t)

                if done: # or t >= config.num_steps_train: # todo
                    episode_reward = info["episode"]["r"]
                    episode_length.append(info["episode"]["l"])
                    episodes_counter += 1

                    if t > config.learning_start:
                        self.update_metrics(rewards, max_q_values, q_values_list)
                        self.add_summary(
                            latest_loss=loss_eval,
                            latest_total_norm=grad_eval,
                            episodes_counter=episodes_counter,
                            episode_length=episode_length,
                            t=t,
                        )
                        if len(rewards) > 0:
                            bar.update(
                                t + 1,
                                exact=[
                                    ("episodes", episodes_counter),
                                    ("avg r", self.avg_reward),
                                    ("max r", np.max(rewards)),
                                    ("max q", self.max_q),
                                    ("eps", self.exp_schedule.epsilon),
                                    ("loss", loss_eval),
                                    ("grads", grad_eval),
                                ],
                                base=config.learning_start,
                            )
                    elif t < config.learning_start:
                        sys.stdout.write(
                            "\rPopulating the memory {}/{}...".format(
                                t, config.learning_start
                            )
                        )
                        sys.stdout.flush()
                        bar.reset_start()
                    break

            rewards.append(episode_reward)

            if t > config.learning_start and last_eval > config.eval_freq:
                last_eval = 0
                self.eval_reward = self.evaluate()

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
                    episode_reward = info['episode']['r']
                    break

            rewards.append(episode_reward)

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
        env = make_evaluation_env(config.env_name)
        self.evaluate(env, 1)

    def update_metrics(
        self, rewards: Deque, max_q_values: Deque, q_values: Deque
    ) -> None:
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self.max_q = np.mean(max_q_values)
        self.avg_q = np.mean(q_values)
        self.std_q = np.sqrt(np.var(q_values) / len(q_values))

    def add_summary(
        self,
        latest_loss: int,
        latest_total_norm: int,
        episodes_counter: int,
        episode_length: Deque,
        t: int,
    ) -> None:
        self.summary_writer.add_scalar("Loss", latest_loss, t)
        self.summary_writer.add_scalar("Gradients Norm", latest_total_norm, t)
        self.summary_writer.add_scalar("Avg reward, 50 eps", self.avg_reward, t)
        self.summary_writer.add_scalar("Max reward, 50 eps", self.max_reward, t)
        self.summary_writer.add_scalar("Std reward", self.std_reward, t)
        self.summary_writer.add_scalar("Avg Q, 1000 ts", self.avg_q, t)
        self.summary_writer.add_scalar("Max Q, 1000 ts", self.max_q, t)
        self.summary_writer.add_scalar("Std Q", self.std_q, t)
        self.summary_writer.add_scalar("Avg evaluated reward", self.eval_reward, t)
        self.summary_writer.add_scalar("Episodes played", episodes_counter, t)
        self.summary_writer.add_scalar("Avg episodes length, 50 eps", np.mean(episode_length), t)

    def save_parameters(self) -> None:
        torch.save(
            self.agent.state_dict(),
            config.model_output,
        )
