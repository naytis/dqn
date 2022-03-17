import os
import sys
from pathlib import Path
from typing import Tuple, Union

import torch
from torch import optim, functional, Tensor

from config import DefaultConfig, NatureConfig, TestConfig
from core.agent import Agent, NetworkType
from core.schedule import ExplorationSchedule
from utils.helpers import calculate_mean_and_ci
from utils.logger import get_logger
from utils.metrics import Metrics
from utils.replay_buffer import ReplayBuffer
from utils.wrappers import make_env_for_record, make_env


class Trainer:
    def __init__(
        self, env_name: str, config: Union[DefaultConfig, NatureConfig, TestConfig]
    ):
        self.output_path = "results/" + env_name + "/"
        self.model_output = self.output_path + "model.weights"
        self.record_path = self.output_path + "videos/"

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.env_name = env_name
        self.env = make_env(env_name)
        print("Running in env", env_name)

        self.config = config
        self.logger = get_logger(filename=self.output_path + "log.txt")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Running model on device {self.device}")

        self.agent = Agent(self.env, self.config, self.device)

        self.metrics = None
        self.replay_buffer = None
        self.exp_schedule = None
        self.optimizer = None

    def run(self) -> None:
        self._setup()
        self.agent.synchronize_networks()

        if self.config.record:
            self._record()

        self._train()

        if self.config.record:
            self._record()

    def _setup(self) -> None:  # todo change name
        self.metrics = Metrics(
            self.config.num_steps_train, self.config.learning_start, self.output_path
        )
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

    def _train(self) -> None:
        t = last_eval = last_record = 0
        self.metrics.avg_eval_reward = self._evaluate()

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

                self.metrics.loss, self.metrics.grad_norm = self._update(t)

                if (
                    t > self.config.learning_start
                    and t % self.config.learning_freq == 0
                ):
                    self.exp_schedule.update_epsilon(t)

                if done:
                    self.metrics.set_episode_results(
                        reward=self.env.get_episode_reward(),
                        length=self.env.get_episode_length(),
                    )

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
                self.metrics.avg_eval_reward = self._evaluate()

            if (
                t > self.config.learning_start
                and last_record > self.config.record_freq
                and self.config.record
            ):
                last_record = 0
                self._record()

        self.logger.info("- Training done.")
        self.metrics.close_file_handler()

        if self.config.save_parameters:
            self._save_parameters()

    def _update(self, t: int) -> Tuple[int, int]:
        loss_eval, grad_eval = 0, 0

        if t > self.config.learning_start and t % self.config.learning_freq == 0:
            loss_eval, grad_eval = self._update_params()

        if t % self.config.target_update_freq == 0:
            self.agent.synchronize_networks()

        if t % self.config.saving_freq == 0 and self.config.save_parameters:
            self._save_parameters()

        return loss_eval, grad_eval

    def _update_params(self) -> Tuple[int, int]:
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

        self.metrics.set_q_values(q_values.squeeze().to("cpu").tolist())

        with torch.no_grad():
            target_q_values = self.agent.get_q_values(
                sp_batch, NetworkType.TARGET_NETWORK
            )

        loss = self._calculate_loss(
            q_values, target_q_values, a_batch, r_batch, done_mask_batch
        )
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.parameters(), self.config.clip_val
        )
        self.optimizer.step()

        return loss.item(), total_norm.item()

    def _calculate_loss(
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
        return functional.F.mse_loss(q_target, q_current)

    def evaluate_trained(self):
        parameters_path = Path(self.model_output)
        assert parameters_path.is_file(), f"Trained parameters not found"
        self.agent.load_state_dict(parameters_path)

        if self.config.record:
           self._record(record_path=self.output_path + "eval_trained/")
        self._evaluate()

    def _evaluate(self, env=None, num_episodes=None) -> float:
        if num_episodes is None:
            self.logger.info("Evaluating")
            num_episodes = self.config.num_episodes_eval

        if env is None:
            env = self.env

        evaluation_replay_buffer = ReplayBuffer(
            self.config.buffer_size, self.config.history_length
        )
        rewards = []

        for i in range(num_episodes):
            sys.stdout.write("\rEpisode #{}".format(i + 1))
            sys.stdout.flush()

            state = env.reset()
            while True:
                index = evaluation_replay_buffer.store_frame(state)
                q_input = evaluation_replay_buffer.encode_recent_observation()

                action = self.agent.get_action(q_input, self.config.soft_epsilon)

                state, reward, done, info = env.step(action)
                evaluation_replay_buffer.store_effect(index, action, reward, done)

                if done:
                    rewards.append(self.env.get_episode_reward())
                    break

        avg_reward, reward_ci = calculate_mean_and_ci(rewards)

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, reward_ci)
            self.logger.info(msg)

        return avg_reward

    def _record(self, record_path=None) -> None:
        self.logger.info("Recording")
        if record_path is None:
            record_path = self.record_path
        env = make_env_for_record(self.env_name, record_path)
        self._evaluate(env, 1)

    def _save_parameters(self) -> None:
        torch.save(
            self.agent.state_dict(),
            self.model_output,
        )
