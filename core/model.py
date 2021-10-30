import os
import sys
from collections import deque

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from core.deep_q_network import DeepQNetwork
from core.timer import Timer
from utils.general import Progbar, get_logger, export_plot
from utils.preprocess import greyscale
from utils.replay_buffer import ReplayBuffer
from utils.wrappers import MaxAndSkipEnv, PreproWrapper


class Model:
    def __init__(self, env, config, logger=None):
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env
        self.timer = Timer(False)
        self.dqn = DeepQNetwork(env, config, self.timer)

        self.summary_writer = SummaryWriter(self.config.output_path, max_queue=1e5)

    def run(self, exp_schedule, lr_schedule):
        self.dqn.synchronize_networks()

        # record one game at the beginning
        if self.config.record:
            self.record()

        # model
        self.train(exp_schedule, lr_schedule)

        # record one game at the end
        if self.config.record:
            self.record()

    def init_averages(self):
        """
        Defines extra attributes for tensorboard
        """
        self.avg_reward = -21.0
        self.max_reward = -21.0
        self.std_reward = 0

        self.avg_q = 0
        self.max_q = 0
        self.std_q = 0

        self.eval_reward = -21.0

    def update_averages(self, rewards, max_q_values, q_values, scores_eval):
        """
        Args:
            rewards: deque
            max_q_values: deque
            q_values: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self.max_q = np.mean(max_q_values)
        self.avg_q = np.mean(q_values)
        self.std_q = np.sqrt(np.var(q_values) / len(q_values))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def train(self, exp_schedule, lr_schedule):
        """
        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

        # initialize replay buffer and variables
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)
        self.init_averages()

        t = last_eval = last_record = 0  # time control of nb of steps
        scores_eval = []  # list of scores computed at iteration time
        scores_eval += [self.evaluate()]

        prog = Progbar(target=self.config.nsteps_train)

        # interact with environment
        while t < self.config.nsteps_train:
            total_reward = 0
            self.timer.start("env.reset")
            state = self.env.reset()
            self.timer.end("env.reset")
            while True:
                t += 1
                last_eval += 1
                last_record += 1
                if self.config.render_train:
                    self.env.render()
                # replay memory stuff
                self.timer.start("replay_buffer.store_encode")
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()
                self.timer.end("replay_buffer.store_encode")

                # chose action according to current Q and exploration
                self.timer.start("get_action")
                best_action, q_vals = self.dqn.get_best_action(q_input)
                action = exp_schedule.get_action(best_action)
                self.timer.end("get_action")

                # store q values
                max_q_values.append(max(q_vals))
                q_values += list(q_vals)

                # perform action in env
                self.timer.start("env.step")
                new_state, reward, done, info = self.env.step(action)
                self.timer.end("env.step")

                # store the transition
                self.timer.start("replay_buffer.store_effect")
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state
                self.timer.end("replay_buffer.store_effect")

                # perform a training step
                self.timer.start("train_step")
                loss_eval, grad_eval = self.train_step(
                    t, replay_buffer, lr_schedule.epsilon
                )
                self.timer.end("train_step")

                # logging stuff
                if (
                    (t > self.config.learning_start)
                    and (t % self.config.log_freq == 0)
                    and (t % self.config.learning_freq == 0)
                ):
                    self.timer.start("logging")
                    self.update_averages(rewards, max_q_values, q_values, scores_eval)
                    self.add_summary(loss_eval, grad_eval, t)
                    exp_schedule.update_epsilon(t)
                    lr_schedule.update_epsilon(t)
                    if len(rewards) > 0:
                        prog.update(
                            t + 1,
                            exact=[
                                ("Loss", loss_eval),
                                ("Avg_R", self.avg_reward),
                                ("Max_R", np.max(rewards)),
                                ("eps", exp_schedule.epsilon),
                                ("Grads", grad_eval),
                                ("Max_Q", self.max_q),
                                ("lr", lr_schedule.epsilon),
                            ],
                            base=self.config.learning_start,
                        )
                    self.timer.end("logging")
                elif (t < self.config.learning_start) and (
                    t % self.config.log_freq == 0
                ):
                    sys.stdout.write(
                        "\rPopulating the memory {}/{}...".format(
                            t, self.config.learning_start
                        )
                    )
                    sys.stdout.flush()
                    prog.reset_start()

                # count reward
                total_reward += reward
                if done or t >= self.config.nsteps_train:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                self.timer.start("eval")
                scores_eval += [self.evaluate()]
                self.timer.end("eval")
                self.timer.print_stat()
                self.timer.reset_stat()

            if (
                (t > self.config.learning_start)
                and self.config.record
                and (last_record > self.config.record_freq)
            ):
                self.logger.info("Recording...")
                last_record = 0
                self.timer.start("recording")
                self.record()
                self.timer.end("recording")

        # last words
        self.logger.info("- Training done.")
        self.save_parameters()
        scores_eval += [self.evaluate()]
        export_plot(scores_eval, "Scores", self.config.plot_output)

    def train_step(self, t, replay_buffer, lr):
        """
        Perform training step

        Args:
            t: (int) nths step
            replay_buffer: buffer for sampling
            lr: (float) learning rate
        """
        loss_eval, grad_eval = 0, 0

        # perform training step
        if t > self.config.learning_start and t % self.config.learning_freq == 0:
            self.timer.start("train_step/update_step")
            loss_eval, grad_eval = self.dqn.update_step(t, replay_buffer, lr)
            self.timer.end("train_step/update_step")

        # occasionally update target network with q network
        if t % self.config.target_update_freq == 0:
            self.timer.start("train_step/synchronize_networks")
            self.dqn.synchronize_networks()
            self.timer.end("train_step/synchronize_networks")

        # occasionally save the weights
        if t % self.config.saving_freq == 0:
            self.timer.start("train_step/save")
            self.save_parameters()
            self.timer.end("train_step/save")

        return loss_eval, grad_eval

    def evaluate(self, env=None, num_episodes=None):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        if num_episodes is None:
            self.logger.info("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env

        # replay memory to play
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = []

        for i in range(num_episodes):
            total_reward = 0
            state = env.reset()
            while True:
                if self.config.render_test:
                    env.render()

                # store last state in buffer
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                action = self.dqn.get_action(q_input)

                # perform action in env
                new_state, reward, done, info = env.step(action)

                # store in replay memory
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # count reward
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
                avg_reward, sigma_reward
            )
            self.logger.info(msg)

        return avg_reward

    def record(self):
        """
        Re create an env and record a video for one episode
        """
        env = gym.make(self.config.env_name)
        env = gym.wrappers.Monitor(
            env,
            self.config.record_path,
            resume=True,  # , video_callable=lambda x: True,
        )
        env = MaxAndSkipEnv(env, skip=self.config.skip_frame)
        env = PreproWrapper(
            env,
            prepro=greyscale,
            shape=(80, 80, 1),
            overwrite_render=self.config.overwrite_render,
        )
        self.evaluate(env, 1)

    def add_summary(self, latest_loss, latest_total_norm, t):
        """
        Tensorboard stuff
        """
        self.summary_writer.add_scalar("loss", latest_loss, t)
        self.summary_writer.add_scalar("grad_norm", latest_total_norm, t)
        self.summary_writer.add_scalar("Avg_Reward", self.avg_reward, t)
        self.summary_writer.add_scalar("Max_Reward", self.max_reward, t)
        self.summary_writer.add_scalar("Std_Reward", self.std_reward, t)
        self.summary_writer.add_scalar("Avg_Q", self.avg_q, t)
        self.summary_writer.add_scalar("Max_Q", self.max_q, t)
        self.summary_writer.add_scalar("Std_Q", self.std_q, t)
        self.summary_writer.add_scalar("Eval_Reward", self.eval_reward, t)

    def save_parameters(self):
        torch.save(self.dqn.get_parameters(), self.config.model_output)
