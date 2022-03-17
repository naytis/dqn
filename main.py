import argparse
import os
from typing import Union

import gym

from config import NatureConfig, DefaultConfig, TestConfig
from core.trainer import Trainer
from utils.benchmark_monitor import BenchmarkMonitor
from utils.helpers import calculate_mean_and_ci
from utils.wrappers import make_env_for_record, make_env


def evaluate(env: Union[gym.Env, BenchmarkMonitor], num_episodes: int):
    rewards = []

    for i in range(num_episodes):
        env.reset()
        while True:
            action = env.action_space.sample()
            done = env.step(action)[2]

            if done:
                rewards.append(env.get_episode_reward())
                break

    avg_reward, reward_ci = calculate_mean_and_ci(rewards)

    msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, reward_ci)
    print(msg)


def evaluate_random_play(env_name: str):
    print("Running in env", env_name)
    record_path = "results/" + env_name + "/eval_random/"
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    print("Recording")
    env = make_env_for_record(env_name, record_path=record_path)
    evaluate(env, 1)

    print("Evaluating")
    env = make_env(env_name)
    evaluate(env, config.num_episodes_eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env",
        default=NatureConfig.default_env,
        help="Name of the environment, default=" + NatureConfig.default_env,
        choices=NatureConfig.available_envs,
    )
    parser.add_argument(
        "-c",
        "--config",
        default="default",
        help="Config and hyperparameters of the training model",
        choices=("default", "nature", "test"),
    )
    parser.add_argument(
        "-t",
        "--task",
        default="train",
        help="Script task: train agent, evaluate trained agent or evaluate random play",
        choices=("train", "eval_trained", "eval_random"),
    )
    args = parser.parse_args()

    if args.config == "default":
        config = DefaultConfig()
    elif args.config == "nature":
        config = NatureConfig()
    else:
        config = TestConfig()

    env_name = args.env + config.env_version

    if args.task == "eval_random":
        evaluate_random_play(env_name)
    else:
        trainer = Trainer(env_name, config)

        if args.task == "train":
            trainer.run()
        elif args.task == "eval_trained":
            trainer.evaluate_trained()
