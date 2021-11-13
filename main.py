import argparse

import gym

from core.trainer import Trainer
from utils.preprocess import greyscale
from utils.test_env import EnvTest
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from core.schedule import LinearExploration, LinearLearningRate

from configs.config import Config
from configs.test_config import TestConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implementation of DQN algorithm.")
    parser.add_argument(
        "-t",
        "--test_env",
        action="store_true",
        help="Run DQN in a test environment",
    )
    args = parser.parse_args()

    if args.test_env:
        print("Running in a test environment")
        config = TestConfig

        env = EnvTest((8, 8, 6))

    else:
        print("Running in Atari")
        config = Config

        env = gym.make(config.env_name)
        env = MaxAndSkipEnv(env, skip=config.skip_frame)
        env = PreproWrapper(
            env,
            prepro=greyscale,
            shape=(80, 80, 1),
            overwrite_render=config.overwrite_render,
        )

    exp_schedule = LinearExploration(
        env, config.epsilon_init, config.epsilon_end, config.epsilon_interp_limit
    )

    lr_schedule = LinearLearningRate(
        config.alpha_init, config.alpha_end, config.alpha_interp_limit
    )

    trainer = Trainer(env, config)
    trainer.run(exp_schedule, lr_schedule)
