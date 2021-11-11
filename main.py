import argparse

import gym

from core.trainer import Trainer
from utils.preprocess import greyscale
from utils.test_env import EnvTest
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from core.schedule import LinearExploration, LinearSchedule

from configs.config import config as main_config
from configs.test_config import config as test_config

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
        config = test_config

        env = EnvTest((8, 8, 6))

    else:
        print("Running in Atari")
        config = main_config

        # make env
        env = gym.make(config.env_name)
        env = MaxAndSkipEnv(env, skip=config.skip_frame)
        env = PreproWrapper(
            env,
            prepro=greyscale,
            shape=(80, 80, 1),
            overwrite_render=config.overwrite_render,
        )
    print("Ignore args")
    # exploration strategy
    exp_schedule = LinearExploration(
        env, config.eps_begin, config.eps_end, config.eps_nsteps
    )

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

    # train model
    model = Trainer(env, config)
    model.run(exp_schedule, lr_schedule)
