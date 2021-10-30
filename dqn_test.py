from core.model import Model
from utils.test_env import EnvTest
from core.schedule import LinearExploration, LinearSchedule

from configs.dqn_test import config


if __name__ == "__main__":
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(
        env, config.eps_begin, config.eps_end, config.eps_nsteps
    )

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

    # train model
    model = Model(env, config)
    model.run(exp_schedule, lr_schedule)
