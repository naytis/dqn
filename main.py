from core.trainer import Trainer
from utils.wrappers import make_env

from core.schedule import ExplorationSchedule

from config import config

if __name__ == "__main__":
    env = make_env(config.env_name)

    exp_schedule = ExplorationSchedule(
        env, config.epsilon_init, config.epsilon_final, config.epsilon_interp_limit
    )

    trainer = Trainer(env, config)
    trainer.run(exp_schedule)
