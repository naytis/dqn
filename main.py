from core.trainer import Trainer
from utils.wrappers import make_env

from config import config

if __name__ == "__main__":
    env = make_env(config.env_name)
    trainer = Trainer(env)
    trainer.run()
