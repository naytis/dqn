import argparse

from core.trainer import Trainer

from config import NatureConfig, DefaultConfig, TestConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e" "--env",
        default=NatureConfig.default_env,
        help="Name of the environment, default=" + NatureConfig.default_env,
        choices=NatureConfig.available_envs,
    )
    parser.add_argument(
        "-c" "--config",
        default="default",
        help="Config and hyperparameters of the training model",
        choices=("default", "nature", "test"),
    )
    args = parser.parse_args()

    if args.config == "default":
        config = DefaultConfig()
    elif args.config == "nature":
        config = NatureConfig()
    else:
        config = TestConfig()

    trainer = Trainer(
        env_name=args.env,
        config=config,
    )
    trainer.run()
