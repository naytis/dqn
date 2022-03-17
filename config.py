class BaseConfig(object):
    default_env = "Pong"
    available_envs = (
        "Pong",
        "Breakout",
        "Boxing",
        "RoadRunner",
        "Enduro",
        "DemonAttack",
        "Assault",
    )
    env_version = "NoFrameskip-v4"
    record = True
    save_parameters = True

    # model and training config
    num_episodes_eval = 10
    clip_val = 10
    saving_freq = 500000
    eval_freq = 250000
    record_freq = 500000


class NatureConfig(BaseConfig):
    num_steps_train = 10000000
    batch_size = 32
    buffer_size = 1000000
    history_length = 4
    target_update_freq = 10000
    gamma = 0.99
    skip_frame = 4
    learning_freq = 4
    learning_rate = 0.0001
    epsilon_init = 1
    epsilon_final = 0.1
    epsilon_interp_limit = 1000000
    soft_epsilon = 0.05
    learning_start = 50000


class DefaultConfig(NatureConfig):
    num_steps_train = 5000000
    buffer_size = 10000
    target_update_freq = 1000
    epsilon_final = 0.02
    soft_epsilon = 0.01
    epsilon_interp_limit = 500000
    learning_start = 10000


class TestConfig(NatureConfig):
    record = False
    save_parameters = False

    # model and training config
    num_episodes_eval = 5
    eval_freq = 10000
    soft_epsilon = 0.01

    # hyper params
    num_steps_train = 400000
    batch_size = 32
    buffer_size = 10000
    target_update_freq = 1000
    epsilon_final = 0.01
    epsilon_interp_limit = 150000
    learning_start = 10000
