class BaseConfig(object):
    env_name = "Pong-v0"

    # output config
    output_path = "results/" + env_name + "/"
    model_output = output_path + "model.weights"
    log_path = output_path + "log.txt"
    record_path = output_path + "monitor/"

    # model and training config
    num_episodes_test = 50
    clip_val = 10
    saving_freq = 500000
    eval_freq = 250000
    record_freq = 500000
    soft_epsilon = 0.05

    # hyper params
    num_steps_train = 6000000
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
    learning_start = 50000


class DevConfig(BaseConfig):
    num_steps_train = 5000000
    buffer_size = 100000
    learning_start = 100000
    target_update_freq = 1000
    epsilon_interp_limit = 500000


config = DevConfig
