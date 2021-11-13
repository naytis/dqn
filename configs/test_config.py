from configs.config import Config


class TestConfig(Config):
    # env config
    record = False

    # output config
    output_path = "results/test/"

    # model and training config
    num_episodes_test = 20
    saving_freq = 5000
    log_freq = 50
    eval_freq = 100
    soft_epsilon = 0

    # hyper params
    num_steps_train = 2000
    buffer_size = 500
    target_update_freq = 500
    alpha_init = 0.00025
    alpha_end = 0.0001
    alpha_interp_limit = num_steps_train / 2
    epsilon_init = 1
    epsilon_end = 0.01
    epsilon_interp_limit = num_steps_train / 2
    learning_start = 200
