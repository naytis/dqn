class Config(object):
    env_name = "Pong-v0"
    high = 255.0

    # output config
    output_path = "results/" + env_name + "/"
    model_output = output_path + "model.weights"
    log_path = output_path + "log.txt"
    record_path = output_path + "monitor/"

    # model and training config
    num_episodes_test = 50
    clip_val = 10
    saving_freq = 500000
    log_freq = 250
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


config = Config
