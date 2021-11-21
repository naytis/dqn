class Config(object):
    # env config
    render_train = False
    render_test = False
    env_name = "DemonAttack-v0"
    overwrite_render = True
    record = True
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

    # nature paper hyper params
    num_steps_train = 10000000
    batch_size = 32
    buffer_size = 1000000
    target_update_freq = 10000
    gamma = 0.99
    learning_freq = 4
    history_length = 4
    skip_frame = 4
    lr_init = 0.00025
    lr_final = 0.00005
    lr_interp_limit = 5000000
    epsilon_init = 1
    epsilon_final = 0.1
    epsilon_interp_limit = 1000000
    learning_start = 50000
