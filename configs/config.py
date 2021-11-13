class Config:
    # env config
    render_train = False
    render_test = False
    env_name = "DemonAttack-v0"
    overwrite_render = True
    record = True
    high = 255.0

    # output config
    output_path = "results/" + env_name + "/"
    model_weights_path = "weights/" + env_name + "/model.weights"
    model_output = output_path + "model.weights"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"
    record_path = output_path + "monitor/"

    # model and training config
    num_episodes_test = 50
    clip_val = 10
    saving_freq = 500e3
    log_freq = 250
    eval_freq = 250e3
    record_freq = 500e3
    soft_epsilon = 0.05

    # nature paper hyper params
    num_steps_train = 10e6
    batch_size = 32
    buffer_size = 1e6
    target_update_freq = 10e3
    gamma = 0.99
    learning_freq = 4
    history_length = 4
    skip_frame = 4
    alpha_init = 0.00008
    alpha_end = 0.00005
    alpha_interp_limit = 500000
    epsilon_init = 0.5
    epsilon_end = 0.1
    epsilon_interp_limit = 1000000
    learning_start = 50000
