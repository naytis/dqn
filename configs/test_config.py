class TestConfig:
    # env config
    render_train = False
    render_test = False
    overwrite_render = True
    record = False
    high = 255.0

    # output config
    output_path = "results/test/"
    model_output = output_path + "model.weights"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"

    # model and training config
    num_episodes_test = 20
    clip_val = 10
    saving_freq = 5000
    log_freq = 50
    eval_freq = 100
    soft_epsilon = 0

    # hyper params
    num_steps_train = 2000
    batch_size = 32
    buffer_size = 500
    target_update_freq = 500
    gamma = 0.99
    learning_freq = 4
    history_length = 4
    alpha_init = 0.00025
    alpha_end = 0.0001
    alpha_interp_limit = num_steps_train / 2
    epsilon_init = 1
    epsilon_end = 0.01
    epsilon_interp_limit = num_steps_train / 2
    learning_start = 200
