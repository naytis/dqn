import numpy as np
from gym import Env as GymEnv


class LearningRateSchedule:
    def __init__(self, alpha_init: float, alpha_end: float, interp_limit: int):
        self.alpha = alpha_init
        self.alpha_init = alpha_init
        self.alpha_end = alpha_end
        self.interp_limit = interp_limit

    def update_alpha(self, frame_number: int):
        self.alpha = (
            np.interp(
                x=frame_number,
                xp=[0, self.interp_limit],
                fp=[self.alpha_init, self.alpha_end],
            )
            if frame_number <= self.interp_limit
            else self.alpha_end
        )


class ExplorationSchedule:
    def __init__(
        self,
        env: GymEnv,
        epsilon_init: float,
        epsilon_end: float,
        interp_limit: int,
    ):
        self.env = env
        self.epsilon = epsilon_init
        self.epsilon_init = epsilon_init
        self.epsilon_end = epsilon_end
        self.interp_limit = interp_limit

    def update_epsilon(self, frame_number: int):
        self.epsilon = (
            np.interp(
                x=frame_number,
                xp=[0, self.interp_limit],
                fp=[self.epsilon_init, self.epsilon_end],
            )
            if frame_number <= self.interp_limit
            else self.epsilon_end
        )

    def get_action(self, best_action: int) -> int:
        return (
            self.env.action_space.sample()
            if np.random.random() <= self.epsilon
            else best_action
        )
