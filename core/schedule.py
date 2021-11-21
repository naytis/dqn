import numpy as np
from gym import Env as GymEnv


class LearningRateSchedule:
    def __init__(self, lr_init: float, lr_end: float, interp_limit: int):
        self.lr = lr_init
        self.lr_init = lr_init
        self.lr_end = lr_end
        self.interp_limit = interp_limit

    def update_lr(self, frame_number: int):
        self.lr = (
            np.interp(
                x=frame_number,
                xp=[0, self.interp_limit],
                fp=[self.lr_init, self.lr_end],
            )
            if frame_number <= self.interp_limit
            else self.lr_end
        )


class ExplorationSchedule:
    def __init__(
        self,
        env: GymEnv,
        epsilon_init: float,
        epsilon_final: float,
        interp_limit: int,
    ):
        self.env = env
        self.epsilon = epsilon_init
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.interp_limit = interp_limit

    def update_epsilon(self, frame_number: int):
        self.epsilon = (
            np.interp(
                x=frame_number,
                xp=[0, self.interp_limit],
                fp=[self.epsilon_init, self.epsilon_final],
            )
            if frame_number <= self.interp_limit
            else self.epsilon_final
        )

    def get_action(self, best_action: int) -> int:
        return (
            self.env.action_space.sample()
            if np.random.random() <= self.epsilon
            else best_action
        )
