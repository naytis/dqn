import numpy as np


class ExplorationSchedule:
    def __init__(
        self,
        epsilon_init: float,
        epsilon_final: float,
        interp_limit: int,
    ):
        self.epsilon = epsilon_init
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.interp_limit = interp_limit

    def update_epsilon(self, frame_number: int) -> None:
        self.epsilon = (
            np.interp(
                x=frame_number,
                xp=[0, self.interp_limit],
                fp=[self.epsilon_init, self.epsilon_final],
            )
            if frame_number <= self.interp_limit
            else self.epsilon_final
        )

    # @get
    # def epsilon():
    #   pass todo
