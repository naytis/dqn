from typing import Union

import numpy as np
from gym import Env as GymEnv

from utils.test_env import EnvTest


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
        env: Union[EnvTest, GymEnv],
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


def test1():
    env = EnvTest((5, 5, 1))
    exp_schedule = ExplorationSchedule(env, 1, 0, 10)

    found_diff = False
    for i in range(10):
        rnd_act = exp_schedule.get_action(0)
        if rnd_act != 0 and rnd_act is not None:
            found_diff = True

    assert found_diff, "Test 1 failed."
    print("Test1: ok")


def test2():
    env = EnvTest((5, 5, 1))
    exp_schedule = ExplorationSchedule(env, 1, 0, 10)
    exp_schedule.update_epsilon(5)
    assert exp_schedule.epsilon == 0.5, "Test 2 failed"
    print("Test2: ok")


def test3():
    env = EnvTest((5, 5, 1))
    exp_schedule = ExplorationSchedule(env, 1, 0.5, 10)
    exp_schedule.update_epsilon(20)
    assert exp_schedule.epsilon == 0.5, "Test 3 failed"
    print("Test3: ok")


if __name__ == "__main__":
    test1()
    test2()
    test3()
