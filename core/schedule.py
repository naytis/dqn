from typing import Union

import numpy as np
from gym import Env as GymEnv

from utils.test_env import EnvTest


class LinearSchedule(object):
    def __init__(self, epsilon_init: float, epsilon_end: float, interp_limit: int):
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


class LinearExploration(LinearSchedule):
    def __init__(
        self,
        env: Union[EnvTest, GymEnv],
        epsilon_init: float,
        epsilon_end: float,
        interp_limit: int,
    ):
        self.env = env
        super(LinearExploration, self).__init__(epsilon_init, epsilon_end, interp_limit)

    def get_action(self, best_action: int) -> int:
        return (
            self.env.action_space.sample()
            if np.random.random() <= self.epsilon
            else best_action
        )


def test1():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0, 10)

    found_diff = False
    for i in range(10):
        rnd_act = exp_strat.get_action(0)
        if rnd_act != 0 and rnd_act is not None:
            found_diff = True

    assert found_diff, "Test 1 failed."
    print("Test1: ok")


def test2():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0, 10)
    exp_strat.update_epsilon(5)
    assert exp_strat.epsilon == 0.5, "Test 2 failed"
    print("Test2: ok")


def test3():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0.5, 10)
    exp_strat.update_epsilon(20)
    assert exp_strat.epsilon == 0.5, "Test 3 failed"
    print("Test3: ok")


if __name__ == "__main__":
    test1()
    test2()
    test3()
