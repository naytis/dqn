from typing import Tuple, Dict, Any, List

import gym
import numpy as np


class BenchmarkMonitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: (gym.Env) The environment
    :param allow_early_resets: (bool) allows the reset of the environment before it is done
    :param reset_keywords: (tuple) extra keywords for the reset call, if extra parameters are needed at reset
    :param info_keywords: (tuple) extra information to log, from the information return of environment.step
    """

    EXT = "monitor.csv"
    file_handler = None

    def __init__(
        self,
        env: gym.Env,
        allow_early_resets: bool = True,
        reset_keywords=(),
        info_keywords=(),
    ):
        super(BenchmarkMonitor, self).__init__(env=env)
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.current_reset_info = (
            {}
        )  # extra info about the current episode, that was passed in during reset()

        # custom properties
        self.episode_reward = 0
        self.episode_length = 0

    def reset(self, **kwargs) -> np.ndarray:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError("Expected you to pass kwarg {} into reset".format(key))
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.needs_reset = True
            self.episode_reward = sum(self.rewards)
            self.episode_length = len(self.rewards)
        return observation, reward, done, info

    # custom methods
    def get_episode_reward(self) -> float:
        return self.episode_reward

    def get_episode_length(self) -> int:
        return self.episode_length
