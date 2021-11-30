__all__ = ["BenchmarkMonitor"]

import csv
import json
import os
import time
from typing import Tuple, Dict, Any, List, Optional

import gym
import numpy as np


class BenchmarkMonitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: (gym.Env) The environment
    :param filename: (Optional[str]) the location to save a log file, can be None for no log
    :param allow_early_resets: (bool) allows the reset of the environment before it is done
    :param reset_keywords: (tuple) extra keywords for the reset call, if extra parameters are needed at reset
    :param info_keywords: (tuple) extra information to log, from the information return of environment.step
    """

    EXT = "monitor.csv"
    file_handler = None

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords=(),
        info_keywords=(),
    ):
        super(BenchmarkMonitor, self).__init__(env=env)
        self.t_start = time.time()
        if filename is None:
            self.file_handler = None
            self.logger = None
        else:
            if not filename.endswith(BenchmarkMonitor.EXT):
                if os.path.isdir(filename):
                    filename = os.path.join(filename, BenchmarkMonitor.EXT)
                else:
                    filename = filename + "." + BenchmarkMonitor.EXT
            self.file_handler = open(filename, "wt")
            self.file_handler.write(
                "#%s\n"
                % json.dumps(
                    {"t_start": self.t_start, "env_id": env.spec and env.spec.id}
                )
            )
            self.logger = csv.DictWriter(
                self.file_handler,
                fieldnames=("r", "l", "t") + reset_keywords + info_keywords,
            )
            self.logger.writeheader()
            self.file_handler.flush()

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = (
            {}
        )  # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs) -> np.ndarray:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: (np.ndarray) the first observation of the environment
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
        """
        Step the environment with the given action

        :param action: (np.ndarray) the action
        :return: (Tuple[np.ndarray, float, bool, Dict[Any, Any]]) observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            eplen = len(self.rewards)
            ep_info = {
                "r": round(ep_rew, 6),
                "l": eplen,
                "t": round(time.time() - self.t_start, 6),
            }
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(ep_info)
                self.file_handler.flush()
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, done, info

    def close(self):
        """
        Closes the environment
        """
        super(BenchmarkMonitor, self).close()
        if self.file_handler is not None:
            self.file_handler.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return: (int)
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes

        :return: ([float])
        """
        return self.episode_rewards

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes

        :return: ([int])
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return: ([float])
        """
        return self.episode_times