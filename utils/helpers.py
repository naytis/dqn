from typing import List, Union, Tuple

import numpy as np


def calculate_mean_and_ci(samples: Union[List, np.ndarray]) -> Tuple[float, float]:
    mean = np.mean(samples)
    std = np.std(samples)
    confidence_level = 1.96  # 95%
    return mean, confidence_level * std / np.sqrt(len(samples))
