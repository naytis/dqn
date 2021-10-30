import time
from collections import defaultdict


class Timer:
    def __init__(self, enabled=False) -> None:
        super().__init__()
        self.enabled = enabled
        self.category_sec_avg = defaultdict(
            lambda: [0.0, 0.0, 0]
        )  # A bucket of [total_secs, latest_start, num_calls]

    def start(self, category):
        if self.enabled:
            stat = self.category_sec_avg[category]
            stat[1] = time.perf_counter()
            stat[2] += 1

    def end(self, category):
        if self.enabled:
            stat = self.category_sec_avg[category]
            stat[0] += time.perf_counter() - stat[1]

    def print_stat(self):
        if self.enabled:
            print("Printing timer stats:")
            for key, val in self.category_sec_avg.items():
                if val[2] > 0:
                    print(
                        f":> category {key}, total {val[0]}, num {val[2]}, avg {val[0] / val[2]}"
                    )

    def reset_stat(self):
        if self.enabled:
            print("Reseting timer stats")
            for val in self.category_sec_avg.values():
                val[0], val[1], val[2] = 0.0, 0.0, 0
