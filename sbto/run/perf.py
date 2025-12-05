import time
import yaml
import os

PERF_FILENAME = "optimization_stats"

class IterationStats():
    def __init__(
        self,
        n_knots_to_opt: int,
        n_sim_steps_rollout: int,
        ):
        self.n_knots_to_opt = n_knots_to_opt
        self.n_sim_steps_rollout = n_sim_steps_rollout

        self._start_time = 0.
        self.duration = 0.
        self.start()

    def start(self,):
        self._start_time = time.time()
    
    def end(self):
        self.duration = time.time() - self._start_time

    def as_dict(self) -> dict:
        return {
            "n_knots_to_opt" : int(self.n_knots_to_opt),
            "n_sim_steps_rollout" : int(self.n_sim_steps_rollout),
            "duration" : float(self.duration),
        }

class OptimizationStats():
    def __init__(self):
        self.i = 0
        self.iterations = {}

    @property
    def n_total_it(self): return len(self.iterations)

    @property
    def duration(self): return sum((it.duration for it in self.iterations.values()))

    def add_iteration(
        self,
        n_knots_to_opt: int,
        n_sim_steps_rollout: int,
        ):
        self.iterations[self.i] = IterationStats(
            n_knots_to_opt,
            n_sim_steps_rollout
            )
    
    def end_iteration(self):
        self.iterations[self.i].end()
        self.i += 1

    def save(self, dir_path: str):
        data = {
            "duration": float(self.duration),
            "n_it": int(self.n_total_it),
            "iterations": {
                it: stats.as_dict()
                for it, stats in self.iterations.items()
            }
        }

        file_path = os.path.join(dir_path, f"{PERF_FILENAME}.yaml")
        with open(file_path, "w") as f:
            yaml.safe_dump(data, f)