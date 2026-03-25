import time
import torch
from transformers import TrainerCallback

class IterTimeCallback(TrainerCallback):
    def __init__(self, warmup_steps=5, measure_steps=10):
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.times = []
        self._last_time = None

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        now = time.time()

        if self._last_time is None:
            self._last_time = now
            return

        if state.global_step >= self.warmup_steps:
            elapsed = now - self._last_time
            self.times.append(elapsed)

            if len(self.times) == self.measure_steps:
                avg = sum(self.times) / len(self.times)
                print(f"Training Time / Iter (s): {avg:.3f}")
                control.should_training_stop = True

        self._last_time = now
