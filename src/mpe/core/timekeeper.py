# src/mpe/core/timekeeper.py

class TimeKeeper:
    def __init__(self, dt: float):
        self.dt = dt
        self.t = 0.0
        self.step_count = 0

    def advance(self):
        self.t += self.dt
        self.step_count += 1
