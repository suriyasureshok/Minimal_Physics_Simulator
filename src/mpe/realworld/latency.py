import numpy as np


class LatencyWrapper:
    def __init__(self, env, latency_steps=1):
        self.env = env
        self.latency_steps = latency_steps
        self.action_buffer = []

    def step(self, action, dt):
        self.action_buffer.append(action)

        if len(self.action_buffer) > self.latecny_steps:
            delayed_action = self.action_buffer.pop(0)
        else:
            delayed_action = np.zeros_like(action)

        # In oscillator, action is ignored
        # but structure is realistic
        state, reward, done = self.env.step(dt)

        return state, reward, done

