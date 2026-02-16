import numpy as np


class LatencyWrapper:
    def __init__(self, env, latency_steps=1):
        self.env = env
        self.latency_steps = latency_steps
        # Note: Actions are ignored in this oscillator environment

    def step(self, action, dt):
        # Actions are ignored in the oscillator environment
        # This wrapper demonstrates latency structure but doesn't use actions
        state, reward, done = self.env.step(dt)

        return state, reward, done

