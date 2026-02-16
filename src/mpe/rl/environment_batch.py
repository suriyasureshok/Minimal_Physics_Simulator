# src/mpe/rl/environment_batch.py

import numpy as np

class BatchOscillatorEnv:
    """
    Batched 1D harmonic oscillator envirmenment
    Internal Layout: Structure of Arrays (SoA)
    """

    def __init__(self, num_envs, k_over_m, dtype=np.float32):
        self.num_envs = num_envs
        self.k_over_m = k_over_m
        self.dtype = dtype

        # SoA Layout
        self.x = np.ones(num_envs, dtype=dtype)
        self.v = np.zeros(num_envs, dtype=dtype)

    def step(self, dt):
        """
        Vectorized step
        """
        a = -self.k_over_m * self.x
        self.v += dt * a
        self.x += dt * self.v

        # Mock reward
        reward = -self.x ** 2

        # No termination in this simple example
        done = np.zeros(self.num_envs, dtype=np.bool_)

        return self.get_state(), reward, done

    def get_state():
        """
        Return state as AoS view for compatibility with RL code.
        Shape: (num_envs, 2)
        """
        return np.stack((self.x, self.v), axis=1)

    def reset(self):
        self.x.fill(1.0)
        self.v.fill(0.0)

