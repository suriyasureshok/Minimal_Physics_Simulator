import numpy as np


class AsyncBatchEnv:
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs

    def step(self, dt):
        # Each env gets slightly different dt
        dt_variation = dt * (1 + 0.01 * np.random.randn(self.num_envs))

        for i in range(self.num_envs):
            a = -self.env.k_over_m * self.env.x[i]
            self.env.v[i] += dt_variation[i] * a
            self.env.x[i] += dt_variation[i] * self.env.v[i]

        reward = -np.sum(self.env.x**2, axis=-1)  # Shape: (num_envs,)
        done = np.zeros(self.num_envs, dtype=np.bool_)

        return self.env.get_state(), reward, done

    def close(self):
        """Cleanup method for AsyncBatchEnv. Currently no-op as no resources are managed."""
        pass

