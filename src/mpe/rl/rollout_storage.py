# src/mpe/rl/rollout_storage.py

import numpy as np

class RolloutStorage:
    """
    On-policy rollout storage.
    Layout: (T, N, ...)
    """

    def __init__(self, horizon, num_envs, state_dim, action_dim, dtype=np.float32):
        self.horizon = horizon
        self.num_envs = num_envs

        self.states = np.zeros((horizon, num_envs, state_dim), dtype=dtype)

        self.actions = np.zeros((horizon, num_envs, action_dim), dtype=dtype)

        self.rewards = np.zeros((horizon, num_envs), dtype=dtype)

        self.dones = np.zeros((horizon, num_envs), dtype=np.bool_)

    def store(self, t, state, action, reward, done):
        """
        Store one timestamp across all environments.
        """
        self.states[t] = state
        self.actions[t] = action
        self.rewards[t] =reward
        self.dones[t] = done

    def memory_megabytes(self):
        total_bytes = (
            self.states.nbytes +
            self.actions.nbytes +
            self.rewards.nbytes +
            self.dones.nbytes
        )

        return total_bytes / (1024 ** 2)

