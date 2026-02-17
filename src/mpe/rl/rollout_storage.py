"""On-policy rollout storage for reinforcement learning.

This module implements a storage system for on-policy RL algorithms,
collecting trajectories from multiple parallel environments.
"""

import numpy as np


class RolloutStorage:
    """Storage for on-policy rollout data.
    
    Stores trajectories collected from parallel environments during
    on-policy RL training (e.g., PPO, A2C). Uses a (T, N, ...) layout
    where T is the time horizon and N is the number of environments.
    
    Attributes:
        horizon (int): Number of time steps to store.
        num_envs (int): Number of parallel environments.
        states (np.ndarray): State storage, shape (horizon, num_envs, state_dim).
        actions (np.ndarray): Action storage, shape (horizon, num_envs, action_dim).
        rewards (np.ndarray): Reward storage, shape (horizon, num_envs).
        dones (np.ndarray): Termination flags, shape (horizon, num_envs).
        
    Notes:
        - Designed for on-policy algorithms (PPO, A2C, etc.)
        - Data is stored in temporal order for GAE computation
        - Memory usage scales with horizon * num_envs
        
    Examples:
        >>> storage = RolloutStorage(horizon=128, num_envs=16, state_dim=2, action_dim=1)
        >>> storage.store(t=0, state=states, action=actions, reward=rewards, done=dones)
        >>> memory_mb = storage.memory_megabytes()
    """

    def __init__(self, horizon, num_envs, state_dim, action_dim, dtype=np.float32):
        """Initialize the rollout storage.
        
        Args:
            horizon (int): Number of time steps to store.
            num_envs (int): Number of parallel environments.
            state_dim (int): Dimension of state space.
            action_dim (int): Dimension of action space.
            dtype (np.dtype, optional): Data type for arrays. Defaults to np.float32.
        """
        self.horizon = horizon
        self.num_envs = num_envs

        # Pre-allocate storage arrays
        self.states = np.zeros((horizon, num_envs, state_dim), dtype=dtype)
        self.actions = np.zeros((horizon, num_envs, action_dim), dtype=dtype)
        self.rewards = np.zeros((horizon, num_envs), dtype=dtype)
        self.dones = np.zeros((horizon, num_envs), dtype=np.bool_)

    def store(self, t, state, action, reward, done):
        """Store one timestep of data across all environments.
        
        Args:
            t (int): Timestep index (0 to horizon-1).
            state (np.ndarray): States from all environments, shape (num_envs, state_dim).
            action (np.ndarray): Actions for all environments, shape (num_envs, action_dim).
            reward (np.ndarray): Rewards for all environments, shape (num_envs,).
            done (np.ndarray): Termination flags, shape (num_envs,).
        """
        self.states[t] = state
        self.actions[t] = action
        self.rewards[t] = reward
        self.dones[t] = done

    def memory_megabytes(self):
        """Compute total memory usage in megabytes.
        
        Returns:
            float: Total memory consumption in MB.
            
        Notes:
            - Includes all arrays (states, actions, rewards, dones)
            - Useful for profiling memory requirements
        """
        total_bytes = (
            self.states.nbytes +
            self.actions.nbytes +
            self.rewards.nbytes +
            self.dones.nbytes
        )

        return total_bytes / (1024 ** 2)

