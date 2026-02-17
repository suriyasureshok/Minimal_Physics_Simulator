"""Off-policy replay buffer for reinforcement learning.

This module implements a circular replay buffer for off-policy RL algorithms,
storing transitions for experience replay.
"""

import numpy as np


class ReplayBuffer:
    """Circular replay buffer for off-policy RL.
    
    Stores transitions (s, a, r, s', done) in a circular buffer for
    off-policy algorithms like DQN, SAC, and TD3. Implements random
    access and automatic overwriting when capacity is reached.
    
    Attributes:
        capacity (int): Maximum number of transitions to store.
        ptr (int): Current write position in the buffer.
        size (int): Current number of stored transitions.
        states (np.ndarray): State storage, shape (capacity, state_dim).
        actions (np.ndarray): Action storage, shape (capacity, action_dim).
        rewards (np.ndarray): Reward storage, shape (capacity,).
        next_states (np.ndarray): Next state storage, shape (capacity, state_dim).
        dones (np.ndarray): Termination flags, shape (capacity,).
        
    Notes:
        - Circular buffer: oldest data is overwritten when full
        - Designed for off-policy algorithms (DQN, SAC, TD3)
        - Supports random sampling for breaking correlations
        
    Examples:
        >>> buffer = ReplayBuffer(capacity=100000, state_dim=4, action_dim=2)
        >>> buffer.add(state, action, reward, next_state, done)
        >>> memory_mb = buffer.memory_megabytes()
    """
    
    def __init__(self, capacity, state_dim, action_dim, dtype=np.float32):
        """Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store.
            state_dim (int): Dimension of state space.
            action_dim (int): Dimension of action space.
            dtype (np.dtype, optional): Data type for arrays. Defaults to np.float32.
        """
        self.capacity = capacity
        self.ptr = 0       # Write pointer
        self.size = 0      # Current size (up to capacity)

        # Pre-allocate storage arrays
        self.states = np.zeros((capacity, state_dim), dtype=dtype)
        self.actions = np.zeros((capacity, action_dim), dtype=dtype)
        self.rewards = np.zeros(capacity, dtype=dtype)
        self.next_states = np.zeros((capacity, state_dim), dtype=dtype)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the replay buffer.
        
        Args:
            state (np.ndarray): Current state, shape (state_dim,).
            action (np.ndarray): Action taken, shape (action_dim,).
            reward (float or np.ndarray): Reward received.
            next_state (np.ndarray): Next state, shape (state_dim,).
            done (bool or np.ndarray): Episode termination flag.
            
        Notes:
            - Overwrites oldest transition when buffer is full
            - Uses circular indexing for efficient storage
        """
        # Store transition at current pointer position
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.capacity  # Circular indexing
        self.size = min(self.size + 1, self.capacity)

    def memory_megabytes(self):
        """Compute total memory usage in megabytes.
        
        Returns:
            float: Total memory consumption in MB.
            
        Notes:
            - Includes all arrays (states, actions, rewards, next_states, dones)
            - Useful for planning buffer capacity based on available memory
        """
        total_bytes = (
            self.states.nbytes +
            self.actions.nbytes +
            self.rewards.nbytes +
            self.next_states.nbytes +
            self.dones.nbytes
        )
    
        return total_bytes / (1024 ** 2)
