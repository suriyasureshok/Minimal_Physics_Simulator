"""Batched reinforcement learning environment.

This module implements a vectorized harmonic oscillator environment for
reinforcement learning, supporting parallel simulation of multiple environments.
"""

import numpy as np


class BatchOscillatorEnv:
    """Batched 1D harmonic oscillator environment for RL.
    
    Implements a vectorized environment that simulates multiple independent
    harmonic oscillators in parallel using Structure-of-Arrays (SoA) layout
    for efficient vectorized computation.
    
    Attributes:
        num_envs (int): Number of parallel environments.
        k_over_m (float): Spring constant divided by mass (k/m) in 1/s².
        dtype (np.dtype): Data type for state arrays.
        x (np.ndarray): Positions of all environments, shape (num_envs,).
        v (np.ndarray): Velocities of all environments, shape (num_envs,).
        
    Notes:
        - Uses Structure-of-Arrays (SoA) layout for memory efficiency
        - All environments evolve independently and deterministically
        - Reward is -x² (penalizes deviation from equilibrium)
        - No episode termination in this simple environment
        
    Examples:
        >>> env = BatchOscillatorEnv(num_envs=1000, k_over_m=10.0)
        >>> state, reward, done = env.step(dt=0.01)
        >>> print(state.shape)  # (1000, 2)
    """

    def __init__(self, num_envs, k_over_m, dtype=np.float32):
        """Initialize the batched oscillator environment.
        
        Args:
            num_envs (int): Number of parallel environments to simulate.
            k_over_m (float): Ratio of spring constant to mass (k/m) in 1/s².
            dtype (np.dtype, optional): Data type for arrays. Defaults to np.float32.
        """
        self.num_envs = num_envs
        self.k_over_m = k_over_m
        self.dtype = dtype

        # SoA Layout: separate arrays for each state component
        self.x = np.ones(num_envs, dtype=dtype)   # Initial position = 1.0
        self.v = np.zeros(num_envs, dtype=dtype)  # Initial velocity = 0.0

    def step(self, dt):
        """Advance all environments by one time step.
        
        Performs vectorized semi-implicit Euler integration for all
        environments simultaneously.
        
        Args:
            dt (float): Time step size in seconds.
            
        Returns:
            tuple: A tuple containing:
                - state (np.ndarray): Current states as AoS, shape (num_envs, 2).
                - reward (np.ndarray): Rewards for each environment, shape (num_envs,).
                - done (np.ndarray): Termination flags (always False), shape (num_envs,).
                
        Notes:
            - Uses semi-implicit Euler: v += dt*a, then x += dt*v
            - Reward = -x² (quadratic penalty for displacement)
        """
        # Vectorized semi-implicit Euler integration
        a = -self.k_over_m * self.x
        self.v += dt * a
        self.x += dt * self.v

        # Compute reward (negative squared displacement)
        reward = -self.x ** 2

        # No termination in this simple example
        done = np.zeros(self.num_envs, dtype=np.bool_)

        return self.get_state(), reward, done

    def get_state(self):
        """Return current state in Array-of-Structures format.
        
        Converts internal SoA representation to AoS format for compatibility
        with standard RL libraries.
        
        Returns:
            np.ndarray: State array with shape (num_envs, 2), where
                state[i] = [x[i], v[i]].
        """
        return np.stack((self.x, self.v), axis=1)

    def reset(self):
        """Reset all environments to initial state.
        
        Sets all positions to 1.0 and all velocities to 0.0.
        
        Returns:
            np.ndarray: Initial state array with shape (num_envs, 2).
        """
        self.x.fill(1.0)
        self.v.fill(0.0)
        return self.get_state()

