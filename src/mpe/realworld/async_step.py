"""Asynchronous batched environment with per-environment dt variation.

This module simulates asynchronous parallel environments where each
environment may advance at a slightly different rate, common in
distributed or multi-threaded simulations.
"""

import numpy as np


class AsyncBatchEnv:
    """Batched environment with asynchronous time evolution.
    
    Wraps a batched environment to simulate asynchronous execution where
    each parallel environment advances with slightly different time steps,
    modeling real distributed or multi-threaded systems.
    
    Attributes:
        env: Wrapped batched environment.
        num_envs (int): Number of parallel environments.
        
    Notes:
        - Each environment gets dt += dt * 0.01 * N(0,1) variation
        - Models distributed systems with independent clocks
        - Can reveal synchronization issues in parallel RL
        - May cause slight divergence between nominally identical envs
        
    Examples:
        >>> base_env = BatchOscillatorEnv(num_envs=100, k_over_m=10.0)
        >>> async_env = AsyncBatchEnv(base_env)
        >>> state, reward, done = async_env.step(dt=0.01)
    """
    
    def __init__(self, env):
        """Initialize the asynchronous batch environment.
        
        Args:
            env: Batched environment to wrap. Should have num_envs,
                k_over_m, x, v attributes.
        """
        self.env = env
        self.num_envs = env.num_envs

    def step(self, dt):
        """Step all environments with per-environment dt variation.
        
        Each environment gets a slightly different time step drawn from
        dt * (1 + 0.01 * N(0,1)), simulating asynchronous execution.
        
        Args:
            dt (float): Nominal time step size in seconds.
            
        Returns:
            tuple: (state, reward, done) where:
                - state: Array of states, shape (num_envs, state_dim)
                - reward: Array of rewards, shape (num_envs,)
                - done: Array of termination flags, shape (num_envs,)
                
        Notes:
            - Uses explicit loop for per-environment dt variation
            - Slower than vectorized step but models async behavior
            - Typical variation is 1% of nominal dt
        """
        # Each env gets slightly different dt (~1% variation)
        dt_variation = dt * (1 + 0.01 * np.random.randn(self.num_envs))

        # Step each environment individually with its own dt
        for i in range(self.num_envs):
            a = -self.env.k_over_m * self.env.x[i]
            self.env.v[i] += dt_variation[i] * a
            self.env.x[i] += dt_variation[i] * self.env.v[i]

        # Compute reward (quadratic penalty)
        reward = -np.sum(self.env.x**2, axis=-1)  # Shape: (num_envs,)
        done = np.zeros(self.num_envs, dtype=np.bool_)

        return self.env.get_state(), reward, done

    def close(self):
        """Cleanup method for AsyncBatchEnv.
        
        Currently a no-op as no external resources are managed.
        """
        pass

