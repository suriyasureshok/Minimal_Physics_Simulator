"""Time step jitter simulation for real-world physics.

This module simulates timing jitter in physics simulations, modeling
the variability in time step duration that occurs in real-time systems.
"""

import numpy as np


def step_with_jitter(env, dt, jitter_std):
    """Step environment with gaussian noise added to the time step.
    
    Simulates real-world timing jitter by adding Gaussian noise to the
    nominal time step. This is common in real-time systems where the
    actual time step varies due to scheduling, I/O, or other factors.
    
    Args:
        env: Environment instance with a step(dt) method.
        dt (float): Nominal time step size in seconds.
        jitter_std (float): Standard deviation of the Gaussian noise
            added to dt, in seconds.
            
    Returns:
        tuple: Result of env.step() - typically (state, reward, done).
        
    Notes:
        - Noisy dt is clamped to a small positive value (1e-8) to prevent
          negative or zero time steps
        - Jitter can cause numerical instability if too large
        - Models real-world variability in game loops, simulations, robotics
        
    Examples:
        >>> env = BatchOscillatorEnv(num_envs=10, k_over_m=10.0)
        >>> state, reward, done = step_with_jitter(env, dt=0.01, jitter_std=1e-4)
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if jitter_std < 0:
        raise ValueError("jitter_std must be non-negative")
    epsilon = 1e-8  # Small positive value to prevent negative dt
    noisy_dt = dt + np.random.normal(0, jitter_std)
    noisy_dt = max(noisy_dt, epsilon)  # Clamp to safe minimum
    
    return env.step(noisy_dt)
