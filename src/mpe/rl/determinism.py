"""Determinism checking utilities for RL environments.

This module provides tools for verifying that physics simulations produce
bit-exact deterministic results across multiple runs.
"""

import numpy as np


def check_determinism(env_class, num_envs, k_over_m, dt, horizon):
    """Verify bit-exact determinism of an environment.
    
    Runs two identical rollouts and checks if all states match exactly
    (bitwise equality). This is crucial for reproducible RL experiments
    and debugging.
    
    Args:
        env_class: Environment class to instantiate (e.g., BatchOscillatorEnv).
        num_envs (int): Number of parallel environments to test.
        k_over_m (float): Spring constant divided by mass parameter.
        dt (float): Time step size for integration.
        horizon (int): Number of steps to simulate.
        
    Returns:
        bool: True if both rollouts are bitwise identical, False otherwise.
        
    Notes:
        - Tests bit-exact equality (not just numerical closeness)
        - Non-determinism can arise from:
            * Uninitialized variables
            * Floating-point operation ordering
            * Multi-threading
            * Random number generation
        - Essential for reproducible RL research
        
    Examples:
        >>> from src.mpe.rl.environment_batch import BatchOscillatorEnv
        >>> is_deterministic = check_determinism(
        ...     BatchOscillatorEnv, num_envs=10, k_over_m=10.0, dt=0.01, horizon=1000
        ... )
        >>> print(f"Environment is deterministic: {is_deterministic}")
    """
    # Create two independent environment instances
    env1 = env_class(num_envs, k_over_m)
    env2 = env_class(num_envs, k_over_m)

    states1 = []
    states2 = []

    # Run both environments for the specified horizon
    for _ in range(horizon):
        s1, _, _ = env1.step(dt)
        s2, _, _ = env2.step(dt)

        # Store copies of states (avoid reference issues)
        states1.append(s1.copy())
        states2.append(s2.copy())

    # Stack all states for comparison
    states1 = np.stack(states1)
    states2 = np.stack(states2)

    # Check bit-exact equality
    identical = np.array_equal(states1, states2)

    return identical
