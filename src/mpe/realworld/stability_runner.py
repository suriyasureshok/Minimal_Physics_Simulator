"""Real-world stability testing utilities.

This module provides utilities for running stability tests on physics
environments under real-world conditions.
"""

import numpy as np


def run_realworld_test(env_class, k_over_m, dt, horizon):
    """Run a stability test and compute energy drift.
    
    Simulates a single environment for the specified horizon and computes
    the mean energy drift as a measure of numerical stability.
    
    Args:
        env_class: Environment class to instantiate (e.g., BatchOscillatorEnv).
        k_over_m (float): Spring constant divided by mass parameter.
        dt (float): Time step size in seconds.
        horizon (int): Number of steps to simulate.
        
    Returns:
        float: Mean absolute energy drift over the simulation.
        
    Notes:
        - Tests a single environment (num_envs=1)
        - Collects full trajectory for detailed analysis
        - Energy drift indicates numerical error accumulation
        - Lower drift indicates better stability
        
    Examples:
        >>> from src.mpe.rl.environment_batch import BatchOscillatorEnv
        >>> drift = run_realworld_test(
        ...     BatchOscillatorEnv, k_over_m=10.0, dt=0.01, horizon=100000
        ... )
        >>> print(f"Energy drift: {drift:.6e}")
    """
    # Create a single environment for testing
    env = env_class(1, k_over_m)

    states = []

    # Capture initial state before any steps
    states.append(env.get_state().copy())

    # Run simulation and collect trajectory
    for _ in range(horizon):
        state, _, _ = env.step(dt)
        states.append(state.copy())

    # Stack all states for analysis
    states = np.stack(states)

    # Extract position and velocity
    x = states[:, :, 0]
    v = states[:, :, 1]

    # Compute total mechanical energy at each step
    energy = 0.5 * v**2 + 0.5 * k_over_m * x**2
    
    # Compute mean absolute drift from initial energy
    drift = np.abs(energy - energy[0]).mean()

    return drift

