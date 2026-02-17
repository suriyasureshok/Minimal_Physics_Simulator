"""Floating-point precision testing for physics simulations.

This module compares float32 vs float64 precision in long-running
physics simulations to quantify numerical drift.
"""

import numpy as np


def run_precision_test(env_class, num_envs, k_over_m, dt, horizon):
    """Compare energy drift between float32 and float64 precision.
    
    Runs identical simulations in float32 and float64 to measure the
    impact of floating-point precision on energy conservation and
    numerical stability.
    
    Args:
        env_class: Environment class to instantiate (e.g., BatchOscillatorEnv).
        num_envs (int): Number of parallel environments to simulate.
        k_over_m (float): Spring constant divided by mass parameter.
        dt (float): Time step size in seconds.
        horizon (int): Number of steps to simulate.
        
    Returns:
        dict: Dictionary mapping dtype string to mean energy drift:
            - "<class 'numpy.float32'>": drift with float32
            - "<class 'numpy.float64'>": drift with float64
            
    Notes:
        - Energy drift quantifies numerical error accumulation
        - float64 typically shows ~100x less drift than float32
        - Memory-bound simulations may prefer float32 despite drift
        - Drift is averaged over all environments and time steps
        
    Examples:
        >>> from src.mpe.rl.environment_batch import BatchOscillatorEnv
        >>> results = run_precision_test(
        ...     BatchOscillatorEnv, num_envs=100, k_over_m=10.0,
        ...     dt=0.01, horizon=100000
        ... )
        >>> print(f"Float32 drift: {results['<class \'numpy.float32\'>']:.6e}")
        >>> print(f"Float64 drift: {results['<class \'numpy.float64\'>']:.6e}")
    """
    results = {}
    
    for dtype in [np.float32, np.float64]:
        # Create environment with specified precision
        env = env_class(num_envs, k_over_m, dtype=dtype)

        # Compute energy drift online to avoid storing the full trajectory
        energy0 = None
        drift_sum = 0.0

        for step_idx in range(horizon):
            state, _, _ = env.step(dt)

            # Extract position and velocity
            x = state[:, 0]
            v = state[:, 1]
            
            # Compute total energy (kinetic + potential)
            energy = 0.5 * v**2 + 0.5 * k_over_m * x**2

            if step_idx == 0:
                # Reference energy at the first step, per environment
                energy0 = energy
            
            # Accumulate mean absolute drift for this step across environments
            drift_sum += np.abs(energy - energy0).mean()

        # Final drift is the mean over all steps and environments
        drift = drift_sum / float(horizon)

        results[str(dtype)] = drift

    return results
