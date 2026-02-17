"""Benchmarking utilities for batch integrators.

This module provides tools to measure and compare the performance of
different batch integration backends.
"""

import time
import numpy as np


def benchmark_backend(integrator, x, v, dt, steps):
    """Benchmark a batch integrator's performance.
    
    Runs the integrator for a specified number of steps and measures
    the elapsed time and throughput.
    
    Args:
        integrator (BatchIntegrator): The batch integrator to benchmark.
        x (np.ndarray): Initial particle positions, shape (n_particles,).
        v (np.ndarray): Initial particle velocities, shape (n_particles,).
        dt (float): Time step size in seconds.
        steps (int): Number of integration steps to perform.
        
    Returns:
        tuple: A tuple containing:
            - steps_per_sec (float): Throughput in steps per second.
            - total_time (float): Total elapsed time in seconds.
            
    Notes:
        - Uses perf_counter for high-resolution timing
        - Input arrays are modified in-place by the integrator
        - For accurate results, use large enough steps to minimize overhead
        
    Examples:
        >>> integrator = NumpyVectorizedIntegrator(k_over_m=10.0)
        >>> x = np.random.randn(1000)
        >>> v = np.random.randn(1000)
        >>> throughput, elapsed = benchmark_backend(integrator, x, v, 0.01, 1000)
        >>> print(f"Throughput: {throughput:.2f} steps/sec")
    """
    # Start timing
    start = time.perf_counter()

    # Run integration loop
    for _ in range(steps):
        x, v = integrator.step(x, v, dt)

    # End timing
    end = time.perf_counter()

    total_time = end - start
    steps_per_sec = steps / total_time

    return steps_per_sec, total_time

