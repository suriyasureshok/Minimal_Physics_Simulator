"""Performance metrics for physics simulations.

This module provides functions for measuring computational performance
of simulators and integrators, including timing and FLOP estimates.
"""

import time
import numpy as np


def measure_ns_per_step(sim, initial_state, dt, steps):
    """Measure performance in nanoseconds per simulation step.
    
    Runs a simulation and computes the average time per step in nanoseconds,
    which is a useful metric for comparing integrator performance.
    
    Args:
        sim: Simulator instance with a run() method.
        initial_state (State1D): Initial state for the simulation.
        dt (float): Time step size in seconds.
        steps (int): Number of steps to simulate.
        
    Returns:
        float: Average nanoseconds per step.
        
    Notes:
        - Uses perf_counter for high-resolution timing
        - Includes all overhead (force computation, state updates, etc.)
        - Lower is better for performance
        - Useful for comparing different integrators
        
    Examples:
        >>> ns_per_step = measure_ns_per_step(sim, initial_state, 0.01, 10000)
        >>> print(f"Performance: {ns_per_step:.2f} ns/step")
    """
    start = time.perf_counter()
    sim.run(initial_state, dt, steps)
    end = time.perf_counter()

    total_ns = (end - start) * 1e9  # Convert seconds to nanoseconds

    return total_ns / steps


def estimate_flops(integrator_name: str):
    """Estimate floating-point operations per step for an integrator.
    
    Provides rough FLOP counts for different integrators, useful for
    comparing computational complexity.
    
    Args:
        integrator_name (str): Name of the integrator. Supported values:
            - "ExplicitEuler": ~10 FLOPs
            - "SemiImplicitEuler": ~10 FLOPs
            - "Verlet": ~20 FLOPs (two force evaluations)
            - "RK4": ~80 FLOPs (four stages)
            
    Returns:
        int or None: Estimated FLOPs per step, or None if integrator unknown.
        
    Notes:
        - These are rough estimates for simple force models
        - Actual FLOPs depend on force model complexity
        - Useful for theoretical performance analysis
        - Does not account for memory bandwidth limitations
        
    Examples:
        >>> flops = estimate_flops("RK4")
        >>> print(f"RK4 uses approximately {flops} FLOPs per step")
        RK4 uses approximately 80 FLOPs per step
    """
    estimates = {
        "ExplicitEuler": 10,
        "SemiImplicitEuler": 10,
        "Verlet": 20,
        "RK4": 80
    }

    return estimates.get(integrator_name, None)
