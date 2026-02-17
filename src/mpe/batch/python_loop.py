"""Python loop-based batch integrator.

This module implements a naive batch integrator using Python loops,
serving as a baseline for performance comparisons with vectorized approaches.
"""

import numpy as np
from src.mpe.batch import BatchIntegrator


class PythonLoopIntegrator(BatchIntegrator):
    """Batch integrator using explicit Python loops.
    
    Implements explicit Euler integration using a Python for-loop over
    particles. This is the slowest approach but serves as a baseline
    for measuring speedup from vectorization.
    
    Attributes:
        k_over_m (float): Spring constant divided by mass (k/m) in 1/s².
        
    Notes:
        - Slowest implementation due to Python loop overhead
        - Useful as a baseline for performance comparisons
        - Scales linearly with number of particles
        - Does not leverage SIMD or vectorization
        
    Examples:
        >>> integrator = PythonLoopIntegrator(k_over_m=10.0)
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> v = np.array([0.0, 0.0, 0.0])
        >>> x_new, v_new = integrator.step(x, v, dt=0.01)
    """
    
    def __init__(self, k_over_m: float):
        """Initialize the Python loop integrator.
        
        Args:
            k_over_m (float): Ratio of spring constant to mass (k/m) in 1/s².
        """
        self.k_over_m = k_over_m

    def step(self, x, v, dt):
        """Advance particles using explicit Euler with Python loop.
        
        Iterates over each particle individually, computing forces and
        updating positions and velocities.
        
        Args:
            x (np.ndarray): Particle positions, shape (n_particles,).
            v (np.ndarray): Particle velocities, shape (n_particles,).
            dt (float): Time step size in seconds.
            
        Returns:
            tuple: Updated (x, v) arrays, both shape (n_particles,).
        """
        n = len(x)

        # Explicit loop over each particle (slow)
        for i in range(n):
            a = -self.k_over_m * x[i]  # Compute acceleration
            x[i] += dt * v[i]            # Update position
            v[i] += dt * a               # Update velocity

        return x, v

