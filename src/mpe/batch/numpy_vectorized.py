"""NumPy vectorized batch integrator.

This module implements a vectorized batch integrator using NumPy array
operations, providing significant performance improvements over Python loops.
"""

import numpy as np
from src.mpe.batch import BatchIntegrator


class NumpyVectorizedIntegrator(BatchIntegrator):
    """Batch integrator using NumPy vectorization.
    
    Implements explicit Euler integration using NumPy's vectorized array
    operations. This eliminates Python loop overhead and leverages SIMD
    instructions for substantial speedup.
    
    Attributes:
        k_over_m (float): Spring constant divided by mass (k/m) in 1/s².
        
    Notes:
        - Much faster than PythonLoopIntegrator (10-100x speedup typical)
        - Leverages NumPy's C-level optimizations and SIMD
        - Memory efficient with in-place operations
        - Cache-friendly for large arrays
        
    Examples:
        >>> integrator = NumpyVectorizedIntegrator(k_over_m=10.0)
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> v = np.array([0.0, 0.0, 0.0])
        >>> x_new, v_new = integrator.step(x, v, dt=0.01)
    """
    
    def __init__(self, k_over_m: float):
        """Initialize the NumPy vectorized integrator.
        
        Args:
            k_over_m (float): Ratio of spring constant to mass (k/m) in 1/s².
        """
        self.k_over_m = k_over_m

    def step(self, x, v, dt):
        """Advance particles using vectorized symplectic Euler.
        
        Performs element-wise operations on entire arrays simultaneously,
        eliminating Python loop overhead.
        
        Note:
            This method modifies x and v in-place for memory efficiency.
        
        Args:
            x (np.ndarray): Particle positions, shape (n_particles,).
            v (np.ndarray): Particle velocities, shape (n_particles,).
            dt (float): Time step size in seconds.
            
        Returns:
            tuple: The same (x, v) arrays after in-place updates, both shape (n_particles,).
        """
        # Vectorized operations (fast)
        a = -self.k_over_m * x  # Compute all accelerations at once
        v += dt * a              # Update all velocities
        x += dt * v              # Update all positions

        return x, v

