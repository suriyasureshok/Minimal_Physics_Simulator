"""Base classes for batch integration.

This module defines the abstract base class for batch integrators that
process multiple particles simultaneously for improved performance.
"""

from abc import ABC, abstractmethod


class BatchIntegrator(ABC):
    """Abstract base class for batch particle integrators.
    
    BatchIntegrators operate on arrays of particles simultaneously, enabling
    vectorized or parallel computation. This is crucial for high-performance
    simulations with many particles.
    
    Notes:
        - All methods operate on batches of particles represented as arrays
        - Implementations can use Python loops, NumPy, PyTorch, or other backends
        - Different backends offer different performance characteristics
        
    Examples:
        >>> class MyBatchIntegrator(BatchIntegrator):
        ...     def step(self, x, v, dt):
        ...         # Batch integration logic
        ...         return new_x, new_v
    """
    
    @abstractmethod
    def step(self, x, v, dt):
        """Perform one integration step for batched particles.
        
        Args:
            x (np.ndarray): Array of particle positions, shape (n_particles,).
            v (np.ndarray): Array of particle velocities, shape (n_particles,).
            dt (float): Time step size in seconds.
            
        Returns:
            tuple: A tuple containing:
                - x (np.ndarray): Updated positions, shape (n_particles,).
                - v (np.ndarray): Updated velocities, shape (n_particles,).
        """
        pass
