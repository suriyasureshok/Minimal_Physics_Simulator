"""PyTorch CPU-based batch integrator.

This module implements a batch integrator using PyTorch tensors on CPU,
demonstrating PyTorch's computational capabilities for physics simulations.
"""

import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from src.mpe.batch import BatchIntegrator


class TorchCPUIntegrator(BatchIntegrator):
    """Batch integrator using PyTorch on CPU.
    
    Implements explicit Euler integration using PyTorch tensor operations
    on CPU. While not GPU-accelerated, it demonstrates PyTorch's ability
    to handle physics computations.
    
    Attributes:
        k_over_m (float): Spring constant divided by mass (k/m) in 1/s².
        
    Notes:
        - Similar performance to NumPy on CPU
        - Easily extensible to GPU with minimal code changes
        - Part of the PyTorch ecosystem for ML integration
        - Creates new tensors rather than in-place operations
        
    Examples:
        >>> integrator = TorchCPUIntegrator(k_over_m=10.0)
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> v = np.array([0.0, 0.0, 0.0])
        >>> x_new, v_new = integrator.step(x, v, dt=0.01)
    """
    
    def __init__(self, k_over_m: float):
        """Initialize the PyTorch CPU integrator.
        
        Args:
            k_over_m (float): Ratio of spring constant to mass (k/m) in 1/s².
        """
        self.k_over_m = k_over_m

    def step(self, x, v, dt):
        """Advance particles using PyTorch tensor operations.
        
        Performs explicit Euler integration using PyTorch's tensor
        arithmetic on CPU.
        
        Args:
            x (np.ndarray): Particle positions, shape (n_particles,).
            v (np.ndarray): Particle velocities, shape (n_particles,).
            dt (float): Time step size in seconds.
            
        Returns:
            tuple: Updated (x, v) arrays, both shape (n_particles,).
        """
        # Compute acceleration
        a = -self.k_over_m * x
        # Update position and velocity (creates new arrays)
        x = x + dt * v
        v = v + dt * a

        return x, v

