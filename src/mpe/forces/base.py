"""Base classes for force models.

This module defines the abstract base class for all force models used in
the Minimal Physics Engine to compute forces acting on particles.
"""

from abc import ABC, abstractmethod
from src.mpe.core import State1D


class ForceModel(ABC):
    """Abstract base class for force computation models.
    
    Force models define the dynamics of a physical system by computing the
    force acting on a particle given its current state and time. Concrete
    implementations must override the compute method.
    
    Examples:
        >>> class MyForce(ForceModel):
        ...     def compute(self, state, t):
        ...         return -10.0 * state.x  # Simple spring force
    """
    
    @abstractmethod
    def compute(self, state: State1D, t: float) -> float:
        """Compute the force acting on a particle.
        
        Args:
            state (State1D): Current state of the particle (position and velocity).
            t (float): Current simulation time in seconds.
            
        Returns:
            float: Scalar force value in Newtons. Positive forces act in the
                positive x-direction.
        """
        pass    
