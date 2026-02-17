"""Spring force model.

This module implements Hooke's law for an ideal spring, providing a restoring
force proportional to displacement from equilibrium.
"""

from src.mpe.forces import ForceModel
from src.mpe.core import State1D


class SpringForce(ForceModel):
    """Ideal spring force model implementing Hooke's law.
    
    Computes a restoring force proportional to displacement: F = -k * x
    where k is the spring constant and x is displacement from equilibrium.
    
    Attributes:
        k (float): Spring constant in N/m. Must be positive for a restoring force.
        
    Notes:
        - Results in simple harmonic motion for an undamped system
        - Natural frequency: ω = sqrt(k/m)
        - Period: T = 2π * sqrt(m/k)
        
    Examples:
        >>> spring = SpringForce(k=10.0)
        >>> state = State1D(x=1.0, v=0.0)
        >>> force = spring.compute(state, t=0.0)
        >>> print(force)  # -10.0 (restoring force)
    """
    
    def __init__(self, k: float):
        """Initialize the spring force model.
        
        Args:
            k (float): Spring constant in N/m. Should be positive.
        """
        self.k = k
    
    def compute(self, state: State1D, t: float) -> float:
        """Compute spring force using Hooke's law.
        
        Args:
            state (State1D): Current state with position x and velocity v.
            t (float): Current time in seconds (unused for time-independent spring).
            
        Returns:
            float: Spring force F = -k * x in Newtons.
        """
        return -self.k * state.x

