"""Damped spring force model.

This module implements a spring-damper system combining Hooke's law with
velocity-dependent damping, commonly used in mechanical systems and vibration analysis.
"""

from src.mpe.forces import ForceModel
from src.mpe.core import State1D


class DampedSpringForce(ForceModel):
    """Damped spring force model with velocity-dependent damping.
    
    Computes combined spring and damping forces: F = -k * x - c * v
    where k is spring constant, x is displacement, c is damping coefficient,
    and v is velocity.
    
    Attributes:
        k (float): Spring constant in N/m.
        c (float): Damping coefficient in N·s/m.
        
    Notes:
        - Damping ratio: ζ = c / (2 * sqrt(k*m))
        - Underdamped: ζ < 1 (oscillatory decay)
        - Critically damped: ζ = 1 (fastest return without oscillation)
        - Overdamped: ζ > 1 (slow return without oscillation)
        
    Examples:
        >>> damped_spring = DampedSpringForce(k=10.0, c=0.5)
        >>> state = State1D(x=1.0, v=2.0)
        >>> force = damped_spring.compute(state, t=0.0)
        >>> print(force)  # -10.0 - 1.0 = -11.0
    """
    
    def __init__(self, k: float, c: float):
        """Initialize the damped spring force model.
        
        Args:
            k (float): Spring constant in N/m. Should be positive.
            c (float): Damping coefficient in N·s/m. Should be non-negative.
        """
        self.k = k
        self.c = c

    def compute(self, state: State1D, t: float) -> float:
        """Compute damped spring force.
        
        Args:
            state (State1D): Current state with position x and velocity v.
            t (float): Current time in seconds (unused for time-independent damping).
            
        Returns:
            float: Combined force F = -k*x - c*v in Newtons.
        """
        return -self.k * state.x - self.c * state.v


