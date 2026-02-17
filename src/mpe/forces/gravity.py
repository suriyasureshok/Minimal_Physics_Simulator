"""Gravitational force model.

This module implements a constant gravitational force acting on a particle,
typically used to simulate motion under Earth's gravity.
"""

from src.mpe.forces import ForceModel
from src.mpe.core import State1D


class GravityForce(ForceModel):
    """Constant gravitational force model.
    
    Computes a constant downward force: F = -m * g
    where m is mass and g is gravitational acceleration.
    
    Attributes:
        m (float): Mass of the particle in kilograms.
        g (float): Gravitational acceleration in m/s² (typically 9.81 on Earth).
        
    Notes:
        - The negative sign indicates downward direction
        - Force is independent of position and velocity
        - Results in uniformly accelerated motion
        
    Examples:
        >>> gravity = GravityForce(m=1.0, g=9.81)
        >>> state = State1D(x=10.0, v=-5.0)
        >>> force = gravity.compute(state, t=0.0)
        >>> print(force)  # -9.81 (downward force)
    """
    
    def __init__(self, m: float, g: float):
        """Initialize the gravitational force model.
        
        Args:
            m (float): Mass of the particle in kilograms. Must be positive.
            g (float): Gravitational acceleration in m/s². Typically 9.81 for Earth.
        """
        self.m = m
        self.g = g

    def compute(self, state: State1D, t: float) -> float:
        """Compute constant gravitational force.
        
        Args:
            state (State1D): Current state (unused for constant gravity).
            t (float): Current time in seconds (unused for constant gravity).
            
        Returns:
            float: Gravitational force F = -m * g in Newtons.
        """
        return -self.m * self.g
