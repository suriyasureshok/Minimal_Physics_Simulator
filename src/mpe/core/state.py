"""State representations for 1D physics simulations.

This module defines the fundamental state representation used throughout
the Minimal Physics Engine for tracking particle positions and velocities
in one-dimensional simulations.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class State1D:
    """Represents the state of a particle in 1D space.
    
    This dataclass encapsulates the position and velocity of a single
    particle in one-dimensional space, providing a lightweight 
    representation for physics simulations.
    
    Attributes:
        x (float): Position of the particle in meters.
        v (float): Velocity of the particle in meters per second.
        
    Examples:
        >>> state = State1D(x=0.0, v=1.0)
        >>> print(f"Position: {state.x}, Velocity: {state.v}")
        Position: 0.0, Velocity: 1.0
    """
    x: float
    v: float
