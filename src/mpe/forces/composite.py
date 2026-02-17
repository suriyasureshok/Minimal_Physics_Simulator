"""Composite force model.

This module implements a composite pattern for combining multiple force models,
enabling complex physical systems with multiple simultaneous forces.
"""

from src.mpe.forces import ForceModel
from src.mpe.core import State1D


class CompositeForce(ForceModel):
    """Combines multiple force models into a single composite force.
    
    Computes the total force by summing the contributions from all
    constituent force models. This allows complex systems to be built
    from simple, reusable force components.
    
    Attributes:
        forces (list): List of ForceModel instances to combine.
        
    Notes:
        - Forces are evaluated sequentially and summed
        - Order of forces in the list doesn't affect the result
        - Enables modular design of complex force systems
        
    Examples:
        >>> spring = SpringForce(k=10.0)
        >>> gravity = GravityForce(m=1.0, g=9.81)
        >>> combined = CompositeForce([spring, gravity])
        >>> state = State1D(x=1.0, v=0.0)
        >>> total_force = combined.compute(state, t=0.0)
    """
    
    def __init__(self, forces):
        """Initialize the composite force model.
        
        Args:
            forces (list): List of ForceModel instances to combine.
                Each element must implement the ForceModel interface.
        """
        self.forces = forces

    def compute(self, state: State1D, t: float) -> float:
        """Compute total force from all constituent forces.
        
        Args:
            state (State1D): Current state with position x and velocity v.
            t (float): Current simulation time in seconds.
            
        Returns:
            float: Sum of all forces in Newtons.
        """
        total = 0.0
        for f in self.forces:
            total += f.compute(state, t)
        return total
