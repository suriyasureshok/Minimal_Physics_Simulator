"""Semi-implicit Euler integration scheme.

This module implements the semi-implicit (symplectic) Euler method, also known
as Euler-Cromer method. It updates velocity first, then uses the new velocity
to update position, providing better energy conservation than explicit Euler.
"""

from src.mpe.integrators import Integrator
from src.mpe.core import State1D
from src.mpe.forces import ForceModel


class SemiImplicitEuler(Integrator):
    """Semi-implicit Euler integration method (Euler-Cromer).
    
    Implements the symplectic Euler method which updates velocity first using
    current acceleration, then updates position using the new velocity. This
    ordering makes it symplectic and provides better long-term energy conservation.
    
    Notes:
        - First-order accurate: O(dt)
        - Symplectic: conserves energy better than explicit Euler
        - More stable for oscillatory systems
        - Widely used in games and real-time simulations
        
    Examples:
        >>> integrator = SemiImplicitEuler()
        >>> state = State1D(x=1.0, v=0.0)
        >>> new_state = integrator.step(state, force_model, mass=1.0, t=0.0, dt=0.01)
    """
    
    def step(
            self,
            state: State1D,
            force_model: ForceModel,
            mass: float,
            t: float,
            dt: float
            ) -> State1D:
        """Advance state using semi-implicit Euler method.
        
        Updates velocity first using current acceleration, then updates
        position using the newly computed velocity.
        
        Args:
            state (State1D): Current state (position and velocity).
            force_model (ForceModel): Force model for computing acceleration.
            mass (float): Particle mass in kilograms.
            t (float): Current time in seconds.
            dt (float): Time step size in seconds.
            
        Returns:
            State1D: Updated state after one time step.
        """
        # Compute acceleration from current state
        force = force_model.compute(state, t)
        a = force / mass

        # Update velocity first (semi-implicit)
        new_v = state.v + dt * a
        # Then update position with new velocity
        new_x = state.x + dt * new_v

        return State1D(new_x, new_v)
