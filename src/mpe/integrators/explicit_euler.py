"""Explicit Euler integration scheme.

This module implements the explicit (forward) Euler method, a first-order
numerical integration scheme. While simple and fast, it is conditionally
stable and can exhibit energy drift in oscillatory systems.
"""

from src.mpe.integrators import Integrator
from src.mpe.core import State1D
from src.mpe.forces import ForceModel


class ExplicitEuler(Integrator):
    """Explicit Euler integration method.
    
    Implements the forward Euler method: a first-order explicit method
    that updates velocity and position using current state values.
    
    Notes:
        - First-order accurate: O(dt)
        - Conditionally stable: dt must be small
        - Not energy-conserving for oscillatory systems
        - Can exhibit instability with large time steps
        
    Examples:
        >>> integrator = ExplicitEuler()
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
        """Advance state using explicit Euler method.
        
        Updates position using current velocity, then updates velocity
        using current acceleration.
        
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

        # Update position and velocity using explicit Euler
        new_x = state.x + dt * state.v
        new_v = state.v + dt * a
        
        return State1D(new_x, new_v)
