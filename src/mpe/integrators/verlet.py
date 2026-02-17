"""Verlet integration scheme.

This module implements the Velocity Verlet method, a second-order symplectic
integrator widely used in molecular dynamics and physics simulations. It provides
excellent energy conservation and time-reversibility.
"""

from src.mpe.integrators import Integrator
from src.mpe.core import State1D
from src.mpe.forces import ForceModel


class Verlet(Integrator):
    """Velocity Verlet integration method.
    
    Implements the Velocity Verlet algorithm, which updates position using
    current velocity and acceleration, then updates velocity using the average
    of current and new accelerations.
    
    Notes:
        - Second-order accurate: O(dt²)
        - Symplectic: excellent energy conservation
        - Time-reversible
        - Requires two force evaluations per step
        - Standard choice for molecular dynamics
        
    Examples:
        >>> integrator = Verlet()
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
        """Advance state using Velocity Verlet method.
        
        Updates position using current velocity and half the current acceleration,
        then updates velocity using the average of accelerations at the old and
        new positions.
        
        Args:
            state (State1D): Current state (position and velocity).
            force_model (ForceModel): Force model for computing acceleration.
            mass (float): Particle mass in kilograms.
            t (float): Current time in seconds.
            dt (float): Time step size in seconds.
            
        Returns:
            State1D: Updated state after one time step.
        """
        # Compute current acceleration
        force = force_model.compute(state, t)
        a = force / mass

        # Update position with current velocity and acceleration
        new_x = state.x + state.v * dt + 0.5 * a * dt * dt

        # Compute acceleration at new position
        temp_state = State1D(new_x, state.v)
        new_force = force_model.compute(temp_state, t + dt)
        new_a = new_force / mass

        # Update velocity using average of old and new accelerations
        new_v = state.v + 0.5 * (a + new_a) * dt

        return State1D(new_x, new_v)
