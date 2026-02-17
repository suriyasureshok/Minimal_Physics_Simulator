"""Fourth-order Runge-Kutta integration scheme.

This module implements the classic RK4 method, one of the most widely used
high-accuracy integrators. It achieves fourth-order accuracy by computing
four intermediate slopes per time step.
"""

from src.mpe.integrators import Integrator
from src.mpe.core import State1D
from src.mpe.forces import ForceModel


class RK4(Integrator):
    """Fourth-order Runge-Kutta integration method.
    
    Implements the classic RK4 algorithm, which computes four intermediate
    evaluations (k1, k2, k3, k4) and combines them with specific weights
    to achieve fourth-order accuracy.
    
    Notes:
        - Fourth-order accurate: O(dt⁴)
        - Highly accurate for smooth problems
        - Requires four force evaluations per step
        - Not symplectic (can exhibit energy drift over long times)
        - Excellent for short-to-medium term accuracy
        
    Examples:
        >>> integrator = RK4()
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
        """Advance state using fourth-order Runge-Kutta method.
        
        Computes four intermediate slopes (k1, k2, k3, k4) and combines
        them with weights 1/6, 1/3, 1/3, 1/6 to achieve O(dt⁴) accuracy.
        
        Args:
            state (State1D): Current state (position and velocity).
            force_model (ForceModel): Force model for computing acceleration.
            mass (float): Particle mass in kilograms.
            t (float): Current time in seconds.
            dt (float): Time step size in seconds.
            
        Returns:
            State1D: Updated state after one time step.
        """
        def accelerate(s, time):
            """Helper function to compute acceleration at a given state."""
            return force_model.compute(s, time) / mass

        # k1: slope at beginning of interval
        a1 = accelerate(state, t)
        k1_v = a1
        k1_x = state.v

        # k2: slope at midpoint using k1
        s2 = State1D(
                state.x + 0.5 * dt * k1_x,
                state.v + 0.5 * dt * k1_v
                )
        a2 = accelerate(s2, t + 0.5 * dt)
        k2_v = a2
        k2_x = s2.v

        # k3: slope at midpoint using k2
        s3 = State1D(
                state.x + 0.5 * dt * k2_x,
                state.v + 0.5 * dt * k2_v
                )
        a3 = accelerate(s3, t + 0.5 * dt)
        k3_v = a3
        k3_x = s3.v

        # k4: slope at end of interval using k3
        s4 = State1D(
                state.x + dt * k3_x,
                state.v + dt * k3_v
                )
        a4 = accelerate(s4, t + dt)
        k4_v = a4
        k4_x = s4.v

        # Weighted average of slopes: (k1 + 2*k2 + 2*k3 + k4) / 6
        new_x = state.x + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        new_v = state.v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        return State1D(new_x, new_v)

