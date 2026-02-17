"""Base classes for numerical integration schemes.

This module defines the abstract base class for all integrators used in
the Minimal Physics Engine to advance particle states through time.
"""

from abc import ABC, abstractmethod
from src.mpe.core import State1D
from src.mpe.forces import ForceModel


class Integrator(ABC):
    """Abstract base class for numerical integrators.
    
    Integrators implement numerical schemes for solving ordinary differential
    equations (ODEs) that govern particle motion. Concrete integrators must
    implement the step method to advance the system state.
    
    Examples:
        >>> class MyIntegrator(Integrator):
        ...     def step(self, state, force_model, mass, t, dt):
        ...         # Custom integration logic
        ...         return new_state
    """
    
    @abstractmethod
    def step(
            self,
            state: State1D,
            force_model: ForceModel,
            mass: float,
            t: float,
            dt: float
            ) -> State1D:
        """Advance the system state by one time step.
        
        Args:
            state (State1D): Current state of the particle (position and velocity).
            force_model (ForceModel): The force model to compute forces.
            mass (float): Mass of the particle in kilograms.
            t (float): Current simulation time in seconds.
            dt (float): Time step size in seconds.
            
        Returns:
            State1D: The new state after one time step.
        """
        pass
