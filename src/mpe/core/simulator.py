"""Core physics simulation engine.

This module provides the main Simulator class that orchestrates physics
simulations by combining integrators, force models, and time management.
"""

import numpy as np
from src.mpe.core import State1D
from src.mpe.integrators import Integrator
from src.mpe.forces import ForceModel
from src.mpe.core import TimeKeeper


class Simulator:
    """Orchestrates physics simulations using pluggable integrators and force models.
    
    The Simulator class provides a high-level interface for running physics
    simulations. It combines an integration scheme, a force model, and mass
    properties to evolve particle states over time.
    
    Attributes:
        integrator (Integrator): The numerical integration method to use.
        force_model (ForceModel): The force model defining system dynamics.
        mass (float): Mass of the particle in kilograms.
        
    Examples:
        >>> from src.mpe.integrators.verlet import VerletIntegrator
        >>> from src.mpe.forces.spring import Spring
        >>> integrator = VerletIntegrator()
        >>> force_model = Spring(k=10.0, x0=0.0)
        >>> sim = Simulator(integrator, force_model, mass=1.0)
        >>> initial = State1D(x=1.0, v=0.0)
        >>> positions, velocities = sim.run(initial, dt=0.01, steps=1000)
    """
    
    def __init__(
            self, 
            integrator: Integrator,
            force_model: ForceModel,
            mass: float
            ):
        """Initialize the Simulator.
        
        Args:
            integrator (Integrator): The numerical integration scheme to use
                for time-stepping the equations of motion.
            force_model (ForceModel): The force model that defines the
                system's dynamics.
            mass (float): Mass of the particle in kilograms. Must be positive.
        """
        self.integrator = integrator
        self.force_model = force_model
        self.mass = mass

    def run(
            self,
            initial_state: State1D,
            dt: float,
            steps: int
            ):
        """Run the physics simulation.
        
        Evolves the particle state over the specified number of time steps
        using the configured integrator and force model.
        
        Args:
            initial_state (State1D): Initial position and velocity of the particle.
            dt (float): Time step size in seconds. Must be positive.
            steps (int): Number of simulation steps to perform. Must be positive.
            
        Returns:
            tuple: A tuple containing:
                - positions (np.ndarray): Array of positions at each step, shape (steps,).
                - velocities (np.ndarray): Array of velocities at each step, shape (steps,).
                
        Examples:
            >>> sim = Simulator(integrator, force_model, mass=1.0)
            >>> initial = State1D(x=0.0, v=1.0)
            >>> pos, vel = sim.run(initial, dt=0.01, steps=100)
            >>> print(pos.shape, vel.shape)
            (100,) (100,)
        """
        state = initial_state
        timekeeper = TimeKeeper(dt)

        # Pre-allocate arrays for storing trajectory
        positions = np.zeros(steps, dtype=np.float64)
        velocities = np.zeros(steps, dtype=np.float64)

        # Main simulation loop
        for i in range(steps):
            # Advance state by one time step
            state = self.integrator.step(
                    state,
                    self.force_model,
                    self.mass,
                    timekeeper.t,
                    dt
                    )
            timekeeper.advance()

            # Store current state
            positions[i] = state.x
            velocities[i] = state.v

        return positions, velocities
