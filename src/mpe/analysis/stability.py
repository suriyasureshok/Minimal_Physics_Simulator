"""Numerical stability analysis for integrators.

This module provides tools for detecting numerical instability and
finding maximum stable time steps for different integration schemes.
"""

import numpy as np
from src.mpe.core import State1D
from src.mpe.analysis import oscillator_energy


def is_unstable(positions, velocities, mass, k, amp_threshold=100, energy_threshold=1.0):
    """Detect numerical instability in simulation results.
    
    Checks multiple criteria to determine if a simulation has become
    numerically unstable, including NaN/Inf detection, amplitude growth,
    and energy drift.
    
    Args:
        positions (np.ndarray): Array of positions over time, shape (n_steps,).
        velocities (np.ndarray): Array of velocities over time, shape (n_steps,).
        mass (float): Particle mass in kilograms.
        k (float): Spring constant in N/m.
        amp_threshold (float, optional): Maximum allowed position amplitude.
            Defaults to 100.
        energy_threshold (float, optional): Maximum allowed relative energy drift.
            Defaults to 1.0 (100% drift allowed).
            
    Returns:
        bool: True if instability detected, False otherwise.
        
    Instability Criteria:
        1. NaN or Inf values in positions or velocities
        2. Position amplitude exceeds amp_threshold
        3. Relative energy drift exceeds energy_threshold
        
    Notes:
        - Designed for oscillatory systems (springs)
        - Energy drift criterion assumes conservative dynamics
        - Used to find maximum stable time steps
        
    Examples:
        >>> unstable = is_unstable(positions, velocities, mass=1.0, k=10.0)
        >>> if unstable:
        ...     print("Simulation became unstable!")
    """
    # Check for NaN in positions
    if np.any(np.isnan(positions)) or np.any(np.isnan(velocities)):
        return True
    
    # Check for Inf in positions or velocities
    if np.any(np.isinf(positions)) or np.any(np.isinf(velocities)):
        return True

    # Check if amplitude has grown too large
    if np.max(np.abs(positions)) > amp_threshold:
        return True

    # Check for excessive energy drift
    energy = oscillator_energy(positions, velocities, mass, k)
    
    # Handle edge case where initial energy is zero
    if np.isclose(energy[0], 0.0):
        # Use absolute drift for zero initial energy
        absolute_drift = np.abs(energy - energy[0])
        if np.max(absolute_drift) > 1e-12:  # Small absolute threshold
            return True
    else:
        # Use relative drift for non-zero initial energy
        relative_drift = np.abs(energy - energy[0]) / energy[0]
        if np.max(relative_drift) > energy_threshold:  # Relative energy drift threshold
            return True

    return False


def find_max_stable_dt(
        simulator_factory,
        integrator,
        force_model,
        mass,
        k,
        initial_state,
        dt_values,
        steps=50000,
    ):
    """Find the maximum stable time step for an integrator.
    
    Tests a range of time step values to find the largest dt that
    produces stable simulation results.
    
    Args:
        simulator_factory: Function that creates a Simulator instance.
            Should accept (integrator, force_model, mass) as arguments.
        integrator (Integrator): The integrator to test.
        force_model (ForceModel): The force model to use.
        mass (float): Particle mass in kilograms.
        k (float): Spring constant in N/m (for stability checking).
        initial_state (State1D): Initial conditions for the test.
        dt_values (list or np.ndarray): Time step values to test, in ascending order.
        steps (int, optional): Number of steps to simulate for each test.
            Defaults to 50000.
            
    Returns:
        tuple: A tuple containing:
            - max_stable_dt (float or None): Largest stable time step found,
                or None if all tested values are unstable.
            - stability_result (dict): Dict mapping dt -> bool (stable or not).
            
    Notes:
        - Tests dt values in order until instability is found
        - Assumes stability region is contiguous (smaller dt more stable)
        - Computationally expensive for many dt values or large steps
        
    Examples:
        >>> dt_test = np.linspace(0.001, 0.1, 20)
        >>> max_dt, results = find_max_stable_dt(
        ...     Simulator, integrator, force_model, mass=1.0, k=10.0,
        ...     initial_state, dt_test
        ... )
        >>> print(f"Maximum stable dt: {max_dt}")
    """
    stability_result = {}
    max_stable_dt = None

    for dt in dt_values:
        # Create a fresh simulator for each test
        sim = simulator_factory(integrator, force_model, mass)

        # Create a copy of initial state to avoid mutation
        state_copy = State1D(initial_state.x, initial_state.v)
        positions, velocities = sim.run(state_copy, dt, steps)

        # Check for instability
        unstable = is_unstable(positions, velocities, mass, k)

        stability_result[dt] = not unstable
    
        if not unstable:
            max_stable_dt = dt  # Update max stable value
        else:
            break  # Stop at first unstable dt
            
    return max_stable_dt, stability_result
