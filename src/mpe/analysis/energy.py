"""Energy analysis utilities for physics simulations.

This module provides functions for computing and analyzing energy in
physics simulations, particularly for detecting numerical energy drift.
"""

import numpy as np


def oscillator_energy(x, v, m, k):
    """Compute total mechanical energy of a harmonic oscillator.
    
    Calculates the sum of kinetic and potential energy for a particle
    attached to a spring.
    
    Args:
        x (float or np.ndarray): Position(s) of the particle in meters.
        v (float or np.ndarray): Velocity(ies) of the particle in m/s.
        m (float): Mass of the particle in kilograms.
        k (float): Spring constant in N/m.
        
    Returns:
        float or np.ndarray: Total mechanical energy in Joules.
            Returns array if x and v are arrays.
            
    Notes:
        - Kinetic energy: KE = (1/2) * m * v²
        - Potential energy: PE = (1/2) * k * x²
        - Total conserved for ideal harmonic oscillator
        
    Examples:
        >>> energy = oscillator_energy(x=1.0, v=0.0, m=1.0, k=10.0)
        >>> print(energy)  # 5.0 (all potential)
    """
    kinetic = 0.5 * m * v ** 2
    potential = 0.5 * k * x ** 2
    return kinetic + potential


def energy_drift(energy):
    """Compute energy drift relative to initial energy.
    
    Calculates how much the energy has drifted from its initial value,
    which indicates numerical integration errors in conservative systems.
    
    Args:
        energy (np.ndarray): Array of energy values over time, shape (n_steps,).
            Must be non-empty.
        
    Returns:
        np.ndarray: Energy drift at each time step, shape (n_steps,).
            Zero at first time step by definition.
            
    Raises:
        ValueError: If energy array is empty.
            
    Notes:
        - Ideal conservative systems should have zero drift
        - Non-zero drift indicates numerical errors or dissipation
        - Symplectic integrators typically show bounded drift
        - Non-symplectic methods may show unbounded drift
        
    Examples:
        >>> energy = np.array([10.0, 10.1, 10.2, 10.3])
        >>> drift = energy_drift(energy)
        >>> print(drift)  # [0.0, 0.1, 0.2, 0.3]
    """
    if energy.size == 0:
        raise ValueError("energy must be non-empty")
    return energy - energy[0]
