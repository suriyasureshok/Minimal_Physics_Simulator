"""Numerical integration schemes.

This module provides various numerical integrators for solving ODEs:
- Explicit Euler: O(dt), conditionally stable
- Semi-Implicit Euler: O(dt), symplectic, better stability
- Verlet: O(dt²), symplectic, excellent energy conservation
- RK4: O(dt⁴), high accuracy, not symplectic
"""

from src.mpe.integrators.base import Integrator
from src.mpe.integrators.explicit_euler import ExplicitEuler
from src.mpe.integrators.semi_implicit_euler import SemiImplicitEuler
from src.mpe.integrators.verlet import Verlet
from src.mpe.integrators.rk4 import RK4

__all__ = [
    'Integrator',
    'ExplicitEuler',
    'SemiImplicitEuler',
    'Verlet',
    'RK4',
]
