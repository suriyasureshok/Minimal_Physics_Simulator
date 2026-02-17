"""Minimal Physics Engine (MPE).

A lightweight, high-performance physics simulator designed for:
- Numerical integration research
- Reinforcement learning environments
- Real-world simulation testing
- Performance benchmarking

Main modules:
- core: Simulation fundamentals (State, TimeKeeper, Simulator)
- integrators: Numerical ODE solvers (Euler, Verlet, RK4)
- forces: Force models (Spring, Gravity, Damped)
- batch: Vectorized parallel processing backends
- analysis: Performance and accuracy analysis tools
- rl: Reinforcement learning integration
- realworld: Real-world simulation effects
"""

# Version
__version__ = '0.1.0'

# Core components
from . import core
from . import integrators
from . import forces
from . import batch
from . import analysis
from . import rl
from . import realworld
# Convenience imports for common usage
from .core import State1D, Simulator
from .integrators import Verlet, RK4, SemiImplicitEuler, ExplicitEuler
from .forces import SpringForce, GravityForce, DampedSpringForce

__all__ = [
    # Version
    '__version__',
    
    # Submodules
    'core',
    'integrators',
    'forces',
    'batch',
    'analysis',
    'rl',
    'realworld',
    
    # Common imports
    'State1D',
    'Simulator',
    'Verlet',
    'RK4',
    'SemiImplicitEuler',
    'ExplicitEuler',
    'SpringForce',
    'GravityForce',
    'DampedSpringForce',
]
