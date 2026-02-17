"""Analysis utilities for physics simulations.

This module provides tools for analyzing numerical integrator performance:
- Energy conservation analysis
- Error measurement against analytic solutions
- Performance metrics (timing, FLOPs estimation)
- Stability detection and testing
"""

from .energy import oscillator_energy, energy_drift
from .error import analytic_solution, absolute_error, l2_error
from .metrics import measure_ns_per_step, estimate_flops
from .stability import is_unstable, find_max_stable_dt
__all__ = [
    # Energy analysis
    'oscillator_energy',
    'energy_drift',
    
    # Error analysis
    'analytic_solution',
    'absolute_error',
    'l2_error',
    
    # Performance metrics
    'measure_ns_per_step',
    'estimate_flops',
    
    # Stability testing
    'is_unstable',
    'find_max_stable_dt',
]
