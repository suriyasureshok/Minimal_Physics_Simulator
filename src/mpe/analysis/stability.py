# src/mpe/analysis/stability

import numpy as np
from src.mpe.core.state import State1D
from src.mpe.analysis.energy import oscillator_energy

def is_unstable(positions, velocities, mass, k, amp_threshold = 100, energy_threshold=1.0):
    """
    detect numerical instability .
    Criteria:
        -NaN od Inf present
        - Magnitude exceedes threshold
    """
    
    if np.any(np.isnan(positions)) or np.any(np.isnan(velocities)):
        return True
    
    if np.any(np.isinf(positions)) or np.any(np.isnan(velocities)):
        return True

    if np.max(np.abs(positions)) > amp_threshold:
        return True

    energy = oscillator_energy(positions, velocities, mass, k)
    relative_drift = np.abs(energy - energy[0])/energy[0]

    if np.max(relative_drift) > 0.1:
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

    stability_result = {}
    max_stable_dt = None

    for dt in dt_values:
        sim = simulator_factory(integrator,force_model,mass)

        state_copy = State1D(initial_state.x, initial_state.v)
        positions,velocities = sim.run(state_copy,dt,steps)

        unstable = is_unstable(positions, velocities, mass, k)

        stability_result[dt] = not unstable
    
        if not unstable:
            max_stable_dt = dt
        else:
            break
    return max_stable_dt , stability_result
