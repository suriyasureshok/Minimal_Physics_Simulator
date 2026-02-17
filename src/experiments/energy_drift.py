"""Energy drift comparison experiment.

This experiment compares energy conservation properties of different
numerical integrators by simulating a harmonic oscillator and measuring
energy drift over time. Ideal integrators should preserve total energy.

The experiment:
    - Simulates a 1D harmonic oscillator (spring system)
    - Compares Semi-Implicit Euler, Verlet, and RK4 integrators
    - Measures energy drift from initial value over 50,000 steps
    - Generates visualization of energy drift vs. time
    
Results:
    - Verlet shows best energy conservation (symplectic)
    - RK4 shows some drift (not symplectic, but high accuracy)
    - Semi-Implicit Euler shows moderate drift
    
Output:
    - plots/energy_drift_without_explicit.png: Energy drift comparison plot
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.mpe.analysis import oscillator_energy
from src.mpe.core import State1D, Simulator
from src.mpe.integrators import Verlet, ExplicitEuler, SemiImplicitEuler, RK4
from src.mpe.forces import SpringForce

# System parameters
m = 1.0    # Mass in kg
k = 10.0   # Spring constant in N/m
dt = 0.01  # Time step in seconds
steps = 50000  # Number of simulation steps

# Initial conditions: displaced spring at rest
initial_state = State1D(1.0, 0.0)
force = SpringForce(k)

# Integrators to compare
integrators = {
        # "Euler" : ExplicitEuler(),  # Commented out: too unstable
        "SemiImplicit" : SemiImplicitEuler(),
        "Verlet" : Verlet(),
        "RK4" : RK4()
        }

# Run simulations and plot energy drift
for name, integrator in integrators.items():
    sim = Simulator(integrator, force, m)
    x, v = sim.run(initial_state, dt, steps)

    # Compute total mechanical energy
    energy = oscillator_energy(x, v, m, k)

    # Plot drift from initial energy
    plt.plot(energy - energy[0], label = name)

# Save results
os.makedirs('plots', exist_ok= True)

plt.legend()
plt.title("Energy Drift")
plt.xlabel("Step")
plt.ylabel("Energy")
plt.savefig("plots/energy_drift_without_explicit.png", dpi=150, bbox_inches='tight')
