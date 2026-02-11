# src/experiments/energy_drift.py

import os
import numpy as np
import matplotlib.pyplot as plt

from src.mpe.analysis.energy import oscillator_energy
from src.mpe.core.state import State1D
from src.mpe.core.simulator import Simulator
from src.mpe.integrators.verlet import Verlet
from src.mpe.integrators.explicit_euler import ExplicitEuler
from src.mpe.integrators.semi_implicit_euler import SemiImplicitEuler
from src.mpe.integrators.rk4 import RK4
from src.mpe.forces.spring import SpringForce

m = 1.0
k = 10.0
dt = 0.01
steps = 50000

initial_state = State1D(1.0, 0.0)
force = SpringForce(k)

integrators = {
        "Euler" : ExplicitEuler(),
        "SemiImplicit" : SemiImplicitEuler(),
        "Verlet" : Verlet(),
        "RK4" : RK4()
        }

for name, integrator in integrators.items():
    sim = Simulator(integrator, force, m)
    x, v = sim.run(initial_state, dt, steps)

    energy = oscillator_energy(x, v, m, k)

    plt.plot(energy - energy[0], label = name)

os.makedirs('plots', exist_ok= True)

plt.legend()
plt.title("Energy Drift")
plt.xlabel("Step")
plt.ylabel("Energy")
plt.savefig("plots/energy_drift_all.png", dpi=300, bbox_inches='tight')
