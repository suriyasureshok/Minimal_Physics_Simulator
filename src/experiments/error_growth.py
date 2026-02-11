# src/experiments/error_growth.py

import os
import numpy as np
import matplotlib.pyplot as plt

from src.mpe.core.state import State1D
from src.mpe.core.simulator import Simulator

from src.mpe.integrators.explicit_euler import ExplicitEuler
from src.mpe.integrators.semi_implicit_euler import SemiImplicitEuler
from src.mpe.integrators.verlet import Verlet
from src.mpe.integrators.rk4 import RK4

from src.mpe.forces.spring import SpringForce

from src.mpe.analysis.error import analytic_solution, absolute_error
from src.mpe.analysis.energy import oscillator_energy

# System parameters
m = 1.0
k = 10.0
omega = np.sqrt(k / m)

dt = 0.001
steps = 1_000_000

initial_state = State1D(1.0, 0.0)
force = SpringForce(k)

integrators = {
        "Euler" : ExplicitEuler(),
        "SemiImplicit" : SemiImplicitEuler(),
        "Verlet" : Verlet(),
        "RK4" : RK4()
        }

t_array = np.arange(steps) * dt
x_true = analytic_solution(t_array, A=1.0, omega=omega)

results = {}

for name, integrator in integrators.items():
    sim = Simulator(integrator, force, m)

    print(f"Running {name}...")
    x_num, v_num = sim.run(initial_state, dt, steps)

    error = absolute_error(x_num, x_true)

    results[name] = error

    print(f"{name} final L2 error: {np.sqrt(np.mean(error**2))}")

# Plot error growth
plt.figure(figsize=(10,6))
for name, error in results.items():
    plt.plot(error, label=name)

os.makedirs('plots', exist_ok=True)

plt.yscale("log")
plt.legend()
plt.title("Error Growth Over Time (log scale)")
plt.xlabel("Step")
plt.ylabel("Absolute Error")
plt.savefig('plots/error_growth_over_time.png', dpi=300, bbox_inches='tight')
