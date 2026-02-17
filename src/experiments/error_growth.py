"""Numerical error growth analysis experiment.

This experiment measures how numerical error accumulates over long simulations
by comparing integrator results against the analytic solution for a harmonic
oscillator.

The experiment:
    - Simulates a 1D harmonic oscillator for 1 million steps
    - Compares all four integrators against exact solution
    - Measures absolute error vs. time on logarithmic scale
    - Demonstrates order of accuracy for each method
    
Expected behavior:
    - Explicit Euler: O(dt) error growth
    - Semi-Implicit Euler: O(dt) error, better stability
    - Verlet: O(dt²) error
    - RK4: O(dt⁴) error (best short-term accuracy)
    
Output:
    - plots/error_growth_over_time.png: Log-scale error growth plot
    - Console output: Final L2 error for each integrator
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.mpe.core import State1D, Simulator
from src.mpe.integrators import ExplicitEuler, SemiImplicitEuler, Verlet, RK4
from src.mpe.forces import SpringForce
from src.mpe.analysis import analytic_solution, absolute_error, oscillator_energy

# System parameters
m = 1.0    # Mass in kg
k = 10.0   # Spring constant in N/m
omega = np.sqrt(k / m)  # Natural frequency

# Simulation parameters
dt = 0.001        # Small time step for accuracy
steps = 1_000_000  # Long simulation to see error growth

# Initial conditions: unit displacement, zero velocity
initial_state = State1D(1.0, 0.0)
force = SpringForce(k)

# Integrators to test
integrators = {
        "Euler" : ExplicitEuler(),
        "SemiImplicit" : SemiImplicitEuler(),
        "Verlet" : Verlet(),
        "RK4" : RK4()
        }

# Compute analytic solution
t_array = np.arange(steps) * dt
x_true = analytic_solution(t_array, A=1.0, omega=omega)

results = {}

# Run each integrator and compute error
for name, integrator in integrators.items():
    sim = Simulator(integrator, force, m)

    print(f"Running {name}...")
    x_num, v_num = sim.run(initial_state, dt, steps)

    # Compute absolute error at each step
    error = absolute_error(x_num, x_true)

    results[name] = error

    # Print final L2 error metric
    print(f"{name} final L2 error: {np.sqrt(np.mean(error**2))}")

# Plot error growth on log scale
plt.figure(figsize=(10,6))
for name, error in results.items():
    plt.plot(error, label=name)

os.makedirs('plots', exist_ok=True)

plt.yscale("log")
plt.legend()
plt.title("Error Growth Over Time (log scale)")
plt.xlabel("Step")
plt.ylabel("Absolute Error")
plt.savefig('plots/error_growth_over_time.png', dpi=150, bbox_inches='tight')
