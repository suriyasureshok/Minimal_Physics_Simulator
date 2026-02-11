# src/experiments/stability_table.py
"""
Stability Analysis for Numerical Integrators

This script generates a comprehensive performance and stability table for various
numerical integration schemes applied to a simple harmonic oscillator.

STABILITY DEFINITION:
---------------------
An integrator is considered "stable" for a given timestep dt if it satisfies ALL of:
1. No NaN or Inf values in position or velocity
2. Amplitude remains bounded (|x| < 100 for initial amplitude ~1)
3. Energy drift < 10% over the simulation period (10,000 steps)

The "Max Stable dt" represents the largest timestep that maintains numerical stability
for a harmonic oscillator with spring constant k=10 and mass m=1 (natural frequency ω≈3.16).

PERFORMANCE METRICS:
--------------------
- ns/step: Nanoseconds required per integration step (lower is faster)
- FLOPs: Floating-point operations per step (includes force evaluations)
- Simulated time/sec: How much simulation time can be advanced per real second
                      = (max_stable_dt × 1e9) / ns_per_step
"""

import numpy as np
import pandas as pd
import time

from src.mpe.core.state import State1D
from src.mpe.integrators.explicit_euler import ExplicitEuler
from src.mpe.integrators.semi_implicit_euler import SemiImplicitEuler
from src.mpe.integrators.verlet import Verlet
from src.mpe.integrators.rk4 import RK4
from src.mpe.forces.spring import SpringForce
from src.mpe.analysis.stability import find_max_stable_dt


from src.mpe.core.simulator import Simulator


def simulator_factory(integrator ,force_model,mass):
    return Simulator(integrator,force_model,mass)


def measure_step_performance(integrator, force, mass, initial_state, dt, num_iterations=10000):
    state = State1D(initial_state.x, initial_state.v)
    
    # Warm-up
    for _ in range(100):
        state = integrator.step(state, force, mass, 0.0, dt)
    
    # Actual timing
    state = State1D(initial_state.x, initial_state.v)
    start = time.perf_counter_ns()
    
    for i in range(num_iterations):
        state = integrator.step(state, force, mass, float(i) * dt, dt)
    
    end = time.perf_counter_ns()
    
    return (end - start) / num_iterations


def estimate_flops(integrator_name):
    flops_map = {
        "Euler": 6,
        "SemiImplicit": 6,
        "Verlet": 13,
        "RK4": 38
    }
    return flops_map.get(integrator_name, 0)


m = 1.0
k = 10.0

initial_state = State1D(1.0,0.0)
force = SpringForce(k)

dt_values = np.linspace(0.0005,1.0,800)

integrators = {
        "Euler" : ExplicitEuler(),
        "SemiImplicit" : SemiImplicitEuler(),
        "Verlet" : Verlet(),
        "RK4" : RK4()
}

results = {}

for name,integrator in integrators.items():
    print(f"Computing stability for {name}...")

    max_dt,stability_map = find_max_stable_dt(
            simulator_factory,
            integrator,
            force,
            m,
            k,
            initial_state,
            dt_values,
            steps=10000
        )
    
    # Measure performance at a small stable timestep
    benchmark_dt = 0.001
    ns_per_step = measure_step_performance(integrator, force, m, initial_state, benchmark_dt)
    
    # Estimate FLOPs
    flops = estimate_flops(name)
    
    # Calculate simulated time per real second
    # If max_dt found and ns_per_step measured:
    # sim_time_per_sec = max_dt * (1e9 / ns_per_step)
    if max_dt is not None and ns_per_step > 0:
        sim_time_per_sec = max_dt * (1e9 / ns_per_step)
    else:
        sim_time_per_sec = 0.0
    
    results[name] = {
        "Max Stable dt": max_dt,
        "ns/step": round(ns_per_step, 2),
        "FLOPs": flops,
        "Sim-time/sec": round(sim_time_per_sec, 2)
    }


df = pd.DataFrame.from_dict(
        results,
        orient="index"
    )

# Reorder columns for better readability
df = df[["Max Stable dt", "ns/step", "FLOPs", "Sim-time/sec"]]

print("\n" + "="*70)
print("STABILITY AND PERFORMANCE TABLE")
print("="*70)
print(df.to_string())
print("="*70)
print("\nNotes:")
print("- Max Stable dt: Largest timestep maintaining numerical stability")
print("- ns/step: Nanoseconds per integration step (lower is faster)")
print("- FLOPs: Floating-point operations per step")
print("- Sim-time/sec: Simulated seconds advanced per real-world second")
