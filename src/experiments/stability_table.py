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

# ===== VISUALIZATIONS =====
import os
import matplotlib.pyplot as plt

os.makedirs('plots', exist_ok=True)

# Figure 1: Max Stable dt Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Max Stable Timestep
ax1 = axes[0, 0]
integrator_names = df.index.tolist()
max_dts = df["Max Stable dt"].fillna(0).values
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
bars1 = ax1.bar(integrator_names, max_dts, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Max Stable dt', fontsize=11, fontweight='bold')
ax1.set_title('Maximum Stable Timestep\n(Higher is Better)', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for i, v in enumerate(max_dts):
    if v > 0:
        ax1.text(i, v + max(max_dts)*0.02, f'{v:.4f}', ha='center', fontsize=9)

# Plot 2: Computational Cost (ns/step)
ax2 = axes[0, 1]
ns_per_step = df["ns/step"].values
bars2 = ax2.bar(integrator_names, ns_per_step, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('ns/step', fontsize=11, fontweight='bold')
ax2.set_title('Computational Cost per Step\n(Lower is Better)', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for i, v in enumerate(ns_per_step):
    ax2.text(i, v + max(ns_per_step)*0.02, f'{v:.0f}', ha='center', fontsize=9)

# Plot 3: Effective Throughput (Sim-time/sec)
ax3 = axes[1, 0]
throughput = df["Sim-time/sec"].values
bars3 = ax3.bar(integrator_names, throughput, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Simulated-seconds per real-second', fontsize=11, fontweight='bold')
ax3.set_title('Effective Throughput\n(Higher is Better)', fontsize=12, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(axis='y', alpha=0.3, linestyle='--', which='both')
for i, v in enumerate(throughput):
    if v > 0:
        ax3.text(i, v * 1.3, f'{v:.0f}', ha='center', fontsize=9)

# Plot 4: FLOPs Comparison
ax4 = axes[1, 1]
flops = df["FLOPs"].values
bars4 = ax4.bar(integrator_names, flops, color=colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('FLOPs per step', fontsize=11, fontweight='bold')
ax4.set_title('Floating-Point Operations\n(Lower is Better)', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3, linestyle='--')
for i, v in enumerate(flops):
    ax4.text(i, v + max(flops)*0.02, f'{v}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('plots/integrator_stability_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization: plots/integrator_stability_comparison.png")

# Figure 2: Efficiency Plot (Throughput vs Cost)
fig2, ax = plt.subplots(figsize=(10, 6))
valid_integrators = df[df["Sim-time/sec"] > 0]
x = valid_integrators["ns/step"].values
y = valid_integrators["Sim-time/sec"].values
names = valid_integrators.index.tolist()

scatter = ax.scatter(x, y, s=300, c=colors[:len(names)], alpha=0.6, edgecolor='black', linewidth=2)
for i, name in enumerate(names):
    ax.annotate(name, (x[i], y[i]), fontsize=11, fontweight='bold', 
                ha='center', va='center')

ax.set_xlabel('Computational Cost (ns/step)', fontsize=12, fontweight='bold')
ax.set_ylabel('Effective Throughput (sim-s/real-s)', fontsize=12, fontweight='bold')
ax.set_title('Integrator Efficiency: Throughput vs Cost\n(Upper-left corner is optimal)', 
             fontsize=13, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('plots/integrator_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: plots/integrator_efficiency.png")

plt.show()
