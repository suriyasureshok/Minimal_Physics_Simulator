"""Oscillator stability visualization experiment.

This experiment creates comprehensive visualizations comparing the stability
and accuracy of different numerical integrators on a harmonic oscillator system.

The experiment:
    - Simulates a 1D spring-mass oscillator with four integrators
    - Generates multi-panel comparison plots showing:
        * Position trajectories over time
        * Phase space portraits (x vs v)
        * Energy drift over the entire simulation
        * Comparative stability analysis
    - Identifies unstable integrators (Explicit Euler)
    
Visualizations:
    1. Comprehensive 4-panel comparison plot
    2. Explicit Euler instability demonstration (separate)
    
Output:
    - plots/oscillator_stability_comparison.png: 4-panel comparison
    - plots/explicit_euler_instability.png: Instability demo
    
Notes:
    - Explicit Euler shows characteristic exponential energy growth
    - Verlet maintains circular phase space orbit (symplectic)
    - RK4 shows high accuracy but phase space spiral (not symplectic)
    - Semi-Implicit Euler provides good stability at low cost
"""

from src.mpe.core import State1D, Simulator
from src.mpe.integrators import ExplicitEuler, SemiImplicitEuler, Verlet, RK4
from src.mpe.forces import SpringForce
from src.mpe.analysis import oscillator_energy

import os
import numpy as np
import matplotlib.pyplot as plt

# System parameters
m = 1.0    # Mass in kg
k = 10.0   # Spring constant in N/m
omega = np.sqrt(k / m)  # Natural frequency
dt = 0.01  # Time step in seconds
steps = 5000  # Number of simulation steps

# Initial conditions: displaced spring at rest
initial_state = State1D(1.0, 0.0)
force = SpringForce(k)

os.makedirs('plots', exist_ok=True)

# Test all integrators
integrators = {
    'Explicit Euler': ExplicitEuler(),
    'Semi-Implicit Euler': SemiImplicitEuler(),
    'Verlet': Verlet(),
    'RK4': RK4()
}

results = {}
for name, integrator in integrators.items():
    print(f"Running {name}...")
    sim = Simulator(integrator, force, m)
    x, v = sim.run(initial_state, dt, steps)
    energy = oscillator_energy(x, v, m, k)
    results[name] = {'x': x, 'v': v, 'energy': energy}

# ===== VISUALIZATIONS =====

# Figure 1: Comprehensive Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = {'Explicit Euler': '#d62728', 'Semi-Implicit Euler': '#ff7f0e', 
          'Verlet': '#2ca02c', 'RK4': '#1f77b4'}

# Plot 1: Position trajectories
ax1 = axes[0, 0]
t = np.arange(steps) * dt
for name, data in results.items():
    ax1.plot(t[:500], data['x'][:500], label=name, linewidth=2, 
             color=colors[name], alpha=0.8)
ax1.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Position', fontsize=11, fontweight='bold')
ax1.set_title('Position Trajectory Comparison\n(First 500 steps)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot 2: Phase space (x vs v)
ax2 = axes[0, 1]
for name, data in results.items():
    ax2.plot(data['x'][:2000], data['v'][:2000], label=name, linewidth=1.5, 
             color=colors[name], alpha=0.7)
ax2.set_xlabel('Position', fontsize=11, fontweight='bold')
ax2.set_ylabel('Velocity', fontsize=11, fontweight='bold')
ax2.set_title('Phase Space Portrait\n(First 2000 steps)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.axis('equal')

# Plot 3: Energy over time
ax3 = axes[1, 0]
for name, data in results.items():
    energy_drift = data['energy'] - data['energy'][0]
    ax3.plot(t, energy_drift, label=name, linewidth=2, 
             color=colors[name], alpha=0.8)
ax3.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Energy Drift', fontsize=11, fontweight='bold')
ax3.set_title('Energy Conservation', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, linestyle='--')

# Plot 4: Relative energy error (%)
ax4 = axes[1, 1]
for name, data in results.items():
    energy_error_pct = (data['energy'] - data['energy'][0]) / data['energy'][0] * 100
    ax4.plot(t, energy_error_pct, label=name, linewidth=2, 
             color=colors[name], alpha=0.8)
ax4.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Relative Energy Error (%)', fontsize=11, fontweight='bold')
ax4.set_title('Energy Drift Percentage', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.5, label='10% Stability Threshold')
ax4.axhline(y=-10, color='red', linestyle='--', linewidth=2, alpha=0.5)

plt.tight_layout()
plt.savefig('plots/oscillator_stability_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization: plots/oscillator_stability_comparison.png")

# Figure 2: Individual Focus on Explicit Euler (Unstable Behavior)
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

euler_data = results['Explicit Euler']
t = np.arange(steps) * dt

# Position trajectory showing amplitude growth
ax1 = axes2[0]
ax1.plot(t, euler_data['x'], linewidth=2, color='#d62728', alpha=0.8)
ax1.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Position', fontsize=11, fontweight='bold')
ax1.set_title('Explicit Euler: Amplitude Explosion\n(dt=0.01, unstable)', 
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Phase space spiral
ax2 = axes2[1]
ax2.plot(euler_data['x'], euler_data['v'], linewidth=1.5, color='#d62728', alpha=0.8)
ax2.set_xlabel('Position', fontsize=11, fontweight='bold')
ax2.set_ylabel('Velocity', fontsize=11, fontweight='bold')
ax2.set_title('Explicit Euler: Phase Space Spiral\n(Energy Injection)', 
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
# Mark initial and final points
ax2.scatter(euler_data['x'][0], euler_data['v'][0], s=100, c='green', 
            marker='o', edgecolor='black', linewidth=2, label='Start', zorder=5)
ax2.scatter(euler_data['x'][-1], euler_data['v'][-1], s=100, c='red', 
            marker='X', edgecolor='black', linewidth=2, label='End', zorder=5)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('plots/explicit_euler_instability.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization: plots/explicit_euler_instability.png")

plt.show()
