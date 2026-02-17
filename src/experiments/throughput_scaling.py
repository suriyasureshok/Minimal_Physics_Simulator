import numpy as np 
import torch
import pandas as pd 

from src.mpe.batch.python_loop import PythonLoopIntegrator
from src.mpe.batch.numpy_vectorized import NumpyVectorizedIntegrator
from src.mpe.batch.torch_cpu import TorchCPUIntegrator
from src.mpe.batch.benchmark import benchmark_backend


m = 1.0
k = 10.0
k_over_m = k/m 

dt = 0.001 
steps = 1000

particle_count = [1,1000,100000]

result = []

for N in particle_count:
    print(f'\nBenchmarking N = {N}')
    x_np = np.ones(N,dtype = np.float32)
    v_np = np.zeros(N,dtype=np.float32)

    x_torch = torch.ones(N,dtype = torch.float32)
    v_torch = torch.zeros(N,dtype = torch.float32)

    backends = {
        'PythonLoop' : PythonLoopIntegrator(k_over_m),
        'NumPy' : NumpyVectorizedIntegrator(k_over_m),
        'TorchCPU' : TorchCPUIntegrator(k_over_m)
    }

    for name , integrator in backends.items():
        if name == 'TorchCPU':
            steps_per_sec , total_time = benchmark_backend(
                integrator,x_torch,v_torch,dt,steps

            )
        else:
            steps_per_sec,total_time = benchmark_backend(
                integrator,x_np.copy(),v_np.copy(),dt,steps
            )


        particle_steps_per_sec = steps_per_sec * N 

        bytes_per_particle = 16
        bandwidth_estimate = (
            particle_steps_per_sec * bytes_per_particle
        ) / 1e9

        result.append([
            N ,
            name,
            steps_per_sec,
            particle_steps_per_sec,
            bandwidth_estimate
        ])

df = pd.DataFrame(
    result,
    columns = [
        'Particles',
        'Backend',
        'Steps/sec',
        'Particle-Steps/sec',
        'Estimated Bandwidth (GB/s)'
    ]
)

print("\nThroughput results:")
print(df)

# ===== VISUALIZATIONS =====
import os
import matplotlib.pyplot as plt

os.makedirs('plots', exist_ok=True)

# Figure 1: Particle-Steps/sec vs Particle Count
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

backends = df['Backend'].unique()
colors = {'PythonLoop': '#d62728', 'NumPy': '#2ca02c', 'TorchCPU': '#1f77b4'}
markers = {'PythonLoop': 'o', 'NumPy': 's', 'TorchCPU': '^'}

# Plot 1: Particle-Steps/sec (Linear Scale)
ax1 = axes[0, 0]
for backend in backends:
    data = df[df['Backend'] == backend]
    ax1.plot(data['Particles'], data['Particle-Steps/sec'], 
             marker=markers[backend], linewidth=2.5, markersize=10,
             color=colors[backend], label=backend, alpha=0.8)
ax1.set_xlabel('Number of Particles', fontsize=11, fontweight='bold')
ax1.set_ylabel('Particle-Steps/sec', fontsize=11, fontweight='bold')
ax1.set_title('Throughput Scaling (Linear)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xscale('log')

# Plot 2: Particle-Steps/sec (Log-Log Scale)
ax2 = axes[0, 1]
for backend in backends:
    data = df[df['Backend'] == backend]
    ax2.plot(data['Particles'], data['Particle-Steps/sec'], 
             marker=markers[backend], linewidth=2.5, markersize=10,
             color=colors[backend], label=backend, alpha=0.8)
ax2.set_xlabel('Number of Particles', fontsize=11, fontweight='bold')
ax2.set_ylabel('Particle-Steps/sec', fontsize=11, fontweight='bold')
ax2.set_title('Throughput Scaling (Log-Log)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--', which='both')
ax2.set_xscale('log')
ax2.set_yscale('log')

# Plot 3: Memory Bandwidth Utilization
ax3 = axes[1, 0]
for backend in backends:
    data = df[df['Backend'] == backend]
    ax3.plot(data['Particles'], data['Estimated Bandwidth (GB/s)'], 
             marker=markers[backend], linewidth=2.5, markersize=10,
             color=colors[backend], label=backend, alpha=0.8)
ax3.set_xlabel('Number of Particles', fontsize=11, fontweight='bold')
ax3.set_ylabel('Bandwidth (GB/s)', fontsize=11, fontweight='bold')
ax3.set_title('Memory Bandwidth Utilization', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10, framealpha=0.9)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xscale('log')
# Add DDR4 theoretical limit line
ax3.axhline(y=51.2, color='red', linestyle='--', linewidth=2, alpha=0.5, label='DDR4 Limit (~51 GB/s)')
ax3.legend(fontsize=9, framealpha=0.9)

# Plot 4: Speedup Comparison (Bar Chart at N=100,000)
ax4 = axes[1, 1]
max_N_data = df[df['Particles'] == df['Particles'].max()]
x_pos = np.arange(len(max_N_data))
bars = ax4.bar(max_N_data['Backend'], max_N_data['Particle-Steps/sec'], 
               color=[colors[b] for b in max_N_data['Backend']], 
               alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Particle-Steps/sec', fontsize=11, fontweight='bold')
ax4.set_title(f'Throughput at N={max_N_data["Particles"].values[0]:,}', fontsize=12, fontweight='bold')
ax4.set_yscale('log')
ax4.grid(axis='y', alpha=0.3, linestyle='--', which='both')
# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, max_N_data['Particle-Steps/sec'])):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height * 1.3,
             f'{val:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/throughput_scaling.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization: plots/throughput_scaling.png")

# Figure 2: Backend Comparison Heatmap
fig2, ax = plt.subplots(figsize=(10, 6))
pivot_data = df.pivot(index='Backend', columns='Particles', values='Particle-Steps/sec')
im = ax.imshow(pivot_data.values, cmap='viridis', aspect='auto', norm=plt.matplotlib.colors.LogNorm())
ax.set_xticks(np.arange(len(pivot_data.columns)))
ax.set_yticks(np.arange(len(pivot_data.index)))
ax.set_xticklabels([f'{int(p):,}' for p in pivot_data.columns])
ax.set_yticklabels(pivot_data.index)
ax.set_xlabel('Number of Particles', fontsize=12, fontweight='bold')
ax.set_ylabel('Backend', fontsize=12, fontweight='bold')
ax.set_title('Throughput Heatmap: Particle-Steps/sec\n(Log Scale)', fontsize=13, fontweight='bold')

# Add value annotations
for i in range(len(pivot_data.index)):
    for j in range(len(pivot_data.columns)):
        value = pivot_data.values[i, j]
        text = ax.text(j, i, f'{value:.2e}', ha="center", va="center", 
                      color="white" if value < pivot_data.values.max()/10 else "black",
                      fontsize=9, fontweight='bold')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Particle-Steps/sec (log scale)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/throughput_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: plots/throughput_heatmap.png")

plt.show()

