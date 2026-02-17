# Minimal Physics Simulator

**A First-Principles Physics Engine for Numerical Analysis and Performance Engineering**

![Project Status](https://img.shields.io/badge/Status-Phase%205%20Complete-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Table of Contents

1. [Overview](#overview)
2. [Project Philosophy](#project-philosophy)
3. [Key Achievements](#key-achievements)
4. [File Structure](#file-structure)
5. [Technical Architecture](#technical-architecture)
6. [Phases & Progress](#phases--progress)
7. [Experimental Results](#experimental-results)
8. [Key Findings & Conclusions](#key-findings--conclusions)
9. [Installation & Usage](#installation--usage)
10. [Future Roadmap](#future-roadmap)
11. [References](#references)

---

## Overview

The **Minimal Physics Simulator** is a comprehensive research project that systematically investigates numerical integration methods, performance optimization strategies, and hardware utilization patterns in computational physics environments. Built entirely from first principles without relying on black-box libraries, this project provides complete transparency into the relationship between algorithm selection, implementation choices, and system performance.

### Research Objectives

- **Numerical Stability Analysis**: Understand how integrator selection affects energy conservation and long-term system behavior
- **Performance Engineering**: Characterize compute-bound vs memory-bound regimes across problem scales
- **Systems Design**: Build modular, deterministic infrastructure for reproducible experiments
- **Real-World Robustness**: Investigate stability under floating-point precision constraints, timing jitter, and latency

### Industrial Relevance

This project directly applies to:
- **Molecular Dynamics** — Pharmaceutical research, materials science
- **Reinforcement Learning** — Environment rollout optimization
- **Orbital Mechanics** — Aerospace trajectory planning
- **Game Physics** — Real-time stable simulation
- **Robotics** — Control loop stability and sim-to-real transfer

---

## Project Philosophy

### Ground-Up Implementation Strategy

All components are implemented from scratch to maintain complete transparency:

| Component | Implementation | Benefit |
|-----------|----------------|---------|
| **Integrators** | Explicit update equations | Full visibility into numerical schemes |
| **Force Models** | Direct physics-to-code translation | No hidden approximations |
| **State Management** | Explicit propagation, zero caches | Complete determinism |
| **Performance Instrumentation** | Direct timer measurement | No profiler artifacts |

### Core Principles

1. **Determinism First**: Zero randomness, fixed-order operations, 100% reproducible
2. **Measurement-Driven**: "If you don't measure it, you don't understand it"
3. **Clean Abstractions**: Modular design enables rapid component substitution
4. **No Black Boxes**: SciPy/MuJoCo/frameworks avoided for complete transparency

---

## Key Achievements

### Phase 1 — Core Architecture
✅ Designed modular, deterministic simulation framework with zero hidden state  
✅ Achieved 100% reproducibility across 10,000+ repeated simulations  
✅ Established clean abstraction boundaries for rapid experimentation  

### Phase 2 — Numerical Analysis
✅ Demonstrated that **symplectic structure preservation** dominates local truncation error order  
✅ Velocity Verlet achieved **245× higher effective throughput** than Explicit Euler  
✅ Established energy-drift stability criteria (10% threshold over 10K steps)  

### Phase 3 — Performance Engineering
✅ Characterized three performance regimes: overhead-bound, interpreter-bound, memory-bandwidth-bound  
✅ Achieved **73% throughput improvement** through in-place array operations  
✅ Final sustained bandwidth: **22.1 GB/s** at **1.38 billion particle-steps/sec**  

### Phase 4 — RL Infrastructure Engineering
✅ Achieved **44.3 million transitions/sec** with 4,096 parallel environments  
✅ Deterministic execution validated (bitwise identical across rollouts)  
✅ Identified **RAM capacity** as primary scaling bottleneck (not compute)  

### Phase 5 — Real-World Stability Constraints
✅ Demonstrated float32 vs float64 show **<0.02% difference** in energy drift for stable integrators  
✅ Established stability hierarchy: **Latency > Jitter > Async > Precision**  
✅ Discovered phase drift (3000% velocity error) dominates despite bounded energy drift (0.5%)  

---

## File Structure

```
Minimal_Physics_Simulator/
│
├── README.md                          # This file
├── pyproject.toml                     # Project configuration
├── requirements.txt                   # Python dependencies
│
├── docs/                              # Documentation
│   ├── PROGRESS_REPORT.MD             # Comprehensive technical report (4,743 lines)
│   ├── phases.md                      # Development roadmap
│   └── file_structure.md              # Original structure documentation
│
├── plots/                             # Generated visualization outputs
│
└── src/                               # Source code
    │
    ├── experiments/                   # Experimental scripts
    │   ├── energy_drift.py            # Energy conservation analysis
    │   ├── error_growth.py            # Numerical error accumulation study
    │   ├── oscillator_stability.py    # Harmonic oscillator stability tests
    │   ├── realworld_stability_test.py # Control loop perturbation experiments
    │   ├── rl_roll_test.py            # RL rollout performance benchmarks
    │   ├── stability_table.py         # Integrator comparison table generator
    │   └── throughput_scaling.py      # Batch performance scaling study
    │
    └── mpe/                           # Main physics engine package
        ├── __init__.py
        │
        ├── core/                      # Fundamental simulation components
        │   ├── __init__.py
        │   ├── state.py               # State representation (State1D class)
        │   ├── simulator.py           # Main simulation loop orchestration
        │   └── timekeeper.py          # Time tracking and management
        │
        ├── integrators/               # Numerical integration schemes
        │   ├── __init__.py
        │   ├── base.py                # Abstract integrator interface
        │   ├── explicit_euler.py      # Forward Euler (unstable baseline)
        │   ├── semi_implicit_euler.py # Symplectic Euler (stable)
        │   ├── verlet.py              # Velocity Verlet (best performer)
        │   └── rk4.py                 # 4th-order Runge-Kutta (high-order)
        │
        ├── forces/                    # Force model implementations
        │   ├── __init__.py
        │   ├── base.py                # Abstract force interface
        │   ├── gravity.py             # Constant gravitational force
        │   ├── spring.py              # Hooke's law spring force
        │   ├── damped_spring.py       # Spring + viscous damping
        │   └── composite.py           # Combined force models
        │
        ├── batch/                     # Batch simulation backends
        │   ├── __init__.py
        │   ├── base.py                # Abstract batch simulator interface
        │   ├── python_loop.py         # Pure Python implementation
        │   ├── numpy_vectorized.py    # NumPy vectorized operations
        │   ├── torch_cpu.py           # PyTorch CPU backend
        │   └── benchmark.py           # Cross-backend performance comparison
        │
        ├── analysis/                  # Analysis and metrics modules
        │   ├── energy.py              # Energy calculation and drift tracking
        │   ├── error.py               # Error metrics and convergence analysis
        │   ├── metrics.py             # Performance metrics computation
        │   └── stability.py           # Stability criteria evaluation
        │
        ├── rl/                        # Reinforcement learning infrastructure
        │   ├── __init__.py
        │   ├── determinism.py         # Determinism validation utilities
        │   ├── environment_batch.py   # Batched environment stepping
        │   ├── replay_buffer.py       # Replay buffer implementation
        │   └── rollout_storage.py     # Structure-of-Arrays rollout storage
        │
        └── realworld/                 # Real-world constraint simulations
            ├── __init__.py
            ├── async_step.py          # Asynchronous stepping simulation
            ├── jitter.py              # Control loop jitter injection
            ├── latency.py             # Fixed latency simulation
            ├── precision_test.py      # float32 vs float64 comparison
            └── stability_runner.py    # Real-world stability test orchestration
```

### Key Modules

- **`mpe/core/`**: State representation, simulator loop, time management
- **`mpe/integrators/`**: Explicit Euler, Semi-Implicit Euler, Velocity Verlet, RK4
- **`mpe/forces/`**: Gravity, spring, damped spring, composite force models
- **`mpe/batch/`**: Python loop, NumPy vectorized, PyTorch CPU backends
- **`mpe/analysis/`**: Energy tracking, stability evaluation, error metrics
- **`mpe/rl/`**: Batched environments, rollout storage (SoA), replay buffers
- **`mpe/realworld/`**: Precision tests, jitter/latency simulation, async stepping

---

## Technical Architecture

### Design Patterns

**1. Strategy Pattern (Integrators)**
```python
class Integrator(ABC):
    @abstractmethod
    def step(self, state, force_model, mass, t, dt) -> State1D:
        pass
```

All integrators implement this interface, enabling runtime swapping without code changes.

**2. Dependency Injection**
```python
simulator = Simulator(
    integrator=VelocityVerlet(),
    force_model=DampedSpring(k=10.0, c=0.1),
    mass=1.0
)
```

Components are injected at construction, not hard-coded.

**3. Stateless Functions**
Integrators are pure functions: `new_state = f(old_state, parameters)` with no internal caches.

### Memory Layout Optimization

**Array-of-Structures (AoS) vs Structure-of-Arrays (SoA)**

❌ **AoS (poor cache locality)**:
```python
particles = [{'x': x1, 'v': v1}, {'x': x2, 'v': v2}, ...]
```

✅ **SoA (vectorization-friendly)**:
```python
positions = np.array([x1, x2, ...])
velocities = np.array([v1, v2, ...])
```

**Result**: 73% throughput improvement from in-place array operations.

### Determinism Guarantees

- **Fixed-precision arithmetic**: All operations use float64
- **Fixed evaluation order**: Single-threaded, sequential operations
- **No environment dependencies**: No randomness, no filesystem I/O during simulation
- **Validation**: 100% bitwise reproducibility verified across platforms

---

## Phases & Progress

### Phase 1: Deterministic Core Engine ✅

**Goal**: Build minimal, predictable, measurable foundation

**Implementation**:
- 1D particle system with position, velocity, mass
- Fixed timestep simulation loop
- Modular force models (gravity, spring, damped oscillator)
- 1D particle system with fixed timestep
- Modular force models (gravity, spring, damped oscillator)
- 10,000+ simulations → 100% identical outputs

### Phase 2: Integrator Comparison ✅
**Test**: 1D harmonic oscillator (m=1, k=10, ω=3.162 rad/s)  
**Metrics**: Energy drift, max stable dt, computational cost, throughput
| Velocity Verlet | 0.2557 | 1,258 | 13 | **203,173** |
| RK4 | Unstable | 3,418 | 38 | 0 |

**Key Insight**: Symplectic integrators (Semi-Implicit Euler, Verlet) preserve energy structure despite lower formal order, achieving **245× higher throughput** than Explicit Euler.

---

### Phase 3: Batch Simulation & Throughput Engineering ✅

**Goal**: Scale from 1 particle to 100,000; characterize performance regimes

**Backends**:
- Python loop (baseline)
- NumPy vectorized
- PyTorch CPU

**Scaling Results**:

| Particles | Backend | Particle-Steps/sec | Bandwidth (GB/s) |
|-----------|---------|-------------------|------------------|
| 1 | Python Loop | 955k | 0.015 |
| | NumPy | 104k | 0.002 |
| 1,000 | Python Loop | 2.0M | 0.032 |
| | NumPy | 299M | 4.8 |
| 100,000 | Python Loop | 1.98M | 0.032 |
| | **NumPy** | **1.38B** | **22.1** |
| | PyTorch CPU | 941M | 15.1 |

**Performance Regimes**:
1. **N < 100**: Overhead-bound (function call overhead dominates)
2. **100 < N < 10,000**: Interpreter-bound (Python loop overhead)
3. **N > 10,000**: Memory-bandwidth-bound (22.1 GB/s near DDR4 limits)

**Optimization**: In-place array updates → **+73% throughput** (842M → 1.38B)

---

### Phase 4: RL-Style Batched Rollouts & Memory Engineering ✅

**Goal**: Production-grade batched rollout systems for RL environments
Throughput Engineering ✅
**Scaling**: 1 → 100,000 particles across Python/NumPy/PyTorch backends44.3M transitions/sec**
- **Memory usage**: 68 MB
- **Determinism**: ✅ Bitwise identical across runs

**Memory Breakdown**:
```
states:  (1024 × 4096 × 2) = 33.6 MB
actions: (1024 × 4096 × 1) = 16.8 MB
rewards: (1024 × 4096)     = 16.8 MB
dones:   (1024 × 4096)     =  4.2 MB
────────────────────────────────────
Total:                      68.0 MB
```

**Scaling Analysis** (8× scale-up to `num_envs=16,384`, `horizon=2,048`):
- Projected memory: **544 MB**
- **First constraint**: RAM capacity (not compute, not bandwidth)

**Engineering Insights**:
- Memory scales as $O(\text{horizon} \times \text{num\_envs} \times \text{state\_dim})$
- SoA layout enables full vectorization
- Chunked rollouts: 4× memory reduction
- Precision red Batched Rollouts ✅
**Config**: 4096 envs × 1024 steps × 2 state_dim
**Experiments**:

#### 1. Floating-Point Precision Comparison
**Test**: float32 vs float64 energy drift in Verlet integrator

**Results**:
- **10K steps**: <0.02% difference
- **1M steps**: <0.02% difference
- **Conclusion**: For stable integrators, **precision is rarely the limiting factor**

#### 2. Control Loop Perturbations
**Perturbation Types**:
- **Jitter**: Timestep variability (dt ± noise)
- **Latency**: Fixed delay between observation and action
- **Async Stepping**: Non-blocking step execution

**Stability Hierarchy** (most destabilizing to least):
```
Latency > Jitter > Async Stepping > Float Precision
  ^^^^     ^^^^       ^^^^             ^^^^
 100×     10-100×      2-5×            <1%
```

**Critical Discovery**: Timing perturbations are **100-1000× more destabilizing** than precision reduction.

#### 3. Phase Drift vs Energy Conservation
**Harmonic oscillator with 1% jitter over 1000 steps**:
- **Energy drift**: 0.5% (bounded)
- **Velocity error**: 3000% (phase desynchronization)

**Conclusion**: **Phase accuracy is critical for control systems**; energy conservation alone is insufficient.

---

## Experimental Results

### Integrator Stability Comparison

#### Precision Comparison
- float32 vs float64: **<0.02% difference** (precision rarely limits stability)

#### Control Loop Perturbations
**Stability Hierarchy**: Latency (100×) > Jitter (10-100×) > Async (2-5×) > Precision (<1%)

#### Phase vs Energy
- 1% jitter over 1000 steps: Energy drift 0.5%, Velocity error **3000%**
- **Conclusion**: Phase accuracy critical for control; energy conservation insufficient
            TorchCPU               941M             15.1
═══════════════════════════════════════════════════════════════
```

**Takeaway**: NumPy vectorization achieves near-DDR4-bandwidth limits at large scales.

---

## Key Findings & Conclusions

### 1. Symplectic Structure > Truncation Error Order

For Hamiltonian systems (energy-conserving), symplectic integrators (Semi-Implicit Euler, Verlet) dramatically outperform high-order non-symplectic methods (RK4).

**Why?** Symplectic integrators preserve geometric properties of phase space, preventing artificial energy drift even with large timesteps.

**Practical Impact**: Verlet with dt=0.25 is more stable than RK4 with any dt.

---

### 2. Performance Regime Transitions

System performance is not monotonic with problem size:

| Regime | Particle Count | Bottleneck | Optimization Strategy |
|--------|----------------|------------|----------------------|
| Overhead-Bound | N < 100 | Function call overhead | Reduce abstraction layers |
| Interpreter-Bound | 100 < N < 10K | Python loop execution | Vectorize with NumPy |
| Memory-Bandwidth-Bound | N > 10K | DRAM throughput | Optimize memory access patterns |

**Engineering Lesson**: Profile first, optimize second. The bottleneck changes with scale.

---

### 3. Memory Layout Dominates Performance

**73% throughput gain** achieved through memory access pattern optimization alone:
- Replace array copies with in-place updates
- Use Structure-of-Arrays (SoA) instead of Array-of-Structures (AoS)
- Align data to cache line boundaries

**Impact**: 842M → 1.38B particle-steps/sec with zero algorithmic changes.

---

### 4. RAM Capacity Limits RL Scaling

For batched RL rollouts:
- **Not compute-limited**: CPU utilization < 50%
- **Not bandwidth-limited**: 22.1 GB/s << DDR4 max (51.2 GB/s)
- **RAM-capacity-limited**: 68 MB at (4096 envs × 1024 steps) scales to 544 MB at 8× scale

**Mitigation Strategies**:
1. Chunked rollouts (stream data to disk)
2. Precision reduction (float32 instead of float64)
3. Sparse state storage (only store deltas)

---

### 5. Latency > Jitter > Precision for Stability

Real-world control system stability hierarchy:

```
┌─────────────────────────────────────┐
│ Latency (100× impact)               │  ← Most destabilizing
├─────────────────────────────────────┤
│ Jitter (10-100× impact)             │
├─────────────────────────────────────┤
│ Async Stepping (2-5× impact)        │
├─────────────────────────────────────┤
│ Float Precision (<1% impact)        │  ← Least destabilizing
└─────────────────────────────────────┘
```

**Engineering Implication**: Invest in low-latency infrastructure before worrying about float32 vs float64.

---

### 6. Phase Drift Dominates Long-Horizon Control

Even when energy is conserved (ΔE < 1%), phase desynchronization causes **3000% velocity error** over 1000 steps with 1% jitter.

**Why It Matters**: For trajectory tracking, feedforward control, and sim-to-real transfer, phase accuracy is paramount.

**Recommendation**: Use timing-aware integrators and compensate for latency/jitter explicitly.

---

### 7. Determinism is Achievable (and Measurable)

**100% bitwise reproducibility** achieved through:
- Fixed-precision arithmetic (float64)
- Fixed evaluation order (single-threaded)
- Zero environment dependencies (no RNG, no I/O)

**Validation**: 10,000+ repeated simulations → identical outputs across Windows/Linux.

**Industrial Value**: Critical for debugging, regression testing, and reproducible research.

---

## Installation & Usage

### Prerequisites

- **Python**: 3.12 or higher
- **Operating System**: Windows, Linux, or macOS

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Minimal_Physics_Simulator
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or using `uv` (recommended):
```bash
uv pip install -r requirements.txt
```

### Dependencies

- `numpy >= 2.4.2` — Vectorized numerical operations
- `torch >= 2.10.0` — PyTorch backend for batch simulation
- `matplotlib >= 3.10.8` — Plotting and visualization
- `pandas >= 3.0.0` — Data analysis and tables

### Quick Start

#### Example 1: Run a Simple Harmonic Oscillator

```python
from src.mpe.core.simulator import Simulator
from src.mpe.core.state import State1D
from src.mpe.integrators.verlet import VelocityVerlet
from src.mpe.forces.spring import Spring

# Setup
simulator = Simulator(
    integrator=VelocityVerlet(),
    force_model=Spring(k=10.0),
    mass=1.0
)

# Initial condition
initial_state = State1D(x=1.0, v=0.0)

# Run
positions, velocities = simulator.run(
    initial_state=initial_state,
    dt=0.01,
    steps=1000
)

print(f"Final position: {positions[-1]:.4f}")
```

#### Example 2: Compare Integrators

```bash
cd src/experiments
python stability_table.py
```

Output:
```
═══════════════════════════════════════════════════════════════
Integrator       Max Stable dt   ns/step   FLOPs   Throughput
───────────────────────────────────────────────────────────────
Euler                 0.000500    602.80        6       829
SemiImplicit          0.045534    649.24        6    70,134
Verlet                0.255691  1,258.49       13   203,173
═══════════════════════════════════════════════════════════════
```

#### Example 3: Run Batch Performance Benchmark

```bash
cd src/experiments
python throughput_scaling.py
```

Generates plots in `plots/` directory showing particle-steps/sec vs particle count.

#### Example 4: Test Real-World Stability

```bash
cd src/experiments
python realworld_stability_test.py
```

Compares float32 vs float64, jitter, latency, and async stepping effects.

---

## Future Roadmap

### Phase 6: GPU Acceleration (Planned)

**Goals**:
- Implement CUDA kernels for custom integrators
- Characterize GPU memory bandwidth vs compute utilization
- Compare PyTorch GPU backend vs raw CUDA

**Expected Results**: 10-100× speedup for N > 100K particles

---

### Phase 7: 2D/3D Rigid Body Dynamics (Planned)

**Expansion**:
- Extend to 2D/3D state vectors
- Add rotation (quaternions/rotation matrices)
- Implement collision detection and response

**Applications**: Game physics, robotics manipulation

---

### Phase 8: Advanced Integrators (Research)

**Candidates**:
- Leapfrog integrator (astronomy)
- RESPA (multiple timesteps)
- Symplectic partitioned Runge-Kutta

**Goal**: Specialized integrators for specific problem classes

---

## References

### Numerical Methods
- Hairer, E., Lubich, C., & Wanner, G. (2006). *Geometric Numerical Integration*.
- Leimkuhler, B., & Reich, S. (2004). *Simulating Hamiltonian Dynamics*.
- **Phase 6**: GPU acceleration with CUDA kernels (10-100× speedup expected)
- **Phase 7**: 2D/3D rigid body dynamics with rotations and collisions
- **Phase 8**: Advanced integrators (Leapfrog, RESPA, symplectic RK)
- GPU batching strategies
- Determinism engineering
- RL rollout architecture design
- Control-loop reliability testing

These are the exact competencies required at:
- Robotics companies (Boston Dynamics, Tesla, etc.)
- Simulation infrastructure teams (NVIDIA, Epic Games)
- RL infrastructure teams (OpenAI, DeepMind)
- GPU systems teams (NVIDIA, AMD)

---

## License

MIT License - See LICENSE file for details.

---

**End of README**
