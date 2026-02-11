# Minimal Physics Engine

A transparent, modular physics simulation engine built from scratch to study numerical integration methods, stability analysis, and performance optimization in computational physics.

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Project Overview

This is **not** a physics toy or game engine. It is a rigorous study in:
- Numerical stability and structure preservation
- Performance tradeoffs in integrator selection
- Long-horizon behavior of conservative systems
- The difference between local accuracy and global reliability

**Core Philosophy:** Complete transparency. Every integrator, force model, and numerical artifact is observable and explainable. No black-box ODE solvers, no hidden abstractions.

---

## ✨ Features

### ✅ Phase 1 & 2 (Complete)

- **Modular Architecture**
  - Clean separation: State, Force, Integrator, Simulator
  - Swappable components without code changes
  - Zero coupling between modules

- **Multiple Integrators**
  - Explicit Euler
  - Semi-Implicit Euler (Symplectic)
  - Velocity Verlet (Symplectic, Time-Reversible)
  - Runge-Kutta 4 (RK4)

- **Comprehensive Analysis Tools**
  - Energy conservation tracking
  - Stability boundary detection
  - L2 error measurement against analytical solutions
  - Performance benchmarking (ns/step, FLOPs, throughput)

- **Force Models**
  - Spring Force (Harmonic oscillator)
  - Gravity Force
  - Damped Spring Force

### 🔄 Phase 3 (Planned)

- NumPy vectorization
- Struct-of-Arrays memory layout
- Multi-particle systems
- Cache optimization

---

## 📊 Key Results

Our stability analysis reveals critical insights about integrator selection:

```
==============================================================================
STABILITY AND PERFORMANCE TABLE
==============================================================================
              Max Stable dt  ns/step  FLOPs  Sim-time/sec
Euler              0.000500   602.80      6        829.46
SemiImplicit       0.045534   649.24      6      70133.99
Verlet             0.255691  1258.49     13     203173.24
RK4                     NaN  3417.92     38          0.00
==============================================================================
```

**Key Findings:**
- ✅ **Velocity Verlet** is the gold standard: 203,000× real-time under stability constraints
- ⚠️ **RK4** fails for long-term Hamiltonian systems despite 4th-order accuracy
- ❌ **Explicit Euler** is unsuitable for oscillatory dynamics
- 🎯 **Symplectic structure** matters more than truncation order

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│                   User Code                       │
│           (experiments/stability_table.py)        │
└─────────────────┬────────────────────────────────┘
                  │
         ┌────────▼────────┐
         │   Simulator     │  ← Orchestration layer
         └────┬────┬───┬───┘
              │    │   │
    ┌─────────┘    │   └─────────┐
    │              │             │
┌───▼────┐   ┌────▼─────┐   ┌──▼──────┐
│ State  │   │ Integrator│   │  Force  │
└────────┘   └──────────┘   └─────────┘
```

### Module Responsibilities

| Module | Responsibility | Does NOT Do |
|--------|---------------|-------------|
| `State1D` | Store position & velocity | Compute derivatives, validate |
| `ForceModel` | Compute forces | Know about integration |
| `Integrator` | Advance state by dt | Store history, compute forces |
| `Simulator` | Loop & record trajectory | Interpret results |
| `TimeKeeper` | Track simulation time | Control timestep size |

---

## 📁 Project Structure

```
Minimal_Physics_Simulator/
├── src/
│   └── mpe/                          # Minimal Physics Engine
│       ├── core/                     # Core simulation components
│       │   ├── __init__.py
│       │   ├── state.py              # State representation
│       │   ├── simulator.py          # Main simulation loop
│       │   └── timekeeper.py         # Time management
│       ├── integrators/              # Numerical integrators
│       │   ├── __init__.py
│       │   ├── base.py               # Integrator interface
│       │   ├── explicit_euler.py
│       │   ├── semi_implicit_euler.py
│       │   ├── verlet.py
│       │   └── rk4.py
│       ├── forces/                   # Force models
│       │   ├── __init__.py
│       │   ├── base.py               # Force interface
│       │   ├── spring.py
│       │   ├── gravity.py
│       │   ├── damped_spring.py
│       │   └── composite.py
│       └── analysis/                 # Analysis tools
│           ├── energy.py             # Energy tracking
│           ├── error.py              # Error metrics
│           ├── stability.py          # Stability detection
│           └── metrics.py
├── experiments/                      # Simulation experiments
│   ├── energy_drift.py
│   ├── error_growth.py
│   ├── oscillator_stability.py
│   └── stability_table.py            # Main results
├── docs/
│   └── PROGRESS_REPORT.MD            # Comprehensive technical report
├── plots/                            # Generated visualizations
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Minimal_Physics_Simulator

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Stability Analysis

```bash
# Generate comprehensive stability table
python -m src.experiments.stability_table

# Outputs:
# - Performance comparison table
# - Max stable timesteps for each integrator
# - Throughput metrics
```

### Example: Simple Harmonic Oscillator

```python
from src.mpe.core.state import State1D
from src.mpe.core.simulator import Simulator
from src.mpe.integrators.verlet import Verlet
from src.mpe.forces.spring import SpringForce

# Configure system
mass = 1.0
k = 10.0
initial_state = State1D(x=1.0, v=0.0)

# Create simulator
integrator = Verlet()
force = SpringForce(k)
sim = Simulator(integrator, force, mass)

# Run simulation
dt = 0.01
steps = 1000
positions, velocities = sim.run(initial_state, dt, steps)
```

---

## 🧪 Experiments

### 1. Energy Drift Analysis
```bash
python -m src.experiments.energy_drift
```
Measures energy conservation over time for each integrator.

### 2. Error Growth
```bash
python -m src.experiments.error_growth
```
Compares numerical solutions to analytical solutions.

### 3. Oscillator Stability
```bash
python -m src.experiments.oscillator_stability
```
Determines stability boundaries as a function of timestep.

### 4. Stability Table (Main Results)
```bash
python -m src.experiments.stability_table
```
Generates comprehensive performance comparison table.

---

## 📖 Technical Documentation

See [docs/PROGRESS_REPORT.MD](docs/PROGRESS_REPORT.MD) for:
- Detailed module-by-module documentation
- Mathematical foundations
- Stability criteria definitions
- Performance analysis methodology
- Engineering lessons learned

---

## 🔬 Key Technical Insights

### 1. Stability Definition Matters

We use **energy-based stability** instead of just explosion detection:

```python
Unstable if:
- NaN or Inf present
- |x| > 100 (amplitude explosion)
- Relative energy drift > 10%  ← Critical criterion
```

This reveals that RK4, while locally accurate, drifts systematically over long simulations.

### 2. Symplectic Structure > Order

**Verlet (2nd order, symplectic)** outperforms **RK4 (4th order, non-symplectic)** for conservative systems.

Why? Symplectic integrators preserve:
- Phase space volume
- Modified Hamiltonian (bounded energy error)
- Qualitative dynamics

### 3. Performance = Throughput Under Constraints

Raw speed (ns/step) is meaningless. What matters:

$$
\text{Throughput} = \frac{\text{Max Stable } \Delta t}{\text{Time per Step}}
$$

Verlet is 2× slower per step than Euler but **245× faster** overall due to larger stable timestep.

### 4. Mathematical Model

1D Harmonic Oscillator:
$$
m \ddot{x} = -kx
$$

First-order system:
$$
\begin{aligned}
\dot{x} &= v \\
\dot{v} &= -\frac{k}{m}x
\end{aligned}
$$

Analytical solution:
$$
x(t) = A \cos(\omega t), \quad \omega = \sqrt{\frac{k}{m}}
$$

Energy:
$$
E = \frac{1}{2}mv^2 + \frac{1}{2}kx^2 = \text{constant}
$$

---

## 📈 Performance Metrics

### Measured Quantities

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Max Stable dt** | Largest dt with <10% energy drift | Determines simulation efficiency |
| **ns/step** | Time per integration step | Raw computational cost |
| **FLOPs** | Operations per step | Theoretical complexity |
| **Sim-time/sec** | $(dt_{\text{max}} \times 10^9) / \text{ns/step}$ | True performance metric |

### Benchmarking Methodology

1. **Warm-up**: 100 iterations to eliminate cold-start overhead
2. **Timing**: 10,000 iterations with `time.perf_counter_ns()`
3. **Averaging**: Mean time per step
4. **Stability Testing**: 800 timestep values, 10,000 steps each

---

## 🛣️ Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| **Phase 1** | ✅ Complete | Core engine, deterministic simulation |
| **Phase 2** | ✅ Complete | Integrator comparison, stability analysis |
| **Phase 3** | 🔄 Planned | NumPy vectorization, multi-particle systems |
| **Phase 4** | 📋 Future | Adaptive timestepping, implicit methods |
| **Phase 5** | 📋 Future | N-body dynamics, constrained systems |
| **Phase 6** | 📋 Future | GPU acceleration, validation studies |

---

## 🎓 Educational Value

This project teaches:

1. **Numerical Methods**
   - Why symplectic integrators matter
   - Local accuracy vs. global reliability
   - Stability analysis techniques

2. **Software Engineering**
   - Modular architecture design
   - Interface-based programming
   - Performance benchmarking

3. **Computational Physics**
   - Hamiltonian mechanics
   - Conservative system simulation
   - Structure preservation

4. **Systems Thinking**
   - How to measure what matters
   - Design for experimentation
   - Coupling vs. cohesion

---

## 📚 References

### Integrator Theory
- Hairer, E., Lubich, C., & Wanner, G. (2006). *Geometric Numerical Integration*
- Leimkuhler, B., & Reich, S. (2004). *Simulating Hamiltonian Dynamics*

### Applications
- Molecular Dynamics: LAMMPS, GROMACS
- Orbital Mechanics: N-body simulations
- Game Physics: Bullet, MuJoCo

**Status**: Phase 2 Complete | **Last Updated**: February 11, 2026

*Built to understand numerical integration from first principles.*