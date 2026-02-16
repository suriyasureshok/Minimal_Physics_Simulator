minimal-physics-engine/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── src/
│   └── mpe/
│       ├── __init__.py
│       │
│       ├── core/
│       │   ├── state.py
│       │   ├── simulator.py
│       │   └── timekeeper.py
│       │
│       ├── batch/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── python_loop.py
│       │   ├── numpy_vectorized.py
│       │   ├── torch_cpu.py
│       │   └── benchmark.py
│       ├── forces/
│       │   ├── base.py
│       │   ├── gravity.py
│       │   ├── spring.py
│       │   ├── damped_spring.py
│       │   └── composite.py
│       │
│       ├── integrators/
│       │   ├── base.py
│       │   ├── explicit_euler.py
│       │   ├── semi_implicit_euler.py
│       │   ├── verlet.py
│       │   └── rk4.py
│       │
│       ├── analysis/
│       │   ├── energy.py
│       │   ├── stability.py
│       │   ├── error.py
│       │   └── metrics.py
│       │
│       ├── benchmarks/
│       │   ├── throughput.py
│       │   └── integrator_compare.py
│       │
│       └── utils/
│           ├── profiling.py
│           └── logging.py
│
├── experiments/
│   ├── oscillator_stability.py
│   ├── energy_drift.py
│   ├── error_growth.py
│   └── dt_sweep.py
│
└── tests/
    ├── test_integrators.py
    ├── test_forces.py
    ├── test_error.py
    └── test_energy_conservation.py

