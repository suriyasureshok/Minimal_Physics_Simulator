"""Real-world simulation effects and imperfections.

This module simulates real-world conditions:
- Jitter: Timing noise and timestep variability
- Latency: Sensor-actuator delays
- Async stepping: Desynchronized parallel environments
- Precision: float32 vs float64 numerical precision effects
- Stability testing under real-world conditions
"""

from src.mpe.realworld.jitter import step_with_jitter
from src.mpe.realworld.latency import LatencyWrapper, step_with_latency
from src.mpe.realworld.async_step import AsyncBatchEnv
from src.mpe.realworld.precision_test import run_precision_test
from src.mpe.realworld.stability_runner import run_realworld_test

__all__ = [
    'step_with_jitter',
    'LatencyWrapper',
    'step_with_latency',
    'AsyncBatchEnv',
    'run_precision_test',
    'run_realworld_test',
]
