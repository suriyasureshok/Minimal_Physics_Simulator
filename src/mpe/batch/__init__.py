"""Batch integration backends for parallel particle processing.

This module provides vectorized integrators for simulating multiple particles:
- PythonLoop: Naive loop-based (baseline reference)
- NumpyVectorized: NumPy vectorization (10-100x speedup)
- TorchCPU: PyTorch CPU backend (ML framework integration, optional)
- Benchmark utilities for performance measurement
"""

from src.mpe.batch.base import BatchIntegrator
from src.mpe.batch.python_loop import PythonLoopIntegrator
from src.mpe.batch.numpy_vectorized import NumpyVectorizedIntegrator
from src.mpe.batch.benchmark import benchmark_backend

# Conditionally import TorchCPUIntegrator if PyTorch is available
try:
    from src.mpe.batch.torch_cpu import TorchCPUIntegrator
    TORCH_AVAILABLE = True
except ImportError:
    TorchCPUIntegrator = None
    TORCH_AVAILABLE = False

__all__ = [
    'BatchIntegrator',
    'PythonLoopIntegrator',
    'NumpyVectorizedIntegrator',
    'benchmark_backend',
]

if TORCH_AVAILABLE:
    __all__.append('TorchCPUIntegrator')
