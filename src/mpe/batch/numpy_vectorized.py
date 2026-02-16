# src/mpe/batch/numpy_vectorized.py

import numpy as np
from src.mpe.batch.base import BatchIntegrator

class NumpyVectorizedIntegrator(BatchIntegrator):
    def __init__(self, k_over_m: float):
        self.k_over_m = k_over_m

    def step(self, x, v, dt):
        a = -self.k_over_m * x
        v += dt * a
        x += dt * v

        return x, v

