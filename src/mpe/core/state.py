# src/mpe/core/state.py

from dataclasses import dataclass

@dataclass
class State1D:
    x: float
    v: float
