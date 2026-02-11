# src/mpe/forces/composite.py

from src.mpe.forces.base import ForceModel
from src.mpe.core.state import State1D

class CompositeForce(ForceModel):
    def __init__(self,forces):
        self.forces = forces

    def compute(self,state:State1D,t:float) -> float:
        total = 0.0
        for f in self.forces:
            total += f.compute(state,t)
        return total
