# src/mpe/forces/damped_spring.py

from mpe.forces.base import ForceModel
from mpe.core.state import State1D

class DampedSpringForce(ForceModel):
    def __init__(self,k:float,c:float):
        self.k=k
        self.c=c

    def compute(self,state:State1D,t:float) -> float:
        return -self.k * state - self.c * state.v


