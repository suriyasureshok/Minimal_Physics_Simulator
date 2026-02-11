#src/mpe/forces/gravity.py

from src.mpe.forces.base import ForceModel
from src.mpe.core.state import State1D

class GravityForce(ForceModel):
    def __init__(self,m:float,g:float):
        self.m = m
        self.g = g

    def compute(self,state:State1D,t:float)->float:
        return -self.m *self.g
