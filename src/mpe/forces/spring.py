# src/mep/forces/spring.py

from src.mpe.forces.base import ForceModel
from src.mpe.core.state import State1D

class SpringForce(ForceModel):
    def __init__(self,k:float):
        self.k = k
    
    def compute(self,state:State1D ,t:float)->float:
        return -self.k * state.x

