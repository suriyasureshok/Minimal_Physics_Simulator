# src/mpe/integrators/base.py

from abc import ABC, abstractmethod
from mpe.core.state import State1D
from mpe.forces.base import ForceModel

class Integrator(ABC):
    @abstractmethod
    def step(
            self,
            state: State1D,
            force_model: ForceModel,
            mass: float,
            t: float,
            dt: float
            ) -> State1D:
        pass
