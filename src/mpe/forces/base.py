# scr/mpe/force/base.py


from abc import ABC,abstractmethod
from mpe.core.state import State1D

class ForceModel(ABC):
    @abstractmethod
    def compute(self,state:State1D,t:float)->float:
        """Return Scalar force value."""
        pass    
