# src/mpe/integrators/explicit_euler.py

from src.mpe.integrators.base import Integrator
from src.mpe.core.state import State1D
from src.mpe.forces.base import ForceModel

class ExplicitEuler(Integrator):
    def step(
            self,
            state: State1D,
            force_model: ForceModel,
            mass: float,
            t: float,
            dt: float
            ) -> State1D:
        
        force = force_model.compute(state, t)
        a = force/mass

        new_x = state.x + dt * state.v
        new_v = state.v + dt * a
        
        return State1D(new_x, new_v)
