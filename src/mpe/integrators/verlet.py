# src/mpe/integrators/verlet.py

from src.mpe.integrtors.base import Integrator
from src.mpe.core.state import State1D
from src.mpe.forces.base import ForceModel

class Verlet(Integrator):
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

        new_x = state.x + state.v * dt + 0.5 * a * dt * dt

        # Compute acceleration at new position
        temp_state = State1D(new_x, state.v)
        new_force = force_model.compute(temp_state, t+dt)
        new_a = new_force/mass

        new_v = state.v + 0.5 * (a + new_a) * dt

        return State1D(new_x, new_v)
