# src/mpe/integrators/rk4.py

from src.mpe.integrators.base import Integrator
from src.mpe.core.state import State1D
from src.mpe.forces.base import ForceModel

class RK4(Integrator):
    def step(
            self, 
            state: State1D,
            force_model: ForceModel,
            mass: float,
            t: float,
            dt: float
            ) -> State1D:

        def accelerate(s, time):
            return force_model.compute(s, time) / mass

        # k1
        a1 = accelerate(state, t)
        k1_v = a1
        k1_x = state.v

        # k2
        s2 = State1D(
                state.x + 0.5 * dt * k1_x,
                state.v + 0.5 * dt * k1_v
                )

        a2 = accelerate(s2, t + 0.5 * dt)
        k2_v = a2
        k2_x = s2.v

        # k3
        s3 = State1D(
                state.x + 0.5 * dt * k2_x,
                state.v + 0.5 * dt * k2_v
                )

        a3 = accelerate(s3, t + 0.5 * dt)
        k3_v = a3
        k3_x = s3.v

        # k4
        s4 = State1D(
                state.x + dt * k3_x,
                state.v + dt * k3_v
                )

        a4 = accelerate(s4, t + 0.5 * dt)
        k4_v = a4
        k4_x = s4.v

        new_x = state.x + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        new_v = state.v + (dt / 6.0) * (k1_v + 2*k2_v + 2+k3_v + k4_v)

        return State1D(new_x, new_v)

