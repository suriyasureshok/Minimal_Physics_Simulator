# src/mpe/core/simulator.py

import numpy as np
from mpe.core.state import State1D
from mpe.integrators.base import Integrator
from mpe.forces.base import ForceModel
from mpe.core.timekeeper import TimeKeeper

class Simulator:
    def __init__(
            self, 
            integrator: Integrator,
            force_model: ForceModel,
            mass: float
            ):
        self.integrator = integrator
        self.force_model = force_model
        self.mass = mass

    def run(
            self,
            initial_state: State1D,
            dt: float,
            steps: int
            ):
        state = initial_state
        timekeeper = TimeKeeper(dt)

        positions = np.zeros(steps, dtype=np.float64)
        velocities = np.zeros(steps, dtype=np.float64)

        for i in range(steps):
            state = self.integrator.step(
                    state,
                    self.force_model,
                    self.mass,
                    timekeeper.t,
                    dt
                    )
            timekeeper.advance()

            positions[i] = state.x
            velocities[i] = state.v

        return positions, velocities
