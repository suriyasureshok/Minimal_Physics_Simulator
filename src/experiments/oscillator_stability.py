# src/experiments/oscillator_stability.py

from src.mpe.core.state import State1D
from src.mpe.core.simulator import Simulator
from src.mpe.integrators.explicit_euler import ExplicitEuler
from src.mpe.forces.spring import SpringForce

import os
import matplotlib.pyplot as plt

m = 1.0
k = 10.0
dt = 0.01
steps = 5000

initial_state = State1D(1.0, 0.0)

force = SpringForce(k)
integrator = ExplicitEuler()
sim = Simulator(integrator, force, m)

x, v = sim.run(initial_state, dt, steps)

os.makedirs('plots', exist_ok=True)

plt.plot(x)
plt.title("Explicit Euler Spring Oscillator")
plt.savefig('plots/explicit_euler_spring_oscillator.png', dpi=300, bbox_inches='tight')
