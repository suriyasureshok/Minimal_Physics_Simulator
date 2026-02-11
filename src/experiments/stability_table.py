import numpy as np
import pandas as pd

from src.mpe.core.state import State1D
from src.mpe.integrators.explicit_euler import ExplicitEuler
from src.mpe.integrators.semi_implicit_euler import SemiImplicitEuler
from src.mpe.integrators.verlet import Verlet
from src.mpe.integrators.rk4 import RK4
from src.mpe.forces.spring import SpringForce
from src.mpe.analysis.stability import find_max_stable_dt


from src.mpe.core.simulator import Simulator


def simulator_factory(integrator ,force_model,mass):
    return Simulator(integrator,force_model,mass)


m = 1.0
k = 10.0

initial_state = State1D(1.0,0.0)
force = SpringForce(k)

dt_values = np.linspace(0.0005,0.2,100)

integrators = {
        "Euler" : ExplicitEuler(),
        "SemiImplicit" : SemiImplicitEuler(),
        "Verlet" : Verlet(),
        "RK4" : RK4()
}

results = {}

for name,integrator in integrators.items():
    print(f"Computing stability for {name}...")

    max_dt,stability_map = find_max_stable_dt(
            simulator_factory,
            integrator,
            force,
            m,
            initial_state,
            dt_values,
            steps=20000
        )
    results[name] = max_dt


df = pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=["Max Stable dt"]
    )


print("\nStability Table:")
print(df)
