import numpy as np


def run_realworld_test(env_class, k_over_m, dt, horizon):
    env = env_class(1, k_over_m)

    states = []

    # Capture initial state before any steps
    states.append(env.get_state().copy())

    for _ in range(horizon):
        state, _, _ = env.step(dt)
        states.append(state.copy())

    states = np.stack(states)

    x = states[:, :, 0]
    v = states[:, :, 1]

    energy = 0.5 * v**2 + 0.5 * k_over_m * x**2
    drift = np.abs(energy - energy[0]).mean()

    return drift

