import numpy as np

def run_precision_test(env_class, num_envs, k_over_m, dt, horizon):
    results = {}
    for dtype in [np.float32, np.float64]:
        env = env_class(num_envs, k_over_m, dtype=dtype)

        states = []

        for _ in range(horizon):
            state, _, _ = env.step(dt)
            states.append(state.copy())

        states = np.stack(states)

        # Compute energy

        x = states[:, :, 0]
        v = states[:, :, 1]

        energy = 0.5 * v**2 + 0.5 * k_over_m * x**2
        drift = np.abs(energy - energy[0]).mean()

        results[str(dtype)] = drift

    return results
