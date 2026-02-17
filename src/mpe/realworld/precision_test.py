import numpy as np

def run_precision_test(env_class, num_envs, k_over_m, dt, horizon):
    results = {}
    for dtype in [np.float32, np.float64]:
        env = env_class(num_envs, k_over_m, dtype=dtype)

        # Compute energy drift online to avoid storing the full trajectory
        energy0 = None
        drift_sum = 0.0

        for step_idx in range(horizon):
            state, _, _ = env.step(dt)

            x = state[:, 0]
            v = state[:, 1]
            energy = 0.5 * v**2 + 0.5 * k_over_m * x**2

            if step_idx == 0:
                # Reference energy at the first step, per environment
                energy0 = energy
            
            # Accumulate mean absolute drift for this step across environments
            drift_sum += np.abs(energy - energy0).mean()

        # Final drift is the mean over all steps and environments
        drift = drift_sum / float(horizon)

        results[str(dtype)] = drift

    return results
