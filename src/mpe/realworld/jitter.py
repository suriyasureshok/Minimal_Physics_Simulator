import numpy as np


def step_with_jitter(env, dt, jitter_std):
    noisy_dt = dt + np.random.normal(0, jitter_std)

    return env.step(noisy_dt)
