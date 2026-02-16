import numpy as np


def step_with_jitter(env, dt, jitter_std):
    epsilon = 1e-8  # Small positive value to prevent negative dt
    noisy_dt = dt + np.random.normal(0, jitter_std)
    noisy_dt = max(noisy_dt, epsilon)  # Clamp to safe minimum
    
    return env.step(noisy_dt)
