# src/mpe/rl/determinism.py


import numpy as np


def check_determinism(env_class, num_envs, k_over_m, dt, horizon):
    """
    Run two rollouts and compare bitwise equality
    """

    env1 = env_class(num_envs, k_over_m)
    env2 = env_class(num_envs, k_over_m)

    states1 = []
    states2 = []

    for _ in range(horizon)
        s1, _, _ = env1.step(dt)
        s2, _, _ = env2.step(dt)

        states1.append(s1.copy())
        states2.append(s2.copy())

    states1 = np.stack(states1)
    states2 = np.stack(states2)

    identical = np.array_equal(states1, states2)

    return identical
