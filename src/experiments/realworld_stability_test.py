import numpy as np

from src.mpe.rl.environment_batch import BatchOscillatorEnv
from src.mpe.realworld.precision_test import run_precision_test
from src.mpe.realworld.jitter import step_with_jitter
from src.mpe.realworld.async_step import AsyncBatchEnv


num_envs = 1
k_over_m = 10.0
dt = 0.001
horizon = 1_000_000


print("===== Precision Comparison =====")
precision_results = run_precision_test(
    BatchOscillatorEnv,
    num_envs,
    k_over_m,
    dt,
    horizon
)

print(precision_results)

print("\n===== Jitter Stability =====")
env = BatchOscillatorEnv(num_envs, k_over_m)

for _ in range(horizon):
    step_with_jitter(env, dt, jitter_std=1e-4)

state = env.get_state()
print("Final state with jitter: ", state)

print("\n===== Async Stopping =====")
env_async = BatchOscillatorEnv(8, k_over_m)
async_env = AsyncBatchEnv(env_async)

for _ in range(horizon):
    async_env.step(dt)

print("Async run complete")

