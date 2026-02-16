import time 
import numpy as np 

from src.mpe.rl.environment_batch import BatchOscillatorEnv
from src.mpe.rl.rollout_storage import RolloutStorage
from src.mpe.rl.determinism import check_determinism

num_envs = 4096
horizon = 1024
state_dim = 2
action_dim = 1
dt = 0.001
k_over_m = 10.0


print('==== RL Rollout Test ====')
print(f"num_envs = {num_envs}")
print(f"horizon = {horizon}")
print()


env = BatchOscillatorEnv(num_envs,k_over_m)
storage = RolloutStorage(horizon,num_envs,state_dim,action_dim)

print("Estimated rollout memory (MB) :",storage.memory_megabytes())

start = time.pref_counter()

for t in range(horizon):
    state , reward ,done = env.step(dt)
    action = np.zeros((num_envs,action_dim) , dtype = np.float32)
    storage.store(t,state,action,reward,done)

end = time.pref_counter()
print('Rollout time (sec):',end-start)
print()

is_deterministic = check_determinism(
    BatchOscillatorEnv,
    num_envs,
    k_over_m,
    dt,
    horizon
)

print('Deterministic rollout:',is_deterministic)

