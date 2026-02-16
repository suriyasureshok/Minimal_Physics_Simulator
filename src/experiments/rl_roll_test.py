import time 
import numpy as np 

from src.mpe.rl.environment_batch import BatchOscillatorEnv
from src.mpe.rl.rollout_storage import RolloutStorage,compute_rollout_memory


num_envs = 4096
horizon = 1024
state_dim = 2
action_dim = 2
action_dim = 1
dt = 0.001
k_over_m = 10.0

env = BatchOscillatorEnv(num_envs,k_over_m)
storage = RolloutStorage(horizon,num_envs,state_dim,action_dim)

print("Estimated rollout memory (MB) :",compute_rollout_memory(storage))

start = time.pref_counter()

for t in range(horizon):
    state , reward ,done = env.step(dt)
    action = np.zeros((num_envs,action_dim) , dtype = np.float32)
    storage.store(t,state,action,reward,done)

end = time.pref_counter()
print('Rollout time (sec):',end-start)
