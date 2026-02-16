import numpy as np 

class ReplayBuffer:
    def __init__(self,capacity,state_dim,action_dim,dtype = np.float32):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity,state_dim),dtype = dtype)
        self.actions = np.zeros((capacity,action_dim),dtype = dtype)
        self.rewards = np.zeros(capacity,dtype = dtype)
        self.next_states = np.zeros((capacity,state_dim),dtype = dtype)
        self.dones = np.zeros(capacity,dtype = np.bool_)

    def add(self,state,action,reward,next_state,done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1 , self.capacity)

    def memory_megabytes(self):
        total_bytes = (
            self.states.nbytes +
            self.actions.nbytes+
            self.rewards.nbytes+
            self.next_states.nbytes+
            self.dones.nbytes
        )
    
        return total_bytes / (1024 ** 2)
