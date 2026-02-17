import numpy as np


class LatencyWrapper:
    def __init__(self, env, latency_steps=1):
        self.env = env
        self.latency_steps = latency_steps
        # Note: Actions are ignored in this oscillator environment

    def step(self, action, dt):
        # Actions are ignored in the oscillator environment
        # This wrapper demonstrates latency structure but doesn't use actions
        state, reward, done = self.env.step(dt)

        return state, reward, done


def step_with_latency(env, dt, latency_steps=0):
    """
    Step the environment with simulated latency.
    
    For latency_steps > 0, this simulates delayed control response
    by stepping the environment multiple times.
    
    Args:
        env: Environment to step
        dt: Base timestep
        latency_steps: Number of additional steps to simulate latency effect
    
    Returns:
        Tuple of (state, reward, done) from the final step
    """
    # Execute latency steps (simulating delay without control)
    for _ in range(latency_steps):
        env.step(dt)
    
    # Execute the actual step
    state, reward, done = env.step(dt)
    
    return state, reward, done


