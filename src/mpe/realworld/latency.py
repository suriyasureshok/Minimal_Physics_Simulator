"""Control latency simulation for real-world physics.

This module simulates control latency in physics simulations, modeling
the delay between sensing and actuation common in robotics and games.
"""

import numpy as np


class LatencyWrapper:
    """Environment wrapper that adds control latency.
    
    Simulates the delay between observation and action application that
    occurs in real systems due to sensing, computation, and actuation delays.
    
    Attributes:
        env: Wrapped environment instance.
        latency_steps (int): Number of steps of delay to simulate.
        
    Notes:
        - Actions are ignored in the simple oscillator environment
        - Demonstrates latency structure for more complex environments
        - Common in robotics (sensor/actuator delays) and networked games
        
    Examples:
        >>> env = BatchOscillatorEnv(num_envs=1, k_over_m=10.0)
        >>> latency_env = LatencyWrapper(env, latency_steps=5)
        >>> state, reward, done = latency_env.step(action=None, dt=0.01)
    """
    
    def __init__(self, env, latency_steps=1):
        """Initialize the latency wrapper.
        
        Args:
            env: Environment to wrap. Should have a step(dt) method.
            latency_steps (int, optional): Number of time steps to delay.
                Defaults to 1.
        """
        self.env = env
        self.latency_steps = latency_steps
        # Note: Actions are ignored in this oscillator environment

    def step(self, action, dt):
        """Step the environment with latency.
        
        Args:
            action: Action to apply (ignored in oscillator environment).
            dt (float): Time step size in seconds.
            
        Returns:
            tuple: (state, reward, done) from the environment.
            
        Notes:
            - Actions are not used in the simple oscillator
            - In a full implementation, actions would be delayed
        """
        # Actions are ignored in the oscillator environment
        # This wrapper demonstrates latency structure but doesn't use actions
        state, reward, done = self.env.step(dt)

        return state, reward, done


def step_with_latency(env, dt, latency_steps=0):
    """Step environment with simulated control latency.
    
    Simulates delayed control response by taking multiple uncontrolled
    steps before the main step. This models the effect of sensing and
    actuation delays in real-world systems.
    
    Args:
        env: Environment instance with a step(dt) method.
        dt (float): Time step size in seconds.
        latency_steps (int, optional): Number of additional uncontrolled
            steps to simulate delay. Defaults to 0 (no latency).
        
    Returns:
        tuple: (state, reward, done) from the final step.
        
    Notes:
        - Latency steps execute without control (environment evolves freely)
        - Models sensor-to-actuator delay in robotics
        - Higher latency makes control more difficult
        - Common latency values: 1-10 steps depending on system
        
    Examples:
        >>> env = BatchOscillatorEnv(num_envs=1, k_over_m=10.0)
        >>> # Simulate 5 steps of latency
        >>> state, reward, done = step_with_latency(env, dt=0.01, latency_steps=5)
    """
    # Execute latency steps (simulating delay without control)
    for _ in range(latency_steps):
        env.step(dt)
    
    # Execute the actual step
    state, reward, done = env.step(dt)
    
    return state, reward, done


