"""Time management for physics simulations.

This module provides the TimeKeeper class for tracking simulation time
and step counts during physics simulations.
"""


class TimeKeeper:
    """Manages simulation time and step counting.
    
    The TimeKeeper class maintains the current simulation time and the
    number of steps taken, providing a centralized mechanism for time
    management in physics simulations.
    
    Attributes:
        dt (float): Time step size in seconds.
        t (float): Current simulation time in seconds.
        step_count (int): Number of simulation steps completed.
        
    Examples:
        >>> tk = TimeKeeper(dt=0.01)
        >>> tk.advance()
        >>> print(f"Time: {tk.t}, Steps: {tk.step_count}")
        Time: 0.01, Steps: 1
    """
    
    def __init__(self, dt: float):
        """Initialize the TimeKeeper.
        
        Args:
            dt (float): Time step size in seconds. Must be positive.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = dt
        self.t = 0.0
        self.step_count = 0

    def advance(self):
        """Advance the simulation time by one time step.
        
        Increments the current time by dt and increments the step count.
        This method should be called after each simulation step.
        """
        self.t += self.dt
        self.step_count += 1
