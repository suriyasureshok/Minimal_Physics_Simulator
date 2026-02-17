"""Error analysis utilities for numerical simulations.

This module provides functions for comparing numerical solutions against
analytic solutions and computing various error metrics.
"""

import numpy as np


def analytic_solution(t_array, A, omega):
    """Compute analytic solution for simple harmonic oscillator.
    
    Calculates the exact solution x(t) = A * cos(ω * t) for an undamped
    harmonic oscillator with initial conditions x(0) = A, v(0) = 0.
    
    Args:
        t_array (np.ndarray): Array of time points in seconds, shape (n_points,).
        A (float): Amplitude of oscillation in meters.
        omega (float): Angular frequency in rad/s. For a spring: ω = sqrt(k/m).
        
    Returns:
        np.ndarray: Position at each time point, shape (n_points,).
        
    Notes:
        - This is the exact solution for undamped harmonic motion
        - Initial conditions: x(0) = A, v(0) = 0
        - Period: T = 2π / ω
        
    Examples:
        >>> t = np.linspace(0, 10, 100)
        >>> x_exact = analytic_solution(t, A=1.0, omega=2.0)
    """
    return A * np.cos(omega * t_array)


def absolute_error(x_numeric, x_analytic):
    """Compute absolute error between numeric and analytic solutions.
    
    Calculates the absolute difference |x_numeric - x_analytic| at each
    time point.
    
    Args:
        x_numeric (np.ndarray): Numerical solution, shape (n_points,).
        x_analytic (np.ndarray): Analytic solution, shape (n_points,).
        
    Returns:
        np.ndarray: Absolute error at each point, shape (n_points,).
        
    Notes:
        - Always non-negative
        - Units match the solution (e.g., meters for position)
        - Useful for visualizing error growth over time
        
    Examples:
        >>> x_num = np.array([1.0, 0.9, 0.8])
        >>> x_ana = np.array([1.0, 0.95, 0.85])
        >>> err = absolute_error(x_num, x_ana)
        >>> print(err)  # [0.0, 0.05, 0.05]
    """
    return np.abs(x_numeric - x_analytic)


def l2_error(x_numeric, x_analytic):
    """Compute L2 (root mean square) error.
    
    Calculates the root mean square deviation between numerical and
    analytic solutions, providing a single scalar error metric.
    
    Args:
        x_numeric (np.ndarray): Numerical solution, shape (n_points,).
        x_analytic (np.ndarray): Analytic solution, shape (n_points,).
        
    Returns:
        float: L2 error (RMS deviation).
        
    Notes:
        - Provides a global measure of error over the entire trajectory
        - Sensitive to large errors (squared differences)
        - Commonly used for comparing integrator accuracy
        - Formula: sqrt(mean((x_num - x_ana)²))
        
    Examples:
        >>> x_num = np.array([1.0, 0.9, 0.8])
        >>> x_ana = np.array([1.0, 0.95, 0.85])
        >>> err = l2_error(x_num, x_ana)
        >>> print(f"L2 error: {err:.6f}")
    """
    return np.sqrt(np.mean((x_numeric - x_analytic) ** 2))


