import numpy as np

def analytic_solution(t_array,A,omega):
    return A * np.cos(omega * t_array)

def absolute_error(x_numeric , x_analytic):
    return np.abs(x_numeric - x_analytic)

def l2_error(x_numeric , x_analytic):
    return np.sqrt(np.mean((x_numeric - x_analytic) ** 2))


