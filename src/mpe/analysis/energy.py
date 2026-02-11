# src/mpe/analysis/energy.py 

import numpy as np

def oscillator_energy(x,v,m,k):
     kinetic = 0.5 * m * v ** 2
     potential = 0.5 * k * x ** 2
     return kinetic + potential


def energy_drift(energy):
     return energy - energy[0]
