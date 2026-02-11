import time
import numpy as np

def measure_ns_per_step(sim,initial_state,dt,steps):
    start= time.pref_counter()
    sim.run(initial_state,dt,steps)
    end = time.pref_counter()

    total_ns = (end-start)*1e9

    return total_ns / steps

def estimate_flops(integrator_name:str):

    estimates ={
            "ExplicitEuler":10,
            "SemiImplicitEuler":10,
            "Verlet":20,
            "RK4":80
    }

    return estimates.get(integrator_name,None)
