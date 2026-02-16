import numpy as np 
import torch
import pandas as pd 

from src.mpe.batch.python_loop import PythonLoopIntegrator
from src.mpe.batch.numpy_vectorized import NumpyVectorizedIntegrator
from src.mpe.batch.torch_cpu import TorchCPUIntegrator
from src.mpe.batch.benchmark import benchmark_backend


m = 1.0
k = 10.0
k_over_m = k/m 

dt = 0.001 
steps = 1000

particle_count = [1,1000,100000]

result = []

for N in particle_count:
    print(f'\nBenchmarking N = {N}')
    x_np = np.ones(N,dtype = np.float32)
    v_np = np.zeros(N,dtype=np.float32)

    x_torch = torch.ones(N,dtype = torch.float32)
    v_torch = torch.zeros(N,dtype = torch.float32)

    backends = {
        'PythonLoop' : PythonLoopIntegrator(k_over_m),
        'NumPy' : NumpyVectorizedIntegrator(k_over_m),
        'TorchCPU' : TorchCPUIntegrator(k_over_m)
    }

    for name , integrator in backends.items():
        if name == 'TorchCPU':
            steps_per_sec , total_time = benchmark_backend(
                integrator,x_torch,v_torch,dt,steps

            )
        else:
            steps_per_sec,total_time = benchmark_backend(
                integrator,x_np.copy(),v_np.copy(),dt,steps
            )


        particle_steps_per_sec = steps_per_sec * N 

        bytes_per_particle = 16
        bandwidth_estimate = (
            particle_steps_per_sec * bytes_per_particle
        ) / 1e9

        result.append([
            N ,
            name,
            steps_per_sec,
            particle_steps_per_sec,
            bandwidth_estimate
        ])

df = pd.DataFrame(
    result,
    columns = [
        'Particles',
        'Backend',
        'Steps/sec',
        'Particle-Steps/sec',
        'Estimated Bandwidth (GB/s)'
    ]
)

print("\nThroughput results:")
print(df)

