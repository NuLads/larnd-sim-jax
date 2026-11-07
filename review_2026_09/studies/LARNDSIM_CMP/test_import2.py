import os, sys
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
sys.path.insert(0, "/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP")
sys.path.insert(0, "/sdf/home/p/pgranger/reference")

import cupy_cpu_shim
sys.modules["cupy"] = cupy_cpu_shim

from larndsim import light_sim
print("light_sim imported OK")

from larndsim import lightLUT
print("lightLUT imported OK")

from numba.cuda.random import create_xoroshiro128p_states
print("numba.cuda.random.create_xoroshiro128p_states imported OK")
states = create_xoroshiro128p_states(10, seed=1)
print("created states:", type(states), len(states))

from numba.cuda import device_array, to_device
import numpy as np
arr = to_device(np.arange(5))
print("to_device OK:", arr.copy_to_host())
darr = device_array(5, dtype=np.float32)
print("device_array OK:", darr.copy_to_host())

print("ALL EXTENDED IMPORTS / CUDASIM PRIMITIVES OK")
