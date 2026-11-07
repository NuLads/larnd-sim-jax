import os, sys
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
sys.path.insert(0, "/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP")
sys.path.insert(0, "/sdf/home/p/pgranger/reference")

import cupy_cpu_shim
sys.modules["cupy"] = cupy_cpu_shim

import numba
print("numba", numba.__version__)
from numba import cuda
print("cuda.is_available (sim) ->", cuda.is_available())
print("cuda.simulator?", cuda.simulator if hasattr(cuda,'simulator') else "n/a")

import larndsim
print("larndsim file:", larndsim.__file__)

from larndsim import consts
print("consts imported OK")

from larndsim import quenching, drifting
print("quenching, drifting imported OK")

from larndsim import detsim
print("detsim imported OK")

from larndsim import fee
print("fee imported OK")

from larndsim import pixels_from_track
print("pixels_from_track imported OK")

from larndsim import active_volume
print("active_volume imported OK")

print("ALL CORE MODULES IMPORTED OK")
