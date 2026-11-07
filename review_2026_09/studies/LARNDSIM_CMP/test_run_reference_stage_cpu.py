import os, sys
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
CMP_DIR = "/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP"
sys.path.insert(0, CMP_DIR)
import cupy_cpu_shim
sys.modules["cupy"] = cupy_cpu_shim
sys.modules["cupy.cuda"] = cupy_cpu_shim.cuda
sys.modules["cupy.cuda.nvtx"] = cupy_cpu_shim._nvtx_mod

ref_dir = CMP_DIR + "/reference_copy"
sys.path.insert(0, ref_dir)
sys.path.insert(0, ref_dir + "/cli")

import fee_overrides
import simulate_pixels
from larndsim import fee
from larndsim.consts import physics, units

for stage in (1, 2, 3):
    applied = fee_overrides.apply_stage(fee, physics, units, stage)
    print("stage", stage, "->", applied)

print("run_reference_stage.py dependency chain imports OK (CPU/CUDASIM, fake cupy)")
