import sys
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
from larndsim import fee
from larndsim.consts import physics, units
import numpy as np

for stage in (1, 2, 3):
    fee_overrides.apply_stage(fee, physics, units, stage)
    base = fee.digitize(np.array([0.0]))
    print("stage", stage, "digitize(0) baseline ADC =", base)
