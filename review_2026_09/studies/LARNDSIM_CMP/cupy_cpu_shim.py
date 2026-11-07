"""
Minimal CPU-only stand-in for `cupy`, for plumbing-testing the reference
DUNE larnd-sim (which hard-imports `cupy` in detsim.py, fee.py,
light_sim.py, active_volume.py, util/cuda_dict.py and cli/simulate_pixels.py)
on a machine with no GPU and no real cupy installed.

This is NOT a numerics-faithful substitute for cupy/CUDA: it only exists to
let the Python-level control flow execute so we can check the pipeline
wires up correctly (shapes, dtypes, file I/O) end to end on CPU. Any timing
or correctness comparison with the real GPU reference must be run for real
on a GPU node with the actual cupy package.

Usage: insert into sys.modules['cupy'] BEFORE anything imports larndsim or
cli.simulate_pixels, and also set NUMBA_ENABLE_CUDASIM=1 before importing
numba, so that numba.cuda kernels execute via numba's pure-Python CUDA
simulator instead of real CUDA.
"""
import numpy as np
import types
import sys

# Re-export the whole numpy namespace under the name "cupy".
_mod = sys.modules[__name__]
for _name in dir(np):
    if not _name.startswith("_"):
        setattr(_mod, _name, getattr(np, _name))


def asnumpy(x):
    return np.asarray(x)


def get_array_module(*args, **kwargs):
    return np


class _FakeMemoryPool:
    def used_bytes(self):
        return 0

    def free_all_blocks(self):
        pass


def get_default_memory_pool():
    return _FakeMemoryPool()


# cupy.cuda submodule: nvtx range push/pop are used in simulate_pixels.py
# purely for NVIDIA profiler markers -- safe no-ops on CPU.
_cuda_mod = types.ModuleType("cupy.cuda")


class _nvtx:
    @staticmethod
    def RangePush(*args, **kwargs):
        pass

    @staticmethod
    def RangePop(*args, **kwargs):
        pass


_nvtx_mod = types.ModuleType("cupy.cuda.nvtx")
_nvtx_mod.RangePush = _nvtx.RangePush
_nvtx_mod.RangePop = _nvtx.RangePop
_cuda_mod.nvtx = _nvtx_mod


class _runtime:
    @staticmethod
    def getDeviceCount():
        return 0

    @staticmethod
    def memGetInfo():
        return (0, 0)


_cuda_mod.runtime = _runtime
_cuda_mod.Device = lambda *a, **k: None
_cuda_mod.MemoryPool = _FakeMemoryPool

sys.modules["cupy.cuda"] = _cuda_mod
sys.modules["cupy.cuda.nvtx"] = _nvtx_mod
_mod.cuda = _cuda_mod

# random submodule -> numpy.random (signatures used: seed, exponential, cumsum via np)
_mod.random = np.random
