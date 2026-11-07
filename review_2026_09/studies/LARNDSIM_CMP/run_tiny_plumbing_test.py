import os, sys, time
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
HERE = "/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP"
REF = "/sdf/home/p/pgranger/reference"
sys.path.insert(0, HERE)
sys.path.insert(0, REF)
sys.path.insert(0, REF + "/cli")

import cupy_cpu_shim
sys.modules["cupy"] = cupy_cpu_shim
sys.modules["cupy.cuda"] = cupy_cpu_shim.cuda
sys.modules["cupy.cuda.nvtx"] = cupy_cpu_shim._nvtx_mod

t0 = time.time()
import simulate_pixels
print("imported simulate_pixels OK after", time.time()-t0, "s")

simulate_pixels.run_simulation(
    input_filename=HERE + "/tiny_input.h5",
    output_filename=HERE + "/tiny_output.h5",
    config="module0",
    pixel_layout=REF + "/larndsim/pixel_layouts/multi_tile_layout-2.2.16.yaml",
    detector_properties=REF + "/larndsim/detector_properties/module0.yaml",
    simulation_properties=REF + "/larndsim/simulation_properties/singles_sim.yaml",
    response_file=REF + "/larndsim/bin/response_44.npy",
    light_lut_filename=REF + "/larndsim/bin/lightLUT.npz",
    rand_seed=12345,
)
print("run_simulation completed in", time.time()-t0, "s total")
