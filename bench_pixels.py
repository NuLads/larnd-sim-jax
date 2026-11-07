import os, sys, time
sys.path.insert(0, 'src'); sys.path.insert(0, '.')
import jax, jax.numpy as jnp
from larndsim.consts_jax import build_params_class, load_detector_properties
from larndsim.fee_jax import get_adc_values_average_noise_chunked

Params = build_params_class(['Ab'])
params = load_detector_properties(
    Params,
    'src/larndsim/detector_properties/module0.yaml',
    'src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml',
)

key = jax.random.PRNGKey(0)
Nticks = 2000

def bench(fn, wfs, n_warm=3, n_time=8):
    for _ in range(n_warm):
        jax.block_until_ready(fn(wfs))
    t0 = time.perf_counter()
    for _ in range(n_time):
        jax.block_until_ready(fn(wfs))
    return (time.perf_counter() - t0) / n_time

fwd = jax.jit(lambda w: get_adc_values_average_noise_chunked(params, w, chunk_size=128))
bwd = jax.jit(jax.value_and_grad(
    lambda w: jnp.sum(get_adc_values_average_noise_chunked(params, w, chunk_size=128)[0])))

print("=== FEE pixel scaling benchmark ===")
print(f"{'Npix':>5}  {'chunks':>6}  {'fwd(ms)':>8}  {'fwd_x':>7}  {'bwd(ms)':>8}  {'bwd_x':>7}")
baseline_fwd = baseline_bwd = None
for npix in [128, 256, 384, 512, 640, 768, 1024]:
    wfs = jax.random.normal(key, (npix, Nticks)) * 5000.0
    tf = bench(fwd, wfs)
    tb = bench(bwd, wfs)
    if baseline_fwd is None:
        baseline_fwd = tf; baseline_bwd = tb
    nchunks = (npix + 127) // 128
    print(f"{npix:5d}  {nchunks:6d}  {tf*1e3:8.1f}  {tf/baseline_fwd:7.2f}  {tb*1e3:8.1f}  {tb/baseline_bwd:7.2f}")
