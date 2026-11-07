"""Equivalence test for LARND_UNIQUE_BUCKET (staged sim_jax): eager value_and_grad w.r.t. (calibration, tracks) of the
data loss on slices of real KN10C batches, plus simulate_wfs's unique_pixels/wfs. Run once with the flag 0 and once 1;
compare the two npz files. usage: python test_unique_bucket.py STAGE out.npz"""
import os, sys, glob, time
STAGE, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, STAGE)
import larndsim.sim_jax as sj, optimize.fit_params as fp
assert sj.__file__.startswith(STAGE) and fp.__file__.startswith(STAGE), (sj.__file__, fp.__file__)
print('UNIQUE_BUCKET', sj._UNIQUE_BUCKET, flush=True)
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/eventdisplay')
from rebuild import FitRebuild
import numpy as np, jax, jax.numpy as jnp
ck = sorted(glob.glob('/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/sci_full_KN10C3/history_iter14000_*.pkl'))[0]
R = FitRebuild(ck); pc = R.pc
assert sys.modules['larndsim.sim_jax'] is sj
from optimize.strategies import LUTProbabilisticSimulation, ProbabilisticLossStrategy
from larndsim.losses_jax import llhd_loss
sim = LUTProbabilisticSimulation(R.response); loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)
P = R.truth_params.replace(**{k: jnp.ones_like(getattr(R.truth_params, k)) * v for k, v in R.calib_fit.items()})
fields = R.sfields; out = {}
for i, frac in ((0, 3), (5, 4)):
    tr = jnp.asarray(R.batch(i, check=False)['fitted']); tr = tr[:tr.shape[0] // frac]
    tgt = {k: jnp.asarray(v) for k, v in R.target(i).items()}
    w, u = sj.simulate_wfs(P, R.response, tr, fields)
    out[f'b{i}_u'] = np.asarray(u); out[f'b{i}_wfs'] = np.asarray(w)
    f = lambda P_, t_: loss.compute(P_, sim.predict(P_, t_, fields, None), tgt)[0]
    t0 = time.perf_counter()
    for rep in range(3):                                   # 3 calls: the first compiles; rows shift by 1 ulp-free amount
        L, (gP, gt) = jax.value_and_grad(f, argnums=(0, 1))(P, tr)
    print(f'batch {i}: {tr.shape[0]} rows, n_unique {int((np.asarray(u) >= 0).sum())}, size {u.shape[0]}, L {float(L):.6f}, 3 calls {time.perf_counter() - t0:.1f} s', flush=True)
    out[f'b{i}_L'] = np.asarray(L); out[f'b{i}_gt'] = np.asarray(gt)
    for k in pc.PARAMS: out[f'b{i}_g_{k}'] = np.asarray(getattr(gP, k))
np.savez(OUT, **out); print('saved', OUT)
