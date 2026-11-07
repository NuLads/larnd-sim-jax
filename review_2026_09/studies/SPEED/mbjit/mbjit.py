"""Jit ceiling for the per-batch data loss (speed, 20 Oct). On NB batches of a finished fit (KN10C3 @ 14000, fitted
tracks, own stored target), value_and_grad w.r.t. (calibration params, tracks):
  A  EAGER, the production path (dynamic unique bucketed by LARND_UNIQUE_BUCKET, running-max ROI): s/call after a
     warm-up pass, cycling over the batches as an epoch does.
  B  JITTED, batch data as traced ARGUMENTS, unique_size and ROI static (= the eager path's own sizes): compile time,
     s/call, host RSS after each compile, and the loss vs A on the same batch.
  C  the persistent-cache entries A wrote (size, name): which eager executables embed batch data as constants.
usage: python mbjit.py NB PARTS(A,B,C) OUT.json      (env MB_CACHE = a fresh persistent-cache dir)"""
import os, sys, time, json
STAGE = '/sdf/scratch/neutrino/pgranger/review_2026_09/SPEED/stage'
NB, PARTS, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]
os.environ['JAX_COMPILATION_CACHE_DIR'] = os.environ['MB_CACHE']
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/eventdisplay'); sys.path.insert(0, STAGE)
import numpy as np, jax, jax.numpy as jnp
import larndsim.sim_jax as sj, optimize.fit_params, optimize.strategies   # from the stage, before rebuild imports the live tree
from rebuild import FitRebuild
from optimize.strategies import LUTProbabilisticSimulation, ProbabilisticLossStrategy
from larndsim.losses_jax import llhd_loss
print('sim_jax from', sj.__file__, '| UNIQUE_BUCKET', sj._UNIQUE_BUCKET, '| XLA_FLAGS', os.environ.get('XLA_FLAGS'), flush=True)
rss = lambda: int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0]) / 2 ** 20

R = FitRebuild(sorted(__import__('glob').glob('/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/sci_full_KN10C3/history_iter14000_*.pkl'))[0])
P = R.truth_params.replace(**{k: jnp.ones_like(getattr(R.truth_params, k)) * R.calib_fit[k] for k in R.pc.PARAMS})
fields = R.sfields; sim = LUTProbabilisticSimulation(R.response); loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)
data = [(jnp.asarray(R.batch(i, check=False)['fitted']), {k: jnp.asarray(v) for k, v in R.target(i).items()}) for i in range(NB)]
print('batches: rows ' + ' '.join(str(t.shape[0]) for t, _ in data) + f' | RSS {rss():.1f} GB', flush=True)
res = dict(NB=NB, xla_flags=os.environ.get('XLA_FLAGS'))
blk = lambda x: jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, 'block_until_ready') else a, x)

def f_eager(P, tracks, tgt):
    pred = sim.predict(P, tracks, fields, None); return loss.compute(P, pred, tgt)[0], pred['unique_pixels']
vg_e = jax.value_and_grad(f_eager, argnums=(0, 1), has_aux=True)
U = {}
if 'A' in PARTS or 'B' in PARTS:                     # warm-up pass: fixes the running-max ROI and each batch's unique size
    for i, (t, g) in enumerate(data):
        t0 = time.perf_counter(); (L, u), _ = blk(vg_e(P, t, g)); u = int(u.shape[0]); U[i] = u
        print(f'[A warm] batch {i} loss {float(L):.3f} U {u} ROI ({sim.padded_small_nb},{sim.padded_large_nb}) {time.perf_counter() - t0:.1f} s RSS {rss():.1f}', flush=True)
    ROI = (int(sim.padded_small_nb), int(sim.padded_large_nb)); res.update(U=U, ROI=ROI)
if 'A' in PARTS:
    LA, TA = {}, []
    for ep in range(3):
        for i, (t, g) in enumerate(data):
            t0 = time.perf_counter(); (L, _), _ = blk(vg_e(P, t, g)); dt = time.perf_counter() - t0
            LA[i] = float(L); TA.append(dt) if ep > 0 else None
            print(f'[A] epoch {ep} batch {i} {dt:.2f} s', flush=True)
    res['A'] = dict(median_s=float(np.median(TA)), times=TA, loss=LA, rss_gb=rss()); print(f'[A] median {np.median(TA):.2f} s/call, RSS {rss():.1f} GB', flush=True)
if 'B' in PARTS:
    def f_jit(P, tracks, tgt, usize, roi):
        return loss.compute(P, sim.predict(P, tracks, fields, None, unique_size=usize, roi_override=roi), tgt)[0]
    vg_j = jax.jit(jax.value_and_grad(f_jit, argnums=(0, 1)), static_argnums=(3, 4))
    B = dict(compile_s={}, lower_s={}, rss_after_compile={}, loss={}, times=[])
    comp = {}
    for i, (t, g) in enumerate(data):
        t0 = time.perf_counter(); lo = vg_j.lower(P, t, g, U[i], ROI); t1 = time.perf_counter(); comp[i] = lo.compile(); t2 = time.perf_counter()
        B['lower_s'][i], B['compile_s'][i], B['rss_after_compile'][i] = t1 - t0, t2 - t1, rss()
        print(f'[B] batch {i} lower {t1 - t0:.1f} s compile {t2 - t1:.1f} s RSS {rss():.1f} GB', flush=True)
        if rss() > 50: print('[B] RSS > 50 GB: stop compiling', flush=True); break
    for ep in range(4):
        for i in comp:
            t, g = data[i]; t0 = time.perf_counter(); L, _ = blk(comp[i](P, t, g)); dt = time.perf_counter() - t0
            B['loss'][i] = float(L); B['times'].append(dt) if ep > 0 else None
            print(f'[B] epoch {ep} batch {i} {dt:.3f} s', flush=True)
    B['median_s'] = float(np.median(B['times'])); res['B'] = B
    print(f'[B] median {B["median_s"]:.3f} s/call; mean compile {np.mean(list(B["compile_s"].values())):.0f} s; RSS {rss():.1f} GB', flush=True)
    if 'A' in res:
        for i in comp: print(f'[B vs A] batch {i}: loss {B["loss"][i]:.4f} vs {res["A"]["loss"][i]:.4f} (rel {B["loss"][i] / res["A"]["loss"][i] - 1:+.1e})', flush=True)
if 'C' in PARTS:
    d = os.environ['MB_CACHE']; ent = sorted(((os.path.getsize(os.path.join(d, f)), f) for f in os.listdir(d)), reverse=True)
    res['C'] = dict(n=len(ent), total_gb=sum(s for s, _ in ent) / 2 ** 30, top=[(s, f) for s, f in ent[:25]])
    print(f'[C] {len(ent)} entries, {res["C"]["total_gb"]:.2f} GB; largest:', flush=True)
    for s, f in ent[:25]: print(f'[C]   {s / 2 ** 20:9.1f} MB  {f}', flush=True)
json.dump(res, open(OUT, 'w'), indent=1); print('done', flush=True)
