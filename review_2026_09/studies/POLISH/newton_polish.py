"""Calibration polish + per-fit uncertainty (20 Oct): one Newton step on the five calibration parameters of a finished
joint fit, with its nuisances (spline coefficients, per-segment log dE/dx of the final checkpoint) held fixed.

The calibration learning rate anneals to ~1e-5 while the nuisances keep moving, so the reported calibration (mean of the
last 1000 iterations, theta_0) can lag its conditional optimum given the final nuisances. For one track sample, over all
its batches and against the fit's own stored stochastic target, the DATA loss (probabilistic model + PPP likelihood; the
nuisance priors do not depend on theta) gives, in log-parameter units (theta_j theta_k d2L/dtheta_j dtheta_k):
  g = sum_b g_b,  H = sum_b H_b   (central finite differences of the AD gradient, step h_k in ln theta_k ~ 1-3x the
  conditional floor of a 400 m sample; the forward-over-reverse HVP of paper_hessian.py needs > 40 GB at 400 cm: OOM 20 Oct)
  Newton step  d = -H^-1 g,  theta_1 = theta_0 exp(d); then the data loss is re-evaluated at theta_1 (actual vs the
  quadratic prediction -g.H^-1.g/2), as a check that the step is valid.
Per-fit uncertainty at the fit's own end point: conditional (nuisances fixed) sigma_k = sqrt((H^-1)_kk); sandwich
H^-1 V H^-1 with V the scatter of the per-batch scores (centred, and uncentred as in paper_hessian.py).
Readout rule: REVIEW.md, 20 Oct (fixed before any result).
usage: python newton_polish.py --run KN10C --iter 14000 --sample s --out out.json [--nb 100] [--suffix b]"""
import os, sys, glob, json, time, argparse
import numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/eventdisplay')
from rebuild import FitRebuild
import jax, jax.numpy as jnp

ap = argparse.ArgumentParser(); ap.add_argument('--run', default='KN10C'); ap.add_argument('--iter', type=int, default=14000)
ap.add_argument('--sample', type=int, required=True); ap.add_argument('--nb', type=int, default=100); ap.add_argument('--out', required=True)
ap.add_argument('--suffix', default='')   # 21 Oct: replica runs, e.g. sci_full_KN10V1b
a = ap.parse_args()
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
ckpt = sorted(glob.glob(f'{FR}/sci_full_{a.run}{a.sample}{a.suffix}/history_iter{a.iter}_*.pkl'))[0]
R = FitRebuild(ckpt); pc = R.pc; PARAMS = pc.PARAMS
for p in PARAMS:                                          # errors are quoted against the fit's own targets
    t = float(np.ravel(R.ck[p + '_target'])[0]); assert abs(t / pc.TRUTH[p] - 1) < 1e-6, (p, t, pc.TRUTH[p])
from optimize.strategies import LUTProbabilisticSimulation, ProbabilisticLossStrategy
from larndsim.losses_jax import llhd_loss
sim = LUTProbabilisticSimulation(R.response); loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)
base = R.truth_params; fields = R.sfields
mk = lambda th: base.replace(**{k: jnp.ones_like(getattr(base, k)) * th[k] for k in PARAMS})
H_STEP = dict(Ab=5e-4, eField=1.5e-5, tran_diff=1e-2, long_diff=2e-2, lifetime=1e-2)   # ln-units FD steps
th0 = dict(R.calib_fit); err = lambda th: {k: 100 * (th[k] / pc.TRUTH[k] - 1) for k in PARAMS}
print(f'{a.run}{a.sample} @ {a.iter}: theta_0 error %', {k: f'{v:+.3f}' for k, v in err(th0).items()}, flush=True)
res = dict(run=a.run, iter=a.iter, sample=a.sample, suffix=a.suffix, ckpt=ckpt, params=PARAMS, truth=pc.TRUTH, theta0=th0, batches=[])

nb = min(a.nb, len(R.ds_sim)); data = []
for i in range(nb):
    t0 = time.perf_counter()
    tracks = jnp.asarray(R.batch(i, check=False)['fitted']); tgt = {k: jnp.asarray(v) for k, v in R.target(i).items()}
    Lb = float(np.asarray(tracks)[:, fields.index('dx')].sum()); data.append((tracks, tgt))
    try:
        f = lambda P, tracks=tracks, tgt=tgt: loss.compute(P, sim.predict(P, tracks, fields, None), tgt)[0]
        vg = jax.value_and_grad(f)
        gln = lambda th: (lambda Lg: (float(Lg[0]), np.array([float(getattr(Lg[1], j)) * th[j] for j in PARAMS])))(vg(mk(th)))
        L0, gb = gln(th0); Hb = np.zeros((5, 5))
        for c, k in enumerate(PARAMS):                    # column c: d(dL/dln theta_j)/dln theta_k
            h = H_STEP[k]
            gp = gln({**th0, k: th0[k] * float(np.exp(h))})[1]; gm = gln({**th0, k: th0[k] * float(np.exp(-h))})[1]
            Hb[:, c] = (gp - gm) / (2 * h)
        res['batches'].append(dict(batch=i, length_cm=Lb, L0=L0, g=gb.tolist(), H=Hb.tolist(),
                                   asym=float(np.abs(Hb - Hb.T).max() / np.abs(Hb).max())))
        print(f'[batch {i}] {Lb:.0f} cm  L0 {L0:.1f}  g ' + ' '.join(f'{x:+.2f}' for x in gb) + '  diag(H) ' + ' '.join(f'{x:.3e}' for x in np.diag(Hb)) + f'  asym {res["batches"][-1]["asym"]:.1e}  {time.perf_counter() - t0:.1f} s', flush=True)
    except Exception as e:
        res['batches'].append(dict(batch=i, length_cm=Lb, error=repr(e)[:300])); print(f'[batch {i}] FAILED {repr(e)[:200]}', flush=True)
    if i % 10 == 0: json.dump(res, open(a.out, 'w'))

ok = [b for b in res['batches'] if 'H' in b]
if not ok: sys.exit('no batch succeeded')
H = sum(np.array(b['H']) for b in ok); H = 0.5 * (H + H.T); G = np.array([b['g'] for b in ok]); g = G.sum(0)
ev = np.linalg.eigvalsh(H); Hi = np.linalg.inv(H)
d = -Hi @ g; th1 = {k: th0[k] * float(np.exp(x)) for k, x in zip(PARAMS, d)}
dL_pred = -0.5 * float(g @ Hi @ g)
L1 = []
for b in ok:                                              # the data loss at theta_1, same batches, same nuisances
    tracks, tgt = data[b['batch']]
    L1.append(float(loss.compute(mk(th1), sim.predict(mk(th1), tracks, fields, None), tgt)[0]))
dL = float(sum(L1) - sum(b['L0'] for b in ok))
N = len(ok); Gc = G - G.mean(0)
Vc = Gc.T @ Gc * N / (N - 1); V0 = G.T @ G
sig = lambda C: (100 * np.sqrt(np.clip(np.diag(C), 0, None))).tolist()
res['summary'] = dict(n_ok=N, n_batches=nb, length_cm=sum(b['length_cm'] for b in ok), H=H.tolist(), g=g.tolist(), eig=ev.tolist(),
                      step_pct=(100 * d).tolist(), theta1=th1, err0=err(th0), err1=err(th1), dL_actual=dL, dL_pred=dL_pred,
                      sigma_cond_pct=sig(Hi), sigma_sandwich_centred_pct=sig(Hi @ Vc @ Hi), sigma_sandwich_uncentred_pct=sig(Hi @ V0 @ Hi),
                      corr=(Hi / np.sqrt(np.outer(np.diag(Hi), np.diag(Hi)))).tolist())
json.dump(res, open(a.out, 'w'), indent=1)
s = res['summary']
print(f'\n{a.run}{a.sample}: {N}/{nb} batches, {s["length_cm"] / 100:.1f} m; eig(H) ' + ' '.join(f'{x:.3e}' for x in ev))
print('  step (%)       ' + '  '.join(f'{k} {x:+.3f}' for k, x in zip(PARAMS, s['step_pct'])))
print('  error before % ' + '  '.join(f'{k} {s["err0"][k]:+.3f}' for k in PARAMS))
print('  error after  % ' + '  '.join(f'{k} {s["err1"][k]:+.3f}' for k in PARAMS))
print(f'  data loss change {dL:+.2f} (quadratic prediction {dL_pred:+.2f})')
print('  sigma cond / sandwich(c) / sandwich(u) % ' + '  '.join(f'{k} {x:.3f}/{y:.3f}/{z:.3f}' for k, x, y, z in
      zip(PARAMS, s['sigma_cond_pct'], s['sigma_sandwich_centred_pct'], s['sigma_sandwich_uncentred_pct'])), flush=True)
print('done', flush=True)
