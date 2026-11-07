"""Schedule-invariance failure (10 Oct): where does the SCHED - R2 difference live?

For one track sample s, the two end points R = sci_full_STRFRZ50<s> (production schedule) and S = sci_full_SCHED<s>
(1.78x the calibration travel) differ by +7 points in lifetime, +5 in D_L, -0.24 in A_b, and S ends 17 per batch
higher in loss. Each end point = calibration theta (mean of the last 1000 iterations, the reported value) + nuisances
nu (spline coefficients and per-segment log dE/dx of the final checkpoint), rebuilt exactly as the fit feeds them to
the simulation (rebuild.FitRebuild).

Per batch, against the fit's own stored stochastic target (identical in both runs, asserted), the DATA loss
(probabilistic model + PPP likelihood; nuisance priors are not included, they do not depend on theta) and its gradient
with respect to the five calibration parameters (log units, theta_k dL/dtheta_k) at the four combinations
    (th_R, nu_R)  (th_S, nu_R)  (th_R, nu_S)  (th_S, nu_S).
Summed over the 100 batches, the conditional Newton step at each end point, delta = -H^-1 g, uses the Hessian of the
same loss at the truth with true inputs (PAPERJOBS/hessian.json, 200 cm batches), scaled to this sample's length:
an approximation to where the calibration would go if the nuisances were held fixed.

Readout rule (fixed before any result, REVIEW.md 10 Oct):
  CALIBRATION-LIMITED  if on >= 2/3 samples L(th_R, nu_S) < L(th_S, nu_S) AND the Newton step from (th_S, nu_S) moves
                       the lifetime toward R2's value by >= half of the SCHED - R2 lifetime difference.
  NUISANCE-COUPLED     if on >= 2/3 samples |Newton lifetime step| < 1.5 points at BOTH end points AND each theta is
                       preferred with its own nu (L(th_S,nu_S) < L(th_R,nu_S) and L(th_R,nu_R) < L(th_S,nu_R)).
  otherwise MIXED (reported as is)."""
import os, sys, glob, json, time, argparse
import numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/eventdisplay')
from rebuild import FitRebuild
import jax, jax.numpy as jnp

ap = argparse.ArgumentParser(); ap.add_argument('--sample', type=int, required=True); ap.add_argument('--nb', type=int, default=100)
ap.add_argument('--out', required=True); a = ap.parse_args()
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
ck = lambda run: sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter10000_*.pkl'))[0]
R = FitRebuild(ck(f'STRFRZ50{a.sample}')); S = FitRebuild(ck(f'SCHED{a.sample}'))
pc = R.pc; PARAMS = pc.PARAMS
from optimize.strategies import LUTProbabilisticSimulation, ProbabilisticLossStrategy
from larndsim.losses_jax import llhd_loss
sim = LUTProbabilisticSimulation(R.response); loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)
base = R.truth_params
mk = lambda th: base.replace(**{k: jnp.ones_like(getattr(base, k)) * th[k] for k in PARAMS})
TH = dict(R=R.calib_fit, S=S.calib_fit)
print('theta_R', {k: f'{100*(v/pc.TRUTH[k]-1):+.3f}%' for k, v in TH['R'].items()})
print('theta_S', {k: f'{100*(v/pc.TRUTH[k]-1):+.3f}%' for k, v in TH['S'].items()}, flush=True)
fields = R.sfields
combos = [('R', 'R'), ('S', 'R'), ('R', 'S'), ('S', 'S')]          # (theta, nu)
res = dict(sample=a.sample, theta=TH, truth=pc.TRUTH, params=PARAMS, batches=[])
for i in range(min(a.nb, len(R.ds_sim))):
    t0 = time.perf_counter()
    tr = dict(R=jnp.asarray(R.batch(i, check=False)['fitted']), S=jnp.asarray(S.batch(i, check=False)['fitted']))
    tR, tS = R.target(i), S.target(i)
    # the two runs regenerate the same stochastic target on the GPU: equal up to float non-determinism (checked 10 Oct:
    # ADC differences ~1e-5, on sample 7 a few hits one tick / 0.6 ADC apart); the loss always uses R2's copy
    assert all(tR[k].shape == tS[k].shape for k in tR) and np.array_equal(tR['pixel_id'], tS['pixel_id']), f'batch {i}: target hits differ'
    nbad = int(((np.abs(tR['adcs'] - tS['adcs']) > 1.0) | (np.abs(tR['ticks'] - tS['ticks']) > 1)).sum())
    assert nbad <= 0.001 * len(tR['adcs']) + 1, f'batch {i}: {nbad} target hits differ by > 1 ADC or > 1 tick'
    tgt = {k: jnp.asarray(v) for k, v in tR.items()}
    row = dict(batch=i, length_cm=float(np.asarray(tr['R'])[:, fields.index('dx')].sum()))
    for th, nu in combos:
        P = mk(TH[th]); tracks = tr[nu]
        f = lambda P: loss.compute(P, sim.predict(P, tracks, fields, None), tgt)[0]
        L, g = jax.value_and_grad(f)(P)
        row[f'L_{th}{nu}'] = float(L); row[f'g_{th}{nu}'] = [float(getattr(g, k)) * TH[th][k] for k in PARAMS]
    res['batches'].append(row)
    print(f"[batch {i}] {row['length_cm']:.0f} cm  L_RR {row['L_RR']:.1f}  L_SR-L_RR {row['L_SR']-row['L_RR']:+.2f}  "
          f"L_SS {row['L_SS']:.1f}  L_RS-L_SS {row['L_RS']-row['L_SS']:+.2f}  {time.perf_counter()-t0:.1f} s", flush=True)
    if i % 10 == 0: json.dump(res, open(a.out, 'w'))
json.dump(res, open(a.out, 'w'))

# ---------------- summary ----------------
B = res['batches']; Ltot = {c: sum(b[f'L_{c[0]}{c[1]}'] for b in B) for c in combos}
G = {c: np.sum([b[f'g_{c[0]}{c[1]}'] for b in B], axis=0) for c in combos}
Lcm = sum(b['length_cm'] for b in B)
hz = json.load(open('/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS/hessian.json'))
ok = [b for b in hz['batches'] if 'H' in b]; H = sum(np.array(b['H']) for b in ok); H = 0.5 * (H + H.T)
H *= Lcm / sum(b['length_cm'] for b in ok)                       # truth Hessian scaled to this sample's length
print(f'\nsample {a.sample}: {len(B)} batches, {Lcm/100:.1f} m')
print('data loss totals (sum over batches):  ' + '  '.join(f'L({c[0]}th,{c[1]}nu) {Ltot[c]:.1f}' for c in combos))
print(f'  at nu_S: L(th_R) - L(th_S) = {Ltot[("R","S")] - Ltot[("S","S")]:+.1f}   at nu_R: L(th_S) - L(th_R) = {Ltot[("S","R")] - Ltot[("R","R")]:+.1f}')
print(f'  end points: L(S,S) - L(R,R) = {Ltot[("S","S")] - Ltot[("R","R")]:+.1f}  (data term only; the fit history gives the total incl. priors)')
dlt = {}
for c in [('R', 'R'), ('S', 'S')]:
    d = -np.linalg.solve(H, G[c]); dlt[c] = d
    print(f'  Newton step from (th_{c[0]}, nu_{c[1]}), % : ' + '  '.join(f'{k} {100*x:+.2f}' for k, x in zip(PARAMS, d))
          + '   g(ln units): ' + ' '.join(f'{x:+.1f}' for x in G[c]))
res['summary'] = dict(L=dict((f'{c[0]}{c[1]}', Ltot[c]) for c in combos), g=dict((f'{c[0]}{c[1]}', G[c].tolist()) for c in combos),
                      newton_RR=dlt[('R', 'R')].tolist(), newton_SS=dlt[('S', 'S')].tolist(), length_cm=Lcm)
json.dump(res, open(a.out, 'w'))
print('done', flush=True)
