"""CPU check: the fit's position prior (MCS, weight from the run config; drift penalty 0 and endpoints off in KN10) at the
fitted spline coefficients vs the truth-projected ones (joint_polish2.truth_geom), per batch. The data-loss gap at the
same (theta, dE/dx) is known from tg_KN10S1 / iter_KN10S1 step 0: truth geometry 23 740 lower over 100 batches. If the
MCS prior at truth costs less than that, the fit's full objective prefers the truth geometry (positions under-converged)."""
import sys, glob, json, numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/eventdisplay')
from rebuild import FitRebuild
import jax.numpy as jnp
ck_path = sorted(glob.glob('/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/sci_full_KN10S1/history_iter10000_*.pkl'))[0]
R = FitRebuild(ck_path); fp, ck = R.fp, R.ck; fields = R.sfields; cfg = ck['config']
di, Ei, xi = fields.index('dEdx'), fields.index('dE'), fields.index('dx')
W = float(cfg.mcs_prior_weight); assert not fp._N_EP and float(__import__('os').environ.get('LARND_CHAIN_DRIFT_W', '0') or 0) == 0
src = open('/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint/joint_polish2.py').read()
body = src.split('def truth_geom(i):')[1].split('\ndef batch_fns')[0]
body = body.replace('    return tw, nkeep', '    return tw, nkeep, coeffs, ctxs, guess')   # also hand back the coefficients
exec('def truth_geom(i):' + body)
nb = int(sys.argv[1]) if len(sys.argv) > 1 else 100; out = []
for i in range(nb):
    bt = R.batch(i, check=False); _, nkeep, ct, ctxs, guess = truth_geom(i)
    meta = fp._build_chain_meta_spline(ctxs, R.knot_cm)
    cf = [jnp.asarray(np.asarray(bt['coeffs'][tuple(c.track_id)], np.float32)) for c in ctxs]
    mcs = {}
    for name, co in (('fit', cf), ('truth', ct)):
        _, disp = fp._chain_warp_spline(jnp.asarray(guess), co, meta, R.col)
        mcs[name] = float(fp._chain_mcs_spline(disp, meta, W))
    out.append(dict(batch=i, T=len(ctxs), nkeep=nkeep, mcs_fit=mcs['fit'], mcs_truth=mcs['truth']))
    print(f'batch {i}: T {len(ctxs)} kept-fitted {nkeep}  MCS prior fit {mcs["fit"]:.3f}  truth {mcs["truth"]:.3f}  diff {mcs["truth"] - mcs["fit"]:+.3f}', flush=True)
d = np.array([o['mcs_truth'] - o['mcs_fit'] for o in out])
print(f'SUM over {nb} batches: MCS prior fit {sum(o["mcs_fit"] for o in out):.2f}  truth {sum(o["mcs_truth"] for o in out):.2f}  '
      f'diff {d.sum():+.2f}  (weight {W}); data-loss gap (100 batches, truth - fit) -23740.5')
json.dump(out, open('/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint/check_geom_prior.json', 'w'), indent=1)
