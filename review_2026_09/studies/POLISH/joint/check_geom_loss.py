"""GPU check: where is the data-loss gap between the fitted and the truth-projected geometry (tg_KN10S1 step 0: -23 740
over 100 batches at the KN10S1 end point)? Per batch, data loss at (theta_0, dE/dx_0) with three geometries:
  fit    : the fitted spline coefficients
  truth  : the least-squares projection of the true positions (joint_polish2.truth_geom)
  hybrid : truth projection, except that tracks the fit froze in the drift direction (LARND_FREEZE_STRAIGHT_CL: fitted
           e2 coefficients exactly 0) keep e2 = 0 -- i.e. the best geometry the fit's own freeze masks allow
  e1only : fitted e2 everywhere, truth e1 everywhere (pixel-plane shape only)
fit - hybrid = what converging the free coefficients could gain; hybrid - truth = what the frozen drift shape costs."""
import sys, json, time, numpy as np
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint'
run, sample = sys.argv[1], sys.argv[2]
sys.argv = ['joint_polish2.py', '--run', run, '--sample', sample, '--iter', '10000', '--out', '/dev/null', '--state', '/dev/null']
src = open(f'{D}/joint_polish2.py').read().split('\n_js = lambda')[0]
exec(src)
import jax, jax.numpy as jnp


def geom(i, mode):
    bt = R.batch(i, check=False); guess, ctxs = bt['guess'], bt['ctxs']; tp = ck['true_positions'][i]; coeffs = []; nfz = 0
    for c in ctxs:
        K = fp._spline_K(c.total_len, R.knot_cm); fc = np.asarray(bt['coeffs'][tuple(c.track_id)], np.float32)
        frozen = bool(np.all(fc[K:] == 0)); nfz += frozen
        tr = tp.get(tuple(map(float, c.track_id)), tp.get(str(tuple(map(float, c.track_id)))))
        if tr is None or mode == 'fit': coeffs.append(jnp.asarray(fc)); continue
        u0 = fp._dir_from_angles(c.theta0_i, c.phi0_i); e1, e2 = fp._transverse_frame(u0)
        L = float(c.total_len); al = np.asarray(c.alpha_mid, float); tr = np.asarray(tr, float); m = min(len(al), len(tr)); al, tr = al[:m], tr[:m]
        d = tr - (np.asarray(c.x0_fixed)[None, :] + al[:, None] * L * u0[None, :]); Bm = fp._spline_basis(al, K)
        b1 = np.linalg.lstsq(Bm, d @ e1, rcond=None)[0]; b2 = np.linalg.lstsq(Bm, d @ e2, rcond=None)[0]
        if mode == 'e1only' or (mode == 'hybrid' and frozen): b2 = fc[K:]
        coeffs.append(jnp.asarray(np.concatenate([b1, b2]).astype(np.float32)))
    tw, _ = fp._chain_warp_spline(jnp.asarray(guess), coeffs, fp._build_chain_meta_spline(ctxs, R.knot_cm), R.col)
    tw = np.asarray(tw).copy(); pid = np.asarray(ck['batch_parent_ids'][i]); ldc = np.asarray(ck['dedx_cache'][i]['log_dedx'])
    ded = np.where(pid >= 0, np.maximum(np.exp(ldc[np.clip(pid, 0, len(ldc) - 1)]), 1e-3), tw[:, di]); tw[:, di] = ded; tw[:, Ei] = ded * tw[:, xi]
    return tw, nfz, len(ctxs)


P0 = mk(th0); out = []
for i in range(100):
    t0 = time.perf_counter(); tgt = {k: jnp.asarray(v) for k, v in R.target(i).items()}; r = dict(batch=i)
    for mode in ('fit', 'truth', 'hybrid', 'e1only'):
        try:
            tw, nfz, T = geom(i, mode)
            r[mode] = float(loss.compute(P0, sim.predict(P0, jnp.asarray(tw), fields, None), tgt)[0])
        except Exception as e:
            r[mode] = float('nan'); r['err'] = repr(e)[:200]
    r['n_frozen'], r['T'] = nfz, T; out.append(r)
    print(f'batch {i}: T {T} frozen {nfz}  L fit {r["fit"]:.2f}  truth-fit {r["truth"] - r["fit"]:+.2f}  hybrid-fit {r["hybrid"] - r["fit"]:+.2f}  '
          f'e1only-fit {r["e1only"] - r["fit"]:+.2f}  ({time.perf_counter() - t0:.0f} s)', flush=True)
    json.dump(out, open(f'{D}/check_geom_loss_{run}{sample}.json', 'w'), indent=1)
ok = [r for r in out if all(np.isfinite(r[m]) for m in ('fit', 'truth', 'hybrid', 'e1only'))]
S = {m: sum(r[m] - r['fit'] for r in ok) for m in ('truth', 'hybrid', 'e1only')}
print(f'SUM over {len(ok)} batches (minus fit): truth {S["truth"]:+.1f}  hybrid {S["hybrid"]:+.1f}  e1only {S["e1only"]:+.1f}  '
      f'| frozen tracks {sum(r["n_frozen"] for r in out)}/{sum(r["T"] for r in out)}')
