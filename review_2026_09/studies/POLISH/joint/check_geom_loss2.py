"""GPU oracle split of the fitted-vs-truth-geometry data-loss gap by TRACK CLASS (rule in REVIEW.md 22 Oct). Per batch,
data loss at (theta_0, dE/dx_0) of the KN10S1 end point when only one class of tracks is moved to its truth projection:
  out200 / out500 : tracks whose fitted mean |r| to truth is > 200 / 500 um (the heavy tail: 10 % of tracks, ~half of |r|)
  in200           : all the other tracks (|r| <= 200 um)
  frozen          : the tracks the hits called straight (fitted e2 exactly 0), full truth projection (e1 and e2)
plus fit and truth (all tracks) as the references of check_geom_loss.py."""
import sys, json, time, numpy as np
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint'
run, sample = sys.argv[1], sys.argv[2]
sys.argv = ['joint_polish2.py', '--run', run, '--sample', sample, '--iter', '10000', '--out', '/dev/null', '--state', '/dev/null']
exec(open(f'{D}/joint_polish2.py').read().split('\n_js = lambda')[0])
import jax, jax.numpy as jnp
MODES = ('fit', 'truth', 'out200', 'out500', 'in200', 'frozen')
cols = [R.col['x'], R.col['y'], R.col['z']]


def geom(i, mode):
    bt = R.batch(i, check=False); guess, ctxs = bt['guess'], bt['ctxs']; tp = ck['true_positions'][i]; coeffs = []; n_sw = 0
    for c in ctxs:
        K = fp._spline_K(c.total_len, R.knot_cm); fc = np.asarray(bt['coeffs'][tuple(c.track_id)], np.float32)
        tr = tp.get(tuple(map(float, c.track_id)), tp.get(str(tuple(map(float, c.track_id)))))
        if tr is None: coeffs.append(jnp.asarray(fc)); continue
        tr = np.asarray(tr, float); idx = np.asarray(c.idxs)[:len(tr)]
        r = 1e4 * float(np.linalg.norm(bt['fitted'][idx][:, cols] - tr[:len(idx)], axis=1).mean())
        sw = dict(fit=False, truth=True, out200=r > 200, out500=r > 500, in200=r <= 200, frozen=bool(np.all(fc[K:] == 0)))[mode]
        if not sw: coeffs.append(jnp.asarray(fc)); continue
        n_sw += 1
        u0 = fp._dir_from_angles(c.theta0_i, c.phi0_i); e1, e2 = fp._transverse_frame(u0)
        L = float(c.total_len); al = np.asarray(c.alpha_mid, float); m = min(len(al), len(tr)); al, trm = al[:m], tr[:m]
        d = trm - (np.asarray(c.x0_fixed)[None, :] + al[:, None] * L * u0[None, :]); Bm = fp._spline_basis(al, K)
        coeffs.append(jnp.asarray(np.concatenate([np.linalg.lstsq(Bm, d @ e1, rcond=None)[0],
                                                  np.linalg.lstsq(Bm, d @ e2, rcond=None)[0]]).astype(np.float32)))
    tw, _ = fp._chain_warp_spline(jnp.asarray(guess), coeffs, fp._build_chain_meta_spline(ctxs, R.knot_cm), R.col)
    tw = np.asarray(tw).copy(); pid = np.asarray(ck['batch_parent_ids'][i]); ldc = np.asarray(ck['dedx_cache'][i]['log_dedx'])
    ded = np.where(pid >= 0, np.maximum(np.exp(ldc[np.clip(pid, 0, len(ldc) - 1)]), 1e-3), tw[:, di]); tw[:, di] = ded; tw[:, Ei] = ded * tw[:, xi]
    return tw, n_sw


P0 = mk(th0); out = []
for i in range(100):
    t0 = time.perf_counter(); tgt = {k: jnp.asarray(v) for k, v in R.target(i).items()}; r = dict(batch=i)
    for mode in MODES:
        try:
            tw, n = geom(i, mode); r['n_' + mode] = n
            r[mode] = float(loss.compute(P0, sim.predict(P0, jnp.asarray(tw), fields, None), tgt)[0])
        except Exception as e:
            r[mode] = float('nan'); r['err'] = repr(e)[:200]
    out.append(r)
    print(f'batch {i}: ' + '  '.join(f'{m}-fit {r[m] - r["fit"]:+.1f} (n {r.get("n_" + m)})' for m in MODES[1:]) + f'  ({time.perf_counter() - t0:.0f} s)', flush=True)
    json.dump(out, open(f'{D}/check_geom_loss2_{run}{sample}.json', 'w'), indent=1)
ok = [r for r in out if all(np.isfinite(r[m]) for m in MODES)]
print(f'SUM over {len(ok)} batches (minus fit): ' + '  '.join(f'{m} {sum(r[m] - r["fit"] for r in ok):+.1f} (tracks {sum(r["n_" + m] for r in ok)})'
                                                       for m in MODES[1:]) + f'  | fit {sum(r["fit"] for r in ok):.1f}')
