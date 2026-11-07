"""CPU, offline: the PRODUCTION alias gate (LARND_HITS_SEED_E1_ALIAS_UM, optimize/fit_params.py) replayed on KN10S1's
kept targets (KN10 env). Per track, the e1 start error vs truth (rms of the seeded shape minus the truth projection)
with the gate off (= the fit's own seeding) and at several thresholds; the truth-based seed classes are those of
seed_replay.py (harmful: seed error > straight + 200 um; helpful: < straight - 200 um, gate off)."""
import os, sys
for k, v in dict(LARND_DRIFT_FROM_HITS='1', LARND_HITS_SEED_E1='1', LARND_FREEZE_STRAIGHT_CL='0.50', LARND_CHAIN_FREEZE_DRIFT='1',
                 LARND_CHAIN_FRAME='drift', LARND_SPLINE_KMIN='3').items(): os.environ[k] = v
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/eventdisplay')
import glob, json, types, logging, numpy as np, optax, jax.numpy as jnp
logging.disable(logging.INFO)
from rebuild import FitRebuild
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint'
R = FitRebuild(sorted(glob.glob('/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/sci_full_KN10S1/history_iter10000_*.pkl'))[0])
fp, ck = R.fp, R.ck
o = {(x['batch'], round(x['L'], 3)): x for x in json.load(open(f'{D}/per_track_KN10S1_10000.json'))}
THR = (0, 300, 500, 800)
rows = []
for i in range(100):
    bt = R.batch(i, check=False); ctxs = bt['ctxs']; tp = ck['true_positions'][i]; tg = R.target(i); S = {}
    for thr in THR:
        os.environ['LARND_HITS_SEED_E1_ALIAS_UM'] = str(thr)
        obj = types.SimpleNamespace(_batch_chain_contexts={i: ctxs}, _chain_optimizer=optax.adam(1e-2), _batch_true_positions={i: {}},
                                    _chain_cache={i: [{'angles': jnp.zeros(2 * fp._spline_K(c.total_len, R.knot_cm), jnp.float32)} for c in ctxs]})
        fp.GradientDescentFitter._seed_drift_from_hits(obj, i, tg['pixel_x'], tg['pixel_y'], tg['pixel_z'], tg['event'], tg['adcs'], tg['pixel_id'], apply_seed=True)
        S[thr] = [np.asarray(st['angles'], float) for st in obj._chain_cache[i]]
    for ti, c in enumerate(ctxs):
        tr = tp.get(tuple(map(float, c.track_id)), tp.get(str(tuple(map(float, c.track_id)))))
        if tr is None: continue
        K = fp._spline_K(c.total_len, R.knot_cm); u0 = fp._dir_from_angles(c.theta0_i, c.phi0_i); e1, _ = fp._transverse_frame(u0)
        L = float(c.total_len); al = np.asarray(c.alpha_mid, float); tr = np.asarray(tr, float); m = min(len(al), len(tr)); al, tr = al[:m], tr[:m]
        d = tr - (np.asarray(c.x0_fixed)[None, :] + al[:, None] * L * u0[None, :]); B = fp._spline_basis(al, K)
        t1 = np.linalg.lstsq(B, d @ e1, rcond=None)[0]; rms = lambda b: 1e4 * float(np.sqrt(np.mean((B @ b) ** 2)))
        x = o.get((i, round(L, 3)))
        r = dict(batch=i, L=L, bad=bool(x['r_mean'] > 200), err_straight=rms(t1))
        for thr in THR:
            r[f'seeded_{thr}'] = bool(np.any(S[thr][ti][:K] != 0)); r[f'err_{thr}'] = rms(S[thr][ti][:K] - t1)
        rows.append(r)
    print(f'batch {i}: {len(rows)} tracks', flush=True)
os.environ['LARND_HITS_SEED_E1_ALIAS_UM'] = '0'
json.dump(rows, open(f'{D}/seed_gate_KN10S1.json', 'w'), indent=1)
a = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
s0 = a['seeded_0'] > 0; harm = s0 & (a['err_0'] > a['err_straight'] + 200); helpf = s0 & (a['err_0'] < a['err_straight'] - 200)
print(f'{len(rows)} tracks; gate off: e1-seeded {s0.sum()} (harmful {harm.sum()}, helpful {helpf.sum()}); outliers at 1e4 {int(a["bad"].sum())}, '
      f'of which seeded {int((a["bad"] > 0)[s0].sum())}')
for thr in THR:
    sk = a[f'seeded_{thr}'] > 0; e = a[f'err_{thr}']
    print(f'  gate {thr:4d} um: seeded {sk.sum():4d} (harmful kept {np.sum(sk & harm)}/{harm.sum()}, helpful kept {np.sum(sk & helpf)}/{helpf.sum()}, '
          f'1e4-outliers kept {np.sum(sk & (a["bad"] > 0))}); start e1 error all tracks mean {e.mean():.0f} median {np.median(e):.0f} (> 1 mm {np.sum(e > 1000)}); '
          f'on the gate-off seeded set mean {e[s0].mean():.0f} (> 1 mm {np.sum(e[s0] > 1000)})')
e = a['err_straight']
print(f'  no e1 seed      : start e1 error all tracks mean {e.mean():.0f} median {np.median(e):.0f} (> 1 mm {np.sum(e > 1000)}); on the gate-off seeded set mean {e[s0].mean():.0f} (> 1 mm {np.sum(e[s0] > 1000)})')
