"""CPU, offline, truth-free predictor of a bad e1 hit seed: PIXEL-GRID ALIASING of the nominal straight line.
The seed fits sine modes to the per-pixel points of a track, whose in-plane position is the PIXEL CENTRE. For a
straight track at a shallow angle to a grid axis, (pixel centre - line) . e1 is a sawtooth of amplitude ~pitch/2 and
long period pitch/sin(angle), which low sine modes read as a mm-scale bend. Per track: sample the nominal line densely,
give each sample its pixel centre, weight each pixel by its path length, and run the seed's own fit (robust
[1, alpha, sin modes]) on these noise-free points -> b_alias = the seed a perfectly straight track would get.
Compared with the replayed seed (as seed_replay.py, KN10 env) and truth:
  (1) does |b_alias| predict the harmful seeds?  (2) does seed - b_alias (de-aliased seed) beat the seed?"""
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
PITCH = 0.443


def grid_origin(px):
    """Pixel-centre lattice offset along one axis from the target pixel centres (mod pitch)."""
    ph = np.exp(2j * np.pi * np.asarray(px, float) / PITCH)
    return float(np.angle(ph.mean()) / (2 * np.pi) * PITCH)


def robust_fit(A, y, sw, kmin):                       # as in _seed_drift_from_hits
    keep = np.ones(len(y), bool)
    for _ in range(4):
        coef = np.linalg.lstsq(A[keep] * sw[keep, None], y[keep] * sw[keep], rcond=None)[0]
        r = y - A @ coef; sig = 1.4826 * np.median(np.abs(r[keep])) + 1e-4
        nk = np.abs(r) < 3 * sig
        if nk.sum() < kmin or (nk == keep).all(): break
        keep = nk
    return coef


def alias_coeffs(x0, u0, L, e1, Kf, ox, oy):
    s = np.arange(0.005, 1.0, 0.01 * 0.05 / max(L, 1e-3) * 100)  # ~0.05 cm steps (fraction of the length)
    s = s[(s > 0.01) & (s < 0.99)]
    X = x0[None, :] + s[:, None] * L * u0[None, :]
    ix = np.round((X[:, 0] - ox) / PITCH); iy = np.round((X[:, 1] - oy) / PITCH)
    key = ix * 100003 + iy; up, inv = np.unique(key, return_inverse=True); w = np.bincount(inv).astype(float)
    C = np.zeros((len(up), 3)); C[:, 0] = np.bincount(inv, ox + ix * PITCH) / w; C[:, 1] = np.bincount(inv, oy + iy * PITCH) / w
    C[:, 2] = np.bincount(inv, X[:, 2]) / w
    if len(up) < 3 * Kf + 2: return None, len(up)
    a = ((C - x0) @ u0) / L; d = C - (x0[None, :] + a[:, None] * L * u0[None, :])
    B = np.sin(np.outer(a, np.pi * np.arange(1, Kf + 1))); sw = np.sqrt(w / w.mean()); one = np.ones((len(a), 1))
    return robust_fit(np.hstack([one, a[:, None], B]), d @ e1, sw, 3 * Kf + 2)[2:], len(up)


rows = []
for i in range(100):
    bt = R.batch(i, check=False); ctxs = bt['ctxs']; tp = ck['true_positions'][i]; tg = R.target(i)
    ox, oy = grid_origin(tg['pixel_x']), grid_origin(tg['pixel_y'])
    obj = types.SimpleNamespace(_batch_chain_contexts={i: ctxs}, _chain_optimizer=optax.adam(1e-2),
                                _chain_cache={i: [{'angles': jnp.zeros(2 * fp._spline_K(c.total_len, R.knot_cm), jnp.float32)} for c in ctxs]},
                                _batch_true_positions={i: {}})
    fp.GradientDescentFitter._seed_drift_from_hits(obj, i, tg['pixel_x'], tg['pixel_y'], tg['pixel_z'], tg['event'], tg['adcs'], tg['pixel_id'], apply_seed=True)
    for c, st in zip(ctxs, obj._chain_cache[i]):
        tr = tp.get(tuple(map(float, c.track_id)), tp.get(str(tuple(map(float, c.track_id)))))
        if tr is None: continue
        K = fp._spline_K(c.total_len, R.knot_cm); Kf = fp._spline_nsine(K); sd = np.asarray(st['angles'], float)
        fc = np.asarray(bt['coeffs'][tuple(c.track_id)], float)
        u0 = fp._dir_from_angles(c.theta0_i, c.phi0_i); e1, e2 = fp._transverse_frame(u0)
        L = float(c.total_len); x0 = np.asarray(c.x0_fixed, float)
        al = np.asarray(c.alpha_mid, float); tr = np.asarray(tr, float); m = min(len(al), len(tr)); al, tr = al[:m], tr[:m]
        d = tr - (x0[None, :] + al[:, None] * L * u0[None, :]); B = fp._spline_basis(al, K)
        t1 = np.linalg.lstsq(B, d @ e1, rcond=None)[0]
        rms = lambda b: 1e4 * float(np.sqrt(np.mean((B @ b) ** 2)))
        ba, npx = alias_coeffs(x0, u0, L, e1, Kf, ox, oy)
        seeded = bool(np.any(sd[:K] != 0)); x = o.get((i, round(L, 3)))
        uxy = np.hypot(u0[0], u0[1]); ang = np.degrees(np.arctan2(min(abs(u0[0]), abs(u0[1])), max(abs(u0[0]), abs(u0[1]))))
        r = dict(batch=i, L=L, bad=bool(x['r_mean'] > 200), r_final=x['r_mean'], seeded=seeded, Lxy=float(L * uxy), grid_angle_deg=float(ang),
                 n_minor=float(L * min(abs(u0[0]), abs(u0[1])) / PITCH), err_straight=rms(t1), err_seed=rms(sd[:K] - t1), err_final=rms(fc[:K] - t1))
        if ba is not None:
            bb = np.zeros(K); bb[:Kf] = ba
            r.update(alias_amp=rms(bb), err_dealiased=rms(sd[:K] - bb - t1) if seeded else float('nan'),
                     cos_alias_seederr=float(np.dot(bb[:Kf], (sd[:K] - t1)[:Kf]) / (np.linalg.norm(bb[:Kf]) * np.linalg.norm((sd[:K] - t1)[:Kf]) + 1e-12)) if seeded else float('nan'))
        rows.append(r)
    print(f'batch {i}: {len(rows)} tracks', flush=True)
json.dump(rows, open(f'{D}/alias_KN10S1.json', 'w'), indent=1)
a = {k: np.array([r.get(k, np.nan) for r in rows], float) for k in rows[0].keys() | {'alias_amp', 'err_dealiased', 'cos_alias_seederr'}}
s = a['seeded'] > 0; ok = np.isfinite(a['alias_amp'])
harm = s & (a['err_seed'] > a['err_straight'] + 200); helpf = s & (a['err_seed'] < a['err_straight'] - 200)
print(f'{len(rows)} tracks, e1-seeded {s.sum()}: harmful {harm.sum()}, helpful {helpf.sum()}')
from scipy.stats import spearmanr
print(f'  Spearman(alias_amp, seed error) on seeded: {spearmanr(a["alias_amp"][s & ok], a["err_seed"][s & ok])[0]:+.2f}; '
      f'(alias_amp, seed harm = seed - straight error): {spearmanr(a["alias_amp"][s & ok], (a["err_seed"] - a["err_straight"])[s & ok])[0]:+.2f}')
print(f'  median alias_amp: harmful {np.nanmedian(a["alias_amp"][harm]):.0f}, helpful {np.nanmedian(a["alias_amp"][helpf]):.0f}, '
      f'unseeded {np.nanmedian(a["alias_amp"][~s]):.0f} um; median cos(b_alias, seed error): harmful {np.nanmedian(a["cos_alias_seederr"][harm]):+.2f}, '
      f'helpful {np.nanmedian(a["cos_alias_seederr"][helpf]):+.2f}')
print(f'  median grid angle: harmful {np.median(a["grid_angle_deg"][harm]):.1f}, helpful {np.median(a["grid_angle_deg"][helpf]):.1f} deg; '
      f'n_minor harmful {np.median(a["n_minor"][harm]):.1f}, helpful {np.median(a["n_minor"][helpf]):.1f}')
for nm, m in (('harmful', harm), ('helpful', helpf), ('all seeded', s)):
    m = m & ok
    print(f'  {nm:10s} n {m.sum():4d}  e1 start error median / mean: straight {np.median(a["err_straight"][m]):.0f} / {np.mean(a["err_straight"][m]):.0f}, '
          f'seed {np.median(a["err_seed"][m]):.0f} / {np.mean(a["err_seed"][m]):.0f}, de-aliased {np.median(a["err_dealiased"][m]):.0f} / {np.mean(a["err_dealiased"][m]):.0f} um')
for thr in (300, 500, 800):
    keep = s & ok & (a['alias_amp'] < thr)
    e = np.where(keep, a['err_seed'], a['err_straight'])
    print(f'  rule "seed only if alias_amp < {thr}": keeps harmful {np.sum(keep & harm)}/{harm.sum()}, helpful {np.sum(keep & helpf)}/{helpf.sum()}; '
          f'start error over seeded-now tracks mean {np.mean(e[s & ok]):.0f} (> 1 mm {np.sum(e[s & ok] > 1000)}) vs current {np.mean(a["err_seed"][s & ok]):.0f} '
          f'({np.sum(a["err_seed"][s & ok] > 1000)}), no seed {np.mean(a["err_straight"][s & ok]):.0f} ({np.sum(a["err_straight"][s & ok] > 1000)})')
