"""By-products of the joint fit: how well recipe R2 (STRFRZ50) reconstructs the trajectories and the per-segment energy
loss, over its nine track samples, from the fit checkpoints (rebuild.py). Per track: mean distance of the fitted and of
the straight-line trajectory to the true one (length-weighted over the chopped rows), and the rms of the fitted error
along the two transverse directions (e1: pixel plane, e2: drift-carrying). Per segment (the fit's ~1 cm segmentation):
fitted dE/dx, true dE/dx (true deposit projected on the segment span, true_dedx_matched_cache), initial value (the
guess file's dE/dx, i.e. the true track average) and length. Output: npz of arrays + json of summaries.
BYPRODUCT_HITSEED=1 also measures the hit-derived STARTING trajectory (d_hit, e1_hit, e2_hit): each batch is re-seeded
from a fresh (straight) chain state with the production GradientDescentFitter._seed_drift_from_hits on the fit's own
kept target hits, exactly as paperfigs/fig_initialisation.py does (cross-checked there against the production log), with
every LARND_* knob of the checkpoint set before fit_params is imported; then warped like the fitted trajectory."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuild import FitRebuild
ST = '/sdf/scratch/neutrino/pgranger/fit_result_staging'; FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
OUT = os.environ.get('BYPRODUCT_OUT', '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/figs/byproduct')
os.makedirs(OUT, exist_ok=True)
samples = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '1,2,3,4,5,6,7,8,9').split(',')]
HIT = os.environ.get('BYPRODUCT_HITSEED', '0') == '1'
if HIT:                                   # every module-level LARND_* knob the seeding reads, BEFORE fit_params is imported
    import glob as _g, pickle as _p
    _a, _i = os.environ.get('BYPRODUCT_ARM', 'STRFRZ50'), os.environ.get('BYPRODUCT_ITER', '10000')
    _ck = sorted(_g.glob(f'{ST}/sci_full_{_a}{samples[0]}/history_iter{_i}_*.pkl') + _g.glob(f'{FR}/sci_full_{_a}{samples[0]}/history_iter{_i}_*.pkl'))[0]
    for _k, _v in _p.load(open(_ck, 'rb'))['env_overrides'].items():
        if _v not in (None, ''): os.environ[_k] = str(_v)
    import logging; logging.disable(logging.INFO)
T = {k: [] for k in ('sample', 'L', 'd_fit', 'd_str', 'e1_fit', 'e2_fit', 'e2_str', 'frozen', 'sag2') + (('d_hit', 'e1_hit', 'e2_hit') if HIT else ())}
S = {k: [] for k in ('sample', 'dx', 'true', 'fit', 'init', 'trk')}
for s in samples:
    import glob
    ARM, IT = os.environ.get('BYPRODUCT_ARM', 'STRFRZ50'), os.environ.get('BYPRODUCT_ITER', '10000')   # CONTR / 14000: continued fits
    ck = sorted(glob.glob(f'{ST}/sci_full_{ARM}{s}/history_iter{IT}_*.pkl') + glob.glob(f'{FR}/sci_full_{ARM}{s}/history_iter{IT}_*.pkl'))[0]
    R = FitRebuild(ck); fp = R.fp; col = R.col; di = R.sfields.index('dEdx'); ntrk0 = len(T['L'])
    for i in range(len(R.ck['chain_cache'])):
        b = R.batch(i, check=False); tp = R.ck['true_positions'][i]
        pid_all = np.asarray(R.ck['batch_parent_ids'][i]); ld = np.asarray(R.ck['dedx_cache'][i]['log_dedx'])
        tm = np.asarray(R.ck['true_dedx_matched_cache'][i]); sdx = np.asarray(R.ck['seg_dx_cache'][i])
        if HIT:                                                    # hit-derived starting trajectory of this batch
            import types, optax, jax.numpy as jnp
            f = R.target(i)
            cs = [types.SimpleNamespace(**c) for c in R.ck['chain_contexts'][i]]
            for c in cs: c.track_id = tuple(float(x) for x in c.track_id)
            tpp = {(tuple(float(x) for x in k.strip('()').split(',')) if isinstance(k, str) else tuple(map(float, k))): v for k, v in tp.items()}
            stub = types.SimpleNamespace(_batch_chain_contexts={i: cs}, _chain_optimizer=optax.adam(1e-3), _batch_true_positions={i: tpp},
                                         _chain_cache={i: [{'angles': np.zeros(len(np.asarray(a['angles'])), np.float32)} for a in R.ck['chain_cache'][i]]})
            fp.GradientDescentFitter._seed_drift_from_hits(stub, i, f['pixel_x'], f['pixel_y'], f['pixel_z'], f['event'], f['adcs'], f['pixel_id'])
            seed = {c.track_id: np.asarray(a['angles']) for c, a in zip(cs, stub._chain_cache[i])}
            meta = fp._build_chain_meta_spline(b['ctxs'], R.knot_cm)
            seeded = np.asarray(fp._chain_warp_spline(jnp.asarray(b['guess']), [jnp.asarray(seed[tuple(map(float, c.track_id))]) for c in b['ctxs']], meta, col)[0])
        for c in b['ctxs']:
            tid = tuple(map(float, c.track_id)); tpos = tp.get(tid, tp.get(str(tid)))
            if tpos is None: continue
            tpos = np.asarray(tpos); idx = np.asarray(c.idxs)
            xyz = lambda arr: arr[idx][:, [col['x'], col['y'], col['z']]]
            u0 = fp._dir_from_angles(c.theta0_i, c.phi0_i); e1, e2 = fp._transverse_frame(u0)
            nom = c.x0_fixed[None, :] + (c.alpha_mid[:, None] * c.total_len) * u0[None, :]
            df, dsr = xyz(b['fitted']) - tpos, xyz(b['guess']) - tpos
            T['sample'].append(s); T['L'].append(c.total_len)
            T['d_fit'].append(1e4 * np.linalg.norm(df, axis=1).mean()); T['d_str'].append(1e4 * np.linalg.norm(dsr, axis=1).mean())
            T['e1_fit'].append(1e4 * np.sqrt(np.mean((df @ e1) ** 2))); T['e2_fit'].append(1e4 * np.sqrt(np.mean((df @ e2) ** 2)))
            T['e2_str'].append(1e4 * np.sqrt(np.mean((dsr @ e2) ** 2)))
            if HIT:
                dh = xyz(seeded) - tpos
                T['d_hit'].append(1e4 * np.linalg.norm(dh, axis=1).mean())
                T['e1_hit'].append(1e4 * np.sqrt(np.mean((dh @ e1) ** 2))); T['e2_hit'].append(1e4 * np.sqrt(np.mean((dh @ e2) ** 2)))
            cf = b['coeffs'].get(tid); T['frozen'].append(bool(cf is not None and np.abs(cf[len(cf) // 2:]).max() == 0.0))
            T['sag2'].append(1e4 * np.abs((tpos - nom) @ e2).max())
            # segments of this track (parent ids of its rows), with the initial value from the guess rows
            pids = np.unique(pid_all[idx][pid_all[idx] >= 0])
            for p in pids:
                if not np.isfinite(tm[p]) or sdx[p] <= 0: continue
                r = idx[pid_all[idx] == p]
                S['sample'].append(s); S['dx'].append(sdx[p]); S['true'].append(tm[p]); S['fit'].append(float(np.exp(ld[p])))
                S['init'].append(float(b['guess'][r[0], di])); S['trk'].append(len(T['L']) - 1)
    print(f'sample {s}: {len(T["L"]) - ntrk0} tracks, {sum(1 for x in S["sample"] if x == s)} segments', flush=True)
    np.savez(f'{OUT}/byproduct_arrays.npz', **{f'T_{k}': np.asarray(v) for k, v in T.items()}, **{f'S_{k}': np.asarray(v) for k, v in S.items()})
# ---------------- summaries ----------------
A = {k: np.asarray(v) for k, v in T.items()}; B = {k: np.asarray(v) for k, v in S.items()}
rs = lambda x: dict(mean=float(np.mean(x)), median=float(np.median(x)), p90=float(np.percentile(x, 90)))
summ = dict(n_tracks=int(len(A['L'])), n_segments=int(len(B['dx'])),
            d_fit_um=rs(A['d_fit']), d_str_um=rs(A['d_str']), e1_fit_um=rs(A['e1_fit']), e2_fit_um=rs(A['e2_fit']), e2_str_um=rs(A['e2_str']),
            frozen_frac=float(A['frozen'].mean()), **({'d_hit_um': rs(A['d_hit']), 'e2_hit_um': rs(A['e2_hit']), 'e1_hit_um': rs(A['e1_hit'])} if HIT else {}),
            length_weighted_d_fit_um=float(np.sum(A['d_fit'] * A['L']) / A['L'].sum()),
            length_weighted_d_str_um=float(np.sum(A['d_str'] * A['L']) / A['L'].sum()))
Cmean = 1.887
for name, est in (('fit', B['fit']), ('init_track_average', B['init']), ('constant_1.887', np.full_like(B['true'], Cmean))):
    r = est / B['true'] - 1; w = B['dx']
    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    summ[f'seg_{name}'] = dict(bias_pct=100 * float(np.average(r, weights=w)), rms_pct=100 * float(np.sqrt(np.average(r ** 2, weights=w))),
                               robust_sigma_pct=100 * float(mad), median_pct=100 * float(np.median(r)))
# track-averaged dE/dx (length-weighted over the track's segments)
tr = B['trk']; w = B['dx']; nt = len(A['L'])
num = lambda v: np.bincount(tr, weights=v * w, minlength=nt); den = np.bincount(tr, weights=w, minlength=nt); ok = den > 0
tt, tf, ti = num(B['true'])[ok] / den[ok], num(B['fit'])[ok] / den[ok], num(B['init'])[ok] / den[ok]
r = tf / tt - 1
summ['track_mean_fit'] = dict(bias_pct=100 * float(np.mean(r)), rms_pct=100 * float(np.sqrt(np.mean(r ** 2))), robust_sigma_pct=100 * float(1.4826 * np.median(np.abs(r - np.median(r)))))
r0 = ti / tt - 1; summ['track_mean_init'] = dict(bias_pct=100 * float(np.mean(r0)), rms_pct=100 * float(np.sqrt(np.mean(r0 ** 2))))
rc = Cmean / tt - 1; summ['track_mean_constant'] = dict(bias_pct=100 * float(np.mean(rc)), rms_pct=100 * float(np.sqrt(np.mean(rc ** 2))))
json.dump(summ, open(f'{OUT}/byproduct_summary.json', 'w'), indent=1)
print(json.dumps(summ, indent=1))
