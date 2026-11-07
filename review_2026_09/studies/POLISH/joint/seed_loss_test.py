"""GPU, offline: can the DATA LOSS tell a good e1 hit seed from a harmful one? (rule in REVIEW.md 22 Oct)
Replays the fit's hit seeding (KN10 env) on KN10S1's kept targets, then per e1-seeded track computes
dL = L(all seeds) - L(all seeds, this track's e1 seed removed) at the calibration the fit starts from (history
iteration 0) and the guess dE/dx, i.e. the information available at the first visit. Compared with the truth-based
class: harmful (seed error > straight error + 200 um) / helpful (< straight - 200 um)."""
import os, sys, json, time, inspect, textwrap
for k, v in dict(LARND_DRIFT_FROM_HITS='1', LARND_HITS_SEED_E1='1', LARND_FREEZE_STRAIGHT_CL='0.50', LARND_CHAIN_FREEZE_DRIFT='1',
                 LARND_CHAIN_FRAME='drift', LARND_SPLINE_KMIN='3').items(): os.environ[k] = v
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint'
sys.argv = ['joint_polish2.py', '--run', 'KN10S', '--sample', '1', '--iter', '10000', '--out', '/dev/null', '--state', '/dev/null']
exec(open(f'{D}/joint_polish2.py').read().split('\n_js = lambda')[0])
import types, logging, optax, numpy as np, jax.numpy as jnp
src = textwrap.dedent(inspect.getsource(fp.GradientDescentFitter._seed_drift_from_hits)); g = dict(fp.__dict__); exec(src, g); seed = g['_seed_drift_from_hits']
th_init = {k: float(np.ravel(ck[k + '_iter'][0])[0]) for k in PARAMS}; P_init = mk(th_init)
print('initial calibration error %', {k: round(100 * (v / pc.TRUTH[k] - 1), 2) for k, v in th_init.items()}, flush=True)


def tracks_for(bt, ctxs, coeffs):
    tw, _ = fp._chain_warp_spline(jnp.asarray(bt['guess']), [jnp.asarray(c) for c in coeffs], fp._build_chain_meta_spline(ctxs, R.knot_cm), R.col)
    tw = np.asarray(tw).copy(); tw[:, Ei] = tw[:, di] * tw[:, xi]; return jnp.asarray(tw)   # guess dE/dx (first visit)


out = []
for i in range(100):
    t0 = time.perf_counter(); bt = R.batch(i, check=False); ctxs = bt['ctxs']; tp = ck['true_positions'][i]
    tgr = R.target(i); tgt = {k: jnp.asarray(v) for k, v in tgr.items()}
    obj = types.SimpleNamespace(_batch_chain_contexts={i: ctxs}, _chain_optimizer=optax.adam(1e-2), _batch_true_positions={i: {}},
                                _chain_cache={i: [{'angles': jnp.zeros(2 * fp._spline_K(c.total_len, R.knot_cm), jnp.float32)} for c in ctxs]})
    logging.disable(logging.INFO); seed(obj, i, tgr['pixel_x'], tgr['pixel_y'], tgr['pixel_z'], tgr['event'], tgr['adcs'], tgr['pixel_id'], apply_seed=True)
    S = [np.asarray(st['angles'], np.float32) for st in obj._chain_cache[i]]
    L_all = float(loss.compute(P_init, sim.predict(P_init, tracks_for(bt, ctxs, S), fields, None), tgt)[0])
    for ti, c in enumerate(ctxs):
        K = fp._spline_K(c.total_len, R.knot_cm)
        if not np.any(S[ti][:K] != 0): continue
        tr = tp.get(tuple(map(float, c.track_id)), tp.get(str(tuple(map(float, c.track_id)))))
        if tr is None: continue
        S2 = [s.copy() for s in S]; S2[ti][:K] = 0.0
        L_rm = float(loss.compute(P_init, sim.predict(P_init, tracks_for(bt, ctxs, S2), fields, None), tgt)[0])
        u0 = fp._dir_from_angles(c.theta0_i, c.phi0_i); e1, _ = fp._transverse_frame(u0)
        Lt = float(c.total_len); al = np.asarray(c.alpha_mid, float); tr = np.asarray(tr, float); m = min(len(al), len(tr)); al, tr = al[:m], tr[:m]
        d = tr - (np.asarray(c.x0_fixed)[None, :] + al[:, None] * Lt * u0[None, :]); B = fp._spline_basis(al, K)
        t1 = np.linalg.lstsq(B, d @ e1, rcond=None)[0]; rms = lambda b: 1e4 * float(np.sqrt(np.mean((B @ b) ** 2)))
        out.append(dict(batch=i, ti=ti, dL=L_all - L_rm, err_seed=rms(S[ti][:K] - t1), err_straight=rms(t1)))
    print(f'batch {i}: {sum(1 for x in out if x["batch"] == i)} seeded tracks ({time.perf_counter() - t0:.0f} s)', flush=True)
    json.dump(out, open(f'{D}/seed_loss_KN10S1.json', 'w'), indent=1)
a = {k: np.array([x[k] for x in out], float) for k in ('dL', 'err_seed', 'err_straight')}
harm = a['err_seed'] > a['err_straight'] + 200; helpf = a['err_seed'] < a['err_straight'] - 200; acc = a['dL'] < 0
print(f'{len(out)} e1-seeded tracks: harmful {harm.sum()}, helpful {helpf.sum()}; loss accepts {acc.sum()}')
print(f'  loss-accepted: harmful {np.sum(acc & harm)}/{harm.sum()}, helpful {np.sum(acc & helpf)}/{helpf.sum()}')
e_cur = a['err_seed']; e_new = np.where(acc, a['err_seed'], a['err_straight'])
print(f'  start e1 error on these tracks: current mean {e_cur.mean():.0f} (> 1 mm: {np.sum(e_cur > 1000)}), loss-checked {e_new.mean():.0f} '
      f'(> 1 mm: {np.sum(e_new > 1000)}), no seed {a["err_straight"].mean():.0f} (> 1 mm: {np.sum(a["err_straight"] > 1000)})')
