"""Rebuild what a finished joint fit predicts for a batch, from its checkpoint alone.

The checkpoint stores the fitted calibration values, the per-track spline coefficients (chain_cache, keyed through
chain_contexts by track id), the per-segment log dE/dx (dedx_cache) and the row -> segment map (batch_parent_ids).
The fit's chain contexts are a pure function of the batch's guess tracks (entry/exit points, projections onto that
axis, 2 cm chain step), so they are rebuilt here with the same code and checked against the saved fields. The
fitted batch is then exactly what loss_wrapper_combined feeds the simulation: guess tracks -> _chain_warp_spline with
the fitted coefficients -> dE/dx replaced by exp(log_dedx[parent]) and dE = dE/dx * dx.
Env knobs that change the geometry (frame, K floor) are taken from the checkpoint's env_overrides BEFORE import."""
import os, sys, pickle, numpy as np
PJ = '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS'
if PJ not in sys.path: sys.path.insert(0, PJ)


def load_ckpt(path):
    ck = pickle.load(open(path, 'rb'))
    for k in ('LARND_CHAIN_FRAME', 'LARND_SPLINE_KMIN'):        # read by fit_params at import time
        v = ck.get('env_overrides', {}).get(k)
        if v not in (None, ''): os.environ[k] = str(v)
    return ck


class FitRebuild:
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.ck = ck = load_ckpt(ckpt_path)
        import paper_common as pc                                  # chdir(REPO), detector config, data selection
        import optimize.fit_params as fp
        self.pc, self.fp = pc, fp
        c = ck['config']
        self.seed = int(c.data_seed)
        assert c.input_file_sim == pc.INPUT_SIM and c.input_file_tgt == pc.INPUT_TGT, 'input files differ from paper_common'
        self.ds_sim, self.ds_tgt, self.tfields = pc.load(float(c.max_batch_len), int(c.max_nbatch), self.seed)
        self.sfields = self.ds_sim.get_track_fields()
        self.step_len, self.knot_cm, self.mom = float(c.chain_step_len), float(c.chain_spline_knot_cm), float(c.chain_momentum_GeV)
        f = self.sfields
        self.col = {'x': f.index('x'), 'y': f.index('y'), 'z': f.index('z'), 'xs': f.index('x_start'), 'ys': f.index('y_start'),
                    'zs': f.index('z_start'), 'xe': f.index('x_end'), 'ye': f.index('y_end'), 'ze': f.index('z_end'), 'dx': f.index('dx')}
        self.truth_params, self.response = pc.params()
        self.calib_fit = {p: float(np.ravel(ck[p + '_iter'])[-1000:].mean()) for p in pc.PARAMS}   # endpoint estimator
        self.calib_init = {p: float(np.ravel(ck[p + '_init'])[0]) for p in pc.PARAMS}
        self.calib_true = {p: float(np.ravel(ck[p + '_target'])[0]) for p in pc.PARAMS}

    # --- the fit's chain contexts, rebuilt (same code as FitParams._build_chain_contexts_for_batch, no endpoint smear)
    def contexts(self, tracks_np):
        fp, c = self.fp, self.col
        ev_idx, tr_idx = self.sfields.index('eventID'), self.sfields.index('trackID')
        evs, trs = tracks_np[:, ev_idx], tracks_np[:, tr_idx]
        seen = {}
        for k in zip(evs.tolist(), trs.tolist()): seen.setdefault(k, None)
        ctxs = []
        for (ev, tr) in seen:
            idxs = np.where((evs == ev) & (trs == tr))[0]
            if len(idxs) < 2: continue
            mids = tracks_np[idxs][:, [c['x'], c['y'], c['z']]]; starts = tracks_np[idxs][:, [c['xs'], c['ys'], c['zs']]]
            ends = tracks_np[idxs][:, [c['xe'], c['ye'], c['ze']]]
            entry, exit_ = starts[0].astype(np.float32), ends[-1].astype(np.float32)
            d = exit_ - entry; L = float(np.linalg.norm(d))
            if L < 1e-4: continue
            n_chain = max(3, int(L / self.step_len)); step = L / n_chain; u = d / L
            theta0 = float(np.arccos(np.clip(float(u[2]), -1., 1.))); phi0 = float(np.arctan2(float(u[1]), float(u[0])))
            proj = lambda p: np.clip(np.dot(p - entry, u) / L, 0., 1.).astype(np.float32)
            ctxs.append(fp._ChainTrackCtx(track_id=(ev, tr), idxs=idxs.astype(np.int32), n_fine=len(idxs), n_chain=n_chain,
                                          step_len=step, total_len=L, x0_fixed=entry, theta0_i=theta0, phi0_i=phi0,
                                          sigma_mcs=fp._highland_sigma_mcs(step, self.mom), alpha_mid=proj(mids),
                                          alpha_start=proj(starts), alpha_end=proj(ends)))
        return ctxs

    def batch(self, i, check=True):
        """Return dict: guess (straight start), fitted, true track arrays for batch i, plus contexts and checks."""
        import jax.numpy as jnp
        fp, ck = self.fp, self.ck
        guess = np.asarray(self.ds_sim[i]); true = np.asarray(self.ds_tgt[i])
        ctxs = self.contexts(guess)
        saved = {tuple(s['track_id']): s for s in ck['chain_contexts'][i]}
        coef = {tuple(s['track_id']): np.asarray(a['angles']) for s, a in zip(ck['chain_contexts'][i], ck['chain_cache'][i])}
        chk = {}
        if check:
            assert len(ck['batch_parent_ids'][i]) == guess.shape[0], (len(ck['batch_parent_ids'][i]), guess.shape)
            assert set(saved) == {tuple(c.track_id) for c in ctxs}, 'track sets differ'
            dm = max(float(np.abs(np.asarray(saved[tuple(c.track_id)]['alpha_mid']) - c.alpha_mid).max()) for c in ctxs)
            dn = max(abs(int(saved[tuple(c.track_id)]['n_chain']) - c.n_chain) for c in ctxs)
            dl = max(abs(float(saved[tuple(c.track_id)]['total_len']) - c.total_len) for c in ctxs)
            chk.update(alpha_mid_maxdiff=dm, n_chain_maxdiff=dn, total_len_maxdiff=dl)
        meta = fp._build_chain_meta_spline(ctxs, self.knot_cm)
        coeffs = [jnp.asarray(coef[tuple(c.track_id)]) for c in ctxs]
        fitted, _ = fp._chain_warp_spline(jnp.asarray(guess), coeffs, meta, self.col)
        fitted = np.asarray(fitted).copy()
        # fitted dE/dx (per segment, broadcast to rows); padding rows (parent -1) keep the input value
        pid = np.asarray(ck['batch_parent_ids'][i]); ld = np.asarray(ck['dedx_cache'][i]['log_dedx'])
        di, Ei, xi = self.sfields.index('dEdx'), self.sfields.index('dE'), self.col['dx']
        ded = np.where(pid >= 0, np.maximum(np.exp(ld[np.clip(pid, 0, len(ld) - 1)]), 1e-3), fitted[:, di])
        fitted[:, di] = ded; fitted[:, Ei] = ded * fitted[:, xi]
        if check:                                                   # position residual vs the stored true positions
            tp = ck['true_positions'][i]; r_fit, r_str = [], []
            for c in ctxs:
                key = tp.get(tuple(map(float, c.track_id)), tp.get(str(tuple(map(float, c.track_id)))))
                if key is None: continue
                tpos = np.asarray(key); idx = np.asarray(c.idxs)
                pm = lambda arr: arr[idx][:, [self.col['x'], self.col['y'], self.col['z']]]
                r_fit.append(np.linalg.norm(pm(fitted) - tpos, axis=1)); r_str.append(np.linalg.norm(pm(guess) - tpos, axis=1))
            chk.update(pos_mean_fit_um=1e4 * float(np.concatenate(r_fit).mean()), pos_mean_straight_um=1e4 * float(np.concatenate(r_str).mean()))
        return dict(guess=guess, fitted=fitted, true=true, ctxs=ctxs, coeffs=coef, checks=chk)

    def target(self, i):
        """The fit's own stochastic target for batch i (kept with LARND_KEEP_TARGETS=1)."""
        run = os.path.basename(os.path.dirname(self.ckpt_path))
        label = os.path.basename(self.ckpt_path).split('_', 2)[2].rsplit('.pkl', 1)[0]      # history_iterN_<label>.pkl
        d = os.path.join(self.pc.REPO, f"target_{run}_{label}_" + "_".join(self.pc.PARAMS))
        z = np.load(os.path.join(d, f'batch{i}_target.npz'))
        return {k: np.asarray(z[k]) for k in z.files}
