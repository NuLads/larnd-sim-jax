"""LUCiD-style Gauss-Newton prototype for CALIBRATION ONLY (truth dEdx + truth positions).

Recipe adapted from CIDeR-ML/LUCiD lucid/fitting/gauss_newton.py to our deterministic
probabilistic forward:
  - residuals per predicted pixel: charge  rQ = sqrt(Qpred+eps) - sqrt(Qobs+eps)
                                   time    rT = w_time * (Tpred - Tobs)   (lit pixels only)
    (LUCiD fit_charge_time structure: pixel <-> PMT, first-hit time <-> arrival time)
  - theta = log-space global params (positivity for free), step clipped (trust region)
  - FULL-DATASET accumulation: H = sum_i Ji^T W Ji, g = sum_i Ji^T W ri over ALL batches
    before each solve (never per-batch solves)
  - damping: ridge*diag(H) + mu*median(diag)*I + eigen floor  (LUCiD ridge_inverse)
  - J by forward-mode jacfwd (5 tangents), refreshed every `refresh` steps
  - NO CRN / noise-batch averaging: our probabilistic forward is deterministic
  - built-in FD-vs-AD Jacobian cross-check at step 0

Setup reuse: point --history at the checkpoint of a standard (1-iteration is enough)
ceiling run (sim = tgt = true segments, no chain, no dEdx). The prototype rebuilds the
identical dataset/params/response from the pickled config, loads the SAME target hit
lists (target_*/batch{i}_target.npz), and reads the true target parameter values.
"""
import os, sys, argparse, pickle, glob
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optimize.dataio import TracksDataset
from optimize.strategies import LUTProbabilisticSimulation
from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut


def ridge_inverse(H, ridge=0.02, mu=0.3):
    """LUCiD damping: median-diagonal ridge + Levenberg term + positive eigen-floor."""
    n = H.shape[0]
    dg = np.clip(np.diag(H), 0, None)
    pos = dg[dg > 1e-30]
    base = np.median(pos) if pos.size else 1.0
    m = mu * base
    A = H + ridge * np.diag(dg) + m * np.eye(n) + 1e-12 * (np.abs(H).max() + 1e-30) * np.eye(n)
    ev, V = np.linalg.eigh(A)
    # SADDLE-FREE: |eigenvalues|. The true NLL Hessian is INDEFINITE far from the minimum;
    # clipping negative curvature to a tiny positive floor turns those directions into huge
    # steps that ASCEND the true function (the observed runaway). |ev| preserves the scale
    # of the curvature while forcing descent (Dauphin et al.; LUCiD avoids this by using
    # PSD JtJ, which we cannot for the NLL objective).
    ev = np.clip(np.abs(ev), 0.5 * m, None)
    return V @ np.diag(1.0 / ev) @ V.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--history', required=True, help='checkpoint pickle of a ceiling run (config + targets)')
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--refresh', type=int, default=5)
    ap.add_argument('--ridge', type=float, default=0.02)
    ap.add_argument('--mu', type=float, default=0.3)
    ap.add_argument('--step_max', type=float, default=0.08)
    ap.add_argument('--w_time', type=float, default=1.0)
    ap.add_argument('--eps', type=float, default=1e-8)
    ap.add_argument('--polyak', type=int, default=0)
    ap.add_argument('--fd_check', action='store_true', default=True)
    ap.add_argument('--max_batches', type=int, default=0, help='use only the first N batches (0=all)')
    ap.add_argument('--warm_start', type=float, default=0.0,
                    help='initialize theta from the history run at this fraction of its iterations (0=off; e.g. 0.2 = Adam@1000/5000). Tests GN-as-polisher.')
    ap.add_argument('--objective', choices=['residual', 'ppp'], default='residual',
                    help='residual: LUCiD-style sqrt-charge+time LSQ (JtJ). ppp: full-batch damped NEWTON on the exact PPP/llhd loss (5x5 Hessian via jacfwd(grad)) — same objective as the Adam fits.')
    ap.add_argument('--out', default='fit_result/gn_proto/result.pkl')
    args = ap.parse_args()

    h = pickle.load(open(args.history, 'rb'))
    cfg = h['config']
    relevant = list(h['norm_params_state'].keys()) if 'norm_params_state' in h else None
    if not relevant:
        raise SystemExit('history has no norm_params_state')
    print(f'[GN-PROTO] params: {relevant}')

    # ── rebuild params exactly as ParamFitter does ──
    Params = build_params_class(relevant)
    ref = load_detector_properties(Params, cfg.detector_props, cfg.pixel_layouts)
    ref = ref.replace(electron_sampling_resolution=cfg.electron_sampling_resolution,
                      number_pix_neighbors=cfg.number_pix_neighbors,
                      signal_length=cfg.signal_length, time_window=cfg.signal_length)
    response, ref = load_lut(cfg.lut_file, ref)
    ref = ref.replace(diffusion_in_current_sim=getattr(cfg, 'diffusion_in_current_sim', False),
                      mc_diff=getattr(cfg, 'mc_diff', False),
                      use_dedx_density=getattr(cfg, 'use_dedx_density', False),
                      dedx_density_mode=getattr(cfg, 'dedx_density_mode', 'histogram'))
    strategy = LUTProbabilisticSimulation(response)

    # replicate the fitter's noise handling: --no-noise removes readout noise from BOTH
    # the guess and the target params (missing this made the prototype fit noisy targets
    # with a noise-smeared guess — a different problem than the Adam reference)
    from optimize.fit_params import remove_noise_from_params
    _no_noise = bool(getattr(cfg, 'no_noise', False))
    # NOTE the fitter's exact convention under --no-noise: the LOSS path uses ref_params with
    # noise sigma INTACT (the probabilistic guess divides by sigma), while only the TARGET
    # params are stripped. Removing noise from the guess causes 1/sigma = 1/0.

    # truth + init values
    truth = {p: float(np.ravel(h[f'{p}_target'])[0]) for p in relevant}
    init = {p: float(getattr(ref, p)) for p in relevant}
    print('[GN-PROTO] truth :', {k: f'{v:.5g}' for k, v in truth.items()})
    print('[GN-PROTO] init  :', {k: f'{v:.5g}' for k, v in init.items()})

    # ── dataset (identical construction to example_run) ──
    ds = TracksDataset(filename=cfg.input_file_sim, nevents=cfg.data_sz, max_nbatch=cfg.max_nbatch,
                       random_nevents=cfg.random_nevents, data_seed=cfg.data_seed,
                       track_len_sel=cfg.track_len_sel, max_abs_costheta_sel=cfg.max_abs_costheta_sel,
                       min_abs_segz_sel=cfg.min_abs_segz_sel, track_z_bound=cfg.track_z_bound,
                       max_batch_len=cfg.max_batch_len, print_input=cfg.print_input,
                       chopped=(not cfg.no_chop), pad=(not cfg.no_pad),
                       electron_sampling_resolution=cfg.electron_sampling_resolution,
                       live_selection=cfg.live_selection,
                       use_dedx_density=getattr(cfg, 'use_dedx_density', False),
                       dedx_density_mode=getattr(cfg, 'dedx_density_mode', 'histogram'))
    fields = ds.get_track_fields()
    nb = len(ds)
    if args.max_batches > 0:
        nb = min(nb, args.max_batches)
    print(f'[GN-PROTO] using {nb} batches')

    # ── targets: regenerate EXACTLY as the fitter does (LUTSimulation at target params,
    # rngkey=i+1 — the identical stochastic data throw of any standard run w/ this config) ──
    from optimize.strategies import LUTSimulation
    target_params = ref.replace(**truth)
    if _no_noise or getattr(cfg, 'no_noise_target', False):
        target_params = remove_noise_from_params(target_params)
        print('[GN-PROTO] readout noise removed from target params')
    tgt_strategy = LUTSimulation(response)
    targets = {}
    for i in range(nb):
        tr = jnp.asarray(ds[i]).reshape(-1, len(fields))
        pr = tgt_strategy.predict(target_params, tr, fields, rngkey=i + 1)
        targets[i] = {'pixel_id': np.asarray(pr['hit_pixels']),
                      'adcs': np.asarray(pr['adcs']),
                      'ticks': np.asarray(pr['ticks'])}
        print(f'[GN-PROTO] target batch {i}: {len(targets[i]["adcs"])} hits')

    Ntick_valid_margin = 3  # loss convention: last ~3 ticks are the no-hit sentinel

    def predict_obs(theta, tracks):
        """theta (P,) log-space -> per-pixel (unique_pixels, Qpred, Tpred). Deterministic."""
        phys = {p: init[p] * jnp.exp(theta[k]) for k, p in enumerate(relevant)}
        params = ref.replace(**phys)
        pred = strategy.predict(params, tracks, fields, None)
        prob = jnp.exp(pred['hit_prob'])                      # (Npix, Nval, Ntick)
        Nt = prob.shape[-1]
        valid_t = jnp.arange(Nt) < (Nt - Ntick_valid_margin)
        prob = prob * valid_t[None, None, :]
        q = pred['adcs_distrib']                              # ADC units; consistent scale w/ target adcs
        Qpred = jnp.sum(prob * q, axis=(1, 2))                # expected total ADC-charge per pixel
        p0 = prob[:, 0, :]                                    # first-hit slot
        t_axis = jnp.arange(Nt)
        Tpred = jnp.sum(p0 * t_axis[None, :], axis=1) / (jnp.sum(p0, axis=1) + 1e-9)
        return pred['unique_pixels'], Qpred, Tpred

    def batch_residual(theta, tracks, tgt):
        up, Qpred, Tpred = predict_obs(theta, tracks)
        # scatter target hits onto the prediction's pixel axis
        tpix = jnp.asarray(tgt['pixel_id'])
        idx = jnp.searchsorted(up, tpix)
        idx = jnp.clip(idx, 0, up.shape[0] - 1)
        ok = (up[idx] == tpix)
        Qobs = jnp.zeros(up.shape[0]).at[idx].add(jnp.where(ok, jnp.asarray(tgt['adcs']), 0.0))
        first = jnp.full(up.shape[0], 1e9).at[idx].min(jnp.where(ok, jnp.asarray(tgt['ticks'], dtype=jnp.float32), 1e9))
        lit = first < 1e8
        pixvalid = up >= 0
        rQ = (jnp.sqrt(Qpred + args.eps) - jnp.sqrt(Qobs + args.eps)) * pixvalid
        rT = args.w_time * (Tpred - jnp.where(lit, first, 0.0)) * (lit & pixvalid)
        return jnp.concatenate([rQ, rT])

    # ── objective 'ppp': the exact PPP/llhd loss our Adam fits use ──
    from optimize.strategies import ProbabilisticLossStrategy
    loss_kw = dict(cfg.loss_fn_kw) if getattr(cfg, 'loss_fn_kw', None) else {}
    ppp_loss = ProbabilisticLossStrategy(**loss_kw)

    # static shapes per batch (unique_size = dynamic padded count at init params; ROI padded
    # to 128-multiples +128 headroom since params move during the fit) -> NLL fully traceable
    from larndsim.sim_jax import simulate_wfs as _sim_wfs, get_roi_counts as _roi_counts
    _static = {}

    def _batch_static(i, tracks):
        if i not in _static:
            wfs_e, _up = _sim_wfs(ref, response, tracks, fields)
            ns, nl = _roi_counts(ref, wfs_e)
            pad128 = lambda n: int(((int(n) + 127) // 128) * 128) + 128
            _static[i] = (int(wfs_e.shape[0]), pad128(ns), pad128(nl))
            print(f'[GN-PROTO] batch {i}: usize {_static[i][0]} roi {_static[i][1:]}')
        return _static[i]

    # BOUNDED parametrization — the fitter's own sigmoid map (theta=0 <-> nominal init).
    # Unbounded log-space let GN descend the tau->infinity plateau that Adam's sigmoid
    # walls make unreachable; same space => same reachable minima.
    from optimize.fit_params import map_norm_to_phys, map_phys_to_norm

    def batch_nll(theta, tracks, tgt, usize, roi):
        phys = {p: map_norm_to_phys(theta[k], p, scheme='sigmoid', scale=1.0) for k, p in enumerate(relevant)}
        params = ref.replace(**phys)
        pred = strategy.predict(params, tracks, fields, None, unique_size=usize, roi_override=roi)
        tgt_data = {'pixel_id': jnp.asarray(tgt['pixel_id']), 'ticks': jnp.asarray(tgt['ticks']),
                    'adcs': jnp.asarray(tgt['adcs'])}
        nll, _aux = ppp_loss.compute(params, pred, tgt_data)
        return nll

    res_fns = {}
    jac_fns = {}
    batch_tracks = {}
    for i in range(nb):
        batch_tracks[i] = jnp.asarray(ds[i]).reshape(-1, len(fields))

    if args.objective == 'ppp':
        # DATA-AS-ARGS: one jitted program shared by all batches (per-batch jitted closures bake
        # data as device constants -> nb x programs+constants -> OOM). Pad to global max shapes;
        # padded target hits use pixel_id=-1 (masked as irrelevant by the loss); padded track
        # rows are all-zero (no charge).
        for i in range(nb):
            _batch_static(i, batch_tracks[i])
        us_g = max(v[0] for v in _static.values())
        rs_g = max(v[1] for v in _static.values())
        rl_g = max(v[2] for v in _static.values())
        Tmax = max(batch_tracks[i].shape[0] for i in range(nb))
        Hmax = max(len(targets[i]['adcs']) for i in range(nb))
        print(f'[GN-PROTO] data-as-args: usize {us_g} roi ({rs_g},{rl_g}) tracks {Tmax} hits {Hmax}')
        tr_pad = {}; tg_pad = {}
        for i in range(nb):
            t = np.asarray(batch_tracks[i])
            tp = np.zeros((Tmax, t.shape[1]), t.dtype); tp[:t.shape[0]] = t
            tr_pad[i] = jnp.asarray(tp)
            npad = Hmax - len(targets[i]['adcs'])
            tg_pad[i] = (jnp.asarray(np.concatenate([targets[i]['pixel_id'], -np.ones(npad, targets[i]['pixel_id'].dtype)])),
                         jnp.asarray(np.concatenate([targets[i]['ticks'], np.zeros(npad, targets[i]['ticks'].dtype)])),
                         jnp.asarray(np.concatenate([targets[i]['adcs'], np.zeros(npad, targets[i]['adcs'].dtype)])))

        def _nll_da(th, tracks, pid, ticks, adcs):
            return batch_nll(th, tracks, {'pixel_id': pid, 'ticks': ticks, 'adcs': adcs},
                             us_g, (rs_g, rl_g))
        _g_da = jax.jit(jax.grad(_nll_da))
        _hvp_da = jax.jit(lambda th, v, tracks, pid, ticks, adcs:
                          jax.jvp(lambda t2: jax.grad(_nll_da)(t2, tracks, pid, ticks, adcs),
                                  (th,), (v,))[1])
        _l_da = jax.jit(_nll_da)
        for i in range(nb):
            res_fns[i] = (lambda th, i=i: _g_da(th, tr_pad[i], *tg_pad[i]))
            def _hess_seq(th, i=i):
                cols = [_hvp_da(th, jnp.eye(len(th))[k], tr_pad[i], *tg_pad[i]) for k in range(len(th))]
                return jnp.stack(cols, axis=1)
            jac_fns[i] = _hess_seq
            res_fns[i+10000] = (lambda th, i=i: _l_da(th, tr_pad[i], *tg_pad[i]))
    else:
        for i in range(nb):
            _r = (lambda th, i=i: batch_residual(th, batch_tracks[i], targets[i]))
            res_fns[i] = _r
            def _jac_seq(th, _r=_r):
                cols = [jax.jvp(_r, (th,), (jnp.eye(len(th))[k],))[1] for k in range(len(th))]
                return jnp.stack(cols, axis=1)
            jac_fns[i] = _jac_seq

    P = len(relevant)
    if args.warm_start > 0:
        ws = {p: float(np.ravel(h[f'{p}_iter'])[int(args.warm_start * (len(np.ravel(h[f'{p}_iter'])) - 1))]) for p in relevant}
        print('[GN-PROTO] warm start from Adam@%.0f%%:' % (100*args.warm_start),
              {k: f'{v:.4g}' for k, v in ws.items()})
        theta = jnp.asarray([float(map_phys_to_norm(ws[p], p, scheme='sigmoid', scale=1.0)) for p in relevant])
    else:
        theta = jnp.asarray([float(map_phys_to_norm(init[p], p, scheme='sigmoid', scale=1.0)) for p in relevant])
    Jc = {}
    hist = []
    acc = np.zeros(P); n_acc = 0

    for s in range(args.steps):
        if s % args.refresh == 0:
            for i in range(nb):
                Jc[i] = np.asarray(jac_fns[i](theta))
            if s == 0 and args.fd_check and args.objective == 'residual':  # NOTE: ppp mode checked via the separate 3-point objective test
                i0 = 0
                r0 = np.asarray(res_fns[i0](theta))
                for k in range(P):
                    ad = Jc[i0][:, k]
                    # central-difference sweep over h: h-stable FD != AD -> AD problem;
                    # h-unstable FD -> discreteness (tick/pixel boundaries) makes FD unreliable
                    line = []
                    for hfd in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
                        rp = np.asarray(res_fns[i0](theta.at[k].add(hfd)))
                        rm = np.asarray(res_fns[i0](theta.at[k].add(-hfd)))
                        fd = (rp - rm) / (2 * hfd)
                        num = float(ad @ fd); den = float(np.linalg.norm(ad) * np.linalg.norm(fd) + 1e-30)
                        line.append(f'h={hfd:g}: cos={num/den:+.3f} |fd|/|ad|={np.linalg.norm(fd)/(np.linalg.norm(ad)+1e-30):.2f}')
                    print(f'[FD-SWEEP] {relevant[k]:10s} ' + ' | '.join(line))
        H = np.zeros((P, P)); g = np.zeros(P); loss = 0.0; n_bad = 0
        for i in range(nb):
            r = np.asarray(res_fns[i](theta))
            if args.objective == 'ppp':
                Hi = np.asarray(Jc[i]); gi = r
                li = float(res_fns[i+10000](theta))
            else:
                Hi = Jc[i].T @ Jc[i]; gi = Jc[i].T @ r
                li = float(r @ r)
            if not (np.all(np.isfinite(Hi)) and np.all(np.isfinite(gi)) and np.isfinite(li)):
                n_bad += 1
                if s == 0:
                    print(f'[GN-WARN] batch {i} NON-FINITE: g finite={np.isfinite(gi).tolist()} '
                          f'Hdiag finite={np.isfinite(np.diag(Hi)).tolist()} loss={li}')
                continue
            H += Hi; g += gi; loss += li
        if n_bad:
            print(f'[GN-WARN] step {s}: skipped {n_bad}/{nb} non-finite batches')
        # Levenberg-Marquardt lambda adaptation with accept/reject on the ACCUMULATED loss:
        # flat directions get large raw Newton steps (eigen-floor), which descend a long shallow
        # valley right past the deep basin. LM rejects steps that don't reduce the true loss and
        # raises lambda until they do — the flat-direction step size becomes self-limiting.
        if not hasattr(main, '_lm_lambda'): main._lm_lambda = args.mu
        lam = main._lm_lambda
        def total_loss(th):
            return sum(float(res_fns[i+10000](th)) for i in range(nb)) if args.objective=='ppp'                    else sum(float(np.asarray(res_fns[i](th)) @ np.asarray(res_fns[i](th))) for i in range(nb))
        L0 = loss
        accepted = False
        for _try in range(6):
            Pinv = ridge_inverse(H, ridge=args.ridge, mu=lam)
            dth_raw = -(Pinv @ g)
            nrm = np.linalg.norm(dth_raw)
            dth = dth_raw * min(1.0, args.step_max*np.sqrt(P)/max(nrm,1e-30))   # NORM trust region (not elementwise)
            L1 = total_loss(theta + jnp.asarray(dth))
            if L1 < L0:
                accepted = True
                lam = max(lam/3.0, 1e-4)
                break
            lam *= 10.0
        main._lm_lambda = lam
        # BOUND-SATURATION GUARD (ported from GaussNewtonCalibFitter._run_gn_loop, fit_params.py):
        # the sigmoid bounds the PHYSICAL value but nothing stops the pre-sigmoid theta drifting to
        # |theta|>>1 via many individually-legitimate descent steps — the parameter then sits pinned
        # at its range bound while the loss keeps decreasing along other directions. THIS is the
        # observed 'runaway': a real descent into a corner, not an ascending/failed step.
        _sat = float(np.max(np.abs(np.asarray(theta + jnp.asarray(dth)))))
        if _sat > 3.5:
            print(f'[GN {s:3d}] norm-space saturation {_sat:.2f} > 3.5 — parameters pinning at bounds; stopping')
            break
        if not accepted:
            print(f'[GN {s:3d}] no acceptable step (lam={lam:.1e}) — stopping')
            break
        theta = theta + jnp.asarray(dth)
        cur = {p: float(map_norm_to_phys(theta[k], p, scheme='sigmoid', scale=1.0)) for k, p in enumerate(relevant)}
        errs = {p: (cur[p] - truth[p]) / truth[p] * 100 for p in relevant}
        hist.append({'step': s, 'loss': loss, 'params': cur, 'errs': errs})
        print(f'[GN {s:3d}] loss {loss:.1f} | ' + ' '.join(f'{p}{errs[p]:+.1f}%' for p in relevant))
        if args.polyak and s >= args.steps - args.polyak:
            acc += np.array(theta); n_acc += 1

    theta_out = acc / n_acc if (args.polyak and n_acc) else np.array(theta)
    final = {p: float(map_norm_to_phys(jnp.asarray(theta_out)[k], p, scheme='sigmoid', scale=1.0)) for k, p in enumerate(relevant)}
    print('[GN-PROTO] FINAL:', ' '.join(f'{p} {final[p]:.5g} ({(final[p]-truth[p])/truth[p]*100:+.2f}%)' for p in relevant))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pickle.dump({'relevant': relevant, 'truth': truth, 'init': init, 'final': final,
                 'history': hist, 'args': vars(args)}, open(args.out, 'wb'))
    print(f'[GN-PROTO] saved {args.out}')


if __name__ == '__main__':
    main()
