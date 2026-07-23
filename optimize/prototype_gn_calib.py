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
    ev = np.clip(ev, 0.5 * m, None)
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

    def batch_nll(theta, tracks, tgt):
        phys = {p: init[p] * jnp.exp(theta[k]) for k, p in enumerate(relevant)}
        params = ref.replace(**phys)
        pred = strategy.predict(params, tracks, fields, None)
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
            res_fns[i] = jax.grad(lambda th, i=i: batch_nll(th, batch_tracks[i], targets[i]))   # gradient (P,)
            jac_fns[i] = jax.jacfwd(jax.grad(lambda th, i=i: batch_nll(th, batch_tracks[i], targets[i])))  # Hessian (P,P)
            res_fns[i+10000] = (lambda th, i=i: batch_nll(th, batch_tracks[i], targets[i]))     # scalar loss
        else:
            res_fns[i] = (lambda th, i=i: batch_residual(th, batch_tracks[i], targets[i]))
            jac_fns[i] = jax.jacfwd(lambda th, i=i: batch_residual(th, batch_tracks[i], targets[i]))

    P = len(relevant)
    theta = jnp.zeros(P)
    Jc = {}
    hist = []
    acc = np.zeros(P); n_acc = 0

    for s in range(args.steps):
        if s % args.refresh == 0:
            for i in range(nb):
                Jc[i] = np.asarray(jac_fns[i](theta))
            if s == 0 and args.fd_check and args.objective == 'residual':
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
        H = np.zeros((P, P)); g = np.zeros(P); loss = 0.0
        for i in range(nb):
            r = np.asarray(res_fns[i](theta))
            if args.objective == 'ppp':
                H += Jc[i]                       # cached per-batch Hessian (P,P)
                g += r                           # r IS the gradient here
                loss += float(res_fns[i+10000](theta))
            else:
                H += Jc[i].T @ Jc[i]
                g += Jc[i].T @ r
                loss += float(r @ r)
        Pinv = ridge_inverse(H, ridge=args.ridge, mu=args.mu)
        dth = -np.clip(Pinv @ g, -args.step_max, args.step_max)
        theta = theta + jnp.asarray(dth)
        cur = {p: init[p] * float(np.exp(theta[k])) for k, p in enumerate(relevant)}
        errs = {p: (cur[p] - truth[p]) / truth[p] * 100 for p in relevant}
        hist.append({'step': s, 'loss': loss, 'params': cur, 'errs': errs})
        print(f'[GN {s:3d}] loss {loss:.1f} | ' + ' '.join(f'{p}{errs[p]:+.1f}%' for p in relevant))
        if args.polyak and s >= args.steps - args.polyak:
            acc += np.array(theta); n_acc += 1

    theta_out = acc / n_acc if (args.polyak and n_acc) else np.array(theta)
    final = {p: init[p] * float(np.exp(theta_out[k])) for k, p in enumerate(relevant)}
    print('[GN-PROTO] FINAL:', ' '.join(f'{p} {final[p]:.5g} ({(final[p]-truth[p])/truth[p]*100:+.2f}%)' for p in relevant))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pickle.dump({'relevant': relevant, 'truth': truth, 'init': init, 'final': final,
                 'history': hist, 'args': vars(args)}, open(args.out, 'wb'))
    print(f'[GN-PROTO] saved {args.out}')


if __name__ == '__main__':
    main()
