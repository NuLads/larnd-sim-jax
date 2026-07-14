#!/usr/bin/env python3
"""Demonstration plots for the chain-position fit: angle (segment) vs spline basis.

(1) Convergence: pos-residual vs iteration, both bases, at 50 cm and 400 cm.
(2) Track reconstruction: for a few example (most-curved) tracks, the transverse
    displacement profile vs arc length — true / linear-guess / angle-fit / spline-fit —
    which shows visually how each parametrization represents the track shape.

Run in the container:
  JAX_PLATFORMS=cpu apptainer exec -B /sdf,/fs,/lscratch larndsim-jax_main.sif \
    bash -c 'cd REPO && PYTHONPATH=$PWD/src:$PWD python3 plot_chain_basis.py'
"""
import pickle, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('plots', exist_ok=True)


def latest(pat):
    g = glob.glob(pat)
    if not g:
        return None
    return max(g, key=lambda x: int(x.split('history_iter')[1].split('_')[0]))


def load(pat):
    p = latest(pat)
    if p is None:
        return None, None
    with open(p, 'rb') as f:
        return pickle.load(f), int(p.split('history_iter')[1].split('_')[0])


def dir_from_angles(theta, phi):
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])


def transverse_frame(u0):
    ref = np.array([0., 0., 1.]) if abs(u0[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(u0, ref); e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(u0, e1);  e2 /= np.linalg.norm(e2) + 1e-12
    return e1, e2


def reconstruct(ctx, params, basis):
    """Return fitted node positions (nc+1, 3) for a track."""
    nc = int(ctx['n_chain']); step = float(ctx['step_len'])
    x0 = np.asarray(ctx['x0_fixed'], float)
    u0 = dir_from_angles(ctx['theta0_i'], ctx['phi0_i'])
    ks = np.arange(nc + 1)
    nominal = x0[None, :] + (ks[:, None] * step) * u0[None, :]
    p = np.asarray(params, float)
    if basis == 'spline':
        K = p.shape[0] // 2
        B = np.sin(np.outer(ks / nc, np.pi * np.arange(1, K + 1)))
        e1, e2 = transverse_frame(u0)
        d1 = B @ p[:K]; d2 = B @ p[K:2 * K]
        return nominal + d1[:, None] * e1[None, :] + d2[:, None] * e2[None, :]
    # angle basis: cumulative walk
    th, ph = p[:nc], p[nc:]
    st, ct = np.sin(th), np.cos(th)
    u = np.stack([st * np.cos(ph), st * np.sin(ph), ct], 1)
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    nodes = np.concatenate([x0[None, :], x0[None, :] + np.cumsum(step * u, 0)], 0)
    return nodes


def interp_at(nodes, alpha, nc):
    k = np.clip(np.floor(alpha * nc).astype(int), 0, nc - 1)
    frac = alpha * nc - k
    return nodes[k] + frac[:, None] * (nodes[k + 1] - nodes[k])


def track_profiles(ctx, params, basis, true_pts):
    """Transverse offset (projected on principal deviation axis) vs arc length, for
    the fitted path and the true path, plus per-point residual."""
    nc = int(ctx['n_chain'])
    x0 = np.asarray(ctx['x0_fixed'], float)
    u0 = dir_from_angles(ctx['theta0_i'], ctx['phi0_i'])
    e1, e2 = transverse_frame(u0)
    alpha = np.asarray(ctx['alpha_mid'], float)
    s = alpha * float(ctx['total_len'])
    nominal_pts = x0[None, :] + alpha[:, None] * float(ctx['total_len']) * u0[None, :]
    fit_pts = interp_at(reconstruct(ctx, params, basis), alpha, nc)
    tr = np.asarray(true_pts, float)
    m = min(len(fit_pts), len(tr), len(s))
    s, fit_pts, tr, nominal_pts = s[:m], fit_pts[:m], tr[:m], nominal_pts[:m]

    def perp(pts):
        d = pts - nominal_pts
        return np.stack([d @ e1, d @ e2], 1)   # (m, 2)

    tr_perp = perp(tr)
    # principal axis of the TRUE deviation, so 1D projection captures the bow
    if np.allclose(tr_perp, 0):
        w = np.array([1., 0.])
    else:
        _, _, V = np.linalg.svd(tr_perp - tr_perp.mean(0), full_matrices=False)
        w = V[0]
    proj = lambda P: P @ w
    resid = np.linalg.norm(fit_pts - tr, axis=1)
    return s, proj(tr_perp), proj(perp(fit_pts)), resid


def main():
    runs = {
        ('angle', 50):  'fit_result/pos_basis/history_iter*_posb_angle_len50_clr1e-4_mcs0.5_*.pkl',
        ('spline', 50): 'fit_result/pos_basis/history_iter*_posb_spline_len50_clr3e-3_mcs0.5_*.pkl',
        ('angle', 400): 'fit_result/pos_basis/history_iter*_posb_angle_len400_clr1e-4_mcs0.5_*.pkl',
        ('spline', 400):'fit_result/pos_basis/history_iter*_posb_spline_len400_clr3e-3_mcs0.5_*.pkl',
    }
    data = {k: load(v) for k, v in runs.items()}

    # ---- (1) convergence ----
    # pos_residual is logged per-BATCH (one batch/iteration), so the raw trace oscillates
    # batch-to-batch; smooth over ~2 epochs to show the convergence envelope.
    def smooth(y, w):
        w = max(3, int(w) | 1)
        k = np.ones(w) / w
        pad = np.r_[np.full(w // 2, y[0]), y, np.full(w // 2, y[-1])]
        return np.convolve(pad, k, 'valid')[:len(y)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, L in zip(axes, [50, 400]):
        for basis, c in [('angle', 'C0'), ('spline', 'C1')]:
            h, it = data[(basis, L)]
            if h is None:
                continue
            pr = np.asarray(h.get('pos_residual_iter', []), float)
            if not len(pr):
                continue
            x = np.linspace(0, it, len(pr))
            sm = smooth(pr, len(pr) // 60)
            ax.plot(x, pr * 1e4, color=c, alpha=0.15, lw=0.6)
            ax.plot(x, sm * 1e4, color=c, lw=2.0,
                    label=f'{basis} (final {np.mean(pr[-40:])*1e4:.0f} µm, iter {it})')
        ax.set_yscale('log'); ax.set_xlabel('iteration'); ax.set_ylabel('mean position residual [µm]')
        ax.set_title(f'{L} cm tracks'); ax.grid(alpha=0.3, which='both'); ax.legend()
    fig.suptitle('Chain position fit convergence: segment (angle) vs spline basis')
    fig.tight_layout(); fig.savefig('plots/chain_convergence.png', dpi=130)
    print('[saved] plots/chain_convergence.png')

    # ---- (2) reconstruction for a few example tracks at each length ----
    for L in [50, 400]:
        ha, _ = data[('angle', L)]; hs, _ = data[('spline', L)]
        if ha is None or hs is None:
            print(f'[skip reconstruction {L}cm: missing run]'); continue
        cca, ctxa, tpa = ha['chain_cache'], ha['chain_contexts'], ha['true_positions']
        ccs = hs['chain_cache']
        # rank tracks by true transverse curvature (most-curved = most illustrative)
        cand = []
        for b in sorted(ctxa.keys()):
            for ti, ctx in enumerate(ctxa[b]):
                tk = str(ctx['track_id'])
                if tk not in tpa.get(b, {}):
                    continue
                s, tr, _, _ = track_profiles(ctx, cca[b][ti]['angles'], 'angle', tpa[b][tk])
                cand.append((np.ptp(tr), b, ti, ctx['track_id']))
        cand.sort(reverse=True)
        picks = cand[:3]
        fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 4.2), squeeze=False)
        for ax, (_, b, ti, tk) in zip(axes[0], picks):
            ctx = ctxa[b][ti]; tks = str(tk)
            s, tr, fa, ra = track_profiles(ctx, cca[b][ti]['angles'], 'angle', tpa[b][tks])
            _, _, fs, rs = track_profiles(ctx, ccs[b][ti]['angles'], 'spline', tpa[b][tks])
            ax.plot(s, tr * 1e4, 'k-', lw=2.5, label='true')
            ax.axhline(0, color='0.6', ls='--', lw=1, label='linear guess (nominal)')
            ax.plot(s, fa * 1e4, 'C0-', lw=1.5, label=f'angle fit ({ra.mean()*1e4:.0f} µm)')
            ax.plot(s, fs * 1e4, 'C1-', lw=1.5, label=f'spline fit ({rs.mean()*1e4:.0f} µm)')
            ax.set_xlabel('arc length [cm]'); ax.set_ylabel('transverse offset [µm]')
            ax.set_title(f'track {tk}  (L={ctx["total_len"]:.0f} cm)')
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle(f'Track-shape reconstruction, {L} cm — transverse displacement vs arc length')
        fig.tight_layout(); out = f'plots/chain_reconstruction_{L}cm.png'
        fig.savefig(out, dpi=130); print(f'[saved] {out}')


if __name__ == '__main__':
    main()
