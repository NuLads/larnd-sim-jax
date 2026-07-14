#!/usr/bin/env python3
"""Per-track debugging plots for the chain-position fit (angle vs spline).

Produces, per track length:
  A) stats: per-track residual histogram; residual vs track length; residual vs true
     bow amplitude; fitted-vs-true peak displacement (under/over-shoot).
  B) gallery: transverse-offset profiles for a grid of tracks spanning the curvature range.
  C) residual vs arc-length (median + IQR band over tracks) — where along the track it fails.
  D) spline mode spectrum: mean |beta_m| per sine mode (which modes are actually used).
"""
import pickle, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('plots', exist_ok=True)
RUNS = {
    ('angle', 50):  'fit_result/pos_basis/history_iter*_posb_angle_len50_clr1e-4_mcs0.5_*.pkl',
    ('spline', 50): 'fit_result/pos_basis/history_iter*_posb_spline_len50_clr3e-3_mcs0.5_*.pkl',
    ('angle', 400): 'fit_result/pos_basis/history_iter*_posb_angle_len400_clr1e-4_mcs0.5_*.pkl',
    ('spline', 400):'fit_result/pos_basis/history_iter*_posb_spline_len400_clr3e-3_mcs0.5_*.pkl',
}


def load(pat):
    g = glob.glob(pat)
    if not g:
        return None
    p = max(g, key=lambda x: int(x.split('history_iter')[1].split('_')[0]))
    with open(p, 'rb') as f:
        return pickle.load(f)


def dir_from_angles(t, p):
    st = np.sin(t)
    return np.array([st*np.cos(p), st*np.sin(p), np.cos(t)])


def frame(u0):
    ref = np.array([0., 0., 1.]) if abs(u0[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(u0, ref); e1 /= np.linalg.norm(e1)+1e-12
    e2 = np.cross(u0, e1);  e2 /= np.linalg.norm(e2)+1e-12
    return e1, e2


def recon_nodes(ctx, params, basis):
    nc = int(ctx['n_chain']); step = float(ctx['step_len'])
    x0 = np.asarray(ctx['x0_fixed'], float); u0 = dir_from_angles(ctx['theta0_i'], ctx['phi0_i'])
    ks = np.arange(nc+1); nominal = x0[None,:] + (ks[:,None]*step)*u0[None,:]
    p = np.asarray(params, float)
    if basis == 'spline':
        K = p.shape[0]//2
        B = np.sin(np.outer(ks/nc, np.pi*np.arange(1, K+1)))
        e1, e2 = frame(u0)
        return nominal + (B@p[:K])[:,None]*e1[None,:] + (B@p[K:2*K])[:,None]*e2[None,:]
    th, ph = p[:nc], p[nc:]
    st, ct = np.sin(th), np.cos(th)
    u = np.stack([st*np.cos(ph), st*np.sin(ph), ct], 1); u /= np.linalg.norm(u,axis=1,keepdims=True)+1e-12
    return np.concatenate([x0[None,:], x0[None,:]+np.cumsum(step*u,0)], 0)


def per_track(h, basis):
    cc, ctx, tp = h['chain_cache'], h['chain_contexts'], h['true_positions']
    rows = []
    for b in sorted(ctx.keys()):
        for ti, c in enumerate(ctx[b]):
            tk = str(c['track_id'])
            if tk not in tp.get(b, {}):
                continue
            nc = int(c['n_chain']); L = float(c['total_len'])
            x0 = np.asarray(c['x0_fixed'], float); u0 = dir_from_angles(c['theta0_i'], c['phi0_i'])
            e1, e2 = frame(u0)
            alpha = np.asarray(c['alpha_mid'], float); s = alpha*L
            nomp = x0[None,:] + alpha[:,None]*L*u0[None,:]
            nodes = recon_nodes(c, cc[b][ti]['angles'], basis)
            k = np.clip(np.floor(alpha*nc).astype(int), 0, nc-1); fr = alpha*nc-k
            fitp = nodes[k] + fr[:,None]*(nodes[k+1]-nodes[k])
            tr = np.asarray(tp[b][tk], float)
            m = min(len(fitp), len(tr), len(s))
            s, fitp, tr, nomp = s[:m], fitp[:m], tr[:m], nomp[:m]
            def perp(P):
                d = P-nomp; return np.stack([d@e1, d@e2], 1)
            trp, fp = perp(tr), perp(fitp)
            if np.allclose(trp, 0):
                w = np.array([1., 0.])
            else:
                _,_,V = np.linalg.svd(trp-trp.mean(0), full_matrices=False); w = V[0]
            resid = np.linalg.norm(fitp-tr, axis=1)
            rows.append(dict(L=L, s=s, tr=trp@w, fit=fp@w, resid=resid,
                             mean_resid=resid.mean(),
                             true_amp=np.ptp(trp@w), fit_amp=np.ptp(fp@w),
                             params=np.asarray(cc[b][ti]['angles'], float)))
    return rows


def figs(L):
    ha, hs = load(RUNS[('angle', L)]), load(RUNS[('spline', L)])
    if ha is None or hs is None:
        print(f'[skip {L}cm: missing]'); return
    A, S = per_track(ha, 'angle'), per_track(hs, 'spline')
    # match by index (same track order in both runs)
    n = min(len(A), len(S))
    A, S = A[:n], S[:n]

    # ---- A) stats ----
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
    ra = np.array([r['mean_resid'] for r in A])*1e4
    rs = np.array([r['mean_resid'] for r in S])*1e4
    bins = np.linspace(0, np.percentile(np.r_[ra, rs], 98), 40)
    ax[0].hist(ra, bins, alpha=0.6, label=f'angle (med {np.median(ra):.0f})', color='C0')
    ax[0].hist(rs, bins, alpha=0.6, label=f'spline (med {np.median(rs):.0f})', color='C1')
    ax[0].set_xlabel('per-track mean residual [µm]'); ax[0].set_ylabel('tracks'); ax[0].legend(); ax[0].set_title('residual distribution')
    Ls = np.array([r['L'] for r in A])
    ax[1].scatter(Ls, ra, s=12, alpha=0.5, color='C0', label='angle')
    ax[1].scatter(Ls, rs, s=12, alpha=0.5, color='C1', label='spline')
    ax[1].set_xlabel('track length [cm]'); ax[1].set_ylabel('mean residual [µm]'); ax[1].legend(); ax[1].set_title('residual vs length'); ax[1].set_yscale('log')
    amp = np.array([r['true_amp'] for r in A])*1e4
    ax[2].scatter(amp, ra, s=12, alpha=0.5, color='C0'); ax[2].scatter(amp, rs, s=12, alpha=0.5, color='C1')
    ax[2].set_xlabel('true bow amplitude [µm]'); ax[2].set_ylabel('mean residual [µm]'); ax[2].set_title('residual vs curvature'); ax[2].set_yscale('log')
    fa = np.array([r['fit_amp'] for r in A])*1e4; fs = np.array([r['fit_amp'] for r in S])*1e4
    lim = np.percentile(amp, 99)
    ax[3].plot([0, lim], [0, lim], 'k--', lw=1)
    ax[3].scatter(amp, fa, s=12, alpha=0.5, color='C0', label='angle')
    ax[3].scatter(amp, fs, s=12, alpha=0.5, color='C1', label='spline')
    ax[3].set_xlabel('true bow amplitude [µm]'); ax[3].set_ylabel('fitted bow amplitude [µm]')
    ax[3].set_xlim(0, lim); ax[3].set_ylim(0, lim*1.1); ax[3].legend(); ax[3].set_title('bow amplitude: fit vs true')
    fig.suptitle(f'Per-track fit diagnostics, {L} cm (n={n})'); fig.tight_layout()
    fig.savefig(f'plots/chain_debug_stats_{L}cm.png', dpi=125); print(f'[saved] plots/chain_debug_stats_{L}cm.png')

    # ---- B) gallery (12 tracks spanning curvature) ----
    order = np.argsort([r['true_amp'] for r in A])
    pick = order[np.linspace(0, len(order)-1, 12).astype(int)]
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    for ax_, idx in zip(axes.ravel(), pick):
        a, s_ = A[idx], S[idx]
        ax_.plot(a['s'], a['tr']*1e4, 'k-', lw=2, label='true')
        ax_.axhline(0, color='0.6', ls='--', lw=0.8)
        ax_.plot(a['s'], a['fit']*1e4, 'C0-', lw=1.3, label=f"angle {a['mean_resid']*1e4:.0f}µm")
        ax_.plot(s_['s'], s_['fit']*1e4, 'C1-', lw=1.3, label=f"spline {s_['mean_resid']*1e4:.0f}µm")
        ax_.set_title(f"L={a['L']:.0f}cm", fontsize=9); ax_.legend(fontsize=7); ax_.grid(alpha=0.3)
    fig.suptitle(f'Track gallery (sorted by curvature), {L} cm — transverse offset [µm] vs arc length [cm]')
    fig.tight_layout(); fig.savefig(f'plots/chain_debug_gallery_{L}cm.png', dpi=115); print(f'[saved] plots/chain_debug_gallery_{L}cm.png')

    # ---- C) residual vs normalized arc-length (median + IQR over tracks) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grid = np.linspace(0, 1, 40)
    for rows, c, lab in [(A, 'C0', 'angle'), (S, 'C1', 'spline')]:
        M = []
        for r in rows:
            a = r['s']/max(r['s'][-1], 1e-9)
            M.append(np.interp(grid, a, r['resid']*1e4))
        M = np.array(M)
        med = np.median(M, 0); q1, q3 = np.percentile(M, [25, 75], 0)
        ax.plot(grid, med, color=c, lw=2, label=f'{lab} (median)')
        ax.fill_between(grid, q1, q3, color=c, alpha=0.2)
    ax.set_xlabel('fractional arc length'); ax.set_ylabel('residual [µm]'); ax.legend()
    ax.set_title(f'Residual vs position along track, {L} cm'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f'plots/chain_debug_arclength_{L}cm.png', dpi=125); print(f'[saved] plots/chain_debug_arclength_{L}cm.png')

    # ---- D) spline mode spectrum ----
    maxK = max(len(r['params'])//2 for r in S)
    spec = np.zeros(maxK); cnt = np.zeros(maxK)
    for r in S:
        p = r['params']; K = len(p)//2
        mag = np.sqrt(p[:K]**2 + p[K:2*K]**2)
        spec[:K] += mag; cnt[:K] += 1
    spec = spec/np.maximum(cnt, 1)*1e4
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(np.arange(1, maxK+1), spec, color='C1')
    ax.set_xlabel('sine mode m'); ax.set_ylabel('mean |β_m| [µm]')
    ax.set_title(f'Spline mode usage, {L} cm (higher modes ~ finer wiggle)'); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(f'plots/chain_debug_modes_{L}cm.png', dpi=125); print(f'[saved] plots/chain_debug_modes_{L}cm.png')


if __name__ == '__main__':
    for L in [50, 400]:
        figs(L)
