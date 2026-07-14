#!/usr/bin/env python3
"""Chain-fit performance as a function of TRACK length (angle vs spline).

Pools per-track results from the 50 cm run (tracks <=50 cm) and the 400 cm run
(tracks up to ~130 cm) so the full length range is covered, and bins by individual
track length. Three views:
  (1) absolute residual vs length  -> raw performance / conditioning growth
  (2) residual / true-bow-amplitude vs length -> fraction of the shape MISSED
      (controls for the signal itself growing with length)
  (3) true bow amplitude vs length -> the signal (MCS displacement ~ L^1.5)
"""
import pickle, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('plots', exist_ok=True)


def load(pat):
    g = glob.glob(pat)
    if not g:
        return None
    p = max(g, key=lambda x: int(x.split('history_iter')[1].split('_')[0]))
    return pickle.load(open(p, 'rb')), int(p.split('history_iter')[1].split('_')[0])


def dir_from_angles(t, p):
    st = np.sin(t); return np.array([st*np.cos(p), st*np.sin(p), np.cos(t)])


def frame(u0):
    ref = np.array([0., 0., 1.]) if abs(u0[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(u0, ref); e1 /= np.linalg.norm(e1)+1e-12
    e2 = np.cross(u0, e1);  e2 /= np.linalg.norm(e2)+1e-12
    return e1, e2


def recon_nodes(ctx, params, basis):
    nc = int(ctx['n_chain']); step = float(ctx['step_len'])
    x0 = np.asarray(ctx['x0_fixed'], float); u0 = dir_from_angles(ctx['theta0_i'], ctx['phi0_i'])
    ks = np.arange(nc+1); nominal = x0[None,:]+(ks[:,None]*step)*u0[None,:]
    p = np.asarray(params, float)
    if basis == 'spline':
        K = p.shape[0]//2; B = np.sin(np.outer(ks/nc, np.pi*np.arange(1, K+1))); e1, e2 = frame(u0)
        return nominal + (B@p[:K])[:,None]*e1[None,:] + (B@p[K:2*K])[:,None]*e2[None,:]
    th, ph = p[:nc], p[nc:]; st, ct = np.sin(th), np.cos(th)
    u = np.stack([st*np.cos(ph), st*np.sin(ph), ct], 1); u /= np.linalg.norm(u,axis=1,keepdims=True)+1e-12
    return np.concatenate([x0[None,:], x0[None,:]+np.cumsum(step*u,0)], 0)


def per_track(h, basis):
    cc, ctx, tp = h['chain_cache'], h['chain_contexts'], h['true_positions']
    out = []
    for b in sorted(ctx.keys()):
        for ti, c in enumerate(ctx[b]):
            tk = str(c['track_id'])
            if tk not in tp.get(b, {}):
                continue
            nc = int(c['n_chain']); L = float(c['total_len'])
            x0 = np.asarray(c['x0_fixed'], float); u0 = dir_from_angles(c['theta0_i'], c['phi0_i'])
            e1, e2 = frame(u0); alpha = np.asarray(c['alpha_mid'], float)
            nomp = x0[None,:]+alpha[:,None]*L*u0[None,:]
            nodes = recon_nodes(c, cc[b][ti]['angles'], basis)
            k = np.clip(np.floor(alpha*nc).astype(int),0,nc-1); fr = alpha*nc-k
            fitp = nodes[k]+fr[:,None]*(nodes[k+1]-nodes[k])
            tr = np.asarray(tp[b][tk], float); m = min(len(fitp),len(tr),len(alpha))
            fitp, tr, nomp = fitp[:m], tr[:m], nomp[:m]
            resid = np.linalg.norm(fitp-tr, axis=1)
            true_amp = np.ptp(((tr-nomp)@e1)) + np.ptp(((tr-nomp)@e2))  # rough 2D bow size
            out.append((L, resid.mean(), max(true_amp, 1e-6)))
    return np.array(out)  # (n, 3): L, mean_resid, true_amp


def collect(basis):
    rows = []
    for L in [50, 400]:
        clr = '1e-4' if basis == 'angle' else ('3e-3')
        r = load(f'fit_result/pos_basis/history_iter*_posb_{basis}_len{L}_clr{clr}_mcs0.5_*.pkl')
        if r is None:
            continue
        h, it = r
        d = per_track(h, basis)
        rows.append((d, L, it))
    if not rows:
        return None
    return np.vstack([d for d, _, _ in rows]), rows


def binned(x, y, edges):
    idx = np.digitize(x, edges)
    med, q1, q3, ctr = [], [], [], []
    for bi in range(1, len(edges)):
        m = idx == bi
        if m.sum() < 2:
            continue
        med.append(np.median(y[m])); q1.append(np.percentile(y[m], 25)); q3.append(np.percentile(y[m], 75))
        ctr.append(0.5*(edges[bi-1]+edges[bi]))
    return np.array(ctr), np.array(med), np.array(q1), np.array(q3)


def main():
    A = collect('angle'); S = collect('spline')
    if A is None or S is None:
        print('missing runs'); return
    da, rowsa = A; ds, rowss = S
    edges = np.arange(0, 140, 12.0)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for d, c, lab in [(da, 'C0', 'angle (segment)'), (ds, 'C1', 'spline')]:
        L, r, amp = d[:, 0], d[:, 1]*1e4, d[:, 2]*1e4
        ax[0].scatter(L, r, s=8, alpha=0.25, color=c)
        cx, md, q1, q3 = binned(L, r, edges)
        ax[0].plot(cx, md, color=c, lw=2.5, label=lab)
        ax[0].fill_between(cx, q1, q3, color=c, alpha=0.18)
        frac = r / amp * 100
        cx2, md2, q12, q32 = binned(L, frac, edges)
        ax[1].plot(cx2, md2, color=c, lw=2.5, label=lab)
        ax[1].fill_between(cx2, q12, q32, color=c, alpha=0.18)
        cx3, ma, _, _ = binned(L, amp, edges)
        ax[2].plot(cx3, ma, color=c, lw=2.0, label=lab)
    ax[0].set_yscale('log'); ax[0].set_xlabel('track length [cm]'); ax[0].set_ylabel('mean residual [µm]')
    ax[0].set_title('absolute residual vs length'); ax[0].legend(); ax[0].grid(alpha=0.3, which='both')
    ax[1].set_xlabel('track length [cm]'); ax[1].set_ylabel('residual / true bow amplitude [%]')
    ax[1].set_title('fraction of track shape MISSED'); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[2].set_xlabel('track length [cm]'); ax[2].set_ylabel('true bow amplitude [µm]')
    ax[2].set_title('the signal: MCS bow vs length'); ax[2].legend(); ax[2].grid(alpha=0.3)
    its = {L: it for _, L, it in rowss}
    note = f"(spline 400cm run in progress, iter {its.get(400,'?')})"
    fig.suptitle(f'Chain-fit performance vs track length: segment (angle) vs spline  {note}')
    fig.tight_layout(); fig.savefig('plots/chain_length_dependence.png', dpi=130)
    print('[saved] plots/chain_length_dependence.png')
    # quick numeric summary
    for d, lab in [(da, 'angle'), (ds, 'spline')]:
        for lo, hi in [(10, 40), (40, 80), (80, 140)]:
            m = (d[:, 0] >= lo) & (d[:, 0] < hi)
            if m.sum():
                print(f'  {lab:6s} L[{lo},{hi}) n={m.sum():3d}  median resid={np.median(d[m,1])*1e4:6.0f}µm  '
                      f'median frac={np.median(d[m,1]/d[m,2])*100:5.1f}%')


if __name__ == '__main__':
    main()
