"""Report figures 23-26 (data: sample-8 production checkpoint + GEOFRZ8 targets; fit checkpoints for the tail)."""
import sys, glob, pickle, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import chi2
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
from optimize.fit_params import _dir_from_angles, _transverse_frame, _DRIFT_AXIS
R = '/sdf/group/neutrino/pgranger/larnd-sim-jax/'; OUT = R + 'review_2026_09/figs/'
TD = R + 'target_sci_fullGEOFRZ8_sci_full_b100_len400_mw2500.0_refit_seed0_Ab_eField_tran_diff_long_diff_lifetime'
ck = pickle.load(open(glob.glob(R + 'fit_result/sci_full_DSREP8/history_iter10000*.pkl')[0], 'rb'))
def adc2q(adc): return (adc / 256 * (1282.71484375 - 284.27734375) + 284.27734375 - 580) / 4e-3 * 1e-3
ag = np.linspace(0, 1, 101)
hit_r, pix_r = [], []; seeds = []; strt = []; pieces = []
for b in range(100):
    f = np.load(f'{TD}/batch{b}_target.npz'); H = np.stack([f['pixel_x'], f['pixel_y'], f['pixel_z']], 1).astype(float)
    ev = f['event']; q = f['adcs'].astype(float); pid = f['pixel_id']; Q = adc2q(q)
    for c in ck['chain_contexts'][b]:
        u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0)
        x0 = np.asarray(c['x0_fixed'], float); L = float(c['total_len'])
        m = ev == int(c['track_id'][0]); P, w, I, QQ = H[m], q[m], pid[m], Q[m]
        a = ((P - x0) @ u0) / L; d = P - x0 - a[:, None] * L * u0
        s = (a > .01) & (a < .99) & (np.linalg.norm(d - (d @ u0)[:, None] * u0, axis=1) < 1.0)
        # classic dQ/dx pieces (all tracks)
        aa = ((P - x0) @ u0); ss = (aa > 0) & (aa < L) & (np.linalg.norm(d - (d @ u0)[:, None] * u0, axis=1) < 1.0)
        for k in range(int(L // 2.0)):
            mm = ss & (aa >= 2 * k) & (aa < 2 * k + 2); zmid = x0[2] + (2 * k + 1) * u0[2]
            if mm.sum() and 0.5 < abs(zmid) < 30.59 - 0.5: pieces.append(((30.58975 - abs(zmid)) / 0.1587, QQ[mm].sum() / 2.0))
        if not (abs(e1[2]) < 1e-6 and np.linalg.norm(np.cross(_DRIFT_AXIS, u0)) > 1e-3): continue
        tr = np.asarray(ck['true_positions'][b][str(tuple(c['track_id']))], float); al = np.asarray(c['alpha_mid'], float)[:len(tr)]; tr = tr[:len(al)]
        K = len(np.asarray(ck['chain_cache'][b][0]['angles'])) // 2 if False else len(np.asarray(ck['chain_cache'][b][list(ck['chain_contexts'][b]).index(c)]['angles'])) // 2
        Bt = np.sin(np.outer(al, np.pi * np.arange(1, K + 1))); dt = tr - x0 - al[:, None] * L * u0
        bt1 = np.linalg.lstsq(Bt, dt @ e1, rcond=None)[0]; bt2 = np.linalg.lstsq(Bt, dt @ e2, rcond=None)[0]
        Bg = np.sin(np.outer(ag, np.pi * np.arange(1, K + 1))); rms = lambda bb: float(np.sqrt(np.mean((Bg @ bb) ** 2)) * 1e4)
        P, w, I = P[s], w[s], I[s]
        if len(np.unique(I)) < 3 * K + 2: continue
        o = np.argsort(al); ttz = np.interp(a[s], al[o], (dt @ e2)[o])          # true e2 offset at hit alpha
        if abs(u0[2]) < 0.5: hit_r += list((d[s] @ e2 - ttz) * 1e4)
        up, inv = np.unique(I, return_inverse=True); ws = np.bincount(inv, w)
        Pm = np.stack([np.bincount(inv, w * P[:, j]) / ws for j in range(3)], 1); wm = ws
        am = ((Pm - x0) @ u0) / L; dm = Pm - x0 - am[:, None] * L * u0
        if abs(u0[2]) < 0.5: pix_r += list((dm @ e2 - np.interp(am, al[o], (dt @ e2)[o])) * 1e4)
        B = np.sin(np.outer(am, np.pi * np.arange(1, K + 1))); sw = np.sqrt(wm / wm.mean())
        out = {}
        for comp, y, bt in (('e1', dm @ e1, bt1), ('e2', dm[:, 2], bt2)):
            A = np.hstack([np.ones((len(am), 1)), am[:, None], B * (e2[2] if comp == 'e2' else 1.0)]); keep = np.ones(len(am), bool)
            for _ in range(4):
                coef = np.linalg.lstsq(A[keep] * sw[keep, None], y[keep] * sw[keep], rcond=None)[0]
                r = y - A @ coef; sig = 1.4826 * np.median(np.abs(r[keep])) + 1e-4; nk = np.abs(r) < 3 * sig
                if nk.sum() < 3 * K + 2 or (nk == keep).all(): break
                keep = nk
            Aw = A[keep] * sw[keep, None]; cov = sig ** 2 * np.linalg.pinv(Aw.T @ Aw); bc = coef[2:]
            T = float(bc @ np.linalg.pinv(cov[2:, 2:]) @ bc); pv = chi2.sf(T, K)
            if comp == 'e2': use = bc if pv < 1e-3 else 0 * bc
            else: use = bc if rms(bc) > 500 else 0 * bc
            out[comp] = (rms(bt), rms(bc - bt), rms(use - bt), pv)
        seeds.append((max(rms(bt1), rms(bt2)),) + out['e1'][:3] + out['e2'])
S = np.array(seeds); hit_r = np.array(hit_r); pix_r = np.array(pix_r)
# ---------- fig23: hit-derived drift information
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
bins = np.linspace(-8000, 8000, 161)
for v, lab, col in ((hit_r, 'single hits', 'C7'), (pix_r, 'per-pixel charge-weighted mean', 'C0')):
    v = v - np.median(v); sig = 1.4826 * np.median(np.abs(v))
    ax[0].hist(v, bins=bins, histtype='step', density=True, lw=1.6, color=col, label=f'{lab}: robust σ = {sig/1000:.1f} mm')
ax[0].set_xlabel('drift-direction position from hit timing − true track [µm]'); ax[0].set_ylabel('density'); ax[0].legend(fontsize=8)
ax[0].set_title('how well hit timing locates the track along the drift')
cv = S[:, 0]; cls = [('near-straight\n(true dev. < 250 µm)', cv < 250), ('all tracks', cv > -1), ('curved\n(true dev. > 500 µm)', cv > 500)]
x = np.arange(len(cls)); wdt = 0.13
for j, (lab, col, idx) in enumerate((('straight line, drift', 'C7', 4), ('raw hit fit, drift', 'C3', 5), ('gated hit seed, drift', 'C2', 6),
                                    ('straight line, pixel plane', '0.6', 1), ('raw hit fit, pixel plane', 'C1', 2), ('gated hit seed, pixel plane', 'C0', 3))):
    ax[1].bar(x + (j - 2.5) * wdt, [np.median(S[m, idx]) for _, m in cls], wdt, color=col, label=lab, edgecolor='k', lw=0.3)
ax[1].set_xticks(x); ax[1].set_xticklabels([c[0] for c in cls], fontsize=8); ax[1].set_ylabel('median shape error vs truth [µm]')
ax[1].set_title('starting-shape error by track class (sample 8, 865 tracks)'); ax[1].legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(OUT + 'fig23_hit_seed_quality.png', dpi=140); plt.close(fig)
# ---------- fig24: classic dQ/dx vs drift time
Pc = np.array(pieces); t, y = Pc[:, 0], Pc[:, 1]; edges = np.linspace(0, 190, 10)
tc, mu, er = [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    v = np.sort(y[(t >= lo) & (t < hi)]); k = int(0.7 * len(v)); w = v[:k]
    tc.append(t[(t >= lo) & (t < hi)].mean()); mu.append(w.mean()); er.append(w.std(ddof=1) / np.sqrt(k))
tc, mu, er = map(np.array, (tc, mu, er))
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.errorbar(tc, mu / mu[1], er / mu[1], fmt='o', color='C0', label='measured: 70% truncated-mean dQ/dx of 2-cm pieces')
tt = np.linspace(0, 190, 50); ax.plot(tt, np.exp(-(tt - tc[1]) / 2406.4), 'k--', label='true attenuation, τ = 2406 µs')
ax.set_xlabel('drift time [µs]'); ax.set_ylabel('dQ/dx relative to the 32-µs bin'); ax.legend(fontsize=8)
ax.set_title('classic lifetime method on the target hits (sample 8)')
fig.tight_layout(); fig.savefig(OUT + 'fig24_classic_dqdx.png', dpi=140); plt.close(fig)
print('fig23/24 done; hits', len(hit_r), 'pixels', len(pix_r), 'tracks', len(S))
