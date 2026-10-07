"""Fine lifetime grid (pass 2) and Ab x lifetime profile (pass 3)."""
import glob, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/E_truthdiag'

def pmin(x, y, half=None):
    """quadratic fit; if half given, restrict to |x - argmin| <= half"""
    if half is not None:
        j = np.argmin(y); m = np.abs(x - x[j]) <= half + 1e-9
        x, y = x[m], y[m]
    c = np.polyfit(x, y, 2)
    return -c[1] / (2 * c[0]), 2 * c[0]

def cmin(x, y):
    c = np.polyfit(x, y, 3); r = np.roots(np.polyder(c)); r = r[np.isreal(r)].real
    r = r[(np.polyval(np.polyder(c, 2), r) > 0)]
    return r[np.argmin(abs(r - 1))] if len(r) else np.nan

# ---------------- pass 2
F = sorted(glob.glob(f'{D}/out/fine_*.npz'))
P = [dict(np.load(f, allow_pickle=True)) for f in F]
cat = lambda k: np.concatenate([p[k] for p in P], 0)
g = P[0]['lt_grid']; b = cat('batch'); print(f'[fine] files {len(F)} batches {len(b)} unique {len(np.unique(b))}; grid {g}')
out = {}
for tag, key in [('prod loss, noise-ON', 'nll_lt_on'), ('prod loss, noise-OFF', 'nll_lt_off'),
                 ('prod loss, noise-OFF tgt, tiny-noise model', 'nll_lt_on_tinymodel_off'),
                 ('no-charge loss, noise-ON', 'nll_nq_on'), ('no-charge loss, noise-OFF', 'nll_nq_off')]:
    a = cat(key)                       # (nb, nlt[, ns])
    if a.ndim == 2: a = a[..., None]
    tot = a.sum(0)                     # (nlt, ns)
    mins = np.array([pmin(g, tot[:, s], half=0.046)[0] for s in range(tot.shape[1])])
    cm = np.array([cmin(g, tot[:, s]) for s in range(tot.shape[1])])
    curv = np.mean([pmin(g, tot[:, s], half=0.046)[1] for s in range(tot.shape[1])])
    pb = a.mean(2); T = pb.sum(0)
    jk = np.array([pmin(g, T - pb[k], half=0.046)[0] for k in range(len(pb))])
    jsd = np.sqrt((len(pb) - 1) * jk.var())
    sd = mins.std(ddof=1) if len(mins) > 1 else np.nan
    out[tag] = dict(tot=tot, mins=mins)
    print(f'{tag:45s} min(local quad) = {mins.mean():.4f}  seed-sd {sd:.4f} (sem {sd/np.sqrt(len(mins)):.4f}); '
          f'cubic {np.nanmean(cm):.4f}; batch-jackknife sd {jsd:.4f}; dNLL=0.5 width {1/np.sqrt(curv):.4f}')

fig, ax = plt.subplots(figsize=(6.4, 4.6))
cols = {'prod loss, noise-ON': 'C0', 'prod loss, noise-OFF': 'C2', 'prod loss, noise-OFF tgt, tiny-noise model': 'C1',
        'no-charge loss, noise-ON': 'C3', 'no-charge loss, noise-OFF': 'C4'}
for tag, d in out.items():
    y = d['tot'].mean(1); y = y - y.min()
    lab = f'{tag}: min {100*(d["mins"].mean()-1):+.2f}%'
    if 'tiny' in tag: lab = f'{tag}: cusp at 0 (quad fit {100*(d["mins"].mean()-1):+.2f}%)'
    if tag == 'no-charge loss, noise-OFF': lab = f'{tag}: monotonic, no min in range'
    ax.plot((g - 1) * 100, y, 'o-', ms=3, color=cols[tag], label=lab)
ax.axvline(0, color='gray', lw=.5); ax.axvline(8, color='k', ls='--', lw=1, label='production fit (+8%)')
ax.set_ylim(0, 400); ax.set_xlabel('lifetime − truth [%]'); ax.set_ylabel('ΔNLL (100 batches; seed-mean)')
ax.legend(fontsize=6.5); ax.set_title('Fine lifetime profile at truth (all else at truth)')
plt.tight_layout(); plt.savefig(f'{D}/fig_E2_fine_lifetime.png', dpi=130)

# ---------------- pass 3
F3 = sorted(glob.glob(f'{D}/out/ab2d_*.npz'))
if F3 and len(sys.argv) < 2:
    Q = [dict(np.load(f, allow_pickle=True)) for f in F3]
    c3 = lambda k: np.concatenate([q[k] for q in Q], 0)
    ab, lt = Q[0]['ab_grid'], Q[0]['lt_grid']; b3 = c3('batch')
    print(f'\n[2D] files {len(F3)} batches {len(b3)} unique {len(np.unique(b3))}; Ab grid {ab}; lt grid {lt}')
    PROF = {}
    def prof2d(A):
        """A: (na, nl). Fit full 2-D quadratic; return profiled lifetime minimum (Ab free),
        lifetime min at Ab=truth, and Ab at optimum."""
        X, Y = np.meshgrid(ab, lt, indexing='ij')
        M = np.vstack([np.ones(X.size), X.ravel(), Y.ravel(), X.ravel()**2, X.ravel()*Y.ravel(), Y.ravel()**2]).T
        c = np.linalg.lstsq(M, A.ravel(), rcond=None)[0]
        H = np.array([[2*c[3], c[4]], [c[4], 2*c[5]]]); gvec = -np.array([c[1], c[2]])
        xa, xl = np.linalg.solve(H, gvec)
        slice_l = -(c[2] + c[4]*1.0) / (2*c[5])
        cov = np.linalg.inv(H)
        return xl, slice_l, xa, np.sqrt(cov[1, 1]), np.sqrt(1/H[1, 1]), cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])
    for tag, key in [('prod loss, noise-ON', 'nll2_on'), ('prod loss, noise-OFF', 'nll2_off'),
                     ('no-charge loss, noise-ON', 'nll2nq_on'), ('no-charge loss, noise-OFF', 'nll2nq_off')]:
        a = c3(key)
        if a.ndim == 3: a = a[..., None]
        tot = a.sum(0)
        r = np.array([prof2d(tot[..., s]) for s in range(tot.shape[-1])])
        # robust alternative: per-Ab LOCAL parabola in lifetime -> (tau*(Ab), NLLmin(Ab)); then a
        # parabola of NLLmin over Ab gives the profiled optimum; tau* interpolated linearly there.
        def prof_robust(A):
            tm, nm = [], []
            for ia in range(len(ab)):
                y = A[ia]; j = int(np.clip(np.argmin(y), 1, len(lt) - 2))
                c = np.polyfit(lt[j-1:j+2], y[j-1:j+2], 2); x0 = -c[1] / (2 * c[0])
                tm.append(x0); nm.append(np.polyval(c, x0))
            tm, nm = np.array(tm), np.array(nm)
            ca = np.polyfit(ab, nm, 2); a0 = -ca[1] / (2 * ca[0])
            return np.polyval(np.polyfit(ab, tm, 1), a0), a0, tm
        rr = [prof_robust(tot[..., s]) for s in range(tot.shape[-1])]
        rt = np.array([x[0] for x in rr]); ra = np.array([x[1] for x in rr])
        tm_mean = np.mean([x[2] for x in rr], 0)
        sdr = rt.std(ddof=1) if len(rt) > 1 else np.nan
        print(f'{"":28s} ROBUST profile: lifetime {rt.mean():.4f} (seed-sd {sdr:.4f}, sem {sdr/np.sqrt(len(rt)):.4f}), Ab* {ra.mean():.5f}; '
              f'tau*(Ab) = ' + ' '.join(f'{v:.4f}' for v in tm_mean))
        PROF[tag] = (rt, ra, tm_mean)
        sd = r[:, 0].std(ddof=1) if len(r) > 1 else np.nan
        print(f'{tag:28s} lifetime PROFILED over Ab: {r[:,0].mean():.4f} (seed-sd {sd:.4f})  | slice at Ab=truth {r[:,1].mean():.4f} '
              f'| Ab at optimum {r[:,2].mean():.5f} | dNLL=.5 width profiled {r[:,3].mean():.4f} slice {r[:,4].mean():.4f} | corr {r[:,5].mean():+.3f}')
    # conditional (matched-tick) charge residual vs drift
    edg = Q[0]['edges_tick']; tcen = ((edg[1:] + edg[:-1]) / 2 - 22.4) * 0.1
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, lab, col in [('cq_on', 'noise-ON target, prod model', 'C0'), ('cq_off', 'noise-OFF target, prod model', 'C2'),
                          ('cq_off_tiny', 'noise-OFF target, tiny-noise model', 'C1')]:
        c = c3(key)
        if c.ndim == 4:      # (nb, ns, 4, NB)
            per = c.sum(0)   # (ns, 4, NB)
            rs = per[:, 0] / per[:, 2]
            r = rs.mean(0); e = rs.std(0, ddof=1) / np.sqrt(len(rs))
        else:
            per = c.sum(0); r = per[0] / per[2]
            jk = np.array([(per[0] - c[k, 0]) / (per[2] - c[k, 2]) for k in range(len(c))])
            e = np.sqrt((len(c) - 1) * jk.var(0))
        w = 1 / e**2; A = np.vstack([np.ones_like(tcen), tcen]).T
        cov = np.linalg.inv(A.T @ (A * w[:, None])); p = cov @ (A.T @ (w * r))
        print(f'  matched-tick charge residual <q_obs - q_model>/<q_obs> [{lab}]: mean {100*np.average(r, weights=w):+.3f}%, '
              f'slope {100*p[1]*190:+.3f}% per 190us (+-{100*np.sqrt(cov[1,1])*190:.3f})')
        ax.errorbar(tcen, 100 * r, 100 * e, fmt='o', ms=3, color=col, capsize=2, label=lab)
    ax.axhline(0, color='gray', lw=.5); ax.set_xlabel('drift time [µs]'); ax.set_ylabel('Σ(q_obs − q_model|tick) / Σq_obs  [%]')
    ax.set_title('Per-hit charge residual at the observed tick'); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f'{D}/fig_E1b_conditional_charge.png', dpi=130)

    # figure: NLL vs lifetime for each Ab (noise-ON, seed mean) + profiled envelope
    a = c3('nll2_on').sum(0).mean(-1)     # (na, nl)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for ia, fa in enumerate(ab):
        ax.plot((lt - 1) * 100, a[ia] - a.min(), 'o-', ms=3, label=f'Ab × {fa:.3f}')
    rt, ra, tm = PROF['prod loss, noise-ON']
    ax.axvline(0.28, color='k', ls=':', lw=1, label='Ab-profiled optimum (+0.3 ± 0.3%)')
    ax.axvline(1.50, color='C0', ls=':', lw=1, label='slice at Ab=truth (+1.5%)')
    ax.axvline(8, color='k', ls='--', lw=1, label='production fit (+8%)')
    ax.set_ylim(0, 400); ax.set_xlabel('lifetime − truth [%]'); ax.set_ylabel('ΔNLL (100 batches, noise-ON, seed mean)')
    ax.set_title('Lifetime × Ab: the slice minimum moves with Ab'); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f'{D}/fig_E2_ab_lifetime_2d_coarse.png', dpi=130)
