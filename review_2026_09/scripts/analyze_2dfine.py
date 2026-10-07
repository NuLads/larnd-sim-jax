"""Fine Ab x lifetime grid: lifetime minimum profiled over Ab (charge-scale nuisance)."""
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/E_truthdiag'
Q = [dict(np.load(f)) for f in sorted(glob.glob(f'{D}/out/ab2dfine_*.npz'))]
c3 = lambda k: np.concatenate([q[k] for q in Q], 0)
ab, lt, b = Q[0]['ab_grid'], Q[0]['lt_grid'], c3('batch')
print(f'files {len(Q)} batches {len(b)} unique {len(np.unique(b))}\nAb {ab}\nlt {lt}')

def quad2d(A):
    X, Y = np.meshgrid(ab - 1, lt - 1, indexing='ij')
    M = np.vstack([np.ones(X.size), X.ravel(), Y.ravel(), X.ravel()**2, X.ravel()*Y.ravel(), Y.ravel()**2]).T
    c, *_ = np.linalg.lstsq(M, A.ravel(), rcond=None)
    rms = np.sqrt(np.mean((M @ c - A.ravel())**2))
    H = np.array([[2*c[3], c[4]], [c[4], 2*c[5]]]); x = np.linalg.solve(H, -c[1:3])
    cov = np.linalg.inv(H)
    return dict(ab=1 + x[0], lt=1 + x[1], slice_lt=1 - c[2] / (2*c[5]), w_prof=np.sqrt(cov[1, 1]),
                w_slice=np.sqrt(1 / H[1, 1]), w_ab=np.sqrt(cov[0, 0]), corr=cov[0, 1] / np.sqrt(cov[0, 0]*cov[1, 1]),
                valley=-c[4] / (2*c[5]), rms=rms)

def robust(A):
    tm, nm = [], []
    for ia in range(len(ab)):
        y = A[ia]; j = int(np.clip(np.argmin(y), 1, len(lt) - 2))
        c = np.polyfit(lt[j-1:j+2], y[j-1:j+2], 2); x0 = -c[1] / (2*c[0]); tm.append(x0); nm.append(np.polyval(c, x0))
    tm, nm = np.array(tm), np.array(nm)
    j = int(np.clip(np.argmin(nm), 1, len(ab) - 2))
    ca = np.polyfit(ab[j-1:j+2], nm[j-1:j+2], 2); a0 = -ca[1] / (2*ca[0])
    return np.polyval(np.polyfit(ab, tm, 1), a0), a0, tm, nm

out = {}
for tag, key in [('prod loss, noise-ON', 'nll2_on'), ('prod loss, noise-OFF', 'nll2_off'),
                 ('no-charge loss, noise-ON', 'nll2nq_on'), ('no-charge loss, noise-OFF', 'nll2nq_off')]:
    a = c3(key)
    if a.ndim == 3: a = a[..., None]
    tot = a.sum(0)
    q = [quad2d(tot[..., s]) for s in range(tot.shape[-1])]
    r = [robust(tot[..., s]) for s in range(tot.shape[-1])]
    lt_q = np.array([x['lt'] for x in q]); lt_r = np.array([x[0] for x in r])
    n = len(lt_q); sd = lambda v: v.std(ddof=1) if n > 1 else np.nan
    # batch jackknife on the seed-mean surface
    pb = a.mean(-1); T = pb.sum(0)
    jk = np.array([quad2d(T - pb[k])['lt'] for k in range(len(pb))]); jsd = np.sqrt((len(pb) - 1) * jk.var())
    qm = {k: np.mean([x[k] for x in q]) for k in q[0]}
    out[tag] = (lt_q, lt_r, qm, np.mean([x[2] for x in r], 0), np.mean([x[3] for x in r], 0))
    print(f'\n{tag}: profiled lifetime (2-D quadratic) {lt_q.mean():.4f}  seed-sd {sd(lt_q):.4f} sem {sd(lt_q)/np.sqrt(n):.4f}  '
          f'batch-jackknife sd {jsd:.4f}\n   robust (per-Ab local parabola) {lt_r.mean():.4f} seed-sd {sd(lt_r):.4f}\n'
          f'   Ab* {qm["ab"]:.5f}; slice(Ab=truth) {qm["slice_lt"]:.4f}; dNLL=0.5 widths: lt profiled {qm["w_prof"]:.4f}, '
          f'lt slice {qm["w_slice"]:.4f}, Ab {qm["w_ab"]:.5f}; corr {qm["corr"]:+.3f}; valley d(lt)/d(Ab) {qm["valley"]:+.1f}; fit rms {qm["rms"]:.2f}')
    print('   tau*(Ab) ' + ' '.join(f'{v:.4f}' for v in out[tag][3]) + '\n   NLLmin(Ab)-min ' + ' '.join(f'{v - out[tag][4].min():.1f}' for v in out[tag][4]))

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
a = c3('nll2_on').sum(0).mean(-1)
ax = axs[0]
for ia, fa in enumerate(ab):
    ax.plot((lt - 1) * 100, a[ia] - a.min(), 'o-', ms=3, label=f'Ab × {fa:.4f}')
ax.set_xlabel('lifetime − truth [%]'); ax.set_ylabel('ΔNLL (100 batches, noise-ON, seed mean)'); ax.legend(fontsize=7)
ax.set_title('NLL vs lifetime at fixed Ab (fine grid)')
ax = axs[1]
lt_q, lt_r, qm, tm, nm = out['prod loss, noise-ON']
ax.plot((ab - 1) * 100, (tm - 1) * 100, 'o-', label='lifetime minimum at fixed Ab')
ax2 = ax.twinx(); ax2.plot((ab - 1) * 100, nm - nm.min(), 's--', color='C3', label='NLL at that minimum'); ax2.set_ylabel('ΔNLL (profile over lifetime)', color='C3')
ax.axvline((qm['ab'] - 1) * 100, color='k', ls=':', lw=1)
ax.axhline((lt_q.mean() - 1) * 100, color='k', ls=':', lw=1)
ax.set_xlabel('Ab − truth [%]'); ax.set_ylabel('lifetime − truth [%]')
ax.set_title(f'Profiled optimum: lifetime {100*(lt_q.mean()-1):+.2f}%, Ab {100*(qm["ab"]-1):+.3f}%')
ax.legend(loc='upper right', fontsize=7)
plt.tight_layout(); plt.savefig(f'{D}/fig_E2_ab_lifetime_2d.png', dpi=130)
