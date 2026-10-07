"""DRIFTTRU (drift coordinate at truth, frozen; pixel-plane geometry and dE/dx fitted) vs DSREP (production)
on all six samples 1/2/3/4/7/8. Samples 3/7/8 = the original test; 1/2/4 = the out-of-sample replication
(resumed from iter 2200 after /sdf/data filled). Every run is read at iteration 10000: endpoint = mean of
iterations 9001-10000; position = median of the per-batch residual over iterations 8001-10000."""
import glob, pickle, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
OUT = '/sdf/scratch/neutrino/pgranger/review_2026_09/GEOFRZ'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
SAMPLES = [3, 7, 8, 1, 2, 4]
f1 = lambda x: float(np.ravel(np.asarray(x))[0])

def read(tag):
    f = sorted(glob.glob(f'{R}/{tag}/history_iter*.pkl'), key=lambda x: int(x.split('iter')[-1].split('_')[0]))[-1]
    d = pickle.load(open(f, 'rb')); r = {'total_iter': d['total_iter']}
    assert d['total_iter'] >= 10000, f'{tag} only reached {d["total_iter"]}'
    for p in P:
        a = np.array([f1(x) for x in d[f'{p}_iter']])[:10001]; t = f1(d[f'{p}_target'][0])
        r[p] = 100 * (a[-1000:].mean() / t - 1); r[p + '_ptp'] = 100 * np.ptp(a[-2000:]) / t
    pr = np.array([f1(x) for x in d['pos_residual_iter']])[:10000]
    r['pos_um'] = float(np.nanmedian(pr[-2000:])) * (1e4 if np.nanmedian(pr) < 1 else 1)
    r['loss'] = float(np.mean([f1(x) for x in d['losses_iter']][9000:10000]))
    return r

res = {(a, s): read(f'sci_full_{a}{s}') for a in ('DSREP', 'DRIFTTRU') for s in SAMPLES}
print(f"{'sample':>6s} {'arm':9s} " + ' '.join(f'{p:>10s}' for p in P) + f" {'pos_um':>7s}  max_ptp")
for s in SAMPLES:
    for a in ('DSREP', 'DRIFTTRU'):
        r = res[(a, s)]
        print(f"{s:6d} {a:9s} " + ' '.join(f"{r[p]:+10.2f}" for p in P) + f" {r['pos_um']:7.0f}  {max(r[p + '_ptp'] for p in P):.2f}")
out = {}
print('\npaired DRIFTTRU - DSREP (mean +- SE; per sample in order 3,7,8 | 1,2,4):')
for grp, ss in [('original 3/7/8', [3, 7, 8]), ('replication 1/2/4', [1, 2, 4]), ('all six', SAMPLES)]:
    for p in ['lifetime', 'long_diff', 'Ab', 'tran_diff', 'eField', 'pos_um', 'loss']:
        v = np.array([res[('DRIFTTRU', s)][p] - res[('DSREP', s)][p] for s in ss])
        out[f'{grp}|{p}'] = dict(mean=v.mean(), se=v.std(ddof=1) / np.sqrt(len(v)), per=v.tolist())
        print(f"  {grp:18s} {p:10s} {v.mean():+9.2f} +- {v.std(ddof=1)/np.sqrt(len(v)):6.2f}  [{', '.join(f'{x:+.1f}' for x in v)}]")
for a in ('DSREP', 'DRIFTTRU'):
    for p in ['lifetime', 'long_diff']:
        v = np.array([res[(a, s)][p] for s in SAMPLES])
        print(f"  {a:9s} {p:10s} level over six: {v.mean():+6.2f} +- {v.std(ddof=1)/np.sqrt(6):.2f}  (|mean| of |err| {np.abs(v).mean():.2f})")
json.dump({'runs': {f'{a}{s}': r for (a, s), r in res.items()}, 'paired': out}, open(f'{OUT}/result_drifttru_n6.json', 'w'), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
cols = {3: '#2a78d6', 7: '#eb6834', 8: '#1baf7a', 1: '#4a3aa7', 2: '#e87ba4', 4: '#eda100'}
for k, p in enumerate(['lifetime', 'long_diff', 'pos_um']):
    for s in SAMPLES:
        ax[k].plot([0, 1], [res[('DSREP', s)][p], res[('DRIFTTRU', s)][p]], 'o-', color=cols[s], lw=1.8, ms=7,
                   ls='-' if s in (3, 7, 8) else '--', label=f'sample {s}' + (' (replication)' if s in (1, 2, 4) else ''))
    if p != 'pos_um': ax[k].axhline(0, color='#52514e', lw=0.8, ls=':')
    ax[k].set_xticks([0, 1]); ax[k].set_xticklabels(['production\n(geometry + dE/dx fitted)', 'drift coordinate\nfrom truth, rest fitted'], fontsize=8.5)
    ax[k].set_xlim(-0.3, 1.3)
    ax[k].set_ylabel({'lifetime': 'lifetime error (%)', 'long_diff': 'long_diff error (%)', 'pos_um': 'position residual (µm)'}[p])
    ax[k].spines[['top', 'right']].set_visible(False); ax[k].grid(axis='y', color='#e5e4df', lw=0.8); ax[k].set_axisbelow(True)
    v = out[f'all six|{p}']; ax[k].set_title(f"({'abc'[k]}) {p}: paired {v['mean']:+.1f} ± {v['se']:.1f} (n=6)", loc='left', fontsize=10)
ax[0].legend(frameon=False, fontsize=7.5, ncol=2)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_drifttru_n6.png', dpi=150); print('figure written')
