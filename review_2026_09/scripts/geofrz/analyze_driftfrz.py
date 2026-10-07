"""DRIFTFRZ (DSREP config + drift-direction geometry frozen) vs the 2x2 arms, paired on samples 3/7/8.
Read at iteration 10000: endpoint = mean of iterations 9001-10000; position = median of the per-batch
pos_residual over the last 2000 iterations (it is a per-batch quantity; only a tail median is meaningful)."""
import glob, pickle, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
ARMS = {'BOTHTRU': 'sci_full_BOTHTRU{s}', 'GEOFRZ': 'sci_fullGEOFRZ{s}', 'DXTRU': 'sci_full_DXTRU{s}',
        'DSREP': 'sci_full_DSREP{s}', 'DRIFTFRZ': 'sci_full_DRIFTFRZ{s}'}
S = [3, 7, 8]
f1 = lambda x: float(np.ravel(np.asarray(x))[0])
res = {}
for arm, pat in ARMS.items():
    for s in S:
        f = sorted(glob.glob(f'{R}/{pat.format(s=s)}/history_iter*.pkl'))[-1]
        d = pickle.load(open(f, 'rb')); r = {}
        for p in P:
            a = np.array([f1(x) for x in d[f'{p}_iter']])[:10001]; t = f1(d[f'{p}_target'][0])
            r[p] = 100 * (a[-1000:].mean() / t - 1); r[p + '_ptp'] = 100 * np.ptp(a[-2000:]) / t
        pr = np.array([f1(x) for x in d['pos_residual_iter']])[:10000]
        r['pos_um'] = float(np.nanmedian(pr[-2000:])) * (1e4 if np.nanmedian(pr) < 1 else 1)
        r['loss'] = float(np.mean([f1(x) for x in d['losses_iter']][9000:10000]))
        res[(arm, s)] = r
print(f"{'arm':9s} " + ' '.join(f'{p:>16s}' for p in P) + f"{'pos(um)':>9s}")
for arm in ARMS:
    m = {p: np.mean([res[(arm, s)][p] for s in S]) for p in P}
    per = ' '.join(f"{m[p]:+6.2f} [" + ','.join(f"{res[(arm, s)][p]:+.1f}" for s in S) + "]" for p in P)
    print(f"{arm:9s} {per}  {np.mean([res[(arm, s)]['pos_um'] for s in S]):7.0f}")
print('\nmax ptp over last 2000 its (DRIFTFRZ):', {p: round(max(res[('DRIFTFRZ', s)][p + '_ptp'] for s in S), 2) for p in P})
def pair(a, b, p):
    v = np.array([res[(a, s)][p] - res[(b, s)][p] for s in S]); return v.mean(), v.std(ddof=1) / np.sqrt(3), v
print('\npaired differences (mean +- SE, n=3, per sample):')
for a, b in [('DRIFTFRZ', 'DSREP'), ('DRIFTFRZ', 'DXTRU'), ('DRIFTFRZ', 'GEOFRZ')]:
    for p in ['lifetime', 'long_diff', 'Ab', 'tran_diff'] + ['pos_um', 'loss']:
        m, se, v = pair(a, b, p)
        print(f"  {a}-{b:8s} {p:10s} {m:+8.2f} +- {se:6.2f}   [{', '.join(f'{x:+.2f}' for x in v)}]")
json.dump({f'{a}_{s}': r for (a, s), r in res.items()}, open('result_driftfrz.json', 'w'), indent=1)
order = ['BOTHTRU', 'GEOFRZ', 'DXTRU', 'DSREP', 'DRIFTFRZ']
xl = ['geo truth\ndE/dx truth', 'geo truth\ndE/dx fitted', 'geo fitted\ndE/dx truth', 'geo fitted\ndE/dx fitted\n(production)', 'geo fitted,\nDRIFT FROZEN\ndE/dx fitted']
col = {3: '#2a78d6', 7: '#eb6834', 8: '#1baf7a'}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for k, p in enumerate(['lifetime', 'long_diff', 'pos_um']):
    for s in S:
        ax[k].plot(range(5), [res[(a, s)][p] for a in order], 'o-', color=col[s], lw=1.5, ms=6, label=f'sample {s}')
    ax[k].plot(range(5), [np.mean([res[(a, s)][p] for s in S]) for a in order], 's', color='#0b0b0b', ms=9, mfc='none', mew=2, label='mean')
    if p != 'pos_um': ax[k].axhline(0, color='#52514e', lw=0.8, ls='--')
    ax[k].axvspan(3.5, 4.5, color='#eda100', alpha=0.12, lw=0)
    ax[k].set_xticks(range(5)); ax[k].set_xticklabels(xl, fontsize=7.5)
    ax[k].set_ylabel({'lifetime': 'lifetime error (%)', 'long_diff': 'long_diff error (%)', 'pos_um': 'position residual (µm, tail median)'}[p])
    ax[k].spines[['top', 'right']].set_visible(False); ax[k].grid(axis='y', color='#e5e4df', lw=0.8); ax[k].set_axisbelow(True)
    ax[k].set_title(f'({"abc"[k]}) {p}', loc='left', fontsize=10)
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle('Freezing the drift-direction geometry (last column) vs the production 2×2, paired on samples 3/7/8', fontsize=10, x=0.01, ha='left')
fig.tight_layout(); fig.savefig('fig_driftfrz.png', dpi=150); print('figure written')
