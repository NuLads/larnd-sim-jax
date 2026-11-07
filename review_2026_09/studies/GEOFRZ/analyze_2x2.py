"""2x2 factorial of the production fit on data samples 3/7/8 (same truth, same noise target):
geometry {frozen at its best truth representation, fitted} x dE/dx {frozen at truth, fitted}.
   BOTHTRU = (geo T, dEdx T)   GEOFRZ = (geo T, dEdx F)   DXTRU = (geo F, dEdx T)   DSREP = (geo F, dEdx F)
Endpoint = mean of the last 1000 iterations; error = endpoint/target - 1 (%). Settledness = peak-to-peak
over the last 2000 iterations. Effects are paired per sample; mean +- SE over n = 3."""
import glob, pickle, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
OUT = '/sdf/scratch/neutrino/pgranger/review_2026_09/GEOFRZ'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
ARMS = {'BOTHTRU': 'sci_full_BOTHTRU{s}', 'GEOFRZ': 'sci_fullGEOFRZ{s}', 'DXTRU': 'sci_full_DXTRU{s}',
        'DSREP': 'sci_full_DSREP{s}', 'DXFRZ': 'sci_full_DXFRZ{s}'}
SEEDS = [3, 7, 8]
f1 = lambda x: float(np.ravel(np.asarray(x))[0])

res = {}
for arm, pat in ARMS.items():
    for s in SEEDS:
        fs = sorted(glob.glob(f'{R}/{pat.format(s=s)}/history_iter*.pkl'))
        if not fs: print('MISSING', arm, s); continue
        d = pickle.load(open(fs[-1], 'rb'))
        row = {'file': fs[-1].split('/')[-1], 'total_iter': d['total_iter'], 'resumed': 'resumed_from' in d}
        for p in P:
            a = np.array([f1(x) for x in d[f'{p}_iter']])[:10001]; t = f1(d[f'{p}_target'][0])   # read every arm at iteration 10000 (GEOFRZ resumes over-ran past it)
            row[p] = 100 * (a[-1000:].mean() / t - 1)
            row[p + '_ptp'] = 100 * np.ptp(a[-2000:]) / t
            row[p + '_n'] = len(a)
        res[(arm, s)] = row

print(f"{'arm':8s} {'s':>2s} {'iters':>6s}  " + '  '.join(f'{p:>10s}' for p in P) + '   (error %, [ptp last 2000])')
for (arm, s), r in res.items():
    print(f"{arm:8s} {s:2d} {r['total_iter']:6d}  " + '  '.join(f"{r[p]:+6.2f}[{r[p+'_ptp']:.2f}]" for p in P)
          + ('  resumed' if r['resumed'] else '') + f"  (history len {r['lifetime_n']})")

def eff(fn):
    v = np.array([fn(s) for s in SEEDS]); return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), v
L = lambda arm, s, p: res[(arm, s)][p]
out = {}
print('\nfactorial effects (paired per sample, mean +- SE, n=3; per-sample values in brackets):')
for p in P:
    geo = eff(lambda s: 0.5 * ((L('DXTRU', s, p) - L('BOTHTRU', s, p)) + (L('DSREP', s, p) - L('GEOFRZ', s, p))))
    ded = eff(lambda s: 0.5 * ((L('GEOFRZ', s, p) - L('BOTHTRU', s, p)) + (L('DSREP', s, p) - L('DXTRU', s, p))))
    inter = eff(lambda s: (L('DSREP', s, p) - L('DXTRU', s, p)) - (L('GEOFRZ', s, p) - L('BOTHTRU', s, p)))
    ded_at_truth_geo = eff(lambda s: L('GEOFRZ', s, p) - L('BOTHTRU', s, p))
    ded_at_fit_geo = eff(lambda s: L('DSREP', s, p) - L('DXTRU', s, p))
    out[p] = dict(geometry=geo[:2], dedx=ded[:2], interaction=inter[:2],
                  dedx_given_true_geo=ded_at_truth_geo[:2], dedx_given_fitted_geo=ded_at_fit_geo[:2])
    fmt = lambda e: f"{e[0]:+6.2f} +- {e[1]:.2f} [{', '.join(f'{x:+.1f}' for x in e[2])}]"
    print(f"  {p:10s} geometry-fit {fmt(geo)}   dEdx-fit {fmt(ded)}   interaction {fmt(inter)}")
    print(f"  {'':10s} dEdx-fit | geometry at truth {fmt(ded_at_truth_geo)}   | geometry fitted {fmt(ded_at_fit_geo)}")
json.dump({f'{a}_{s}': r for (a, s), r in res.items()} | {'effects': out}, open(f'{OUT}/result_2x2.json', 'w'), indent=1, default=float)

# figure: lifetime and long_diff per arm, samples connected
order = ['BOTHTRU', 'GEOFRZ', 'DXTRU', 'DSREP']
xl = ['geometry truth\ndE/dx truth', 'geometry truth\ndE/dx fitted', 'geometry fitted\ndE/dx truth', 'geometry fitted\ndE/dx fitted']
col = {3: '#2a78d6', 7: '#eb6834', 8: '#1baf7a'}
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for k, p in enumerate(['lifetime', 'long_diff', 'Ab']):
    for s in SEEDS:
        ax[k].plot(range(4), [L(a, s, p) for a in order], 'o-', color=col[s], lw=1.5, ms=6, label=f'sample {s}')
    m = [np.mean([L(a, s, p) for s in SEEDS]) for a in order]
    ax[k].plot(range(4), m, 's', color='#0b0b0b', ms=9, mfc='none', mew=2, label='mean')
    ax[k].axhline(0, color='#52514e', lw=0.8, ls='--')
    ax[k].set_xticks(range(4)); ax[k].set_xticklabels(xl, fontsize=8)
    ax[k].set_ylabel(f'{p} error (%)'); ax[k].spines[['top', 'right']].set_visible(False)
    ax[k].grid(axis='y', color='#e5e4df', lw=0.8); ax[k].set_axisbelow(True)
    ax[k].set_title(f'({"abc"[k]}) {p}', loc='left', fontsize=10)
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle('Production fit, 2×2: which block, when fitted, creates the lifetime bias? (paired on samples 3/7/8)', fontsize=10, x=0.01, ha='left')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_2x2.png', dpi=150)
print('figure written')
