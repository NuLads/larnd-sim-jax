"""Fig. kscan (20 Oct 2026, review R022: "K plot to appendix"). Beam width K of the probabilistic electronics: hit
probability of the model minus the frequency in 500 stochastic throws of the same waveforms, averaged over the pixels
of two 400-cm batches of through-going muons, for K = 10, 25, 50, 100, 200. Data: PAPERJOBS/kscan.json
(paper_kscan.py, job 40213579, 8 Oct 2026), binned in sampled frequency; here combined over all bins and over the
intermediate bins 0.2-0.8 (weights = pixel counts). usage: python fig_kscan.py [out_stem]"""
import sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = sys.argv[1] if len(sys.argv) > 1 else '/sdf/scratch/neutrino/pgranger/review_2026_09/FIG4/kscan/kscan'
d = json.load(open('/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS/kscan.json'))
KS = [10, 25, 50, 100, 200]
def comb(rows, lo=None, hi=None):
    r = [x for x in rows if lo is None or (x[0] >= lo - 1e-9 and x[1] <= hi + 1e-9)]
    n = np.array([x[2] for x in r], float); m = np.array([x[3] for x in r]); se = np.array([x[4] for x in r]); w = n / n.sum()
    return (w * m).sum(), np.sqrt((w ** 2 * se ** 2).sum()), int(n.sum())
num = {}
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.3, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6, 'font.family': 'sans-serif'})
C0, C1 = '#0072B2', '#D55E00'
fig, ax = plt.subplots(figsize=(3.4, 2.3))
for s, col, lab in ((0, C0, 'muons: first hit'), (1, C1, 'muons: second hit')):
    a = [comb(d[f'K{K}_slot{s}']) for K in KS]
    ax.errorbar(KS, [x[0] for x in a], yerr=[x[1] for x in a], fmt='o-', color=col, ms=3.5, lw=1.1, capsize=1.5, label=lab)
    num[f'slot{s}_all'] = [(K, round(x[0], 4), round(x[1], 4), x[2]) for K, x in zip(KS, a)]
    if s == 1:
        b = [comb(d[f'K{K}_slot{s}'], 0.2, 0.8) for K in KS]
        ax.errorbar(KS, [x[0] for x in b], yerr=[x[1] for x in b], fmt='s--', color=col, ms=3, lw=0.9, mfc='white', capsize=1.5,
                    label='muons: second hit, probability 0.2–0.8')
        num['slot1_mid'] = [(K, round(x[0], 4), round(x[1], 4), x[2]) for K, x in zip(KS, b)]
# 20 Oct (review R101): stopping protons, one 167-cm batch, 1e4 throws (FIG4/proton/proton_K_numbers.json)
pk = json.load(open('/sdf/scratch/neutrino/pgranger/review_2026_09/FIG4/proton/proton_K_numbers.json'))['K']
KP = sorted(int(k) for k in pk)
ax.errorbar(KP, [pk[str(k)]['slot1_mean'] for k in KP], yerr=[pk[str(k)]['slot1_se'] for k in KP], fmt='^-', color='#009E73', ms=3.5,
            lw=1.1, capsize=1.5, label='protons: second hit')
num['proton_slot1'] = [(k, round(pk[str(k)]['slot1_mean'], 4)) for k in KP]
ax.axhline(0, color='k', lw=0.6); ax.axvline(100, color='0.4', ls=':', lw=0.9)
ax.text(100 * 1.06, -0.13, '$K=100$\n(used)', fontsize=6.3, color='0.35', va='bottom')
XT = [10, 25, 50, 100, 200, 400, 800]; ax.set_xscale('log'); ax.set_xticks(XT); ax.set_xticklabels([str(k) for k in XT]); ax.minorticks_off()
ax.set_xlabel('paths kept in the beam search, $K$'); ax.set_ylabel('hit probability:\nmodel $-$ sampled')
ax.set_ylim(-0.33, 0.03); ax.grid(alpha=0.25, lw=0.5)
ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), frameon=False)
fig.tight_layout(); fig.savefig(OUT + '.pdf'); fig.savefig(OUT + '.png', dpi=250)
json.dump(num, open(OUT + '_numbers.json', 'w'), indent=1); print(json.dumps(num))
