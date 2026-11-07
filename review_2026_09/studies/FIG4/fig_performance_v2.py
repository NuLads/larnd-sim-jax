"""Fig. performance, version 2 (19 Oct 2026, review R023: "fwd and fwd+grad on the right plot; maybe more ranges;
PyTorch forward is weird"). Reads PAPERJOBS/bench2_{prob,stoch}_{len}_{fwd,grad}.json (paper_bench2.py, one A100,
forward alone and forward + gradient in separate processes, batch-length settings 25 cm to 10 m).
  (a) time per evaluation against the batch track length, forward (open, dashed fit) and forward + gradient (filled,
      solid fit), probabilistic and stochastic, with the references: PyTorch forward + gradient (Ref. Gasiorowski 2023,
      1 m, 25 s) and the reference larnd-sim forward (CUDA, measured here). The former PyTorch forward line (drawn from a
      quoted rate, not a measurement) is dropped.
  (b) peak GPU memory against the length of the longest batch of the run, same four series, PyTorch (checkpointed,
      14 GB at 1 m), A100 limit.
usage: python fig_performance_v2.py [out_stem]"""
import sys, json, glob, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
B = '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS/bench2_run2/'
OUT = sys.argv[1] if len(sys.argv) > 1 else '/sdf/scratch/neutrino/pgranger/review_2026_09/FIG4/performance_v2'
C = dict(prob='#0072B2', stoch='#D55E00'); MK = dict(prob='o', stoch='^'); A100_GB = 40.0
PT_GRAD = (1.0, 25.0); PT_MEM = (1.0, 14.0)
REF_FWD = np.array([[2.16, 4.40], [4.38, 4.91], [21.61, 12.89]])

def load(mode, part):
    rows = []
    for f in sorted(glob.glob(f'{B}bench2_{mode}_*_{part}.json')):
        d = json.load(open(f))
        rows.append(d)
    return sorted(rows, key=lambda d: d['max_batch_len'])

num = {}
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.3, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6, 'font.family': 'sans-serif'})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.75))
codes = set()
for mode in ('prob', 'stoch'):
    for part, filled, ls in (('fwd', False, '--'), ('grad', True, '-')):
        rows = load(mode, part)
        if not rows: continue
        codes |= {(d['git'], bool(d['code_dirty'])) for d in rows}
        L = np.array([bb['length_cm'] / 100 for d in rows for bb in d['batches']])
        T = np.array([bb['time_s'] for d in rows for bb in d['batches']])
        m, c = np.polyfit(L, T, 1)
        kw = dict(s=14, marker=MK[mode], lw=0.7, zorder=3, edgecolor=C[mode], facecolor=C[mode] if filled else 'none')
        a.scatter(L, T, **kw)
        xx = np.logspace(np.log10(L.min()), np.log10(L.max()), 60); a.plot(xx, m * xx + c, ls, color=C[mode], lw=0.8)
        Lmax = np.array([max(bb['length_cm'] for bb in d['batches']) / 100 for d in rows]); P = np.array([d['peak_bytes'] / 1e9 for d in rows])
        b.plot(Lmax, P, ls, marker=MK[mode], color=C[mode], ms=3.5, mfc=C[mode] if filled else 'white', lw=1.0)
        num[f'{mode}_{part}'] = dict(slope_s_per_m=float(m), intercept_s=float(c), lengths_m=[round(x, 2) for x in L],
                                     settings_cm=[d['max_batch_len'] for d in rows], peak_gb=[round(x, 2) for x in P],
                                     longest_m=[round(x, 2) for x in Lmax])
a.scatter(*PT_GRAD, marker='s', s=22, color='k', zorder=4)
a.scatter(REF_FWD[:, 0], REF_FWD[:, 1], marker='D', s=16, facecolor='none', edgecolor='k', lw=0.8, zorder=4)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('batch track length (m)'); a.set_ylabel('time per evaluation (s)')
a.set_xlim(0.15, 40); a.set_ylim(0.04, 80); a.grid(alpha=0.25, lw=0.5)
a.text(0.03, 0.97, '(a)', transform=a.transAxes, va='top', fontsize=8)
b.scatter(*PT_MEM, marker='s', s=22, color='k', zorder=4)
b.axhline(A100_GB, color='k', ls=':', lw=0.9); b.text(0.97, A100_GB, 'A100 (40 GB)', transform=b.get_yaxis_transform(), ha='right', va='bottom', fontsize=6.3)
b.set_xlabel('longest batch of the run (m)'); b.set_ylabel('peak GPU memory (GB)'); b.set_ylim(0, 44)
b.set_xlim(0, 11); b.grid(alpha=0.25, lw=0.5)
b.annotate('10 m with gradient:\nexceeds 40 GB', xy=(10.47, 38.5), xytext=(9.6, 31.5), ha='center', fontsize=6.0, color='0.3',
           arrowprops=dict(arrowstyle='->', lw=0.6, color='0.3'))
b.text(0.03, 0.97, '(b)', transform=b.transAxes, va='top', fontsize=8)
for ax in (a,):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
handles = [Line2D([], [], color=C['prob'], marker='o', ls='-', ms=3.5, label='probabilistic'),
           Line2D([], [], color=C['stoch'], marker='^', ls='-', ms=3.5, label='stochastic'),
           Line2D([], [], color='0.4', marker='o', mfc='white', ls='--', ms=3.5, label='forward'),
           Line2D([], [], color='0.4', marker='o', ls='-', ms=3.5, label='forward + gradient'),
           Line2D([], [], color='k', marker='s', ls='none', ms=4, label='PyTorch, fwd + grad'),
           Line2D([], [], color='k', marker='D', mfc='none', ls='none', ms=3.5, label='larnd-sim, fwd (CUDA)')]
fig.legend(handles=handles, loc='upper center', ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.0,
           handletextpad=0.4)
fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.5)
fig.savefig(OUT + '.pdf'); fig.savefig(OUT + '.png', dpi=200)
num['code'] = sorted(f'{g}{" dirty" if d else ""}' for g, d in codes)
json.dump(num, open(OUT + '_numbers.json', 'w'), indent=1); print(json.dumps(num, indent=1)); print('wrote', OUT)
