"""Fig 73 (reworked for legibility) — the data-replica ensemble, one panel per conclusion.

DESIGN: every parameter panel uses the SAME grammar, so any one of them reads in seconds --
  green line   = truth (zero error)
  blue dots    = the 8 replicas (independent track samples, truth pinned)
  black marker = ensemble mean, with the +-SE band shaded
  red star     = the campaign's single fixed sample (what every previous result used)
Bias is real when the black band sits off the green line; the red star's position vs the blue
cloud says whether the campaign's sample was representative. Linear axes throughout (the first
version's symlog forest plot buried the story).

Study: truth pinned at the seed-0 draw; ONLY the track sample varies (data_seed 1-8, 600 random
events each, 0.33% pairwise overlap). Enabled by fixing data_seed, a silent no-op campaign-wide.
"""
import glob, os, pickle, io, contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with contextlib.redirect_stdout(io.StringIO()):
    import robust_convergence as R

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00', grey='#888888',
         ink='#222222')
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .22,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
P = R.P
FLOOR = {'Ab': 0.048, 'eField': 0.0005, 'tran_diff': 0.45, 'long_diff': 0.9, 'lifetime': 1.0}
W = 200


def sm(v, w=W):
    v = np.asarray(v, float)
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1) if len(v) > w else v


def read(path):
    h = pickle.load(open(path, 'rb'))
    return {p: 100 * (np.median(sm(np.ravel(h[p + '_iter'])[1:])[-300:])
                      / float(np.ravel(h[p + '_target'])[0]) - 1) for p in P}


REP = {ds: read(glob.glob(f'fit_result/sci_full_DSREP{ds}/history_iter10000_*.pkl')[0])
       for ds in range(1, 9)}
OLD = read(glob.glob('fit_result/sci_full_BT2/history_iter10000_*seed0.pkl')[0])

PANELS = [
    ('lifetime', 'LIFETIME — the bad news', 
     'REAL bias: all 8 replicas land ABOVE truth (+8.6 ± 1.6%).\nThe campaign’s sample was a lucky outlier (p ≈ 1%).'),
    ('Ab', 'A$_b$ — the good news',
     'The famous “+0.27% bias” EVAPORATES: replicas centre on truth.\nIt was the campaign’s sample all along (red star, 3.5σ off).'),
    ('long_diff', 'LONGITUDINAL DIFFUSION',
     'REAL bias, and big: all 8 replicas above truth (+14.0 ± 3.4%).'),
    ('tran_diff', 'TRANSVERSE DIFFUSION',
     'Suggestive only (+1.1 ± 0.5%, 2.2σ) — not established.'),
    ('eField', 'E FIELD',
     'A real bias, but tiny: −0.036 ± 0.006% (5.9σ).'),
]

fig = plt.figure(figsize=(16.4, 9.6))
gs = fig.add_gridspec(2, 3, hspace=.52, wspace=.16)
rng = np.random.default_rng(3)

for k, (p, name, verdict) in enumerate(PANELS):
    a = fig.add_subplot(gs[k // 3, k % 3])
    v = np.array([REP[d][p] for d in range(1, 9)])
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(8)
    jit = rng.uniform(-.16, .16, 8)
    a.axvline(0, color=C['green'], lw=3, zorder=1)
    a.axvspan(m - se, m + se, ymin=.25, ymax=.75, color=C['ink'], alpha=.14, zorder=1)
    a.plot(v, jit, 'o', ms=11, color=C['blue'], alpha=.75, zorder=3,
           label='8 replicas (independent samples)' if k == 0 else None)
    a.plot(m, 0, 'D', ms=13, color=C['ink'], zorder=4,
           label='ensemble mean ± SE' if k == 0 else None)
    a.plot(OLD[p], 0, '*', ms=22, color=C['red'], zorder=5,
           label='the campaign’s single sample' if k == 0 else None)
    lo = min(v.min(), OLD[p], 0); hi = max(v.max(), OLD[p], 0)
    pad = .18 * (hi - lo)
    a.set_xlim(lo - pad, hi + pad)
    a.set_ylim(-.6, .6); a.set_yticks([])
    a.annotate('truth', (0, .52), fontsize=9, color=C['green'], ha='center', fontweight='bold')
    a.set_xlabel(f'{name.split(" — ")[0].title() if "—" in name else name.title()} error at the pinned truth (%)',
                 fontsize=9)
    a.set_title(f'{name}\n{verdict}', fontsize=9.6)
    if k == 0:
        a.legend(fontsize=8.2, frameon=True, framealpha=.95, loc='lower right')

# ---------------- (f) the budget ----------------
a = fig.add_subplot(gs[1, 2])
labs = ['lifetime', 'long. diff.', 'tran. diff.', 'A$_b$']
keys = ['lifetime', 'long_diff', 'tran_diff', 'Ab']
y = np.arange(len(keys))
fl = [FLOOR[k] for k in keys]
sp = [np.array([REP[d][k] for d in range(1, 9)]).std(ddof=1) for k in keys]
a.barh(y - .19, fl, height=.38, color=C['green'], edgecolor='white',
       label='floor for ONE sample (fig 72)')
a.barh(y + .19, sp, height=.38, color=C['orange'], edgecolor='white',
       label='scatter BETWEEN samples (this study)')
for i in range(len(keys)):
    a.text(sp[i] + .25, y[i] + .19, f'×{sp[i]/fl[i]:.1f} larger', va='center', fontsize=9,
           color=C['orange'])
a.set_yticks(y); a.set_yticklabels(labs, fontsize=10); a.invert_yaxis()
a.set_xlim(0, max(sp) * 1.45)
a.set_xlabel('% of the fitted value')
a.set_title('THE ERROR BUDGET\nWhich tracks you drew matters MORE than\nthe statistics of the tracks you have.',
            fontsize=9.6)
a.legend(fontsize=8.2, frameon=False, loc='lower right')

fig.suptitle(
    'Fig 73 — same truth, eight different input datasets: which biases are real?\n'
    'green = truth  |  blue = 8 independent track samples  |  black ◆ = their mean ± SE  |  '
    'red ★ = the single sample every previous result used',
    fontsize=11.6, y=.995)
fig.tight_layout(rect=[0, 0, 1, .915])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig73_replica_ensemble.png'); plt.close(fig)
print('wrote', f'{OUT}/fig73_replica_ensemble.png')
