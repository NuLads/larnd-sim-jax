"""Fig 42: every completed run of the production configuration — what it achieves, and why the
error bars are what they are.

THE SUBTLETY THIS FIGURE EXISTS TO SHOW. Three directories hold the SAME configuration:
`sci_full_ANNEALLONG`, `sci_full_ANNEALLONG2` and `sci_full_MDX` (the last is a no-op variant --
its minimum-length cut never fires on the guess file). Seeds 0/1/2 therefore appear three times
each, and `--seed` draws the TARGET, so those are repeats of one experiment rather than new
samples. Pooling them as independent understates the uncertainty.

Splitting the two sources is the point:
  * WITHIN a seed  -- identical target, identical config, only GPU non-determinism
                      (`--non_deterministic` is on in every production run).
  * BETWEEN seeds  -- different target draws, spanning lifetime 984-4901 us, i.e. very different
                      problems, not just different noise.

Panel (d) checks whether the extreme targets are what drives the spread.
"""
import glob, os, re, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}
DIRS = [('sci_full_ANNEALLONG', 'set A', C['blue']),
        ('sci_full_ANNEALLONG2', 'set B', C['orange']),
        ('sci_full_MDX', 'set C', C['green'])]

runs = []
for d, lab, col in DIRS:
    for f in sorted(glob.glob(f'fit_result/{d}/history_iter*.pkl')):
        if int(re.search(r'iter(\d+)', f).group(1)) != 10000:
            continue                                    # completed runs only
        s = int(re.search(r'seed(\d+)', f).group(1))
        h = pickle.load(open(f, 'rb'))
        e = {}
        for p in P:
            v = np.ravel(h[p + '_iter']); t = float(np.ravel(h[p + '_target'])[0])
            e[p] = 100 * (np.median(v[int(len(v) * .8):]) / t - 1)
        pr = np.ravel(h['pos_residual_iter']) * 1e4
        e['pos'] = float(np.median(pr[int(len(pr) * .8):]))
        e['tau_tgt'] = float(np.ravel(h['lifetime_target'])[0])
        runs.append(dict(seed=s, set=lab, col=col, **e))
print(f'{len(runs)} completed runs, {len(set(r["seed"] for r in runs))} distinct seeds')

seeds = sorted({r['seed'] for r in runs})
bys = {s: [r for r in runs if r['seed'] == s] for s in seeds}
permean = {p: np.array([np.mean([r[p] for r in bys[s]]) for s in seeds]) for p in P + ['pos']}

fig, ax = plt.subplots(2, 2, figsize=(11.6, 8.0))

# (a) lifetime per seed, repeats visible
a = ax[0, 0]
_seen = set()
for k, s in enumerate(seeds):
    for r in bys[s]:
        lab = r['set'] if r['set'] not in _seen else None
        _seen.add(r['set'])
        a.scatter([k], [r['lifetime']], color=r['col'], s=46, zorder=3, label=lab)
    m = np.mean([r['lifetime'] for r in bys[s]])
    a.plot([k - .3, k + .3], [m, m], color=C['ink'], lw=2, zorder=4)
    a.annotate(f"τ={bys[s][0]['tau_tgt']:.0f}", (k, a.get_ylim()[0]), fontsize=6.4,
               ha='center', va='bottom', color=C['grey'], rotation=90)
h_, l_ = a.get_legend_handles_labels()
a.legend(h_, l_, fontsize=7.4, frameon=False, title='repeat set', title_fontsize=7.4)
a.axhline(0, color=C['grey'], lw=1.2)
a.set_xticks(range(len(seeds))); a.set_xticklabels([f'seed {s}' for s in seeds], fontsize=7.4)
a.set_ylabel('lifetime error (%)')
a.set_title('(a) seeds 0–2 were run THREE times each (same target).\n'
            'Bars = per-seed mean; the scatter within a seed is pure non-determinism', fontsize=9.2)

# (b) all parameters, per-seed means
a = ax[0, 1]
xs = np.arange(len(P))
for k, s in enumerate(seeds):
    a.scatter(xs + (k - len(seeds) / 2) * 0.055, [np.mean([r[p] for r in bys[s]]) for p in P],
              s=26, color=C['grey'], alpha=.65, zorder=2)
mu = [permean[p].mean() for p in P]
se = [permean[p].std(ddof=1) / np.sqrt(len(seeds)) for p in P]
a.errorbar(xs, mu, yerr=se, lw=0, elinewidth=2.5, marker='o', ms=9, capsize=5,
           color=C['red'], zorder=5, label=f'mean ± s.e.m. (n = {len(seeds)} seeds)')
a.axhline(0, color=C['grey'], lw=1.2)
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in P], fontsize=8, rotation=15, ha='right')
a.set_ylabel('error vs truth (%)')
a.set_title('(b) production performance\ngrey = per-seed means', fontsize=9.2)
a.legend(fontsize=7.6, frameon=False)

# (c) variance decomposition
a = ax[1, 0]
within = []
for p in P:
    w = [np.std([r[p] for r in bys[s]], ddof=1) for s in seeds if len(bys[s]) > 1]
    within.append(np.mean(w) if w else np.nan)
between = [permean[p].std(ddof=1) for p in P]
a.bar(xs - .18, within, width=.34, color=C['purple'], edgecolor='white',
      label='WITHIN seed (run-to-run, same target)')
a.bar(xs + .18, between, width=.34, color=C['ink'], edgecolor='white',
      label='BETWEEN seeds (different targets)')
a.set_yscale('log')
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in P], fontsize=8, rotation=15, ha='right')
a.set_ylabel('s.d. (%)')
a.set_title('(c) two sources of scatter — repeats alone move\nlifetime by ±2.2 points', fontsize=9.2)
a.legend(fontsize=7.4, frameon=False)

# (d) is it the extreme targets?
a = ax[1, 1]
for s in seeds:
    t = bys[s][0]['tau_tgt']
    a.scatter([t] * len(bys[s]), [r['lifetime'] for r in bys[s]], color=C['blue'], s=40, zorder=3)
    a.plot([t, t], [min(r['lifetime'] for r in bys[s]), max(r['lifetime'] for r in bys[s])],
           color=C['blue'], lw=1, alpha=.5)
a.axvline(2200, color=C['grey'], ls='--', lw=1.4)
a.text(2200, a.get_ylim()[1], ' nominal 2200 µs', fontsize=7.4, color=C['grey'], va='top')
a.axhline(0, color=C['grey'], lw=1.2)
a.set_xlabel('this seed\'s TARGET lifetime (µs)'); a.set_ylabel('lifetime error (%)')
a.set_title('(d) targets span 984–4901 µs, so seeds are\nnot equally hard problems', fontsize=9.2)

fig.suptitle('Fig 42 — every completed run of the production configuration. Seeds 0–2 are repeats '
             'of one experiment,\nnot independent samples: pooling them as independent overstates '
             'the precision.', fontsize=10.2)
fig.tight_layout(rect=[0, 0, 1, .93])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig42_production_runs.png'); plt.close(fig)
print('wrote', f'{OUT}/fig42_production_runs.png')
for p in P + ['pos']:
    v = permean[p]
    print(f'  {p:11s} {v.mean():+8.2f} ± {v.std(ddof=1)/np.sqrt(len(v)):5.2f}'
          f'   between-seed sd {v.std(ddof=1):5.2f}')
