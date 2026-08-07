"""Fig 47 (study 2, detail): every loss component at every evaluated point.

Fig 46 collapsed the objective into LIKELIHOOD and PRIORS, which is the split that answers the
question. This figure keeps all six non-zero terms separate, because two of them behave
differently from the group they were lumped into:

  * `no_match_penalty` (targets with no simulated hit) falls 239.5 -> 110.5 when geometry alone
    moves to truth -- a 54% drop. It is a likelihood term and it carries a third of the geometry
    signal, so quoting only the hit log-intensity understates what geometry buys.
  * `dedx_mean_penalty` RISES at truth (7.0 -> 11.3). Truth's length-weighted mean dEdx is 1.8851
    against the fit's 1.8786 and a constraint target of 1.887 -- so truth is CLOSER to the target
    and still pays more, because the penalty is quadratic in a quantity the fit has been pulled to
    match exactly. It is small either way (+2.6 out of +565).

Four terms are identically zero throughout and are dropped: dedx_barrier, chain_drift_penalty,
dedx_drift_penalty, spatial_moment. `log_likelihood_charge` is also zero, but that one is a
reporting stub -- the charge residual lives inside `log_likelihood_tick` (see fig 44).

Medians throughout, because the per-batch distribution is skewed: mean total delta is -518 against
a median of +231, driven by a minority of batches where truth wins by 20-45%.
"""
import glob, pickle, re, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
RUN = 'fit_result/sci_full_TP'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})

POINTS = ['P0_fit', 'P1_truth', 'P2_calib_true', 'P3_dedx_true', 'P4_geom_true']
PLBL = {'P0_fit': 'fit', 'P1_truth': 'truth\n(all blocks)', 'P2_calib_true': 'calib\n→ truth',
        'P3_dedx_true': 'dE/dx\n→ truth', 'P4_geom_true': 'geometry\n→ truth'}
# (key, label, colour, is_likelihood)
TERMS = [('log_likelihood_tick', 'joint hit log-intensity', C['blue'], True),
         ('no_match_penalty', 'unmatched-target penalty', C['red'], True),
         ('expected_total_hits', 'expected hit count', C['green'], True),
         ('dedx_prior', 'dE/dx student-t prior', C['orange'], False),
         ('dedx_mean_penalty', 'dE/dx mean constraint', C['grey'], False),
         ('mcs_prior', 'MCS prior', C['purple'], False)]
TK = [t[0] for t in TERMS]

rows = []
for f in sorted(glob.glob(RUN + '/history_iter*.pkl')):
    seed = int(re.search(r'seed(\d+)', f).group(1))
    for rec in pickle.load(open(f, 'rb'))['truthpoint']:
        d = {'seed': seed}
        for p in POINTS:
            a = rec['points'][p]['aux']
            d[p] = rec['points'][p]['loss']
            for t in TK:
                d[f'{p}|{t}'] = float(a.get(t, 0.0))
        rows.append(d)
g = lambda k: np.array([r[k] for r in rows], float)
S = sorted({r['seed'] for r in rows})
n = len(rows)
print(f'{n} batch evaluations, {len(S)} seeds')

fig, ax = plt.subplots(2, 3, figsize=(16.6, 9.2))

# ---------------- (a) composition at each point ----------------
a = ax[0, 0]
xs = np.arange(len(POINTS))
bw = .8 / len(TERMS)
for k, (t, lab, col, _) in enumerate(TERMS):
    v = [np.median(g(f'{p}|{t}')) for p in POINTS]
    a.bar(xs + (k - (len(TERMS) - 1) / 2) * bw, v, width=bw * .9, color=col,
          edgecolor='white', linewidth=.4, label=lab)
a.axhline(0, color=C['ink'], lw=1.4)
a.set_xticks(xs); a.set_xticklabels([PLBL[p] for p in POINTS], fontsize=7.6)
a.set_ylabel('median term value')
a.set_title('(a) what the objective is made of at each point.\n'
            'The dE/dx prior is the only NEGATIVE term.', fontsize=9.3)
a.legend(fontsize=6.8, frameon=False, ncol=1, loc='upper right')

# ---------------- (b) delta heatmap ----------------
a = ax[0, 1]
M = np.array([[np.median(g(f'{p}|{t}') - g(f'P0_fit|{t}')) for p in POINTS[1:]] for t in TK])
lim = np.abs(M).max()
im = a.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
a.set_xticks(range(len(POINTS) - 1))
a.set_xticklabels([PLBL[p] for p in POINTS[1:]], fontsize=7.6)
a.set_yticks(range(len(TERMS)))
a.set_yticklabels([t[1] for t in TERMS], fontsize=7.6)
for r_ in range(M.shape[0]):
    for c_ in range(M.shape[1]):
        v = M[r_, c_]
        a.text(c_, r_, f'{v:+.0f}', ha='center', va='center', fontsize=7.6,
               color='white' if abs(v) > .55 * lim else C['ink'])
a.grid(False)
fig.colorbar(im, ax=a, label='median Δ vs the fit point')
a.set_title('(b) the full Δ matrix (PER-TERM medians; they do not\nsum to the median total — the spread is skewed)',
            fontsize=9.3)

# ---------------- (c) waterfall on the median batch ----------------
a = ax[0, 2]
tot_d = g('P1_truth') - g('P0_fit')
j = int(np.argsort(tot_d)[len(tot_d) // 2])          # the batch AT the median total delta
contrib = [(lab, g(f'P1_truth|{t}')[j] - g(f'P0_fit|{t}')[j], col) for t, lab, col, _ in TERMS]
contrib.sort(key=lambda z: -abs(z[1]))
run = g('P0_fit')[j]
a.axhline(run, color=C['grey'], ls=':', lw=1.4)
for k, (lab, dv, col) in enumerate(contrib):
    a.bar(k, dv, bottom=run, color=col, edgecolor='white', width=.66)
    a.annotate(f'{dv:+.0f}', (k, run + dv), textcoords='offset points',
               xytext=(0, 5 if dv >= 0 else -12), ha='center', fontsize=7.4)
    run += dv
a.axhline(run, color=C['red'], ls='--', lw=1.6)
a.set_xticks(range(len(contrib)))
a.set_xticklabels([c[0].replace(' ', '\n', 1) for c in contrib], fontsize=6.6, rotation=25, ha='right')
a.set_ylabel('running total loss')
_lo, _hi = g('P0_fit')[j], run
_span = max(abs(x[1]) for x in contrib)
a.set_ylim(min(_lo, _hi) - 1.35 * _span, max(_lo, _hi) + 1.35 * _span)
a.set_title('(c) fit → truth on the MEDIAN batch, term by term\n'
            f'({_lo:.0f} → {_hi:.0f}; exact for this batch — medians do NOT sum)', fontsize=9.3)

# ---------------- (d) spread, not just the median ----------------
a = ax[1, 0]
data = [g(f'P1_truth|{t}') - g(f'P0_fit|{t}') for t in TK]
bp = a.boxplot(data, vert=False, widths=.62, showfliers=False, patch_artist=True,
               medianprops=dict(color=C['ink'], lw=1.8))
for patch, (_, _, col, _) in zip(bp['boxes'], TERMS):
    patch.set_facecolor(col); patch.set_alpha(.75); patch.set_edgecolor('white')
a.axvline(0, color=C['ink'], lw=1.5)
a.set_yticks(range(1, len(TERMS) + 1))
a.set_yticklabels([t[1] for t in TERMS], fontsize=7.6)
a.set_xscale('symlog', linthresh=10)
a.set_xlabel('Δ (truth − fit) per batch   [symlog, linear within ±10]')
a.set_title('(d) per-batch spread. The dE/dx prior charge is\npositive in EVERY batch; the data term is not.',
            fontsize=9.3)

# ---------------- (e) the trade-off, batch by batch ----------------
a = ax[1, 1]
dl = sum(g(f'P1_truth|{t}') - g(f'P0_fit|{t}') for t, _, _, isl in TERMS if isl)
dp = sum(g(f'P1_truth|{t}') - g(f'P0_fit|{t}') for t, _, _, isl in TERMS if not isl)
win = (dl + dp) < 0
a.scatter(dl[~win], dp[~win], s=13, color=C['red'], alpha=.55, label=f'fit wins  ({(~win).sum()})')
a.scatter(dl[win], dp[win], s=13, color=C['blue'], alpha=.55, label=f'truth wins  ({win.sum()})')
xl = np.linspace(min(dl.min(), -3000), 500, 50)
a.plot(xl, -xl, color=C['ink'], lw=1.6, ls='--', label='total Δ = 0')
a.axvline(0, color=C['grey'], lw=1.0); a.axhline(0, color=C['grey'], lw=1.0)
a.set_xscale('symlog', linthresh=100); a.set_yscale('symlog', linthresh=100)
a.set_xlabel('Δ likelihood (truth − fit)   [symlog]')
a.set_ylabel('Δ priors (truth − fit)   [symlog]')
a.set_title('(e) every batch sits ABOVE the axis (priors always\nprefer the fit) and mostly LEFT of it (data prefers truth)',
            fontsize=9.3)
a.legend(fontsize=7.4, frameon=False, loc='upper left')

# ---------------- (f) per-seed consistency ----------------
a = ax[1, 2]
xs = np.arange(len(TERMS))
for si, s in enumerate(S):
    m = g('seed') == s
    v = [np.median((g(f'P1_truth|{t}') - g(f'P0_fit|{t}'))[m]) for t in TK]
    a.scatter(xs + (si - len(S) / 2) * .085, v, s=26, color=C['grey'], alpha=.75, zorder=2)
allm = [np.median(g(f'P1_truth|{t}') - g(f'P0_fit|{t}')) for t in TK]
a.scatter(xs, allm, s=95, marker='_', color=C['red'], zorder=4, linewidths=2.6,
          label='all seeds pooled')
a.axhline(0, color=C['ink'], lw=1.4)
a.set_yscale('symlog', linthresh=10)
a.set_xticks(xs)
a.set_xticklabels([t[1].replace(' ', '\n', 1) for t in TERMS], fontsize=6.6, rotation=25, ha='right')
a.set_ylabel('median Δ (truth − fit)')
a.set_title(f'(f) every term has the same sign in all {len(S)} seeds\n(grey = per-seed medians)',
            fontsize=9.3)
a.legend(fontsize=7.4, frameon=False)

fig.suptitle('Fig 47 — study 2 in detail: all six non-zero loss components at all five evaluated '
             'points.\nThe unmatched-target penalty carries a third of the geometry signal; the '
             'dE/dx prior charge is positive in 600/600 batches.', fontsize=10.4)
fig.tight_layout(rect=[0, 0, 1, .92])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig47_truthpoint_components.png'); plt.close(fig)
print('wrote', f'{OUT}/fig47_truthpoint_components.png')

print('\nmedian term values')
print(f'  {"term":26s}' + ''.join(f'{PLBL[p].replace(chr(10)," "):>16s}' for p in POINTS))
for t, lab, _, _ in TERMS:
    print(f'  {lab:26s}' + ''.join(f'{np.median(g(f"{p}|{t}")):>16.1f}' for p in POINTS))
print('\ngeometry -> truth, likelihood terms only')
for t, lab, _, isl in TERMS:
    if isl:
        d_ = np.median(g(f'P4_geom_true|{t}') - g(f'P0_fit|{t}'))
        print(f'  {lab:26s} {d_:+9.1f}   (better in '
              f'{100*((g(f"P4_geom_true|{t}") - g(f"P0_fit|{t}")) < 0).mean():.0f}% of batches)')
print('\nsign consistency of each term (truth - fit)')
for t, lab, _, _ in TERMS:
    d_ = g(f'P1_truth|{t}') - g(f'P0_fit|{t}')
    print(f'  {lab:26s} positive in {100*(d_>0).mean():5.1f}% of batches')
