"""Fig 49 (study 1): do the three learning-rate schedules matter?

Five ceiling arms x 6 seeds, 10 000 iterations, differing ONLY in a schedule:

  S1A0  calibration decay 0.91/epoch, dE/dx LR constant      (baseline)
  S1A1  calibration decay 0.80        (faster anneal, LESS total travel)
  S1A2  calibration decay 0.97        (slower anneal, MORE total travel)
  S1A3  dE/dx LR annealed to 0.1x  by the end of the run
  S1A4  dE/dx LR annealed to 0.01x by the end of the run

EVERY COMPARISON IS PAIRED. All five arms use the same six seeds, and --seed draws the TARGET, so
seed s poses the identical problem in every arm. Differencing within a seed removes the
target-to-target variation that dominates the raw s.d. -- at ceiling that variation is 4.1 points
on lifetime, which is larger than any effect here. Unpaired, this study resolves almost nothing;
paired, it resolves A_b and longitudinal diffusion cleanly.

THE PHYSICAL AXIS IS TOTAL OPTIMISER TRAVEL, sum of LR over the run, not decay_rate. Lifetime has
tracked travel across this campaign, so panel (b) plots against travel and the three calibration
arms span a 4.5x range of it.

Two arms (S1A5 dE/dx hard-freeze, and the 6th seed of some arms) were cancelled mid-campaign once
study 2 showed the dE/dx block is prior-limited rather than optimiser-limited. What survives still
brackets both axes.
"""
import glob, os, pickle, re, io, contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with contextlib.redirect_stdout(io.StringIO()):
    import robust_convergence as R

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}
# arm, label, calib decay, dEdx final fraction, colour
ARMS = [('S1A0', 'baseline\n0.91 / const', 0.91, None, C['ink']),
        ('S1A1', 'calib 0.80\n(less travel)', 0.80, None, C['blue']),
        ('S1A2', 'calib 0.97\n(more travel)', 0.97, None, C['green']),
        ('S1A3', 'dE/dx → 0.1×', 0.91, 0.1, C['orange']),
        ('S1A4', 'dE/dx → 0.01×', 0.91, 0.01, C['red'])]
ITERS, NB, WARM, PEAK = 10000, 100, 500, 0.1


def calib_lr(decay):
    """optax.warmup_exponential_decay_schedule: linear warmup to PEAK over WARM steps, then
    exponential decay indexed in EPOCHS (transition_steps = epoch_size, staircase)."""
    t = np.arange(ITERS)
    lr = np.where(t < WARM, PEAK * t / WARM,
                  PEAK * decay ** np.floor(np.maximum(t - WARM, 0) / NB))
    return lr


def dedx_lr(final_frac, base=1e-2):
    """LARND_DEDX_LR_FINAL_FRAC: the per-batch Adam state steps once per EPOCH, so the decay is
    per epoch and the rate is derived from the actual epoch count (see fit_params.py)."""
    t = np.arange(ITERS)
    if final_frac is None:
        return np.full(ITERS, base)
    nep = ITERS / NB
    rate = final_frac ** (1.0 / nep)
    return base * rate ** (t / NB)


def perseed(tag):
    out = {}
    for f in sorted(glob.glob(f'fit_result/sci_ceiling_{tag}/history_iter10000_*.pkl')):
        s = int(re.search(r'seed(\d+)', f).group(1))
        h = pickle.load(open(f, 'rb'))
        d = {}
        for p in R.P:
            v = np.ravel(h[p + '_iter']); t = float(np.ravel(h[p + '_target'])[0])
            d[p] = 100 * (np.asarray(v, float) / t - 1)          # full trace
            d[p + '_f'] = float(np.median(d[p][int(len(d[p]) * .8):]))
        d['tau'] = float(np.ravel(h['lifetime_target'])[0])
        a = np.asarray([x for x in h.get('dedx_mae_iter', []) if x == x], float)
        d['gap'] = 100 * (1 - a[-50:].mean() / a[:50].mean()) if len(a) > 200 else np.nan
        out[s] = d
    return out


A = {t: perseed(t) for t, *_ in ARMS}
SEEDS = sorted(set.intersection(*[set(v) for v in A.values()]))
TRAVEL = {t: calib_lr(dc).sum() for t, _, dc, _, _ in ARMS}
DTRAVEL = {t: dedx_lr(ff).sum() for t, _, _, ff, _ in ARMS}
print(f'{len(SEEDS)} paired seeds {SEEDS}')
for t, lab, dc, ff, _ in ARMS:
    print(f'  {t}: calib travel Sum(LR) = {TRAVEL[t]:7.1f}   dEdx travel = {DTRAVEL[t]:6.2f}')

fig, ax = plt.subplots(2, 4, figsize=(21.4, 8.8))
xs = np.arange(len(R.P))

# ---------------- (a) paired differences vs baseline ----------------
a = ax[0, 0]
for k, (t, lab, _, _, col) in enumerate(ARMS[1:]):
    m, e = [], []
    for p in R.P:
        d = np.array([A[t][s][p + '_f'] - A['S1A0'][s][p + '_f'] for s in SEEDS])
        m.append(d.mean()); e.append(d.std(ddof=1) / np.sqrt(len(d)))
    a.errorbar(xs + (k - 1.5) * .16, m, yerr=e, lw=0, elinewidth=2.2, marker='o', ms=6.5,
               capsize=4, color=col, label=lab.replace('\n', ' '))
a.axhline(0, color=C['ink'], lw=1.5)
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in R.P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('PAIRED difference vs baseline (points)')
a.set_yscale('symlog', linthresh=0.5)
a.set_title('(a) paired vs the baseline arm. Only A$_b$ and long.\n'
            'diffusion move resolvably; lifetime does not.', fontsize=9.3)
a.legend(fontsize=7.0, frameon=False)

# ---------------- (b) lifetime vs total calibration travel ----------------
a = ax[0, 1]
CAL = ['S1A1', 'S1A0', 'S1A2']
for t in CAL:
    col = dict(ARMS_=0) and [x[4] for x in ARMS if x[0] == t][0]
    a.scatter([TRAVEL[t]] * len(SEEDS), [A[t][s]['lifetime_f'] for s in SEEDS],
              s=26, color=col, alpha=.55, zorder=2)
mu = [np.mean([A[t][s]['lifetime_f'] for s in SEEDS]) for t in CAL]
se = [np.std([A[t][s]['lifetime_f'] for s in SEEDS], ddof=1) / np.sqrt(len(SEEDS)) for t in CAL]
a.errorbar([TRAVEL[t] for t in CAL], mu, yerr=se, lw=2.0, elinewidth=2.4, marker='o', ms=10,
           capsize=5, color=C['ink'], zorder=4, label='arm mean ± s.e.m.')
for t, m in zip(CAL, mu):
    a.annotate(f'  {t}', (TRAVEL[t], m), fontsize=7.6, va='center')
a.axhline(0, color=C['red'], lw=1.5, ls='--', label='truth')
a.set_xscale('log')
a.set_xlabel('total calibration travel,  $\\sum$LR over the run')
a.set_ylabel('lifetime error (%)')
a.set_title('(b) a 4.5× range of optimiser travel moves lifetime\n'
            'by less than its own error bar — it has ASYMPTOTED', fontsize=9.3)
a.legend(fontsize=7.4, frameon=False)

# ---------------- (c) the dE/dx null ----------------
a = ax[0, 2]
g = [[A[t][s]['gap'] for s in SEEDS] for t, *_ in ARMS]
bp = a.boxplot(g, widths=.6, patch_artist=True, showfliers=False,
               medianprops=dict(color=C['ink'], lw=1.8))
for patch, (_, _, _, _, col) in zip(bp['boxes'], ARMS):
    patch.set_facecolor(col); patch.set_alpha(.6); patch.set_edgecolor('white')
for k, (t, *_ ) in enumerate(ARMS):
    a.scatter([k + 1] * len(SEEDS), [A[t][s]['gap'] for s in SEEDS], s=16,
              color=C['grey'], alpha=.7, zorder=3)
a.set_xticks(range(1, len(ARMS) + 1))
a.set_xticklabels([x[1] for x in ARMS], fontsize=7.0)
a.set_ylabel('dE/dx: % of the prior→truth gap closed')
a.set_title('(c) THE dE/dx SCHEDULE IS A NULL: 45.9→47.4% across\n'
            'every arm. Study 2 predicted exactly this.', fontsize=9.3)

# ---------------- (d) the schedules themselves ----------------
a = ax[1, 0]
for t, lab, dc, ff, col in ARMS:
    a.plot(calib_lr(dc), color=col, lw=2.0,
           ls='-' if ff is None else ':', label=f'{t} calib (Σ={TRAVEL[t]:.0f})')
a.set_yscale('log')
a.set_xlabel('iteration'); a.set_ylabel('calibration learning rate')
a.set_title('(d) the calibration schedules. Decay is indexed in\nEPOCHS, so batch count changes it silently.',
            fontsize=9.3)
a.legend(fontsize=6.8, frameon=False)

# ---------------- (e) lifetime convergence traces ----------------
a = ax[1, 1]
W = 200
def smooth(v, w=W):
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(np.asarray(v, float), w), axis=1) if len(v) > w else v
for t, lab, _, _, col in ARMS:
    cs = [smooth(A[t][s]['lifetime'][1:]) for s in SEEDS]
    n = min(len(c) for c in cs)
    a.plot(np.median([c[:n] for c in cs], axis=0), color=col, lw=2.0, label=lab.replace('\n', ' '))
a.axhline(0, color=C['red'], lw=1.4, ls='--')
a.set_xlabel('iteration'); a.set_ylabel('lifetime error (%)')
a.set_title('(e) lifetime traces (across-seed median). The slow arm\n'
            'is still moving at 10 000 iterations.', fontsize=9.3)
a.legend(fontsize=7.0, frameon=False)

# ---------------- (f) is each arm converged? ----------------
a = ax[1, 2]
for k, (t, lab, _, _, col) in enumerate(ARMS):
    with contextlib.redirect_stdout(io.StringIO()):
        r = R.analyse('x', 'fit_result/sci_ceiling_' + t, True)
    dr = [np.mean(r['drifts'][p]) for p in R.P]
    a.scatter(xs + (k - 2) * .14, dr, s=42, color=col, label=lab.replace('\n', ' '), zorder=3)
a.axhline(0, color=C['ink'], lw=1.4)
a.axhspan(-0.3, 0.3, color=C['green'], alpha=.13)
a.text(len(R.P) - .6, 0.32, 'settled', fontsize=7.2, color=C['green'], ha='right')
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in R.P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('residual drift over the last 20% (points)')
a.set_title('(f) the convergence diagnostic. Only the SLOW arm has\n'
            'not settled — more travel buys motion, not accuracy.', fontsize=9.3)
a.legend(fontsize=6.8, frameon=False)


# ---------------- (g) the seed scatter is NOT noise: it is the per-seed lever arm ----------------
# Every seed draws its own target lifetime, and the detector depth is FIXED, so each seed's problem
# has a different amount of signal: the anode-to-cathode charge swing is 1 - exp(-t_drift/tau).
# Seed 3 (tau 4518) has a 4.15% swing; seed 1 (tau 1160) has 15.21%. Those are not equally hard
# problems, and pooling them as if they were is what produces the "irreducible 4-point seed s.d.".
a = ax[0, 3]
T_DRIFT = 0.087 * 2200.0
for t, lab, _, _, col in ARMS:
    sw = [100 * (1 - np.exp(-T_DRIFT / A[t][s]['tau'])) for s in SEEDS]
    a.scatter(sw, [abs(A[t][s]['lifetime_f']) for s in SEEDS], s=34, color=col, alpha=.75,
              label=lab.replace('\n', ' '), zorder=3)
allsw = np.array([100 * (1 - np.exp(-T_DRIFT / A[t][s]['tau'])) for t, *_ in ARMS for s in SEEDS])
allerr = np.array([abs(A[t][s]['lifetime_f']) for t, *_ in ARMS for s in SEEDS])
o = np.argsort(allsw)
a.plot(allsw[o], np.poly1d(np.polyfit(allsw, allerr, 1))(allsw[o]), color=C['ink'], lw=1.8, ls='--',
       label=f'r = {np.corrcoef(allsw, allerr)[0,1]:+.2f}')
a.axvline(8.3, color=C['grey'], lw=1.4)
a.annotate(' nominal\n 8.3%', (8.3, allerr.max() * .82), fontsize=7.2, color=C['grey'])
a.set_xlabel("this seed's anode-to-cathode charge swing (%)")
a.set_ylabel('|lifetime error| (%)')
a.set_title('(g) the seed scatter is LEVER ARM, not noise: seeds\n'
            'below the median swing are 3.9× worse', fontsize=9.3)
a.legend(fontsize=6.8, frameon=False)

ax[1, 3].axis('off')

fig.suptitle('Fig 49 — study 1: the schedules barely matter. The dE/dx schedule is a clean null '
             '(c), exactly as study 2 predicted\nfor a prior-limited block; a 4.5× range of '
             'calibration travel leaves lifetime inside its own error bar (b).', fontsize=10.3)
fig.tight_layout(rect=[0, 0, 1, .92])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig49_study1_schedules.png'); plt.close(fig)
print('wrote', f'{OUT}/fig49_study1_schedules.png')
for t, lab, *_ in ARMS:
    v = [A[t][s]['lifetime_f'] for s in SEEDS]
    print(f'  {t}: lifetime {np.mean(v):+6.2f} ± {np.std(v,ddof=1)/np.sqrt(len(v)):4.2f}   '
          f'gap {np.mean([A[t][s]["gap"] for s in SEEDS]):.1f}%   travel {TRAVEL[t]:.0f}')
