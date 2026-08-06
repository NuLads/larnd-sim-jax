"""Fig 45: where we stand — the best setup, what improves it, what does not, and the hard limits.

This is the summary figure. Everything in it is a re-read of arms already in the report; nothing
new is fitted. It exists so the campaign's bottom line can be checked against data in one place.

PRODUCTION = the S4 configuration: straight-line guess geometry, jointly fitting positions, ~4000
per-segment dE/dx and the five calibration parameters; 100 batches x 400 cm, 10 000 iterations,
annealed calibration LR (0.91/epoch), spline geometry basis, chain_lr 1e-2 / decay 0.9997, dE/dx
prior 5, mean constraint 1e5, noise ON.

SEEDS ARE POOLED PER SEED, NOT PER RUN. Three directories hold the identical configuration, so
seeds 0-2 appear three times each; `--seed` draws the TARGET, so those are repeats of one
experiment. Every error bar here is the s.e.m. over DISTINCT SEEDS after averaging repeats within a
seed. Pooling runs instead of seeds understates the uncertainty by ~1.8x on lifetime.

CEILING = the same fit with the true geometry frozen. It is the venue for single-knob decisions
because it removes the geometry variance that dominates production (seed s.d. shrinks 11-70x).
"""
import glob, os, re, pickle, io, contextlib
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
P = R.P
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}
PROD_DIRS = ['sci_full_ANNEALLONG', 'sci_full_ANNEALLONG2', 'sci_full_MDX']


def prod_runs(dirs, iters=10000):
    """One record per COMPLETED run; keyed by seed so repeats can be averaged."""
    out = []
    for d in dirs:
        for f in sorted(glob.glob(f'fit_result/{d}/history_iter*.pkl')):
            if int(re.search(r'iter(\d+)', f).group(1)) != iters or 'len400' not in f:
                continue
            h = pickle.load(open(f, 'rb'))
            e = {'seed': int(re.search(r'seed(\d+)', f).group(1))}
            for p in P:
                v = np.ravel(h[p + '_iter']); t = float(np.ravel(h[p + '_target'])[0])
                e[p] = 100 * (np.median(v[int(len(v) * .8):]) / t - 1)
            pr = np.ravel(h['pos_residual_iter']) * 1e4
            e['pos'] = float(np.median(pr[int(len(pr) * .8):]))
            out.append(e)
    return out


def per_seed(runs):
    seeds = sorted({r['seed'] for r in runs})
    return seeds, {k: np.array([np.mean([r[k] for r in runs if r['seed'] == s]) for s in seeds])
                   for k in P + ['pos']}


def ceiling(d):
    with contextlib.redirect_stdout(io.StringIO()):
        r = R.analyse('x', 'fit_result/' + d, True)
    return r


def dedx_gap(d):
    """% of the prior->truth dE/dx gap closed, averaged over seeds."""
    m = []
    for f in sorted(glob.glob(f'fit_result/{d}/history_iter*.pkl')):
        h = pickle.load(open(f, 'rb'))
        a = np.asarray([x for x in h.get('dedx_mae_iter', []) if x == x], float)
        if len(a) > 200:
            m.append((a[:50].mean(), a[-50:].mean()))
    m = np.array(m)
    return 100 * (1 - m[:, 1].mean() / m[:, 0].mean()) if len(m) else np.nan


PRUNS = prod_runs(PROD_DIRS)
SEEDS, PS = per_seed(PRUNS)
NS = len(SEEDS)
CEIL = {k: ceiling(v) for k, v in
        [('base', 'sci_ceiling_CEILBASE'), ('p05', 'sci_ceiling_CEILP05'),
         ('w3k', 'sci_ceiling_CEILW3KP05'), ('nonoise', 'sci_ceiling_CEILP05NN'),
         ('mdx', 'sci_ceiling_CEILMDX')]}

fig, axes = plt.subplots(2, 3, figsize=(16.4, 9.0))
xs = np.arange(len(P))

# ---------------- (a) where we stand ----------------
a = axes[0, 0]
mu = [PS[p].mean() for p in P]
se = [PS[p].std(ddof=1) / np.sqrt(NS) for p in P]
cm = [np.mean(CEIL['p05']['rows'][p]) for p in P]
cs = [np.std(CEIL['p05']['rows'][p]) / np.sqrt(CEIL['p05']['n']) for p in P]
a.axhspan(-1, 1, color=C['green'], alpha=.10)
a.text(len(P) - 0.55, 1.2, '±1%', fontsize=7.2, color=C['green'], ha='right')
a.errorbar(xs - .13, mu, yerr=se, lw=0, elinewidth=2.6, marker='o', ms=8.5, capsize=5,
           color=C['blue'], label=f'PRODUCTION, guess geometry (n = {NS} seeds)')
a.errorbar(xs + .13, cm, yerr=cs, lw=0, elinewidth=2.6, marker='s', ms=8.0, capsize=5,
           color=C['red'], label=f"CEILING, true geometry + prior 0.5 (n = {CEIL['p05']['n']})")
for k, p in enumerate(P):
    a.annotate(f'{mu[k]:+.2f}', (k - .13, mu[k]), textcoords='offset points', xytext=(-7, -4),
               fontsize=7.2, ha='right', color=C['blue'])
a.axhline(0, color=C['grey'], lw=1.3)
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('error vs truth (%)')
a.set_title('(a) WHERE WE STAND. A$_b$ and E field are solved;\n'
            f'the diffusions and lifetime carry a few percent.  pos = {PS["pos"].mean():.0f} µm',
            fontsize=9.4)
a.legend(fontsize=7.4, frameon=False, loc='lower left')

# ---------------- (b) lever 1: the dE/dx prior ----------------
a = axes[0, 1]
for key, col, lab in [('base', C['red'], 'prior 5  (what production uses)'),
                      ('p05', C['blue'], 'prior 0.5  (the script default)')]:
    r = CEIL[key]
    a.errorbar(xs + (-.13 if key == 'base' else .13), [np.mean(r['rows'][p]) for p in P],
               yerr=[np.std(r['rows'][p]) for p in P], lw=0, elinewidth=2.4, marker='o', ms=7,
               capsize=4, color=col, label=f"{lab}  (n={r['n']})")
a.axhline(0, color=C['grey'], lw=1.3)
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('error vs truth (%)  —  ceiling')
a.set_title('(b) LEVER 1 — drop the dE/dx prior 5 → 0.5.\n'
            'The largest single win available, and free.', fontsize=9.4)
a.legend(fontsize=7.4, frameon=False, loc='lower left')

# ---------------- (c) lever 2: data volume ----------------
# SEED-MATCHED. nb200 has only seeds 0-1 complete and nb400 only 0-2; comparing their
# between-seed s.d. against an 8-seed production s.d. is not a comparison at all (the 8-seed
# set includes harder targets, and a 2-seed s.d. carries one degree of freedom). Each arm is
# therefore compared with production restricted to that arm's OWN seeds.
a = axes[0, 2]
nb2 = prod_runs(['sci_full_NB200'], 20000)
nb4 = prod_runs(['sci_full_NB400FIX'], 10000)
S2 = sorted({r['seed'] for r in nb2}); S4 = sorted({r['seed'] for r in nb4})


def sub(runs, keep):
    ss = sorted(keep)
    return {k: np.array([np.mean([r[k] for r in runs if r['seed'] == s]) for s in ss])
            for k in P + ['pos']}


BARS = [(sub(PRUNS, S2), f'production, seeds {S2}', C['blue']),
        (sub(nb2, S2), f'2× data + 2× iters, seeds {S2}', C['green']),
        (sub(PRUNS, S4), f'production, seeds {S4}', C['grey']),
        (sub(nb4, S4), f'4× data, iters FIXED, seeds {S4}', C['red'])]
bw = .8 / len(BARS)
for k, (dd, lab, col) in enumerate(BARS):
    a.bar(xs + (k - (len(BARS) - 1) / 2) * bw, [abs(dd[p].mean()) for p in P],
          width=bw * .9, color=col, edgecolor='white', label=lab,
          hatch='' if k % 2 else '//')
a.set_yscale('log')
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('|error vs truth| (%)  —  lower is better')
a.set_title('(c) LEVER 2 — more data, iterations SCALED alongside.\n'
            'position ' + ' / '.join(f'{d["pos"].mean():.0f}' for d, _, _ in BARS) + ' µm: '
            'fixed iters starves the geometry block', fontsize=9.4)
a.legend(fontsize=6.4, frameon=False, loc='upper left', ncol=1)

# ---------------- (d) lever 3: how many seeds ----------------
a = axes[1, 0]
rng = np.random.default_rng(0)
for p, col in [('lifetime', C['red']), ('long_diff', C['orange']), ('tran_diff', C['blue'])]:
    v = PS[p]
    ns = np.arange(2, NS + 1)
    sem = [np.mean([np.std(rng.choice(v, n, replace=False), ddof=1) / np.sqrt(n)
                    for _ in range(4000)]) for n in ns]
    a.plot(ns, sem, marker='o', ms=5, lw=2, color=col, label=PL[p])
a.axvline(6, color=C['grey'], ls='--', lw=1.5)
a.text(6.1, a.get_ylim()[1] * .93, ' ≥6 seeds', fontsize=7.6, color=C['grey'], va='top')
a.set_xlabel('number of distinct seeds'); a.set_ylabel('expected error bar (s.e.m., %)')
a.set_title('(d) LEVER 3 — run at ≥ 6 seeds.\n'
            'Two campaign conclusions died from n = 3.', fontsize=9.4)
a.legend(fontsize=7.6, frameon=False)

# ---------------- (e) what NOT to do ----------------
a = axes[1, 1]


def rms(vals):
    return float(np.sqrt(np.mean(np.square(vals))))


mdx_runs = prod_runs(['sci_full_MDX'])
_, mdx_ps = per_seed(mdx_runs)
al_runs = prod_runs(['sci_full_ANNEALLONG'])          # matched seeds 0-2, same tree
_, al_ps = per_seed(al_runs)


def bias_var(mean_v, sd_v):
    return rms(mean_v), float(np.mean(sd_v))


# each entry: label, (bias, scatter) of the ARM, of its CONTROL
CASES = [
    ('dE/dx min-length cut 0.15 cm\n(production, seeds 0–2)',
     bias_var([mdx_ps[p].mean() for p in P], [mdx_ps[p].std(ddof=1) for p in P]),
     bias_var([al_ps[p].mean() for p in P], [al_ps[p].std(ddof=1) for p in P])),
    ('dE/dx min-length cut 0.15 cm\n(ceiling)',
     bias_var([np.mean(CEIL['mdx']['rows'][p]) for p in P],
              [np.std(CEIL['mdx']['rows'][p]) for p in P]),
     bias_var([np.mean(CEIL['base']['rows'][p]) for p in P],
              [np.std(CEIL['base']['rows'][p]) for p in P])),
    ('mean-constraint weight 1e5 → 3000\n(ceiling, at prior 0.5)',
     bias_var([np.mean(CEIL['w3k']['rows'][p]) for p in P],
              [np.std(CEIL['w3k']['rows'][p]) for p in P]),
     bias_var([np.mean(CEIL['p05']['rows'][p]) for p in P],
              [np.std(CEIL['p05']['rows'][p]) for p in P])),
    ('readout noise ON → OFF\n(ceiling, at prior 0.5)',
     bias_var([np.mean(CEIL['nonoise']['rows'][p]) for p in P],
              [np.std(CEIL['nonoise']['rows'][p]) for p in P]),
     bias_var([np.mean(CEIL['p05']['rows'][p]) for p in P],
              [np.std(CEIL['p05']['rows'][p]) for p in P])),
]
ys = np.arange(len(CASES))
db = [c[1][0] - c[2][0] for c in CASES]
dv = [c[1][1] - c[2][1] for c in CASES]
a.barh(ys - .19, db, height=.34, color=C['blue'], edgecolor='white',
       label='Δ BIAS   (RMS of the 5 errors)')
a.barh(ys + .19, dv, height=.34, color=C['purple'], edgecolor='white',
       label='Δ SCATTER (mean seed s.d.)')
for k in range(len(CASES)):
    for off, v in [(-.19, db[k]), (.19, dv[k])]:
        a.text(v + (0.06 if v >= 0 else -0.06), k + off, f'{v:+.2f}', va='center',
               ha='left' if v >= 0 else 'right', fontsize=7.0)
a.axvline(0, color=C['ink'], lw=1.4)
a.set_yticks(ys); a.set_yticklabels([c[0] for c in CASES], fontsize=7.0)
a.set_xlabel('change vs its own control (percentage points) — right is worse')
a.set_title('(e) WHAT NOT TO DO. The min-length cut is a clean null.\n'
            'Weight 3000 buys no bias and costs 7.5× the lifetime scatter.', fontsize=9.4)
a.legend(fontsize=7.0, frameon=False, loc='lower right')
a.invert_yaxis()

# ---------------- (f) the two hard limits ----------------
a = axes[1, 2]
tau = 2200.0                      # nominal lifetime, µs
frac = 0.087                      # detector depth in lifetimes (§6k)
t = np.linspace(0, frac * tau, 200)
a.plot(t, 100 * np.exp(-t / tau), color=C['blue'], lw=2.4)
a.annotate('surviving charge = exp(−t/τ)', (t[60], 100 * np.exp(-t[60] / tau)),
           textcoords='offset points', xytext=(6, 7), fontsize=7.8, color=C['blue'])
swing = 100 * (1 - np.exp(-frac))
a.annotate('', xy=(frac * tau, 100), xytext=(frac * tau, 100 - swing),
           arrowprops=dict(arrowstyle='<->', color=C['red'], lw=2))
a.text(frac * tau * .97, 100 - swing / 2, f'  the ENTIRE signal:\n  {swing:.1f}% over full drift',
       fontsize=8.2, color=C['red'], ha='right', va='center')
a.axhspan(100 - 4.67, 100, color=C['orange'], alpha=.22)
a.text(3, 100 - swing * 1.32, "shaded: σ$_Q$ = 500 e⁻ ≈ 4.7% of a median (10.7 ke) hit —\n"
       "the loss's assumed per-hit charge resolution is\ncomparable to the whole lifetime signal",
       fontsize=7.4, color=C['orange'], va='top')
a.set_xlabel('drift time (µs), nominal τ = 2200 µs')
a.set_ylabel('surviving charge (%)')
a.set_ylim(100 - swing * 2.5, 100.4)
a.set_title('(f) HARD LIMIT 1 — the detector is only 0.087 τ deep.\n'
            'No optimiser or prior creates more lever arm.', fontsize=9.4)
g_on, g_off = dedx_gap('sci_ceiling_CEILP05'), dedx_gap('sci_ceiling_CEILP05NN')
a.text(.03, .06, f'HARD LIMIT 2 — the dE/dx block is objective-limited:\n'
                 f'with PERFECT geometry it closes only {g_on:.0f}% of the prior→truth gap\n'
                 f'({g_off:.0f}% with noise off). Over half the per-segment\nstructure is unreachable.',
       transform=a.transAxes, fontsize=7.2, color=C['ink'], va='bottom',
       bbox=dict(fc='white', ec=C['grey'], alpha=.85, lw=.8))

fig.suptitle('Fig 45 — the state of play. (a) is the deliverable; (b–d) are the three levers that '
             'measurably improve it, in order of gain;\n(e) is everything tested that does not; '
             '(f) is what no amount of tuning will fix.', fontsize=10.4)
fig.tight_layout(rect=[0, 0, 1, .93])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig45_state_of_play.png'); plt.close(fig)
print('wrote', f'{OUT}/fig45_state_of_play.png')
print(f'PRODUCTION  {len(PRUNS)} completed runs, {NS} distinct seeds {SEEDS}')
for p in P:
    print(f'  {p:11s} {PS[p].mean():+7.2f} ± {PS[p].std(ddof=1)/np.sqrt(NS):5.2f}'
          f'   (seed s.d. {PS[p].std(ddof=1):5.2f})')
print(f'  {"pos":11s} {PS["pos"].mean():7.0f} ± {PS["pos"].std(ddof=1)/np.sqrt(NS):5.0f} µm')
for k in ['base', 'p05', 'w3k', 'nonoise', 'mdx']:
    r = CEIL[k]
    print(f'CEILING {k:8s} n={r["n"]}  ' +
          '  '.join(f'{p}={np.mean(r["rows"][p]):+.2f}' for p in P))
print(f'dE/dx gap closed: noise ON {g_on:.1f}%   noise OFF {g_off:.1f}%')
