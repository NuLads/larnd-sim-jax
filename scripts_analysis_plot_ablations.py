"""Figures for the CONSTDEDX (dE/dx ablation) and PROBTGT (probabilistic-target bias check)
arms, placed alongside the existing annealed family (ANNEALLONG, SLOWANNEAL).

All runs: 10 000 iterations, len400, b100, LR decay 0.91, chain_lr 1e-2. Each seed draws its
OWN target, so every error is computed per seed against that seed's own <param>_target.
Robust estimator: median over the last 20% of iterations.
"""
import pickle, glob, re, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = sys.argv[1]
PARAMS = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
NICE = {'Ab': r'$A_b$', 'eField': r'$E$ field', 'tran_diff': 'tran. diff.',
        'long_diff': 'long. diff.', 'lifetime': 'lifetime'}
ARMS = [('ANNEALLONG', 'fit_result/sci_full_ANNEALLONG', '#1f5fa9', 'baseline: fit dE/dx, deterministic target'),
        ('SLOWANNEAL', 'fit_result/sci_full_SLOWANNEAL', '#6a9bd1', 'slower anneal (1.78x travel)'),
        ('CONSTDEDX',  'fit_result/sci_full_CONSTDEDX',  '#c1392b', 'dE/dx FROZEN at the mean'),
        ('PROBTGT',    'fit_result/sci_full_PROBTGT',    '#2e8b57', 'probabilistic target')]


def load(d):
    runs = {}
    for f in sorted(glob.glob(d + '/history_iter*_sci_full_b100_len400_seed*.pkl')):
        runs[re.search(r'seed(\d+)', f).group(1)] = pickle.load(open(f, 'rb'))
    return runs


DATA = {lab: load(d) for lab, d, _, _ in ARMS}


def err_tail(h, p):
    t = np.asarray(h[p + '_iter'], dtype=float)
    tgt = float(np.asarray(h[p + '_target']).ravel()[0])
    return 100.0 * (np.median(t[int(0.8 * len(t)):]) / tgt - 1.0)


def err_traj(h, p):
    t = np.asarray(h[p + '_iter'], dtype=float)
    tgt = float(np.asarray(h[p + '_target']).ravel()[0])
    return 100.0 * (t / tgt - 1.0)


# =====================================================================
# FIG 23 — global comparison of the annealed family
# =====================================================================
fig, ax = plt.subplots(figsize=(11.4, 5.2))
w = 0.2
xs = np.arange(len(PARAMS))
for k, (lab, _, col, _) in enumerate(ARMS):
    if not DATA[lab]:
        continue
    mus, sds, pts = [], [], []
    for p in PARAMS:
        e = [err_tail(h, p) for h in DATA[lab].values() if p + '_iter' in h]
        mus.append(np.mean(e)); sds.append(np.std(e, ddof=1) if len(e) > 1 else 0.0); pts.append(e)
    off = (k - (len(ARMS) - 1) / 2) * w
    ax.bar(xs + off, mus, w * 0.9, yerr=sds, capsize=3, color=col, label=lab,
           error_kw=dict(lw=1.1, ecolor='#333333'))
    for x, e in zip(xs + off, pts):
        ax.plot([x] * len(e), e, 'o', ms=3, color='#222222', alpha=0.55, zorder=4)
ax.axhline(0, color='#000000', lw=1.0)
ax.axhspan(-2, 2, color='#999999', alpha=0.13, lw=0)
ax.text(4.42, 2.4, '±2% band', fontsize=8.5, color='#555555', ha='right')
ax.set_xticks(xs); ax.set_xticklabels([NICE[p] for p in PARAMS], fontsize=11)
ax.set_ylabel('error vs each seed\'s own target  (%)')
ax.set_title('Global comparison of the annealed family — 10 000 it, len400, b100, 3 seeds each\n'
             'dots = individual seeds; bars = mean ± sd', fontsize=11)
ax.legend(frameon=False, fontsize=9, ncol=4, loc='lower left')
ax.grid(alpha=0.25, axis='y', lw=0.5)
ax.set_ylim(-28, 9)
ax.annotate('freezing dE/dx costs\nlong. diffusion 4x',
            xy=(3.3, -21.1), xytext=(2.0, -25.5), fontsize=9, color='#c1392b',
            arrowprops=dict(arrowstyle='->', color='#c1392b', lw=1.3))
fig.tight_layout()
fig.savefig(OUT + '/fig23_ablation_global.png', dpi=155)
print('fig23')

# =====================================================================
# FIG 24 — trajectories of the two parameters that move
# =====================================================================
fig2, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
for j, p in enumerate(['lifetime', 'long_diff']):
    a = axes[j]
    for lab, _, col, _ in ARMS:
        if lab == 'SLOWANNEAL':
            continue                      # keep the panel readable: 3 arms
        hs = [h for h in DATA[lab].values() if p + '_iter' in h]
        if not hs:
            continue
        tr = [err_traj(h, p) for h in hs]
        n = min(len(t) for t in tr)
        tr = np.array([t[:n] for t in tr])
        it = np.arange(n)
        for t in tr:
            a.plot(it, t, color=col, lw=0.7, alpha=0.30)
        a.plot(it, np.median(tr, 0), color=col, lw=2.2, label=lab)
    a.axhline(0, color='#000000', lw=1.0)
    a.set_ylim(-30, 40) if p == 'lifetime' else a.set_ylim(-45, 40)
    a.axvspan(8000, 10000, color='#999999', alpha=0.12, lw=0)
    a.text(0.90, 0.93, 'read-out\nwindow', transform=a.transAxes,
           fontsize=8, ha='center', va='top', color='#555555')
    a.set_xlabel('iteration'); a.set_ylabel(f'{NICE[p]} error (%)')
    a.set_title(f'{NICE[p]}: freezing dE/dx pushes it away from truth', fontsize=10.5)
    a.legend(frameon=False, fontsize=9)
    a.grid(alpha=0.25, lw=0.5)
fig2.suptitle('Effect of the dE/dx nuisance block and of a probabilistic target', fontsize=12)
fig2.tight_layout(rect=[0, 0, 1, 0.93])
fig2.savefig(OUT + '/fig24_ablation_traces.png', dpi=155)
print('fig24')

# =====================================================================
# FIG 25 — what it costs elsewhere: position and dE/dx MAE
# =====================================================================
fig3, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
a = axes[0]
labs, mus, sds, cols = [], [], [], []
for lab, _, col, _ in ARMS:
    v = [np.median(np.asarray(h['pos_residual_iter'], dtype=float)[
             int(0.8 * len(h['pos_residual_iter'])):]) * 1e4
         for h in DATA[lab].values() if 'pos_residual_iter' in h]
    if not v:
        continue
    labs.append(lab); mus.append(np.mean(v)); sds.append(np.std(v, ddof=1) if len(v) > 1 else 0)
    cols.append(col)
a.bar(np.arange(len(labs)), mus, 0.55, yerr=sds, capsize=4, color=cols,
      error_kw=dict(lw=1.1, ecolor='#333333'))
for i, (m, s) in enumerate(zip(mus, sds)):
    a.text(i, m + s + 4, f'{m:.0f} µm', ha='center', fontsize=9)
a.set_xticks(np.arange(len(labs))); a.set_xticklabels(labs, fontsize=9, rotation=12)
a.set_ylabel('position residual (µm)')
a.set_title('(a) position residual: all four arms agree within errors\n'
            '(CONSTDEDX trends lower but 158±9 vs 168±14 is NOT significant)', fontsize=10)
a.grid(alpha=0.25, axis='y', lw=0.5); a.set_ylim(0, max(mus) * 1.35)

b = axes[1]
for lab, _, col, _ in ARMS:
    hs = [h for h in DATA[lab].values() if 'dedx_mae_iter' in h]
    if not hs:
        continue
    n = min(len(h['dedx_mae_iter']) for h in hs)
    tr = np.array([np.asarray(h['dedx_mae_iter'], dtype=float)[:n] for h in hs])
    med = np.median(tr, 0)
    # dedx_mae_iter is a PER-BATCH metric and swings 0.02-0.06 within one epoch (100 batches).
    # Smooth over one full epoch so the trend is visible rather than the batch-to-batch sawtooth.
    W = 100
    ker = np.ones(W) / W
    sm = np.convolve(med, ker, mode='valid')
    b.plot(np.arange(len(sm)) + W // 2, sm, color=col, lw=2.0, label=lab)
b.set_xlabel('iteration'); b.set_ylabel('dE/dx MAE')
b.set_title('(b) dE/dx MAE, smoothed over one epoch (100 batches)\n'
            'probabilistic target changes nothing; CONSTDEDX absent (not fitted)', fontsize=10)
b.legend(frameon=False, fontsize=9); b.grid(alpha=0.25, lw=0.5)
fig3.tight_layout()
fig3.savefig(OUT + '/fig25_ablation_cost.png', dpi=155)
print('fig25')
