"""Figures for the noise-ON campaign report (S1 reprocessing -> present)."""
import pickle, glob, os, re, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = 'plots/noise_report'
os.makedirs(OUT, exist_ok=True)
P = ['Ab', 'eField', 'lifetime', 'tran_diff', 'long_diff']
PL = {'Ab': 'A$_b$ (recomb.)', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}
# Okabe-Ito colourblind-safe
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', sky='#56B4E9', yellow='#F0E442', grey='#666666')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})


def load(d, require_len400=True):
    """Latest checkpoint per seed -> {seed: history}.

    NOTE: several stage-ladder directories contain BOTH the 400 cm and 50 cm runs, so an
    unfiltered glob silently mixes batch lengths across seeds. Filter explicitly.
    """
    seeds = {}
    for f in glob.glob(d + '/history_iter*.pkl'):
        if require_len400 and 'len400' not in f:
            continue
        m = re.search(r'seed(\d+)', f)
        if not m:
            continue
        s = int(m.group(1)); it = int(re.search(r'iter(\d+)', f).group(1))
        if s not in seeds or it > seeds[s][0]:
            seeds[s] = (it, f)
    out = {}
    for s, (it, f) in seeds.items():
        try:
            out[s] = pickle.load(open(f, 'rb'))
        except Exception:
            pass
    return out


def errs(h):
    """Final % error per parameter."""
    o = {}
    for p in P:
        if p + '_iter' not in h:
            continue
        v = np.ravel(np.array(h[p + '_iter'])); t = np.ravel(h[p + '_target'])[0]
        o[p] = (v[-1] / t - 1) * 100
    return o


def pos_tail(h):
    """Median of last 20% of the PER-BATCH position residual (um)."""
    pr = h.get('pos_residual_iter')
    if pr is None:
        return None
    pr = np.ravel(pr) * 1e4
    return float(np.median(pr[int(len(pr) * .8):])) if len(pr) else None


R = 'fit_result/'
STAGES = [('S1\ntrue dEdx+pos', R + 'sci_ceiling_noise_s1'),
          ('S2\n+fitted dEdx', R + 'sci_ceiling_noise_s2'),
          ('S3\n+wrong frozen geom', R + 'sci_floor_noise_s3'),
          ('S4\n+fitted geom', R + 'sci_full_noise_s4')]

# ── Fig 1 ── stage ladder ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.4, 3.9))
w = 0.19
for i, p in enumerate(P):
    means, sds = [], []
    for _, d in STAGES:
        e = [errs(h).get(p, np.nan) for h in load(d).values()]
        e = [x for x in e if np.isfinite(x)]
        means.append(np.mean(np.abs(e)) if e else np.nan)
        sds.append(np.std(np.abs(e)) if e else np.nan)
    x = np.arange(len(STAGES)) + (i - 2) * w
    ax.bar(x, means, w, yerr=sds, capsize=2, label=PL[p],
           color=list(C.values())[i], edgecolor='white', linewidth=.6)
ax.axhline(5, ls='--', lw=1, color=C['grey'])
ax.text(3.42, 5.6, '5% target', color=C['grey'], fontsize=8, ha='right')
ax.set_yscale('log'); ax.set_ylabel('|error|  (%, mean of 3 seeds)')
ax.set_xticks(range(len(STAGES))); ax.set_xticklabels([s for s, _ in STAGES])
ax.set_title('Fig 1 — Noise-ON stage ladder @400 cm: calibration is solved through S2;\n'
             'both stages that touch geometry break lifetime and long. diffusion', loc='left')
ax.legend(ncol=5, fontsize=8, frameon=False, loc='upper left')
fig.savefig(f'{OUT}/fig1_stage_ladder.png'); plt.close(fig)

# ── Fig 2 ── S4 variant history ──────────────────────────────────────────────
VAR = [('baseline', 'sci_full_noise_s4'), ('10k iters', 'sci_full_noise_s4_10k'),
       ('drift-profile pen.', 'sci_full_noise_s4_dpw'), ('dEdx prior x10', 'sci_full_noise_s4_dxp5'),
       ('slow chain decay', 'sci_full_noise_s4_slowdecay'), ('slowdecay+prior', 'sci_full_noise_s4_sd_dxp5'),
       ('  “winner” @8k (6 seeds)', 'sci_full_noise_s4_WIN'), ('slowdecay+freeze', 'sci_full_noise_s4_sdfrz'),
       ('two-pass frozen geom', 'sci_full_noise_s4_pass2'), ('triple combo', 'sci_full_noise_s4_triple'),
       ('LR ANNEAL (new)', 'sci_full_noise_s4_anneal')]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
for ax, p in zip(axes, ['lifetime', 'long_diff']):
    labs, mus, sds, cols = [], [], [], []
    for lab, d in VAR:
        e = [errs(h).get(p, np.nan) for h in load(R + d).values()]
        e = [x for x in e if np.isfinite(x)]
        if not e:
            continue
        labs.append(lab); mus.append(np.mean(e)); sds.append(np.std(e))
        cols.append(C['green'] if 'ANNEAL' in lab else C['blue'])
    y = np.arange(len(labs))
    ax.barh(y, mus, xerr=sds, color=cols, capsize=2.5, edgecolor='white', linewidth=.6)
    ax.axvline(0, color='k', lw=.9)
    ax.axvspan(-5, 5, color=C['grey'], alpha=.13, zorder=0)
    ax.set_yticks(y)
    if p == 'lifetime':
        ax.set_yticklabels(labs, fontsize=8)
    else:
        ax.tick_params(labelleft=False)
    ax.invert_yaxis()
    ax.set_xlabel(f'{PL[p]} error (%)'); ax.set_title(PL[p], loc='left', fontsize=9)
fig.suptitle('Fig 2 — Every S4 variant tried under noise-ON. Shaded band = ±5%.\n'
             'Only LR annealing (green) brings both parameters inside it.', x=.02, ha='left', y=1.06)
fig.savefig(f'{OUT}/fig2_s4_variants.png'); plt.close(fig)

# ── Fig 3 ── lifetime trajectories: biased walk vs oscillation ───────────────
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), sharey=True)
for ax, (lab, d) in zip(axes, [('S4 baseline', 'sci_full_noise_s4'),
                               ('S4 “winner” @8k', 'sci_full_noise_s4_WIN'),
                               ('S4 + LR ANNEAL', 'sci_full_noise_s4_anneal')]):
    for s, h in sorted(load(R + d).items()):
        if 'lifetime_iter' not in h:
            continue
        v = np.ravel(np.array(h['lifetime_iter'])); t = np.ravel(h['lifetime_target'])[0]
        ax.plot(np.arange(len(v)), (v / t - 1) * 100, lw=1.1, alpha=.85, label=f'seed {s}')
    ax.axhline(0, color='k', lw=.9); ax.axhspan(-5, 5, color=C['grey'], alpha=.15, zorder=0)
    ax.set_title(lab, loc='left'); ax.set_xlabel('iteration')
    ax.legend(fontsize=7, frameon=False, ncol=2)
axes[0].set_ylabel('lifetime error (%)'); axes[0].set_ylim(-40, 160)
fig.tight_layout(rect=[0,0,1,.86]); fig.suptitle('Fig 3 — Why endpoints deceived us: baseline and “winner” perform a BIASED WALK '
             '(transient truth-crossings);\nannealing instead oscillates about truth inside ±5%.',
             x=.02, ha='left')
fig.savefig(f'{OUT}/fig3_trajectories.png'); plt.close(fig)

# ── Fig 4 ── plateau audit ───────────────────────────────────────────────────
if os.path.exists('plateau_audit.json'):
    rows = json.load(open('plateau_audit.json'))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    fr = np.array([r['plateau_it'] / max(r['iters'], 1) for r in rows])
    axes[0].hist(fr, bins=24, color=C['blue'], edgecolor='white')
    axes[0].axvline(.98, color=C['red'], ls='--', lw=1.2)
    axes[0].text(.965, axes[0].get_ylim()[1] * .85, 'never plateaued', color=C['red'],
                 fontsize=8, ha='right')
    axes[0].set_xlabel('plateau iteration / total iterations'); axes[0].set_ylabel('runs')
    axes[0].set_title(f'{int((fr>.98).sum())} of {len(fr)} runs ({(fr>.98).mean()*100:.0f}%) were '
                      'still moving\nwhen they were read out', loc='left', fontsize=9)
    mv = [(r.get('long_diff_end', np.nan) - r.get('long_diff_plat', np.nan)) for r in rows
          if r['plateau_it'] < r['iters'] * .98]
    mv = [m for m in mv if np.isfinite(m)]
    axes[1].hist(mv, bins=14, color=C['orange'], edgecolor='white')
    axes[1].axvline(0, color='k', lw=.9)
    axes[1].set_xlabel('long. diff.:  value at end  −  value at plateau  (% points)')
    axes[1].set_ylabel('runs')
    axes[1].set_title('Among runs that DID plateau, the reported number\n'
                      f'shifts by up to {np.max(np.abs(mv)):.0f} points depending on read-out point',
                      loc='left', fontsize=9)
    fig.suptitle('Fig 4 — Convergence audit over all 105 campaign runs', x=.02, ha='left')
    fig.savefig(f'{OUT}/fig4_plateau_audit.png'); plt.close(fig)

# ── Fig 5 ── Hessian eigen-spectrum ──────────────────────────────────────────
H = json.load(open('fit_result/hessian_analysis/hessian_results.json'))
eig = H['calib5x5']['eigen']
vals = [e['value'] for e in eig]
fig, ax = plt.subplots(figsize=(8.0, 3.4))
cols = [C['red'] if v < 1e4 else C['blue'] for v in vals]
ax.barh(range(len(vals)), vals, color=cols, edgecolor='white')
for i, e in enumerate(eig):
    ax.text(e['value'] * 1.6, i, e['composition'], va='center', fontsize=7.4)
ax.set_xscale('log'); ax.set_xlim(1e2, 1e13)
ax.set_yticks(range(len(vals))); ax.set_yticklabels([f'mode {i+1}' for i in range(len(vals))])
ax.set_xlabel('Hessian eigenvalue (log scale)')
s = H['calib5x5']['sigma_pct_logspace']
ax.set_title('Fig 5 — Curvature at truth (S2@400, noise-ON): the two softest modes are a MIXED\n'
             f"(lifetime, long. diff.) plane — statistical floors σ(lifetime)={s['lifetime']}%, "
             f"σ(long_diff)={s['long_diff']}%", loc='left')
ax.legend(handles=[Patch(color=C['red'], label='soft / degenerate'),
                   Patch(color=C['blue'], label='stiff / well-determined')],
          fontsize=8, frameon=False, loc='lower right')
fig.savefig(f'{OUT}/fig5_hessian.png'); plt.close(fig)

# ── Fig 6 ── two measurement traps ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
h = load(R + 'sci_full_noise_s4_anneal').get(0)
if h is not None and h.get('pos_residual_iter') is not None:
    pr = np.ravel(h['pos_residual_iter']) * 1e4
    axes[0].plot(pr, lw=.7, color=C['blue'], alpha=.8)
    k = 201
    axes[0].plot(np.convolve(pr, np.ones(k) / k, 'same'), lw=2, color=C['red'],
                 label='running median-scale average')
    axes[0].set_xlim(len(pr) * .5, len(pr)); axes[0].set_ylim(0, 2000)
    axes[0].set_xlabel('iteration'); axes[0].set_ylabel('position residual (µm)')
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title('Trap 1: the residual is PER-BATCH and swings\n'
                      '120→1400 µm within one converged run', loc='left', fontsize=9)


def coeff_mag(d):
    hs = load(d)
    if not hs:
        return None
    out = []

    def walk(o):
        if isinstance(o, dict):
            [walk(v) for v in o.values()]
        elif isinstance(o, (list, tuple)):
            [walk(v) for v in o]
        else:
            a = np.abs(np.ravel(np.asarray(o)))
            if a.size and np.issubdtype(a.dtype, np.number):
                out.append(float(a.max()))
    walk(list(hs.values())[0].get('chain_cache', {}))
    return np.array(out)


mags = [('LR correct\n(anneal)', coeff_mag(R + 'sci_full_noise_s4_anneal')),
        ('LR 100x too small\n(invalidated arm)', coeff_mag(R + 'sci_full_FRAMEONLY'))]
mags = [(l, m) for l, m in mags if m is not None and m.size]
if mags:
    axes[1].boxplot([m * 1e4 for _, m in mags], tick_labels=[l for l, _ in mags],
                    showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor=C['sky'], edgecolor=C['grey']),
                    medianprops=dict(color=C['red'], lw=1.6))
    axes[1].set_yscale('log'); axes[1].set_ylabel('max |spline coefficient| per track (µm)')
    axes[1].set_title('Trap 2: a mis-copied chain LR left geometry stalled\n'
                      '(87× smaller) — caught by 3 seeds agreeing to 0.1%', loc='left', fontsize=9)
fig.suptitle('Fig 6 — Two measurement traps found (and fixed) during this campaign', x=.02, ha='left')
fig.savefig(f'{OUT}/fig6_traps.png'); plt.close(fig)

# ── Fig 7 ── ANNEAL parameter recovery ───────────────────────────────────────
hs = load(R + 'sci_full_noise_s4_anneal')
fig, ax = plt.subplots(figsize=(7.6, 3.4))
w = .26
for j, (s, h) in enumerate(sorted(hs.items())):
    e = errs(h)
    ax.bar(np.arange(len(P)) + (j - 1) * w, [e.get(p, np.nan) for p in P], w,
           label=f'seed {s}', color=list(C.values())[j], edgecolor='white', linewidth=.6)
ax.axhline(0, color='k', lw=.9); ax.axhspan(-5, 5, color=C['grey'], alpha=.15, zorder=0)
ax.set_xticks(range(len(P))); ax.set_xticklabels([PL[p] for p in P])
ax.set_ylabel('error (%)'); ax.legend(fontsize=8, frameon=False, ncol=3)
pt = np.mean([pos_tail(h) for h in hs.values() if pos_tail(h)])
ax.set_title('Fig 7 — Current best S4 result (LR anneal, 3 seeds x 5000 it).\n'
             f'All five parameters inside ±5%; position residual ≈ {pt:.0f} µm', loc='left')
fig.savefig(f'{OUT}/fig7_anneal_recovery.png'); plt.close(fig)

print('wrote figures to', OUT)
for f in sorted(os.listdir(OUT)):
    print('  ', f)

# ── Fig 8 ── per-parameter trajectories for the ANNEAL runs ──────────────────
hs = load(R + 'sci_full_noise_s4_anneal')
fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.2))
seedcol = [C['blue'], C['orange'], C['green']]
for ax, p in zip(axes, P):
    for j, (s, h) in enumerate(sorted(hs.items())):
        v = np.ravel(np.array(h[p + '_iter'])); t = np.ravel(h[p + '_target'])[0]
        ini = np.ravel(h[p + '_init'])[0]
        ax.plot((v / t - 1) * 100, lw=1.0, color=seedcol[j % 3], alpha=.9,
                label=f'seed {s}  (start {(ini/t-1)*100:+.0f}%)')
    ax.axhline(0, color='k', lw=.9)
    ax.axhspan(-5, 5, color=C['grey'], alpha=.18, zorder=0)
    ax.set_title(PL[p], loc='left', fontsize=9.5)
    ax.set_xlabel('iteration'); ax.legend(fontsize=6.5, frameon=False)
    lim = max(12, min(140, np.percentile([abs((np.ravel(np.array(h[p+'_iter']))/np.ravel(h[p+'_target'])[0]-1)*100).max()
                                          for h in hs.values()], 100) * 1.05))
    ax.set_ylim(-lim, lim)
axes[0].set_ylabel('error (%)')
fig.suptitle('Fig 8 — LR-anneal runs: every calibration parameter vs iteration (3 seeds, shaded = ±5%).\n'
             'Legend gives each seed\'s STARTING offset — seed targets differ, so seeds do not face equally hard problems.',
             x=.02, ha='left', y=1.10)
fig.savefig(f'{OUT}/fig8_anneal_param_traces.png'); plt.close(fig)

# ── Fig 9 ── position residual, dEdx MAE, loss vs iteration ──────────────────
def roll(a, k=151, f=np.median):
    a = np.asarray(a, float); n = len(a)
    return np.array([f(a[max(0, i - k // 2):min(n, i + k // 2 + 1)]) for i in range(n)])

fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.5))
for j, (s, h) in enumerate(sorted(hs.items())):
    pr = np.ravel(h['pos_residual_iter']) * 1e4
    axes[0].plot(pr, lw=.4, color=seedcol[j % 3], alpha=.20)
    axes[0].plot(roll(pr), lw=1.6, color=seedcol[j % 3], label=f'seed {s}')
    mae = np.ravel(h['dedx_mae_iter'])
    axes[1].plot(roll(mae, 101, np.mean), lw=1.5, color=seedcol[j % 3], label=f'seed {s}')
    axes[2].plot(roll(np.ravel(h['losses_iter']), 101, np.mean), lw=1.5, color=seedcol[j % 3],
                 label=f'seed {s}')
axes[0].set_yscale('log'); axes[0].set_ylabel('position residual (µm)')
axes[0].set_title('Position precision: still descending at 5000 it\n(faint = raw per-batch, bold = rolling median)', loc='left', fontsize=9)

axes[1].set_ylabel('per-segment dE/dx MAE (MeV/cm)')
axes[1].set_title('dE/dx nuisance accuracy: degrades sharply, then only\npartially recovers — ends ~30% WORSE than it started', loc='left', fontsize=9)
axes[2].set_yscale('log'); axes[2].set_ylabel('loss')
axes[2].set_title('Total loss: still decreasing', loc='left', fontsize=9)
for ax in axes:
    ax.set_xlabel('iteration'); ax.legend(fontsize=7.5, frameon=False)
fig.tight_layout(rect=[0,0,1,.90])
fig.suptitle('Fig 9 — LR-anneal runs: position precision, dE/dx nuisance accuracy and loss vs iteration.\nNone of the three has converged by 5000 iterations.', x=.02, ha='left', y=1.02)
fig.savefig(f'{OUT}/fig9_anneal_pos_dedx.png'); plt.close(fig)

# ── Fig 10 ── gap-closed: how much of the init->target offset was removed ────
fig, ax = plt.subplots(figsize=(8.2, 3.3))
w = .26
for j, (s, h) in enumerate(sorted(hs.items())):
    cl = []
    for p in P:
        v = np.ravel(np.array(h[p + '_iter'])); t = np.ravel(h[p + '_target'])[0]
        ini = np.ravel(h[p + '_init'])[0]
        ie = (ini / t - 1) * 100; fe = (v[-1] / t - 1) * 100
        cl.append(100 * (1 - abs(fe) / max(abs(ie), 1e-12)))
    b = ax.bar(np.arange(len(P)) + (j - 1) * w, cl, w, color=seedcol[j % 3],
               label=f'seed {s}', edgecolor='white', linewidth=.6)
    for r, c in zip(b, cl):
        ax.text(r.get_x() + r.get_width() / 2, max(c, 0) + 2, f'{c:.0f}', ha='center', fontsize=6.5)
ax.axhline(0, color='k', lw=.9); ax.set_ylim(-10, 118)
ax.set_xticks(range(len(P))); ax.set_xticklabels([PL[p] for p in P])
ax.set_ylabel('% of initial offset removed')
ax.legend(fontsize=8, frameon=False, ncol=3)
ax.set_title('Fig 10 — How much work each fit actually did. A small final error is NOT proof of recovery\n'
             'when the seed\'s target started close to the initial guess (low bars = fit barely moved).',
             loc='left')
fig.savefig(f'{OUT}/fig10_gap_closed.png'); plt.close(fig)
print('added figs 8-10')
