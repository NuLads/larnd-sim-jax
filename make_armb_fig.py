"""Fig 92 — ARM B: the bias survives a change of truth point, and the lever arm explains its size.

THE TEST. Every number in this campaign was conditional on ONE truth draw (--seed 0), which fixes
both the truth parameters AND the target noise realisation; --data_seed only varies which tracks
are used. Arm B crosses 3 NEW truth points with the SAME 4 track samples: a paired 4x4 design.

THE RESULT.
  * The bias REPRODUCES at every truth point: +11.7 / +3.0 / +11.6 / +6.0%, 15 of 16 fits above
    truth, grand mean +8.0%. It is a property of the procedure, not of the seed-0 draw.
  * The truth point matters more than the sample does (F = 3.19 vs 1.22) -- so "how biased are we"
    has no single answer; it depends on the detector's true lifetime.
  * WHY: the drift lever arm. Truth seed 1 has tau = 1160 us -- half our nominal -- so 30 cm of
    drift costs 15.9% of the charge instead of 7.7%, and the bias falls to +3.0%. More attenuation
    signal in the data => less room for the transient to wander. Seed 3 (tau = 4518, lever 4.1%)
    is NOT worse than seed 0, which is what kills the naive '1/lever' law (r = -0.53 only).
  * The absolute offset is ~270-280 us at three of four truth points and 34 us at the short one:
    suggestive of a fixed displacement along the degeneracy rather than a fixed relative error,
    but n = 4 truth points cannot establish that.
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
BLUE, ORANGE, GREEN, RED, GREY, INK = ('#0072B2', '#E69F00', '#009E73', '#D55E00',
                                       '#888888', '#222222')
plt.rcParams.update({'font.size': 9.5, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
DRIFT_US = 184.0


def get(ts, ds, k='lifetime'):
    tag = f'DSREP{ds}' if ts == 0 else f'TRU{ts}D{ds}'
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter10000_*.pkl')
    if not fs:
        return None, None
    h = pickle.load(open(fs[0], 'rb'))
    v = np.ravel(h[k + '_iter'])
    t = float(np.ravel(h[k + '_target'])[0])
    return 100 * (np.median(v[int(len(v) * .9):]) / t - 1), t


TS, DS = [0, 1, 2, 3], [1, 3, 7, 8]
E = np.array([[get(t, d)[0] for d in DS] for t in TS])
TAU = np.array([get(t, 1)[1] for t in TS])
COLS = {0: INK, 1: BLUE, 2: ORANGE, 3: RED}

fig, ax = plt.subplots(1, 3, figsize=(13.8, 4.3))

# (a) the 4x4 matrix
a = ax[0]
rng = np.random.default_rng(2)
for i, t in enumerate(TS):
    a.scatter(E[i], np.full(4, i) + rng.uniform(-.12, .12, 4), s=55, color=COLS[t],
              edgecolors='white', lw=.6, zorder=3)
    m = E[i].mean(); se = E[i].std(ddof=1) / 2
    a.errorbar(m, i + .28, xerr=se, fmt='D', color=COLS[t], ms=7, capsize=4, zorder=4)
    a.annotate(f'{m:+.1f} ± {se:.1f}', (m, i + .28), textcoords='offset points',
               xytext=(0, 9), ha='center', fontsize=8.5, color=COLS[t])
a.axvline(0, color=GREEN, lw=2)
a.annotate('truth', (0, 3.62), ha='center', color=GREEN, fontweight='bold', fontsize=9)
a.set_yticks(TS, [f'truth pt {t}\nτ = {tau:.0f} µs' for t, tau in zip(TS, TAU)], fontsize=8.5)
a.set_ylim(-.5, 3.8)
a.set_xlabel('lifetime error (%)')
a.set_title('(a) the bias REPRODUCES at every truth point\n15/16 fits above truth, grand mean +8.0%',
            fontsize=9.5)

# (b) bias vs lever arm
a = ax[1]
lever = 100 * (1 - np.exp(-DRIFT_US / TAU))
for i, t in enumerate(TS):
    a.errorbar(lever[i], E[i].mean(), yerr=E[i].std(ddof=1) / 2, fmt='o', color=COLS[t],
               ms=9, capsize=4, zorder=3)
    a.annotate(f'  τ={TAU[i]:.0f}', (lever[i], E[i].mean()), fontsize=8.5, color=COLS[t],
               va='center')
a.axhline(0, color=GREEN, lw=1.8)
a.set_xlabel('attenuation signal available: charge lost over 30 cm (%)')
a.set_ylabel('lifetime bias (%)')
a.set_title('(b) more attenuation signal → smaller bias,\nbut not a simple 1/lever law (r = −0.53)',
            fontsize=9.5)

# (c) absolute offset
a = ax[2]
abs_off = TAU * E.mean(1) / 100
for i, t in enumerate(TS):
    a.bar(i, abs_off[i], color=COLS[t])
    a.annotate(f'{abs_off[i]:+.0f} µs', (i, abs_off[i]), textcoords='offset points',
               xytext=(0, 4), ha='center', fontsize=9, color=COLS[t])
a.axhline(np.mean(abs_off[[0, 2, 3]]), color=GREY, lw=1.6, ls='--',
          label=f'mean of the three long-τ points: {np.mean(abs_off[[0,2,3]]):.0f} µs')
a.set_xticks(range(4), [f'τ={t:.0f}' for t in TAU], fontsize=8.5)
a.set_ylabel('absolute lifetime offset (µs)')
a.set_title('(c) in ABSOLUTE terms the offset is ~275 µs at three\nof four truth points — suggestive, n = 4',
            fontsize=9.5)
a.legend(frameon=False, fontsize=8.5, loc='lower left')

fig.suptitle('Fig 92 — ARM B: three NEW truth points (new truth parameters AND new target noise) × the same four track samples.\n'
             'The +8.85% bias is NOT an artefact of the seed-0 draw — it reproduces everywhere. But its SIZE depends on the truth point\n'
             '(F = 3.19 vs 1.22 for the sample), and the driver is how much attenuation signal the detector’s true lifetime leaves in the data.',
             fontsize=10.5, y=1.08)
fig.tight_layout()
fig.savefig(f'{OUT}/fig92_armb_truthpoints.png')
print('wrote fig92')
print('means', E.mean(1), 'grand', E.mean(), 'positive', (E > 0).sum(), '/16')
