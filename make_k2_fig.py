"""Fig 60 — dropping the third spline mode: it costs nothing, and it gains nothing either.

Per-mode analysis of BT2's fitted geometry against the truth projection found mode 3 recovering
-61.7% -- its error EXCEEDS its signal, so fitting it injects noise. K2 drops it via
LARND_SPLINE_KMIN=2. 2 seeds, 3 000 iterations, with BT2's own trace as the matched control.

WHAT ACTUALLY CHANGED. K = clip(round(L/40), KMIN, 12), so lowering the floor only affects tracks
whose length does not already demand 3 modes: 839 of 864 tracks (97%) dropped to K=2 and 25 kept
K=3. Effectively the whole sample.

THE RESULT IS A NULL ON POSITION, and panel (c) says why that should have been expected. Removing
mode 3 cuts the coefficient-space error from 0.0639 to 0.0604 cm, 5.5%, because the residual is
DOMINATED by mode 1 -- which carries 4x the true amplitude of the others and is only 51.6%
recovered. Perfect mode-1 recovery would cut the coefficient error 54%. Basis truncation was never
going to be the lever; mode 1 is.

AND A TRAP THIS FIGURE EXISTS TO DOCUMENT. At iteration 2 900 K2's calibration looks far better
than BT2's -- lifetime |error| 2.81 against 6.00, transverse diffusion 3.36 against 8.64. None of
that survives: BT2 is still in its transient there and reaches 1.56 by iteration 5 000, passing K2.
A 3 000-iteration screen can rank POSITION, which descends monotonically, but not calibration,
which is nowhere near converged. The geometry screen (fig 57) carried the same caveat and this is
the concrete demonstration of it.
"""
import glob, os, pickle, re, io, contextlib, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with contextlib.redirect_stdout(io.StringIO()):
    import robust_convergence as R

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00', grey='#666666',
         ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
SEEDS = [0, 1]
W = 200
# per-mode rms error and true amplitude of BT2's geometry against the truth projection
MODE_TRUE = [0.1172, 0.0289, 0.0129]
MODE_ERR = [0.0567, 0.0207, 0.0209]


def sm(v, w=W):
    v = np.asarray(v, float)
    if len(v) <= w:
        return v
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1)


def get(tag, s):
    f = max(glob.glob(f'fit_result/sci_full_{tag}/history_iter*seed{s}.pkl'),
            key=lambda p: int(re.search(r'iter(\d+)', p).group(1)))
    h = pickle.load(open(f, 'rb'))
    d = {'pos': np.ravel(h['pos_residual_iter']) * 1e4, 'st': np.asarray(h['step_time'], float)}
    for p in R.P:
        t = float(np.ravel(h[p + '_target'])[0])
        d[p] = 100 * (np.ravel(h[p + '_iter'])[1:] / t - 1)
    return d


fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.8))

# ---------------- (a) position ----------------
a = ax[0]
for tag, col, lab in [('BT2', C['ink'], 'BT2 — 3 modes'), ('K2', C['green'], 'K2 — 2 modes')]:
    cs = [sm(get(tag, s)['pos']) for s in SEEDS]
    n = min(len(c) for c in cs)
    for c in cs:
        a.plot(c[:n], color=col, lw=.8, alpha=.35)
    a.plot(np.median([c[:n] for c in cs], axis=0), color=col, lw=2.4, label=lab)
a.set_yscale('log'); a.set_xlim(0, 3000)
a.set_xlabel('iteration'); a.set_ylabel('position residual (µm)')
b = np.mean([sm(get('BT2', s)['pos'])[2900 - W // 2] for s in SEEDS])
k = np.mean([sm(get('K2', s)['pos'])[2900 - W // 2] for s in SEEDS])
ps = [(np.mean([sm(get(t, s)['pos'])[2900 - W // 2]]) for s in SEEDS) for t in ('BT2', 'K2')]
pb = [sm(get('BT2', s)['pos'])[2900 - W // 2] for s in SEEDS]
pk = [sm(get('K2', s)['pos'])[2900 - W // 2] for s in SEEDS]
a.set_title(f'(a) position is UNCHANGED: {b:.0f} → {k:.0f} µm ({100*(k/b-1):+.1f}%),\n'
            'and that −2.5% is one seed of two', fontsize=9.4)
a.annotate('per seed:  ' + '   '.join(f'{x:.0f}→{y:.0f}' for x, y in zip(pb, pk)) + ' µm\n'
           'K=2 took effect on 839/864 tracks (97%);\nthe other 25 are long enough to demand K=3',
           (.03, .04), xycoords='axes fraction', fontsize=7.4, color=C['grey'], va='bottom')
a.legend(fontsize=7.6, frameon=False, loc='upper right')

# ---------------- (b) the phase trap ----------------
a = ax[1]
for tag, col, lab in [('BT2', C['ink'], 'BT2 — 3 modes, run to 10 000'),
                      ('K2', C['green'], 'K2 — 2 modes, stopped at 3 000')]:
    cs = [np.abs(sm(get(tag, s)['lifetime'])) for s in SEEDS]
    n = min(len(c) for c in cs)
    a.plot(np.median([c[:n] for c in cs], axis=0), color=col, lw=2.4, label=lab)
a.axvline(2900, color=C['red'], lw=1.8, ls='--')
a.annotate(' the screen reads here', (2900, 12), fontsize=7.6, color=C['red'])
a.set_yscale('log')
a.set_xlabel('iteration'); a.set_ylabel('|lifetime error| (%)')
a.set_title('(b) THE TRAP. At 2 900 K2 looks 2× better; BT2 is merely\n'
            'behind, and passes it by iteration 5 000.', fontsize=9.4)
a.legend(fontsize=7.4, frameon=False)

# ---------------- (c) where the geometry error actually lives ----------------
a = ax[2]
xs = np.arange(3)
a.bar(xs - .19, MODE_TRUE, width=.36, color=C['blue'], edgecolor='white', label='true amplitude')
a.bar(xs + .19, MODE_ERR, width=.36, color=C['red'], edgecolor='white', label='fit error')
for i in range(3):
    a.text(i + .19, MODE_ERR[i] + .004, f'{100*(1-MODE_ERR[i]/MODE_TRUE[i]):+.0f}%', ha='center',
           fontsize=7.8, color=C['red'])
a.set_xticks(xs); a.set_xticklabels(['mode 1', 'mode 2', 'mode 3'], fontsize=9)
a.set_ylabel('rms spline coefficient (cm)')
tot = np.sqrt(sum(np.array(MODE_ERR) ** 2))
no3 = np.sqrt(sum(np.array(MODE_ERR[:2]) ** 2))
perf1 = np.sqrt(sum(np.array(MODE_ERR[1:]) ** 2))
a.set_title(f'(c) why: dropping mode 3 cuts the coefficient error only\n'
            f'{100*(1-no3/tot):.1f}% — mode 1 dominates, and would give {100*(1-perf1/tot):.0f}%',
            fontsize=9.4)
a.legend(fontsize=7.6, frameon=False)

st_b = np.mean([np.median(get('BT2', s)['st']) for s in SEEDS])
st_k = np.mean([np.median(get('K2', s)['st']) for s in SEEDS])
fig.suptitle('Fig 60 — dropping the noise-injecting third spline mode is a NULL on position '
             f'({b:.0f} → {k:.0f} µm) though it is {100*(1-st_k/st_b):.0f}% cheaper per iteration. '
             'Mode 3 was\nindeed noise, but the residual is dominated by MODE 1, which is only 51.6% '
             'recovered — basis truncation was never the lever.', fontsize=10.1)
fig.tight_layout(rect=[0, 0, 1, .86])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig60_k2_basis.png'); plt.close(fig)
print('wrote', f'{OUT}/fig60_k2_basis.png')
print(f'  position {b:.1f} -> {k:.1f} um ({100*(k/b-1):+.1f}%)')
print(f'  step time {st_b*1e3:.0f} -> {st_k*1e3:.0f} ms ({100*(1-st_k/st_b):.0f}% cheaper)')
print(f'  coefficient error: all modes {tot:.4f}, without mode 3 {no3:.4f}, with perfect mode 1 {perf1:.4f}')
