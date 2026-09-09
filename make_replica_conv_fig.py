"""Fig 74 — how the 8 replicas CONVERGE (loss, position, dEdx), and the mechanism hiding in it.

Panels (a-d): the convergence traces, one line per replica, COLOURED BY THAT REPLICA'S FINAL
LIFETIME ERROR (blue = small, red = large), with the campaign's fixed sample dashed black. If
lifetime's bias came from optimisation trouble, the red traces would converge visibly worse.
They do not: every sample descends the same way, to sample-appropriate levels.

PANEL (e) TESTS THE OBVIOUS MECHANISM AND DISFAVOURS IT. The dEdx mean constraint pulls every
sample's <dEdx> to the SAME imposed target 1.887 while each sample's TRUE mean differs, and the
lever arm amplifies a charge-scale error ~11x into lifetime -- a clean candidate for the +8.6%
bias. The data says no: across replicas the mismatch does not correlate with the lifetime error
(r = -0.10), and the simple charge-trade sign is the WRONG WAY (target > true mean should push
the fitted lifetime DOWN, the bias is UP).

WHAT (e) DOES ESTABLISH is a systematic defect worth fixing anyway: the imposed target exceeds
the true mean of EVERY ONE of the 8 random samples (+0.44 to +0.71%, mean +0.53), because 1.887
was tuned on the unrepresentative fixed sample (true mean 1.8803, the highest of all nine). The
constraint is mis-centred for typical data even though it is not, by itself, the lifetime story.

Panel (f): final position vs lifetime error gives r = +0.52 -- suggestive that harder-to-fit
samples bias lifetime more, but far from significant at n = 8 (p ~ 0.19). The one visible
convergence anomaly supports it qualitatively: the worst-lifetime replica (+16.4%) is also the one
whose position PLATEAUS at ~240 um from iteration ~6 000 while every other sample keeps descending.
Worth keeping an eye on, not concluding from.
"""
import glob, os, pickle, io, contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with contextlib.redirect_stdout(io.StringIO()):
    import robust_convergence as R

OUT = 'plots/noise_report'
C = dict(green='#009E73', ink='#222222', grey='#888888')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
W = 200
TGT_MEAN = 1.887          # the imposed dEdx mean-constraint target, common to every sample


def sm(v, w=W):
    v = np.asarray(v, float)
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1) if len(v) > w else v


def load(path):
    h = pickle.load(open(path, 'rb'))
    d = {}
    t = float(np.ravel(h['lifetime_target'])[0])
    d['life'] = 100 * (np.median(sm(np.ravel(h['lifetime_iter'])[1:])[-300:]) / t - 1)
    d['loss'] = sm(np.asarray(h['losses_iter'], float))
    d['pos'] = sm(np.ravel(h['pos_residual_iter']) * 1e4)
    d['mae'] = sm(np.asarray([x for x in h['dedx_mae_truth_iter'] if x == x], float))
    d['mean'] = sm(np.asarray([x for x in h['mean_dedx_iter'] if x == x], float))
    # the SAMPLE's true length-weighted mean dEdx
    num = den = 0.0
    for b, td in h['true_dedx_cache'].items():
        l = np.asarray(h['seg_dx_cache'][b], float)[:len(td)]
        td = np.asarray(td, float)
        ok = (td > 0) & (l > 0)
        num += float((l[ok] * td[ok]).sum()); den += float(l[ok].sum())
    d['true_mean'] = num / den
    return d


REP = {ds: load(glob.glob(f'fit_result/sci_full_DSREP{ds}/history_iter10000_*.pkl')[0])
       for ds in range(1, 9)}
OLD = load(glob.glob('fit_result/sci_full_BT2/history_iter10000_*seed0.pkl')[0])

lv = np.array([REP[d]['life'] for d in range(1, 9)])
norm = matplotlib.colors.Normalize(vmin=lv.min(), vmax=lv.max())
cmap = matplotlib.cm.coolwarm
col = {d: cmap(norm(REP[d]['life'])) for d in range(1, 9)}

fig, ax = plt.subplots(2, 3, figsize=(16.8, 9.6))
fig.subplots_adjust(hspace=.42, top=.86)

TRACES = [('loss', 'loss (PPP NLL)', True, ax[0, 0], '(a) LOSS: every sample converges the same '
           'way, to its own level.\nNothing distinguishes the high-bias (red) samples.'),
          ('pos', 'position residual (µm)', True, ax[0, 1], '(b) POSITION: parallel descent — with ONE '
           'exception: the worst-\nlifetime sample (deep red) plateaus at ~240 µm from iteration ~6 000.'),
          ('mae', 'per-segment dE/dx MAE vs truth', False, ax[0, 2], '(c) dE/dx MAE: descends '
           'identically in every sample.'),
          ('mean', 'length-weighted ⟨dE/dx⟩ (MeV/cm)', False, ax[1, 0], '(d) ⟨dE/dx⟩: every sample '
           'is PULLED TO THE SAME imposed target\n(grey line) — but their TRUE means (ticks, '
           'right) all differ.')]
for key, ylab, logy, a, title in TRACES:
    for d in range(1, 9):
        a.plot(REP[d][key], color=col[d], lw=1.6, alpha=.85)
    a.plot(OLD[key], color=C['ink'], lw=2.0, ls='--', label='the campaign’s fixed sample')
    if key == 'mean':
        a.axhline(TGT_MEAN, color=C['grey'], lw=2.2)
        a.annotate(f'  imposed target {TGT_MEAN}', (100, TGT_MEAN), fontsize=7.6,
                   color=C['grey'], va='bottom')
        for d in range(1, 9):
            a.plot([9700, 9950], [REP[d]['true_mean']] * 2, color=col[d], lw=2.4)
        a.plot([9700, 9950], [OLD['true_mean']] * 2, color=C['ink'], lw=2.4, ls='--')
        a.annotate('true means →', (9650, max(REP[d]['true_mean'] for d in range(1, 9)) + .002),
                   fontsize=7.4, ha='right')
    if logy:
        a.set_yscale('log')
    a.set_xlabel('iteration'); a.set_ylabel(ylab)
    a.set_title(title, fontsize=9.2)
    if key == 'loss':
        a.legend(fontsize=7.4, frameon=False, loc='upper right')

# ---------------- (e) the mechanism ----------------
a = ax[1, 1]
mm = np.array([100 * (TGT_MEAN / REP[d]['true_mean'] - 1) for d in range(1, 9)])
a.scatter(mm, lv, s=110, c=[col[d] for d in range(1, 9)], edgecolor=C['ink'], zorder=4)
mo = 100 * (TGT_MEAN / OLD['true_mean'] - 1)
a.plot(mo, OLD['life'], '*', ms=20, color=C['ink'], zorder=5)
a.annotate('  fixed sample', (mo, OLD['life']), fontsize=7.8)
r = np.corrcoef(mm, lv)[0, 1]
b1, b0 = np.polyfit(mm, lv, 1)
xs = np.linspace(min(mm.min(), mo) - .05, mm.max() + .05, 10)
a.plot(xs, b0 + b1 * xs, color=C['grey'], lw=1.6, ls=':')
a.axhline(0, color=C['green'], lw=1.6); a.axvline(0, color=C['grey'], lw=1)
a.set_xlabel('imposed ⟨dE/dx⟩ target − sample TRUE mean  (%)')
a.set_ylabel('lifetime error (%)')
a.set_title(f'(e) THE OBVIOUS MECHANISM, DISFAVOURED: mean-target mismatch vs\n'
            f'lifetime error gives r = {r:+.2f} (and the wrong sign for a charge trade).\n'
            f'But note: the imposed target exceeds the true mean of ALL 8 samples.',
            fontsize=9.0)

# ---------------- (f) convergence quality does not predict the bias ----------------
a = ax[1, 2]
pf = np.array([REP[d]['pos'][-1] for d in range(1, 9)])
a.scatter(pf, lv, s=110, c=[col[d] for d in range(1, 9)], edgecolor=C['ink'], zorder=4)
a.plot(OLD['pos'][-1], OLD['life'], '*', ms=20, color=C['ink'], zorder=5)
r2 = np.corrcoef(pf, lv)[0, 1]
a.axhline(0, color=C['green'], lw=1.6)
a.set_xlabel('final position residual (µm)')
a.set_ylabel('lifetime error (%)')
a.set_title(f'(f) convergence quality vs the bias: r = {r2:+.2f} — suggestive\n'
            f'that harder samples bias more, but not significant at n = 8.', fontsize=9.2)

sm_ = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm_, ax=ax, fraction=.025, pad=.01)
cb.set_label('final lifetime error of the replica (%)')

fig.suptitle('Fig 74 — convergence across the 8 data replicas (colour = that replica’s lifetime '
             'error; dashed black = the campaign’s fixed sample).\nEvery sample converges cleanly '
             'and identically in shape — the bias is not a convergence pathology. The obvious '
             'mechanism (the fixed dE/dx mean\ntarget vs each sample’s true mean) is DISFAVOURED '
             '(e) — though that target does exceed every random sample’s true mean and should be '
             're-centred.', fontsize=10.2)
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig74_replica_convergence.png'); plt.close(fig)
print('wrote', f'{OUT}/fig74_replica_convergence.png')
print(f'\nmechanism test: r = {r:+.3f}, slope = {b1:.2f} (lever-arm ~1/0.087 = 11.5)')
for d in range(1, 9):
    print(f'  rep{d}: true mean {REP[d]["true_mean"]:.4f}  mismatch {100*(TGT_MEAN/REP[d]["true_mean"]-1):+.3f}%  life {REP[d]["life"]:+.2f}%')
print(f'  fixed: true mean {OLD["true_mean"]:.4f}  mismatch {mo:+.3f}%  life {OLD["life"]:+.2f}%')
print(f'convergence-quality test: r = {r2:+.3f}')
