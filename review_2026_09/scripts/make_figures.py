import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

OUT = '/lscratch/pgranger/tmp/claude-45184/-sdf-group-neutrino-pgranger-larnd-sim-jax/e77eab56-46d4-44e7-b115-2992a0b283ba/scratchpad/C_curve'

# Okabe-Ito colorblind-safe categorical palette (fixed order)
C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_SKY, C_YELLOW, C_GREY, C_INK = (
    '#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442',
    '#888888', '#222222')
DSEED_COLOR = {1: C_BLUE, 3: C_ORANGE, 7: C_GREEN, 8: C_RED}

plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .25,
                      'axes.spines.top': False, 'axes.spines.right': False,
                      'figure.dpi': 150, 'savefig.bbox': 'tight',
                      'savefig.dpi': 150})

df = pd.read_csv(f'{OUT}/all_runs.csv')
curve = pd.read_csv(f'{OUT}/curve_runs.csv')

# ============================================================ G1 ============================================================
VALID_TAUS = [800, 1200, 1800, 2400, 3200]
TAUS_ALL = sorted(curve.tau.unique())
DSS = sorted(curve.dseed.unique())

sub = curve[curve.tau.isin(VALID_TAUS)]
Ti = sub.tau.values.astype(float)
Fi = sub.lifetime_endpoint.values.astype(float)
a_i, b_i = np.polyfit(Ti, Fi, 1)

# LODO
lodo_pred = {}
for held in DSS:
    train = sub[sub.dseed != held]
    a_h, b_h = np.polyfit(train.tau.values.astype(float), train.lifetime_endpoint.values.astype(float), 1)
    lodo_pred[held] = (a_h, b_h)

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

# (a) tau_fit vs tau_true
ax = axes[0]
for ds in DSS:
    g = curve[curve.dseed == ds]
    valid_mask = g.tau.isin(VALID_TAUS)
    ax.scatter(g.tau[valid_mask], g.lifetime_endpoint[valid_mask], s=55, color=DSEED_COLOR[ds],
               edgecolors='white', lw=.6, zorder=4, label=f'D-seed {ds}')
    ax.scatter(g.tau[~valid_mask], g.lifetime_endpoint[~valid_mask], s=55, color=DSEED_COLOR[ds],
               marker='x', zorder=4)
xx = np.linspace(600, 4700, 10)
ax.plot(xx, xx, '--', color=C_GREY, lw=1.6, label='unbiased (y = x)', zorder=2)
ax.plot(xx[xx <= 3400], a_i * xx[xx <= 3400] + b_i, '-', color=C_INK, lw=2.2,
        label=f'in-sample fit (800-3200 µs):\n{a_i:.3f}·τ − {abs(b_i):.0f}', zorder=3)
ax.plot(xx[xx >= 3200], a_i * xx[xx >= 3200] + b_i, ':', color=C_INK, lw=1.4, zorder=3)
ax.set_xlabel('true lifetime τ (µs)')
ax.set_ylabel('fitted lifetime τ_fit (µs)')
ax.set_title('(a) response curve — × marks τ=4400 (noise-limited,\nexcluded from the fit)')
ax.legend(frameon=False, fontsize=7.8, loc='upper left')

# (b) residuals: in-sample vs LODO out-of-sample
ax = axes[1]
in_resid = Fi - (a_i * Ti + b_i)
lodo_resid_all = []
for ds in DSS:
    g = sub[sub.dseed == ds]
    a_h, b_h = lodo_pred[ds]
    pred = a_h * g.tau.values.astype(float) + b_h
    r = g.lifetime_endpoint.values.astype(float) - pred
    lodo_resid_all.extend(r.tolist())
    ax.scatter(g.tau, r, s=60, color=DSEED_COLOR[ds], marker='D', zorder=4,
               edgecolors='white', lw=.6)
ax.scatter(Ti, in_resid, s=30, color=C_GREY, marker='o', alpha=.55, zorder=3,
           label=f'in-sample residual (RMS {np.std(in_resid, ddof=2):.0f} µs)')
lodo_resid_all = np.array(lodo_resid_all)
rms_lodo = np.sqrt(np.mean(lodo_resid_all**2))
ax.axhline(0, color=C_GREEN, lw=1.6)
ax.axhspan(-rms_lodo, rms_lodo, color=C_RED, alpha=.08, zorder=1)
ax.set_xlabel('true lifetime τ (µs)')
ax.set_ylabel('τ_fit − line prediction (µs)')
ax.set_title(f'(b) diamonds = honest leave-one-D-seed-out residual\n(RMS {rms_lodo:.0f} µs = {100*rms_lodo/np.mean(Ti):.1f}%) '
             f'vs quoted ±35 µs (2.9%)')
handles = [plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=C_GREY, markersize=8, label='LODO out-of-sample'),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C_GREY, alpha=.55, markersize=6, label='in-sample (4-seed fit)')]
ax.legend(handles=handles, frameon=False, fontsize=8, loc='upper left')

# (c) Delta(1/tau) vs tau_true
ax = axes[2]
sub2 = curve.copy()
sub2['d_inv'] = 1.0 / sub2.lifetime_endpoint - 1.0 / sub2.tau
valid_mask = sub2.tau.isin(VALID_TAUS)
for ds in DSS:
    g = sub2[sub2.dseed == ds]
    vm = g.tau.isin(VALID_TAUS)
    ax.scatter(g.tau[vm], g.d_inv[vm] * 1e5, s=55, color=DSEED_COLOR[ds], zorder=4, edgecolors='white', lw=.6)
    ax.scatter(g.tau[~vm], g.d_inv[~vm] * 1e5, s=55, color=DSEED_COLOR[ds], marker='x', zorder=4)
per_tau = sub2[valid_mask].groupby('tau').d_inv.agg(['mean', 'std', 'count'])
per_tau['se'] = per_tau['std'] / np.sqrt(per_tau['count'])
ax.errorbar(per_tau.index, per_tau['mean'] * 1e5, yerr=per_tau['se'] * 1e5, fmt='o', color=C_INK,
            ms=8, capsize=4, zorder=5, label='mean ± SE (n=4)')
w = 1 / per_tau['se'].values**2
const = np.sum(per_tau['mean'].values * w) / np.sum(w)
const_err = 1 / np.sqrt(np.sum(w))
chi2dof = np.sum(w * (per_tau['mean'].values - const)**2) / (len(per_tau) - 1)
ax.axhline(const * 1e5, color=C_RED, lw=2,
           label=f'weighted const: {const*1e5:.2f}±{const_err*1e5:.2f}e-5/µs\n(χ²/dof={chi2dof:.2f})')
ax.axhspan((const - const_err) * 1e5, (const + const_err) * 1e5, color=C_RED, alpha=.1)
ax.set_xlabel('true lifetime τ (µs)')
ax.set_ylabel('Δ(1/τ) = 1/τ_fit − 1/τ_true  (×10⁻⁵ /µs)')
ax.set_title('(c) attenuation-rate bias: is it constant?\n(spread of per-τ means: %.0f%% of mean)' % (
    100 * per_tau['mean'].std(ddof=1) / abs(per_tau['mean'].mean())))
ax.legend(frameon=False, fontsize=7.8, loc='lower right')

fig.suptitle('G1 — Independent recompute of the CURVE2400/CALIB response curve (TAGs CURVE{800..4400}D{1,3,7,8}, n=4 D-seeds/τ).\n'
             'The in-sample fit (a) reproduces the reported line, but the honest leave-one-D-seed-out residual (b) is ~%.0f µs (%.1f%%) — about %.1fx the quoted ±35 µs (2.9%%).\n'
             'The Δ(1/τ) constant-offset model (c) is statistically consistent with the data (χ²/dof≈%.1f), supporting that part of the claim.' % (
                 rms_lodo, 100*rms_lodo/np.mean(Ti), rms_lodo/35, chi2dof), fontsize=10.5, y=1.09)
fig.tight_layout()
fig.savefig(f'{OUT}/G1_calibration_curve.png')
print('wrote G1')

# ============================================================ G2 ============================================================
PARAMS = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']

def mean_abs_err(row):
    return np.mean([abs(row[f'{p}_err_pct']) for p in PARAMS])

df['mean_abs_err'] = df.apply(mean_abs_err, axis=1)
baseline = df[df.tag.str.startswith('DSREP')].copy()
baseline['dseed'] = baseline.tag.str.extract(r'DSREP(\d+)').astype(int)
baseline = baseline.set_index('dseed')
arm = df[df.tag.str.startswith('NARR040D')].copy()
arm['dseed'] = arm.tag.str.extract(r'NARR040D(\d+)').astype(int)
arm = arm.set_index('dseed')

TUNED = [3, 7, 8]
FRESH = [1, 2, 4, 5, 6]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

for ax, metric, title in [(axes[0], 'mean_abs_err', 'mean |error| over 5 calibration params (%)'),
                            (axes[1], 'lifetime_err_pct', '|lifetime error| (%)')]:
    for grp, col, lab, off in ((TUNED, C_RED, 'tuned seeds (3,7,8)', -0.10),
                                (FRESH, C_BLUE, 'fresh seeds (1,2,4,5,6)', 0.10)):
        if metric == 'lifetime_err_pct':
            b = np.abs(baseline.loc[grp, metric].values)
            x = np.abs(arm.loc[grp, metric].values)
        else:
            b = baseline.loc[grp, metric].values
            x = arm.loc[grp, metric].values
        for i in range(len(b)):
            ax.plot([0 + off, 1 + off], [b[i], x[i]], '-', color=col, lw=1, alpha=.4, zorder=2)
        d = x - b
        se = d.std(ddof=1) / np.sqrt(len(d))
        ax.plot([0 + off, 1 + off], [b.mean(), x.mean()], 'o-', color=col, lw=2.8, ms=10, zorder=4,
                label=f'{lab}: Δ = {d.mean():+.2f} ± {se:.2f}  ({int((d<0).sum())}/{len(d)} improved)')
    ax.set_xticks([0, 1], ['production\n(×1.0, DSREP)', 'x0.40 prior\n(NARR040D)'])
    ax.set_xlim(-.4, 1.4)
    ax.set_ylabel(title)
    ax.legend(frameon=False, fontsize=8, loc='upper right')

axes[0].set_title('(a) mean |error| across all 5 calib params')
axes[1].set_title('(b) |lifetime error| alone')
fig.suptitle('G2 — independent recompute: paired ×0.40-vs-×1.0 change, tuned vs fresh D-seeds (baseline = DSREP{k}, matched by data_seed).\n'
             'Confirms the headline "evaporates" on fresh seeds: mean|err| Δ = −4.31±1.48 (tuned) vs −0.34±0.65 (fresh, not significant).',
             fontsize=10.3, y=1.06)
fig.tight_layout()
fig.savefig(f'{OUT}/G2_paired_NARR040.png')
print('wrote G2')

# ============================================================ G3 ============================================================
fig, ax = plt.subplots(figsize=(7.2, 6.2))

FAMILY_COLOR = {
    'CURVE': C_GREY, 'DSREP': C_BLUE, 'NARR': C_ORANGE, 'WIDE': C_PURPLE, 'LDPIN': C_RED,
}
def family(tag):
    for k in FAMILY_COLOR:
        if tag.startswith(k):
            return k
    return 'other'

df['family'] = df.tag.apply(family)
for fam, col in FAMILY_COLOR.items():
    g = df[df.family == fam]
    ax.scatter(g.long_diff_err_pct, g.lifetime_err_pct, s=45, color=col, alpha=.75,
               edgecolors='white', lw=.4, label=f'{fam} (n={len(g)})', zorder=3)

allx = df.long_diff_err_pct.values
ally = df.lifetime_err_pct.values
r, p = stats.pearsonr(allx, ally)
b1, b0 = np.polyfit(allx, ally, 1)
xx = np.linspace(allx.min(), allx.max(), 10)
ax.plot(xx, b1 * xx + b0, '--', color=C_INK, lw=1.6, zorder=2,
        label=f'linear fit, all runs: r={r:+.2f} (p={p:.1e})')
ax.axhline(0, color=C_GREY, lw=1, alpha=.6)
ax.axvline(0, color=C_GREY, lw=1, alpha=.6)
ax.set_xlabel('long_diff error (%)')
ax.set_ylabel('lifetime error (%)')
ax.set_title(f'G3 — long_diff error vs lifetime error, ALL {len(df)} loaded runs\n'
             '(CURVE/DSREP/NARR/WIDE/LDPIN families) — soft-plane degeneracy check')
ax.legend(frameon=False, fontsize=7.5, loc='upper left', ncol=1)
fig.tight_layout()
fig.savefig(f'{OUT}/G3_soft_plane_scatter.png')
print('wrote G3')
print(f'G3 overall Pearson r = {r:.3f}, p = {p:.2e}, n = {len(df)}')
