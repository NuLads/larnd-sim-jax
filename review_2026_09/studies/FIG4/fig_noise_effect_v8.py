"""Fig. noise_effect, version 8 (19 Oct 2026, option A for newcomers): mechanism first, then the batch.
The reset noise adds a constant to the cumulative charge of a pixel, so one noise draw is the noiseless curve moved up or
down, and the hit fires where the moved curve crosses the threshold (fee_jax.get_adc_values).
  (a) a typical pixel well above threshold: noiseless curve, curves moved by symmetric noise quantiles, and below the
      distribution of the crossing time over 1e4 draws: the average crossing is earlier than the noiseless one;
  (b) a pixel whose charge levels off just below threshold: no hit without noise, a hit in a fraction of the draws;
  (c) all pixels with a first hit without noise: average minus noiseless first-hit time (prob_vs_stoch cache, 1e4 throws
      of the production stochastic electronics), with the probabilistic model.
The crossing-time distributions of (a) and (b) use a numpy replica of the first-hit logic of fee_jax.get_adc_values
(reset-noise baseline, upward crossing, acceptance of the sample at the end of the hold window with the uncorrelated
noise); the replica is checked against the 1e4-throw cache below. Quoted probabilities and shifts are from the cache.
usage: python fig_noise_effect_v8.py [out_stem]"""
import sys, json
import numpy as np
from scipy.stats import norm
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/'
OUT = sys.argv[1] if len(sys.argv) > 1 else D + 'FIG4/noise_effect_v8'
PIX_A, PIX_B = 15, 731
p = np.load(D + 'SAMPLING_STEP/prob_vs_stoch_cache.npz'); v = np.load(D + 'SAMPLING_STEP/noise_effect_v3_cache.npz')
b = np.load(D + 'FIG4/noiseless_quantities.npz')
c = np.load('/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/paperfigs/fig3_census_cache.npz')
w = c['wfs'][c['valid']]; ts = float(c['t_sampling']); thr = float(c['Vth']); sr = float(c['sigma_r']); su = float(c['sigma_u'])
interval = int(c['interval']); N = int(p['N'])
h0, t0 = v['hit0'], v['t0']; assert np.array_equal(h0.sum(1), b['n0'])
cum = np.cumsum(w, 1) * ts
f1 = p['cnt'][:, 0] / N; mt = p['st'][:, 0] / np.maximum(p['cnt'][:, 0], 1)

def crossings(j, n=10000, seed=1):
    """first-hit tick per draw (nan if none), replica of the first pass of fee_jax.get_adc_values"""
    rng = np.random.default_rng(seed); eps = rng.normal(0, sr, n); eu = rng.normal(0, su, n)
    x = cum[j][None, :] + eps[:, None]
    up = (x[:, 1:] >= thr) & (x[:, :-1] <= thr)
    has = up.any(1); idx = np.where(has, up.argmax(1), -1)
    end = np.clip(idx + 1 + interval, 0, x.shape[1] - 1)
    ok = has & (x[np.arange(n), end] + eu >= thr)
    return np.where(ok, idx, np.nan)

sel = h0[:, 0] & (p['cnt'][:, 0] >= 100)
dt_mc = ((mt - t0[:, 0]) * ts)[sel]; dt_mod = ((p['Et'][:, 0] - t0[:, 0]) * ts)[sel]
cA, cB = crossings(PIX_A), crossings(PIX_B)
num = dict(pix_a=PIX_A, a_qmax_ke=float(cum[PIX_A].max() / 1e3), a_shift_us_cache=float((mt[PIX_A] - t0[PIX_A, 0]) * ts),
           a_shift_us_replica=float((np.nanmean(cA) - t0[PIX_A, 0]) * ts), a_p_hit_cache=float(f1[PIX_A]),
           pix_b=PIX_B, b_qmax_ke=float(cum[PIX_B].max() / 1e3), b_noiseless_hit=bool(h0[PIX_B, 0]),
           b_p_hit_cache=float(f1[PIX_B]), b_p_hit_replica=float(np.isfinite(cB).mean()),
           n_dt=int(sel.sum()), dt_median_us=float(np.median(dt_mc)), dt_frac_earlier=float((dt_mc < 0).mean()),
           dt_model_median_us=float(np.median(dt_mod)))
print(json.dumps(num, indent=1))

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D; from matplotlib.patches import Patch
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6, 'font.family': 'sans-serif'})
CB, CG, CA = '#0072B2', '#9A9A9A', '#D55E00'
fig = plt.figure(figsize=(7.0, 3.3))
gs = fig.add_gridspec(2, 3, height_ratios=[2.3, 1], width_ratios=[1, 1, 1.05], hspace=0.08, wspace=0.42,
                      left=0.065, right=0.985, top=0.87, bottom=0.19)
shifts = sr * norm.ppf(np.linspace(0.06, 0.94, 11))          # symmetric noise draws, for display

def mechanism(col, j, origin, xl, tag, title, shifts=shifts):
    ax = fig.add_subplot(gs[0, col]); st = fig.add_subplot(gs[1, col], sharex=ax)
    tt = (np.arange(cum.shape[1]) - origin) * ts
    for e in shifts:
        ax.plot(tt, (cum[j] + e) / 1e3, color=CG, lw=0.5, zorder=1)
        y = cum[j] + e; up = np.where((y[1:] >= thr) & (y[:-1] <= thr))[0]
        if len(up): ax.plot(tt[up[0]], thr / 1e3, 'o', ms=2.2, color=CG, zorder=2)
    ax.plot(tt, cum[j] / 1e3, color='k', lw=1.4, zorder=3)
    ax.axhline(thr / 1e3, color=CA, lw=0.9, ls='--', zorder=0)
    ax.text(xl[0] + 0.03 * (xl[1] - xl[0]), thr / 1e3 + 0.15, 'threshold', color=CA, fontsize=6.3, va='bottom')
    ax.set_xlim(*xl); ax.set_ylabel(r'charge collected (ke$^-$)'); ax.tick_params(labelbottom=False)
    ax.text(0.03, 0.97, tag, transform=ax.transAxes, va='top', fontsize=8)
    ax.set_title(title, fontsize=7.5, pad=3)
    return ax, st, tt

def t_cross(j, e):
    y = cum[j] + e; i = np.where((y[1:] >= thr) & (y[:-1] <= thr))[0][0]
    return (i + (thr - y[i]) / (y[i + 1] - y[i])) * ts

# (a)
oA = int(t0[PIX_A, 0])
ax, st, tt = mechanism(0, PIX_A, oA, (-1.6, 1.2), '(a)', 'pixel well above threshold', shifts=np.array([sr, -sr]))
ax.set_ylim(0, 9.5)
tA0 = t_cross(PIX_A, 0.); tup, tdn = t_cross(PIX_A, sr) - tA0, t_cross(PIX_A, -sr) - tA0
num.update(a_cross_up_1sig_us=float(tup), a_cross_down_1sig_us=float(tdn))
x0 = tA0 - oA * ts
ax.text(1.17, (cum[PIX_A][oA + 11] + sr) / 1e3 + 0.25, r'noise $+\sigma_r$', ha='right', fontsize=6.3, color='#606060')
ax.text(-1.55, (cum[PIX_A][oA - 16] - sr) / 1e3 - 0.35, r'noise $-\sigma_r$', ha='left', va='top', fontsize=6.3, color='#606060')
bins = np.arange(-1.65, 1.25, 0.1)
st.hist((cA[np.isfinite(cA)] - oA) * ts, bins=bins, color=CB, alpha=0.45, weights=np.full(np.isfinite(cA).sum(), 1 / len(cA)))
st.axvline(0, color='k', lw=1.4); st.axvline(num['a_shift_us_cache'], color=CB, lw=1.4, ls='--')
for tx, lab, ha in ((tup, r'$+\sigma_r$: ' + f'{tup:+.2f}', 'right'), (tdn, r'$-\sigma_r$: ' + f'{tdn:+.2f}', 'left')):
    st.axvline(tx, color='#606060', lw=0.7, ls=':')
    st.text(tx + 0.04 if ha == 'left' else -1.57, 0.40, lab, ha='left', va='top', fontsize=6.0, color='#606060')
st.text(num['a_shift_us_cache'] - 0.04, 0.22, f"average\n{num['a_shift_us_cache']:+.2f}", ha='right', va='top', fontsize=6.0, color=CB)
st.set_ylim(0, 0.42); st.set_ylabel('draws', labelpad=8); st.set_yticks([])
st.set_xlabel(r'time from the hit without noise ($\mu$s)')
# (b)
cb = cB[np.isfinite(cB)]; oB = int(np.median(cb))
ax, st, tt = mechanism(1, PIX_B, oB, (-2.5, 2.0), '(b)', 'pixel just below threshold', shifts=sr * norm.ppf(np.linspace(0.08, 0.92, 9)))
ax.set_ylim(0, 7.2)
ax.text(0.97, 0.06, 'without noise:\nno hit', transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5)
bins = np.arange(-2.55, 2.05, 0.1)
st.hist((cb - oB) * ts, bins=bins, color=CB, alpha=0.45, weights=np.full(len(cb), 1 / len(cB)))
st.text(0.97, 0.88, f"with noise: hit in {100 * num['b_p_hit_cache']:.0f}% of draws", transform=st.transAxes, ha='right', va='top',
        fontsize=6.5, color=CB)
st.set_ylabel('draws', labelpad=8); st.set_yticks([]); st.set_ylim(0, None)
st.set_xlabel(r'time from the median hit with noise ($\mu$s)')
# (c)
axc = fig.add_subplot(gs[:, 2])
bins = np.arange(-0.6, 0.12, 0.02)
axc.hist(np.clip(dt_mc, bins[0], bins[-1] - 1e-9), bins=bins, color=CB, alpha=0.45)
axc.hist(np.clip(dt_mod, bins[0], bins[-1] - 1e-9), bins=bins, histtype='step', color='k', lw=0.8)
axc.axvline(0, color='k', lw=1.4)
axc.set_xlim(bins[0], bins[-1]); axc.set_xlabel(r'shift of the first hit ($\mu$s):' + '\nwith noise (average) $-$ without'); axc.set_ylabel('pixels')
axc.text(0.03, 0.97, '(c)', transform=axc.transAxes, va='top', fontsize=8)
axc.set_title(f"all {num['n_dt']} pixels with a hit", fontsize=7.5, pad=3)
axc.text(0.03, 0.86, f"earlier on {100 * num['dt_frac_earlier']:.0f}% of pixels,\nmedian {num['dt_median_us']:+.2f} $\\mu$s\n($\\simeq 170\\,\\mu$m of drift)",
         transform=axc.transAxes, va='top', fontsize=6.5)
ya = axc.get_ylim()[1]
axc.annotate('pixel (a)', xy=(num['a_shift_us_cache'], 0.45 * ya), xytext=(-0.42, 0.45 * ya), fontsize=6.3,
             arrowprops=dict(arrowstyle='->', lw=0.6), ha='center', va='center')
handles = [Line2D([], [], color='k', lw=1.4, label='without noise'),
           Line2D([], [], color=CG, lw=0.6, marker='o', ms=2.2, label='individual noise draws'),
           Patch(color=CB, alpha=0.45, label='with noise, over $10^4$ draws'),
           Line2D([], [], color=CB, lw=1.4, ls='--', label='average with noise'),
           Patch(fill=False, ec='k', lw=0.8, label='probabilistic model')]
fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.52, 1.0), handlelength=1.6, columnspacing=1.2)
fig.savefig(OUT + '.pdf'); fig.savefig(OUT + '.png', dpi=200)
json.dump(num, open(OUT + '_numbers.json', 'w'), indent=1)
print('wrote', OUT)
