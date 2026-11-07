"""Fig. noise_effect, version 4 (12 Oct 2026, simpler, Pierre): two pixels from the waveforms of Fig. prob_vs_stoch
(fig3_census_cache.npz, production parameters), noiseless readout against 2000 noise throws (fee_jax.get_adc_values, the
production noise model; noiseless = the same function with RESET_NOISE_CHARGE = UNCORRELATED_NOISE_CHARGE = 0).
  (a) a pixel near threshold: one noiseless hit; with noise the hit is absent in a fraction of throws and its time is
      spread, with an earlier mean;
  (b) retriggering: one noiseless hit, after which the charge collected since the reset stays below threshold, so a second
      hit is impossible without noise; the reset noise (sigma_r) lets it cross, and a second hit occurs in a large fraction
      of throws.
Runs on CPU in about a minute. usage: JAX_PLATFORMS=cpu python fig_noise_effect_v4.py"""
import os, sys, json
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(HERE, 'fig3_census_cache.npz')
OUT = '/sdf/group/neutrino/pgranger/NuLads_draft/figures/noise_effect_v4'
N, CHUNK = 2000, 200
PIX_A, PIX_B = 852, 597      # near-threshold single hit; single noiseless hit whose post-reset charge plateaus 0.2 ke below threshold

sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS')
import jax, jax.numpy as jnp
import paper_common as pc
from larndsim.fee_jax import get_adc_values
truth, _ = pc.params()
c = np.load(CENSUS); wfs_all = np.asarray(c['wfs'][c['valid']])
ts = float(truth.t_sampling); thr = float(truth.DISCRIMINATION_THRESHOLD); sig_r = float(truth.RESET_NOISE_CHARGE)
interval = round((3 * truth.CLOCK_CYCLE + truth.ADC_HOLD_DELAY * truth.CLOCK_CYCLE) / ts)
nf = truth.replace(RESET_NOISE_CHARGE=0., UNCORRELATED_NOISE_CHARGE=0.)
res = {}
for tag, j in (('a', PIX_A), ('b', PIX_B)):
    w = jnp.asarray(wfs_all[j:j + 1])
    q0, t0 = get_adc_values(nf, w, jax.random.key(0)); q0, t0 = np.asarray(q0)[0], np.asarray(t0)[0]
    n0 = int((q0 > 0).sum()); first0 = float(t0[0])
    T1, T2, NH = [], [], []
    key = jax.random.key(11 if tag == 'a' else 12)
    run = jax.jit(lambda ww, k: get_adc_values(truth, ww, k))
    for i in range(N // CHUNK):
        key, sub = jax.random.split(key)
        q, tk = run(jnp.tile(w, (CHUNK, 1)), sub); q, tk = np.asarray(q), np.asarray(tk)
        NH.append((q > 0).sum(1)); T1.append(np.where(q[:, 0] > 0, tk[:, 0], np.nan)); T2.append(np.where(q[:, 1] > 0, tk[:, 1], np.nan))
    T1, T2, NH = np.concatenate(T1), np.concatenate(T2), np.concatenate(NH)
    cum = np.cumsum(np.asarray(w)[0]) * ts
    # noiseless reset: the integration window closes `interval` ticks after the crossing, the charge is removed one tick later
    t_reset = int(first0) + 1 + interval + 1
    resid = cum - cum[min(t_reset, len(cum) - 1)]
    res[tag] = dict(j=j, n0=n0, first0=first0, T1=T1, T2=T2, NH=NH, cum=cum, t_reset=t_reset, resid=resid, q=float(cum.max()))
    print(f'({tag}) pixel {j}: {cum.max()/1e3:.1f} ke, noiseless hits {n0}; with noise: P(no hit) {np.mean(NH == 0):.2f}, P(>=2) {np.mean(NH >= 2):.2f}, '
          f'mean first-hit shift {np.nanmean(T1 - first0) * ts:+.2f} us, residual after reset max {resid[t_reset:].max()/1e3:.2f} ke', flush=True)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.3, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6, 'font.family': 'sans-serif'})
C0, C1, CG = '#D55E00', '#0072B2', '#7F7F7F'
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6))
# (a)
r = res['a']; t0 = r['first0']; tt = (np.arange(len(r['cum'])) - t0) * ts
a.plot(tt, r['cum'] / 1e3, color='k', lw=1.1, label='charge on the pixel, no noise')
a.axhline(thr / 1e3, color=CG, ls=':', lw=0.7); a.text(3.9, thr / 1e3 * 1.03, 'threshold', ha='right', fontsize=6, color=CG)
a.axvline(0, color=C0, lw=1.2, label='hit without noise')
ok = np.isfinite(r['T1'])
a2 = a.twinx(); a2.hist((r['T1'][ok] - t0) * ts, bins=np.arange(-4, 4.01, 0.2), color=C1, alpha=0.45, label='first hit with noise')
a2.set_yticks([]); a2.set_ylim(0, a2.get_ylim()[1] * 2.4)
a.axvline(np.nanmean(r['T1'] - t0) * ts, color=C1, lw=1.2, ls='--', label='its mean')
a.set_xlim(-4, 4); a.set_ylim(0, r['cum'].max() / 1e3 * 1.6)
a.set_xlabel('time from the hit without noise ($\\mu$s)'); a.set_ylabel('charge (ke$^-$)')
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, loc='upper left', frameon=False)
a.text(0.97, 0.06, f'no hit in {100*np.mean(r["NH"] == 0):.0f}% of throws', transform=a.transAxes, ha='right', fontsize=6.3, color=C1)
a.text(0.97, 0.94, '(a)', transform=a.transAxes, ha='right', va='top')
# (b)
r = res['b']; t0 = r['first0']; tr = r['t_reset']; tt = (np.arange(len(r['cum'])) - t0) * ts
m = np.arange(len(r['cum'])) >= tr
b.plot(tt[~m], r['cum'][~m] / 1e3, color='k', lw=1.1, label='charge, no noise')
b.plot(tt[m], r['resid'][m] / 1e3, color='k', lw=1.1, ls='--', label='charge since the reset, no noise')
b.fill_between(tt[m], (r['resid'][m] - sig_r) / 1e3, (r['resid'][m] + sig_r) / 1e3, color=C1, alpha=0.18, lw=0,
               label='$\\pm 1\\sigma$ reset noise')
b.axhline(thr / 1e3, color=CG, ls=':', lw=0.7)
b.axvline(0, color=C0, lw=1.2, label='hit without noise (only one)')
b.axvline((tr - t0) * ts, color=CG, lw=0.6, ls='-.')
b.text((tr - t0) * ts + 0.05, thr / 1e3 * 0.35, 'reset', fontsize=6, color=CG, rotation=90, va='bottom')
ok2 = np.isfinite(r['T2'])
b2 = b.twinx(); b2.hist((r['T2'][ok2] - t0) * ts, bins=np.arange(-0.5, 6.01, 0.1), color=C1, alpha=0.45, label='second hit with noise')
b2.set_yticks([]); b2.set_ylim(0, b2.get_ylim()[1] * 2.4)
b.set_xlim(-0.5, 6); b.set_ylim(-0.8, thr / 1e3 * 2.1)   # zoom on the threshold region; the first pulse runs off scale
b.set_xlabel('time from the first hit ($\\mu$s)'); b.set_ylabel('charge (ke$^-$)')
h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
b.legend(h1 + h2, l1 + l2, loc='upper right', frameon=False)
b.text(0.97, 0.24, f'second hit in {100*np.mean(r["NH"] >= 2):.0f}% of throws', transform=b.transAxes, ha='right', fontsize=6.3, color=C1)
b.text(0.03, 0.94, '(b)', transform=b.transAxes, va='top')
for x in (a, b): x.grid(alpha=0.25, lw=0.5)
fig.tight_layout(w_pad=1.6)
fig.savefig(OUT + '.pdf', bbox_inches='tight'); fig.savefig(OUT + '.png', dpi=250, bbox_inches='tight'); print('saved', OUT)
json.dump({k: dict(pixel=int(v['j']), charge_ke=v['q'] / 1e3, noiseless_hits=v['n0'], p_no_hit=float(np.mean(v['NH'] == 0)),
                   p_two_or_more=float(np.mean(v['NH'] >= 2)), mean_first_shift_us=float(np.nanmean(v['T1'] - v['first0']) * ts),
                   resid_max_ke=float(v['resid'][v['t_reset']:].max() / 1e3)) for k, v in res.items()},
          open('/sdf/scratch/neutrino/pgranger/review_2026_09/SAMPLING_STEP/noise_effect_v4_numbers.json', 'w'), indent=1)
