"""Fig. prob_vs_stoch (rework, 9 Oct 2026): the probabilistic electronics model against 1e4 stochastic throws of the
SAME waveforms, per pixel and per hit (first and second): hit probability lambda, mean hit time E[t] and mean ADC.
Waveforms: the census of fig3_probabilistic_method.py (one batch of real through-going muons, production
parameters, 1024 pixel slots). Model: production beam search (get_adc_values_average_noise_chunked, K=100) and
get_average_hit_values. Stochastic: fee_jax.get_adc_values (production noise model), N throws in chunks.
usage: JAX_PLATFORMS=cpu python fig_prob_vs_stoch.py [--compute] [--n_throws 10000] [--chunk 50]"""
import os, sys, time, argparse
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(HERE, 'fig3_census_cache.npz')
CACHE = '/sdf/scratch/neutrino/pgranger/review_2026_09/SAMPLING_STEP/prob_vs_stoch_cache.npz'
OUT = '/sdf/group/neutrino/pgranger/NuLads_draft/figures/probabilistic_vs_stochastic'
ap = argparse.ArgumentParser(); ap.add_argument('--compute', action='store_true')
ap.add_argument('--n_throws', type=int, default=10000); ap.add_argument('--chunk', type=int, default=50)
ap.add_argument('--time_only', action='store_true'); a = ap.parse_args()

if a.compute or not os.path.exists(CACHE):
    sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS')
    import jax, jax.numpy as jnp
    import paper_common as pc
    from larndsim.fee_jax import get_adc_values_average_noise_chunked, digitize, get_average_hit_values, get_adc_values
    truth, _ = pc.params()
    c = np.load(CENSUS); valid = c['valid']; wfs = jnp.asarray(c['wfs'][valid]); npx = wfs.shape[0]
    t0 = time.time()
    lp, ch = get_adc_values_average_noise_chunked(truth, wfs, chunk_size=128)
    Et, Eadc, lam = get_average_hit_values(lp, digitize(truth, ch))
    Et, Eadc, lam = (np.asarray(x) for x in (Et, Eadc, lam))
    print(f'model: {npx} pixels, {time.time()-t0:.1f}s', flush=True)
    NH = 2
    cnt = np.zeros((npx, NH)); st = np.zeros((npx, NH)); sa = np.zeros((npx, NH)); sa2 = np.zeros((npx, NH)); st2 = np.zeros((npx, NH))
    tot = np.zeros(npx)
    run = jax.jit(lambda w, k: get_adc_values(truth, w, k))
    nchunks = a.n_throws // a.chunk; key = jax.random.key(2026)
    t0 = time.time()
    for i in range(nchunks):
        key, sub = jax.random.split(key)
        q, tk = run(jnp.tile(wfs, (a.chunk, 1)), sub)
        adc = np.asarray(digitize(truth, q)).reshape(a.chunk, npx, -1); qq = np.asarray(q).reshape(a.chunk, npx, -1)
        tk = np.asarray(tk).reshape(a.chunk, npx, -1)
        hit = qq > 0
        tot += hit.sum(axis=(0, 2))
        for n in range(NH):
            h = hit[:, :, n]
            cnt[:, n] += h.sum(0); st[:, n] += np.where(h, tk[:, :, n], 0).sum(0); st2[:, n] += np.where(h, tk[:, :, n] ** 2, 0).sum(0)
            sa[:, n] += np.where(h, adc[:, :, n], 0).sum(0); sa2[:, n] += np.where(h, adc[:, :, n] ** 2, 0).sum(0)
        if i == 0: print(f'first chunk ({a.chunk} throws): {time.time()-t0:.1f}s', flush=True)
        if a.time_only: sys.exit(0)
        if (i + 1) % 20 == 0: print(f'{(i+1)*a.chunk} throws, {time.time()-t0:.0f}s', flush=True)
    N = nchunks * a.chunk
    np.savez(CACHE, lam=lam[:, :NH], Et=Et[:, :NH], Eadc=Eadc[:, :NH], lam_all=lam, cnt=cnt, st=st, st2=st2, sa=sa, sa2=sa2,
             tot=tot, N=N, t_sampling=float(truth.t_sampling))
    print('cached', CACHE, flush=True)

# ------------------------------------------------------------------ figure ------------------------------------------
import json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
z = dict(np.load(CACHE)); N = float(z['N'])
C0, C1, BLACK = '#0072B2', '#D55E00', '#000000'
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6, 'font.family': 'sans-serif'})
f = z['cnt'] / N                                   # stochastic hit frequency per (pixel, hit)
mt = z['st'] / np.maximum(z['cnt'], 1); sdt = np.sqrt(np.maximum(z['st2'] / np.maximum(z['cnt'], 1) - mt ** 2, 0))
ma = z['sa'] / np.maximum(z['cnt'], 1); sda = np.sqrt(np.maximum(z['sa2'] / np.maximum(z['cnt'], 1) - ma ** 2, 0))
dl = z['lam'] - f; dt = z['Et'] - mt; da = z['Eadc'] - ma
MINC = 200                                          # pixels with at least this many stochastic hits for mean comparisons
nums = {}
# 12 Oct (Pierre): only the three distributions are kept (the former bottom row); the scatter panels were dropped.
# The Monte Carlo error of the sampled estimates (N throws) is computed per (pixel, hit) and reported, since it accounts
# for part of the widths.
from matplotlib.lines import Line2D
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.25)); fig.subplots_adjust(wspace=0.42)
lab = ['first hit', 'second hit']; col = [C0, C1]
se = {'lambda': np.sqrt(np.clip(f * (1 - f), 0, None) / N), 'time': sdt / np.sqrt(np.maximum(z['cnt'], 1)),
      'adc': sda / np.sqrt(np.maximum(z['cnt'], 1))}
for j, (res, name, unit, lim, spread) in enumerate(((dl, 'lambda', '', 0.03, None), (dt, 'time', ' ticks', 1.5, sdt), (da, 'adc', ' counts', 0.3, sda))):
    ax = axs[j]; bins = np.linspace(-lim, lim, 61)
    for n in (0, 1):
        m = (z['cnt'][:, n] >= (1 if j == 0 else MINC)) | ((j == 0) & (z['lam'][:, n] > 1e-3))
        r = res[m, n]; med = np.median(r); lo, hi = np.percentile(r, [16, 84])
        e = se[name][m, n]
        nums[f'{name}_{n}'] = dict(n=int(m.sum()), median=float(med), p16=float(lo), p84=float(hi), mean=float(np.mean(r)),
                                   mean_abs=float(np.mean(np.abs(r))), spread_median=float(np.median(spread[m, n])) if spread is not None else None,
                                   mc_se_median=float(np.median(e)), half68=float(0.5 * (hi - lo)),
                                   frac_within_2se=float(np.mean(np.abs(r) <= 2 * e)))
        ax.hist(np.clip(r, -lim, lim), bins=bins, histtype='step', color=col[n], lw=1.1)
    ax.set_title(['(a) hit probability', '(b) mean hit time', '(c) mean charge'][j], fontsize=8)
    ax.set_yscale('log'); ax.set_ylabel('pixels')
    ax.set_ylim(top=ax.get_ylim()[1] * 8)
    ax.set_xlabel([r'$\Delta\lambda$', r'$\Delta E[t]$ (ticks)', r'$\Delta$ mean ADC (counts)'][j])
    if spread is not None:
        s_ = np.median(spread[z['cnt'][:, 0] >= MINC, 0])
        ax.text(0.97, 0.95, f'noise spread per hit:\n{s_:.1f}{unit} (median)', transform=ax.transAxes, ha='right', va='top', fontsize=5.8, color='0.3')
axs[0].legend(handles=[Line2D([0], [0], color=c, lw=1.1, label=l) for c, l in zip(col, lab)], frameon=False, loc='upper left', fontsize=6)
for ax in axs:
    ax.grid(alpha=0.25, lw=0.5)
tot_model = float(z['lam_all'].sum()); tot_stoch = float(z['tot'].sum() / N)
nums.update(total_hits_model=tot_model, total_hits_stoch=tot_stoch, total_rel=tot_model / tot_stoch - 1, N=N, pixels=int(f.shape[0]))
fig.savefig(OUT + '.pdf', bbox_inches='tight'); fig.savefig(OUT + '.png', dpi=250, bbox_inches='tight')
json.dump(nums, open(CACHE.replace('.npz', '_numbers.json'), 'w'), indent=1); print(json.dumps(nums, indent=1))
