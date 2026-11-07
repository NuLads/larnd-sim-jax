"""Fig. noise_effect v5 (option A, 19 Oct 2026): per-pixel noise throws on the waveforms of Fig. prob_vs_stoch
(fig3_census_cache.npz, production parameters, 926 active pixels), with the production stochastic electronics
(fee_jax.get_adc_values); noiseless = same function with RESET_NOISE_CHARGE = UNCORRELATED_NOISE_CHARGE = 0.
Stores, per pixel: noiseless hits, peak cumulative charge Q_max, peak charge collected after the first noiseless reset
R_max, and over N throws the counts of >=1 and >=2 hits and the hit-multiplicity histogram.
usage: JAX_PLATFORMS=cpu python noise_throws.py N_THROWS TILE OUT.npz"""
import os, sys, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
N, TILE, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS')
import jax, jax.numpy as jnp
import paper_common as pc
from larndsim.fee_jax import get_adc_values
truth, _ = pc.params()
c = np.load('/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/paperfigs/fig3_census_cache.npz')
wfs = np.asarray(c['wfs'][c['valid']]); P = len(wfs)
ts = float(truth.t_sampling); thr = float(truth.DISCRIMINATION_THRESHOLD); sig_r = float(truth.RESET_NOISE_CHARGE)
interval = round((3 * truth.CLOCK_CYCLE + truth.ADC_HOLD_DELAY * truth.CLOCK_CYCLE) / ts)
nf = truth.replace(RESET_NOISE_CHARGE=0., UNCORRELATED_NOISE_CHARGE=0.)
q0, t0 = get_adc_values(nf, jnp.asarray(wfs), jax.random.key(0)); q0, t0 = np.asarray(q0), np.asarray(t0)
n0 = (q0 > 0).sum(1)
cum = np.cumsum(wfs, 1) * ts; Qmax = cum.max(1)
Rmax = np.full(P, np.nan)
for j in np.where(n0 >= 1)[0]:                     # charge collected after the first noiseless reset (as in fee_jax)
    tr = min(int(t0[j, 0]) + 1 + interval + 1, cum.shape[1] - 1)
    Rmax[j] = np.clip(cum[j, tr:] - cum[j, tr], 0, None).max()
run = jax.jit(lambda w, k: get_adc_values(truth, w, k))
W = jnp.asarray(np.tile(wfs, (TILE, 1)))
ge1 = np.zeros(P); ge2 = np.zeros(P); hist = np.zeros((P, 11), np.int64); key = jax.random.key(2026)
t_start = time.time(); done = 0
while done < N:
    key, sub = jax.random.split(key)
    q, _ = run(W, sub); nh = (np.asarray(q) > 0).sum(1).reshape(TILE, P)
    ge1 += (nh >= 1).sum(0); ge2 += (nh >= 2).sum(0)
    for k in range(TILE): hist[np.arange(P), nh[k]] += 1
    done += TILE
    if done % (10 * TILE) == 0 or done >= N: print(f'{done}/{N} throws, {time.time() - t_start:.0f} s', flush=True)
np.savez_compressed(OUT, n0=n0, Qmax=Qmax, Rmax=Rmax, ge1=ge1, ge2=ge2, hist=hist, N=done, thr=thr, sig_r=sig_r,
                    sig_u=float(truth.UNCORRELATED_NOISE_CHARGE))
print('noiseless total hits', int(n0.sum()), ' mean total with noise', float((hist * np.arange(11)).sum() / done))
