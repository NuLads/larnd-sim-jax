"""Pairs of single stochastic throws (two noise seeds) on the 926 pixels of Fig. prob_vs_stoch, with the production
stochastic electronics (fee_jax.get_adc_values + digitize), to show the scale of the differences between two noise
realizations. Stores, per throw, pixel and hit slot (first two): presence, tick and ADC.
usage: JAX_PLATFORMS=cpu python seed_pairs.py N_PAIRS /abs/path/out.npz"""
import os, sys
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
M, OUT = int(sys.argv[1]), sys.argv[2]
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS')
import jax, jax.numpy as jnp
import paper_common as pc                       # note: chdir to the repo, hence the absolute OUT path
from larndsim.fee_jax import get_adc_values, digitize
truth, _ = pc.params()
c = np.load('/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/paperfigs/fig3_census_cache.npz')
wfs = np.asarray(c['wfs'][c['valid']]); P = len(wfs); T = 2 * M
run = jax.jit(lambda w, k: get_adc_values(truth, w, k))
q, tk = run(jnp.asarray(np.tile(wfs, (T, 1))), jax.random.key(7357))
adc = np.asarray(digitize(truth, q)).reshape(T, P, -1)[:, :, :2]; q = np.asarray(q).reshape(T, P, -1)[:, :, :2]
tk = np.asarray(tk).reshape(T, P, -1)[:, :, :2]
np.savez_compressed(OUT, hit=q > 0, tick=tk, adc=adc, M=M)
print('throws', T, 'pixels', P, 'mean hits/throw (first two slots)', float((q > 0).sum() / T))
