import os, sys, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS'); sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/NEIGH')
import numpy as np, paper_common as pc, jax, jax.numpy as jnp
from neigh_exact import simulate_wfs_variant
from larndsim.sim_jax import simulate_stochastic
from larndsim.fee_jax import get_adc_values
truth, response = pc.params()
ds_sim, ds_tgt, fields = pc.load(200., 400, 1)
i = int(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else None
tracks = jnp.asarray(np.asarray(ds_tgt[i])[:n]); print('cm', pc.track_length_cm(tracks, fields), flush=True)
W = {k: simulate_wfs_variant(truth, response, tracks, fields, k, chunk=256) for k in ['prod', 'L']}
up = np.asarray(W['prod'][1])
q = {}; t = {}
for k, (w, u) in W.items():
    qq, tt = get_adc_values(truth, w, jax.random.key(i + 1)); q[k] = np.asarray(qq); t[k] = np.asarray(tt)
    print(k, 'wfs diff max', float(np.abs(np.asarray(w) - np.asarray(W['prod'][0])).max()), flush=True)
hit0 = q['prod'] > 0; hitL = q['L'] > 0
print('slots with hit prod/L/both', hit0.sum(), hitL.sum(), (hit0 & hitL).sum())
both = hit0 & hitL
dt = (t['L'] - t['prod'])[both]; dq = (q['L'] - q['prod'])[both]
print('ticks differ on', int((dt != 0).sum()), 'of', int(both.sum()), 'hits; |dt| distribution', np.unique(np.abs(dt), return_counts=True))
print('charge diff: median |dq|', float(np.median(np.abs(dq))), 'max', float(np.abs(dq).max()))
# same draws? compare with a different seed for scale
qs, ts = get_adc_values(truth, W['prod'][0], jax.random.key(i + 2)); qs, ts = np.asarray(qs), np.asarray(ts)
b2 = hit0 & (qs > 0); print('other seed: ticks differ on', int(((ts - t['prod'])[b2] != 0).sum()), 'of', int(b2.sum()))
