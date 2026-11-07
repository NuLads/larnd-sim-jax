import os, sys, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS'); sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/NEIGH')
import numpy as np
import paper_common as pc
import jax, jax.numpy as jnp
from neigh_exact import simulate_wfs_variant
truth, response = pc.params()
ds_sim, ds_tgt, fields = pc.load(200., 1, 1)
tr = np.asarray(ds_tgt[0]); print('rows', tr.shape, 'fields', fields[:6], flush=True)
tracks = jnp.asarray(tr[:int(sys.argv[1]) if len(sys.argv) > 1 else 2000])
print('track length cm', pc.track_length_cm(tracks, fields), flush=True)
out = {}
for k in ['prod', 'P', 'L', 'T', 'E']:
    t0 = time.time(); w, up = simulate_wfs_variant(truth, response, tracks, fields, k, chunk=128); w = np.asarray(w)
    out[k] = w; print(k, w.shape, f'{time.time()-t0:.1f}s', 'sum', w.sum(), flush=True)
up = np.asarray(up); valid = up >= 0
ts = float(truth.t_sampling)
for k in ['P', 'L', 'T', 'E']:
    d = np.cumsum(out[k] - out['prod'], 1) * ts
    q = np.cumsum(out['prod'], 1) * ts
    dm = np.abs(d[valid]).max(1); qm = q[valid].max(1)
    print(f'{k}: max|dQ| e- median {np.median(dm):.2f} p99 {np.percentile(dm,99):.2f} max {dm.max():.2f}; '
          f'final dQ/Q median {np.median((d[valid,-1])/np.maximum(q[valid,-1],1)):.2e}; npix {valid.sum()}', flush=True)
