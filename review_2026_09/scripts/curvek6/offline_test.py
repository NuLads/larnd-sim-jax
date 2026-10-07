"""Offline check of FitParams._seed_drift_from_hits using a saved checkpoint's contexts + target hits."""
import sys, pickle, logging, numpy as np, types
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/src')
logging.basicConfig(level=logging.INFO)
import jax; jax.config.update('jax_platform_name', 'cpu')
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax'); import optax; from optimize import fit_params as fp
ck = pickle.load(open(sys.argv[1], 'rb')); tdir = sys.argv[2]; batches = [int(b) for b in sys.argv[3].split(',')]
for b in batches:
    ctxs = [types.SimpleNamespace(**c) for c in ck['chain_contexts'][b]]
    for c in ctxs: c.track_id = tuple(c.track_id)
    K = [len(np.asarray(s['angles'])) for s in ck['chain_cache'][b]]
    self = types.SimpleNamespace(_batch_chain_contexts={b: ctxs}, _chain_cache={b: [{'angles': np.zeros(k, np.float32)} for k in K]},
                                 _chain_optimizer=optax.adam(1e-3),
                                 _batch_true_positions={b: {tuple(float(x) for x in k.strip('()').split(',')): v for k, v in ck['true_positions'][b].items()}})
    f = np.load(f'{tdir}/batch{b}_target.npz')
    fp.GradientDescentFitter._seed_drift_from_hits(self, b, f['pixel_x'], f['pixel_y'], f['pixel_z'], f['event'], f['adcs'], f['pixel_id'])
