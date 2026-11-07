"""Offline check of FitParams._seed_drift_from_hits using a saved checkpoint's contexts + target hits."""
import sys, pickle, logging, numpy as np, types
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/src')
logging.basicConfig(level=logging.INFO)
import jax; jax.config.update('jax_platform_name', 'cpu')
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax'); import optax; from optimize import fit_params as fp
ck = pickle.load(open(sys.argv[1], 'rb')); tdir = sys.argv[2]; batches = [int(b) for b in sys.argv[3].split(',')]
ALL = []
for b in batches:
    ctxs = [types.SimpleNamespace(**c) for c in ck['chain_contexts'][b]]
    for c in ctxs: c.track_id = tuple(float(x) for x in c.track_id)
    K = [len(np.asarray(s['angles'])) for s in ck['chain_cache'][b]]
    self = types.SimpleNamespace(_batch_chain_contexts={b: ctxs}, _chain_cache={b: [{'angles': np.zeros(k, np.float32)} for k in K]},
                                 _chain_optimizer=optax.adam(1e-3),
                                 _batch_true_positions={b: {tuple(float(x) for x in k.strip('()').split(',')): v for k, v in ck['true_positions'][b].items()}})
    f = np.load(f'{tdir}/batch{b}_target.npz')
    fp.GradientDescentFitter._seed_drift_from_hits(self, b, f['pixel_x'], f['pixel_y'], f['pixel_z'], f['event'], f['adcs'], f['pixel_id'])
    D = np.array(getattr(self, '_geom_hits_diag', [])); ALL.extend(D.tolist()) if len(D) else None
D = np.array(ALL); print(f"AGG tracks {len(D)}: e1 median {np.median(D[:,0]):.0f} (straight {np.median(D[:,1]):.0f}) mean {D[:,0].mean():.0f} ({D[:,1].mean():.0f}) | e2 median {np.median(D[:,2]):.0f} (straight {np.median(D[:,3]):.0f}) mean {D[:,2].mean():.0f} ({D[:,3].mean():.0f})")
cv = np.maximum(D[:,1], D[:,3])
for lab, m in (('curved>500', cv > 500), ('straightish<250', cv < 250)):
    print(f"AGG {lab} n={m.sum()}: e1 {np.median(D[m,0]):.0f} vs {np.median(D[m,1]):.0f} | e2 {np.median(D[m,2]):.0f} vs {np.median(D[m,3]):.0f}")
