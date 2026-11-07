import sys, pickle, types, logging, numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax'); logging.basicConfig(level=logging.WARNING)
import jax; jax.config.update('jax_platform_name', 'cpu')
import optax
from optimize import fit_params as fp
ck = pickle.load(open(sys.argv[1], 'rb')); tdir = sys.argv[2]
nfrz = ndf = 0; errs_frozen = []; errs_free = []
for b in range(0, 100, 5):
    ctxs = [types.SimpleNamespace(**c) for c in ck['chain_contexts'][b]]
    for c in ctxs: c.track_id = tuple(float(x) for x in c.track_id)
    K = [len(np.asarray(s['angles'])) for s in ck['chain_cache'][b]]
    self = types.SimpleNamespace(_batch_chain_contexts={b: ctxs}, _chain_cache={b: [{'angles': np.zeros(k, np.float32)} for k in K]},
                                 _chain_optimizer=optax.adam(1e-3),
                                 _batch_true_positions={b: {tuple(float(x) for x in k.strip('()').split(',')): v for k, v in ck['true_positions'][b].items()}})
    f = np.load(f'{tdir}/batch{b}_target.npz')
    fp.GradientDescentFitter._seed_drift_from_hits(self, b, f['pixel_x'], f['pixel_y'], f['pixel_z'], f['event'], f['adcs'], f['pixel_id'], apply_seed=True)
    masks = fp.GradientDescentFitter._drift_freeze_masks(self, b, self._chain_cache[b])
    for ti, (c, m) in enumerate(zip(ctxs, masks)):
        m = np.asarray(m); Kt = len(m) // 2
        u0 = fp._dir_from_angles(c.theta0_i, c.phi0_i); e1, e2 = fp._transverse_frame(u0)
        if not (abs(e1[2]) < 1e-6): continue
        ndf += 1
        frozen = m[Kt:].sum() == 0
        assert frozen == (ti in self._straight_freeze[b]), 'mask/classification mismatch'
        if frozen:
            nfrz += 1; assert np.allclose(np.asarray(self._chain_cache[b][ti]['angles'])[Kt:], 0), 'frozen track not at straight line'
print(f'drift-frame tracks {ndf}, frozen (called straight) {nfrz} = {nfrz/ndf:.2f}; masks consistent with classification; frozen e2 = straight line')
