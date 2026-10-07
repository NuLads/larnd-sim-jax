import sys, pickle, numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
from optimize.fit_params import _dir_from_angles, _transverse_frame
ck = pickle.load(open(sys.argv[1], 'rb')); f = np.load(sys.argv[2] + f'/batch{sys.argv[3]}_target.npz'); b = int(sys.argv[3])
H = np.stack([f['pixel_x'], f['pixel_y'], f['pixel_z']], 1).astype(float); ev = f['event']
print('target events', np.unique(ev))
for c in ck['chain_contexts'][b]:
    tid = tuple(c['track_id']); tr = np.asarray(ck['true_positions'][b][str(tid)], float)
    u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0); x0 = np.asarray(c['x0_fixed'], float); L = c['total_len']
    print(f"track {tid} L={L:.1f} u0={np.round(u0,3)} e1={np.round(e1,3)} e2={np.round(e2,3)} x0={np.round(x0,2)}")
    print('   truth range', np.round(tr.min(0),2), np.round(tr.max(0),2))
    m = ev == int(tid[0]); P = H[m]
    a = ((P - x0) @ u0) / L; d = P - x0 - a[:, None] * L * u0
    print(f"   hits in event {m.sum()}, alpha range {a.min():.2f}..{a.max():.2f}; |d.e1| <1cm: {(np.abs(d@e1)<1).sum()}; d.e2 percentiles", np.round(np.percentile(d@e2,[5,50,95]),3))
    # nearest-truth distance for hits
    from scipy.spatial import cKDTree
    dd, _ = cKDTree(tr).query(P); print('   hit->truth-mid nearest dist percentiles', np.round(np.percentile(dd,[5,50,95]),3))
    dz = P[:, 2] - tr[cKDTree(tr[:, :2]).query(P[:, :2])[1], 2]; print('   hit z - truth z (matched in xy) pct', np.round(np.percentile(dz,[5,50,95]),3))
