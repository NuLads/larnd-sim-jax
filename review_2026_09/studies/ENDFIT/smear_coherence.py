"""ENDFIT, 25 Oct: the LARND_ENDPOINT_SMEAR_MM draw is seeded by (eventID, trackID), which are batch-LOCAL ids, so the
same few dozen smear vectors repeat in every batch. Measure the coherent part: the drift-distance shift of the given
(smeared) endpoints vs the true ones, s_z * sign(z - z_anode), batch means and overall; the transverse rms; and the
best endpoint slope the zero-mean drift constraint allows (b with its batch-mean drift shift removed, min-norm, as
_project_endpoint_drift). usage (container, PYTHONPATH = live tree): python3 smear_coherence.py <run> [...]"""
import sys, zlib, glob, pickle, numpy as np
from optimize.fit_params import _dir_from_angles, _transverse_frame
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
ZB = np.array([[30.59, 0.159], [-30.59, -0.159]])
for run in sys.argv[1:]:
    d = pickle.load(open(sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter*.pkl'))[-1], 'rb'))
    allsh, bm, ids, sl_num, sl_den = [], [], set(), 0., 0.
    for bi in sorted(d['chain_contexts']):
        sh, W, Bv = [], [], []
        for c in d['chain_contexts'][bi]:
            u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); _, e2 = _transverse_frame(u0)
            g = np.random.default_rng(zlib.crc32(f"{tuple(c['track_id'])}|0".encode())); ids.add(tuple(c['track_id']))
            x0 = np.asarray(c['x0_fixed'], float); ends = (x0, x0 + c['total_len'] * u0)
            for k in range(2):
                s = g.normal(0., .1, 3); z = ends[k][2]; t = int(np.argmin(np.abs(ZB.mean(1) - z)))
                sg = np.sign(z - ZB[t, 0]); sh.append(s[2] * sg)        # given - true drift distance (cm)
                W.append(float(e2[2]) * sg); Bv.append(-s @ e2)          # constraint weight; ideal e2 coefficient
        sh, W, Bv = np.array(sh), np.array(W), np.array(Bv)
        allsh += list(sh); bm.append(1e4 * sh.mean())
        bp = Bv - W * (W @ Bv) / (W @ W)                                   # ideal coefficients under the constraint
        sl_num += bp @ Bv; sl_den += Bv @ Bv
    allsh = np.array(allsh); bm = np.array(bm)
    print(f'{run}: {len(allsh)} ends, {len(ids)} distinct (ev, tr) ids over {len(bm)} batches')
    print(f'   given - true drift distance: mean {1e4 * allsh.mean():+7.1f} um (SE if independent {1e4 * allsh.std() / np.sqrt(len(allsh)):.1f}); '
          f'batch means {bm.mean():+.1f} +- {bm.std():.1f} um (min {bm.min():+.0f}, max {bm.max():+.0f})')
    print(f'   best e2 slope allowed by the zero-mean constraint: {sl_num / sl_den:.3f}')
