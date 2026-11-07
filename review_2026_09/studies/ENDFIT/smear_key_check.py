"""ENDFIT (g), 25 Oct: check of LARND_ENDPOINT_SMEAR_KEY=entry from dry-run checkpoints (training_history
['endpoint_smear'] = {batch: {(ev, tr): (s_entry, s_exit)}}). Per sample: distinct smear vectors (must = 2 x tracks),
per-axis rms (must be ~1000 um), coherent drift-distance offset of the given ends (+ = away from the anode; must be
within ~2 SE of 0, vs -43..-69 um with the legacy key), batch-mean scatter; across samples: shared vectors (only
physical tracks present in both samples may share one).
usage (container, PYTHONPATH = live tree): python3 smear_key_check.py <ckpt_s1.pkl> [<ckpt_s2.pkl> ...]"""
import sys, pickle, numpy as np
from optimize.fit_params import _dir_from_angles
ZB = np.array([[30.59, 0.159], [-30.59, -0.159]])
V = []
for f in sys.argv[1:]:
    d = pickle.load(open(f, 'rb')); sm = d['endpoint_smear']; cx = d['chain_contexts']
    vec, sh, bm = [], [], []
    for bi, m in sm.items():
        byid = {tuple(c['track_id']): c for c in cx[bi]}; b = []
        for tid, (s1, s2) in m.items():
            c = byid[tuple(tid)]; u0 = _dir_from_angles(c['theta0_i'], c['phi0_i'])
            x0 = np.asarray(c['x0_fixed'], float); xs = (x0, x0 + c['total_len'] * u0)
            for k, s in enumerate((s1, s2)):
                vec.append(np.round(np.asarray(s, float), 7)); z = xs[k][2]
                t = int(np.argmin(np.abs(ZB.mean(1) - z))); b.append(s[2] * np.sign(z - ZB[t, 0]))
        sh += b; bm.append(np.mean(b))
    vec, sh, bm = np.array(vec), 1e4 * np.array(sh), 1e4 * np.array(bm)
    nd = len({tuple(v) for v in vec}); V.append({tuple(v) for v in vec})
    print(f'{f.split("/fit_result/")[-1]}: {len(vec) // 2} tracks in {len(sm)} batches, {len(vec)} ends, {nd} distinct smear vectors')
    print(f'   rms x / y / z: ' + ' / '.join(f'{1e4 * np.sqrt(np.mean(vec[:, i] ** 2)):.0f}' for i in range(3)) + ' um')
    print(f'   given - true drift distance: mean {sh.mean():+.1f} um (SE {sh.std() / np.sqrt(len(sh)):.1f}); batch means +- {bm.std():.0f} um')
for i in range(len(V)):
    for j in range(i + 1, len(V)):
        print(f'shared smear vectors, sample file {i + 1} vs {j + 1}: {len(V[i] & V[j])}')
