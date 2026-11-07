"""First-order calibration shift from bias.json (run_neigh.py --part bias), with the per-batch Hessians of hessian.json.
usage: python analyze_bias.py [bias.json]"""
import json, sys, numpy as np
B = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '/sdf/scratch/neutrino/pgranger/review_2026_09/NEIGH/bias.json'))
H = {b['batch']: b for b in json.load(open('/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS/hessian.json'))['batches'] if 'H' in b}
S = json.load(open('/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS/hessian_summary.json'))
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
ok = B['batches']; Hs = sum(np.array(H[b['batch']]['H']) for b in ok); Hs = 0.5 * (Hs + Hs.T); Hi = np.linalg.inv(Hs)
L = sum(b['length_cm'] for b in ok) / 100
print(f'{len(ok)} batches, {L:.1f} m')
print('floor (398 m) %   ' + ' '.join(f'{p} {100*x:.3f}' for p, x in zip(P, S['floor'])))
for k in ['L', 'T', 'E']:
    D = np.array([np.array(b['g'][k]) - np.array(b['g']['prod']) for b in ok])
    dth = -Hi @ D.sum(0)
    # uncertainty: jackknife over batches
    jk = np.array([-np.linalg.inv(Hs - 0.5*(np.array(H[b['batch']]['H'])+np.array(H[b['batch']]['H']).T)) @ (D.sum(0) - D[i]) for i, b in enumerate(ok)])
    n = len(ok); s = np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))
    print(f'{k}: shift %  ' + ' '.join(f'{p} {100*x:+.3f}+-{100*e:.3f}' for p, x, e in zip(P, dth, s)))
    nh = np.array([b['g'][k + '_nhits'] - b['g']['prod_nhits'] for b in ok]); n0 = sum(b['g']['prod_nhits'] for b in ok)
    print(f'   recorded hits {k} - prod: {nh.sum():+d} of {n0} ({100*nh.sum()/n0:+.3f}%)')
