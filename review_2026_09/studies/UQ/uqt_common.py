"""UQ TEST (3 Nov) shared readout pieces (rules: REVIEW.md "UQ TEST of the recommendations"). Errors in % vs truth."""
import glob, pickle, re, sys, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ')
import uq_lib as U
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ/dumps'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: float(np.mean(x))
# frozen training reference (train_ref.py on EXP1-12C75H @ 20000, n = 12; REVIEW.md table)
BIAS = np.array([0.013, 0.131, 1.599, 2.827, 0.847])
SD = np.array([0.152, 0.035, 1.285, 4.064, 3.866])
NTR = 12


def ckpt(run, it):
    fs = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))
    if not fs:
        raise SystemExit(f'{run}: no checkpoint at {it} -> set not complete, no readout')
    d = pickle.load(open(fs[0], 'rb')); assert int(d['total_iter']) == it, (run, d['total_iter'])
    return d


def end(run, it=20000, check=None):
    """Error % (last-1000 mean / truth - 1) at the end point, truth, and the config check (data_seed, env)."""
    d = ckpt(run, it)
    truth = np.array([f1(d[f'{p}_target'][0]) for p in P])
    e = np.array([100 * (np.mean([f1(x) for x in d[f'{p}_iter'][-1000:]]) / t - 1) for p, t in zip(P, truth)])
    last = np.array([100 * (f1(d[f'{p}_iter'][-1]) / t - 1) for p, t in zip(P, truth)])
    argv = ' '.join(d['provenance'].get('argv', [])); env = d.get('env_overrides', {})
    m = re.search(r'--data_seed (\d+)', argv)
    cfg = dict(seed=int(m.group(1)) if m else None, coarse=env.get('LARND_CHAIN_COARSE_ITER'),
               smear=env.get('LARND_ENDPOINT_SMEAR_MM'), key=env.get('LARND_ENDPOINT_SMEAR_KEY'))
    if check is not None:
        assert cfg == dict(seed=check, coarse='7500', smear='1.0', key='entry'), (run, cfg)
    return dict(e=e, last=last, truth=truth, cfg=cfg)


def e1(name, truth=None):
    """E1 of a dump: sigma (%), Newton step d (%), z; the dumped point's error (%) if truth is given."""
    d = U.load(f'{D}/{name}.npz'); assert list(d['params']) == P
    x = U.estimators(d)['E1']
    out = dict(sig=100 * x['sig'], d=100 * x['d'], z=x['z'], edm=x['edm'], it=int(d['iteration']),
               calib_only=bool(d['calib_only']) if 'calib_only' in d else False, phys=d.get('phys'))
    if truth is not None and out['phys'] is not None:
        out['e'] = 100 * (out['phys'] / truth - 1)
    return out


def tq(q):
    return np.mean(np.abs(q) < 1), np.mean(np.abs(q) < 2)
