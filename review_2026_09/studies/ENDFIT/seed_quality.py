"""ENDFIT (d), 25 Oct: quality of the hit-based endpoint estimates (training_history['endpoint_seed'], written by
_seed_drift_from_hits with LARND_HITS_SEED_ENDPOINTS or LARND_SEED_DIAG_ONLY) vs the truth. Per endpoint and axis:
b = true - given (the smear: stored endpoint_smear, else re-drawn with the legacy (ev, tr) key), p = the hit estimate.
  e1 (pixel plane): rms(p - b) vs rms(b) (before); slope p~b; pull rms (p - b) / se; by standard-error bin.
  e2 (drift, from hit times): drift-distance residual delta = w (p - b), w = e2.z sign(z - z_anode) [latency: mean,
  median, slope vs drift distance]; coefficient residual after removing the mean latency, by |e2.z| bin.
Reference (EZS1FR, fitted from the smeared ends, 14000 iterations): rms(a - b) e1 467 / e2 544 um; before 856 / 1000.
usage (container, PYTHONPATH = live tree): python3 seed_quality.py <checkpoint.pkl>"""
import sys, zlib, pickle, numpy as np
from optimize.fit_params import _dir_from_angles, _transverse_frame
ZB = np.array([[30.59, 0.159], [-30.59, -0.159]])
d = pickle.load(open(sys.argv[1], 'rb')); cx, rec = d['chain_contexts'], d['endpoint_seed']; sm = d.get('endpoint_smear')
um = lambda v: 1e4 * float(np.sqrt(np.mean(np.square(v)))) if len(v) else float('nan')
R = {k: [] for k in ('b1', 'p1', 'se1', 'b2', 'p2', 'se2', 'w', 'dd', 'ez', 'st')}
ntr = sum(len(v) for v in cx.values())
for bi, rs in rec.items():
    for r in rs:
        c = cx[bi][r['ti']]; assert tuple(c['track_id']) == tuple(r['track_id'])
        u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0)
        if sm is not None:
            s = sm[bi][tuple(r['track_id'])]
        else:
            g = np.random.default_rng(zlib.crc32(f"{tuple(c['track_id'])}|0".encode()))
            s = (g.normal(0., .1, 3), g.normal(0., .1, 3))
        x0 = np.asarray(c['x0_fixed'], float); ends = (x0, x0 + c['total_len'] * u0)
        for k in range(2):
            z = ends[k][2]; t = int(np.argmin(np.abs(ZB.mean(1) - z))); sg = np.sign(z - ZB[t, 0])
            R['b1'].append(-s[k] @ e1); R['p1'].append(r['e1'][k]); R['se1'].append(r['e1_se'][k])
            R['b2'].append(-s[k] @ e2); R['p2'].append(r['e2'][k]); R['se2'].append(r['e2_se'][k])
            R['w'].append(float(e2[2]) * sg); R['dd'].append(abs(z - ZB[t, 0])); R['ez'].append(abs(float(e2[2])))
            R['st'].append(bool(r['straight']))
R = {k: np.array(v) for k, v in R.items()}
n = len(R['b1'])
print(f'{n // 2} seeded tracks of {ntr} ({n} endpoints); smear from {"checkpoint" if sm is not None else "legacy re-draw"}')
r1 = R['p1'] - R['b1']
print(f'\ne1: before {um(R["b1"]):.0f} um -> hit estimate {um(r1):.0f} um; slope p~b {R["p1"] @ R["b1"] / (R["b1"] @ R["b1"]):.3f}; '
      f'pull rms {np.sqrt(np.mean((r1 / R["se1"]) ** 2)):.2f}; median se {1e4 * np.median(R["se1"]):.0f} um')
for lo, hi in ((0, 100e-4), (100e-4, 200e-4), (200e-4, 500e-4), (500e-4, 1.0)):
    m = (R['se1'] >= lo) & (R['se1'] < hi)
    if m.any():
        print(f'   se {1e4 * lo:4.0f}-{1e4 * hi:5.0f} um: {m.sum():4d} ends ({100 * m.mean():3.0f}%)  before {um(R["b1"][m]):5.0f}  estimate {um(r1[m]):5.0f} um')
for cap in (0.5, 0.2, 0.1):
    m = np.abs(R['p1']) < cap
    print(f'   applied with |p| < {10 * cap:.0f} mm: {100 * m.mean():.0f}% of ends; error after seeding {um(np.where(m, r1, -R["b1"])):.0f} um')
r2 = R['p2'] - R['b2']; dl = R['w'] * r2                            # drift-distance residual (cm), + = away from anode
print(f'\ne2 latency (drift-distance residual of the hit estimate, + = away from the anode): mean {1e4 * dl.mean():+.0f} um, '
      f'median {1e4 * np.median(dl):+.0f} um')
A = np.stack([np.ones(n), R['dd']], 1); cf = np.linalg.lstsq(A, dl, rcond=None)[0]
print(f'   vs drift distance: {1e4 * cf[0]:+.0f} um + {1e4 * cf[1]:+.1f} um/cm x d  (d from {R["dd"].min():.1f} to {R["dd"].max():.1f} cm)')
for lo, hi in ((0, 10), (10, 20), (20, 31)):
    m = (R['dd'] >= lo) & (R['dd'] < hi)
    if m.any(): print(f'   d {lo:2d}-{hi:2d} cm: {m.sum():4d} ends, median latency {1e4 * np.median(dl[m]):+5.0f} um')
lat = np.median(dl); r2c = r2 - lat / R['w']                         # remove the mean latency, in coefficient units
print(f'   coefficient error after removing the median latency, by |e2.z| (before = smear):')
for lo, hi in ((0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)):
    m = (R['ez'] >= lo) & (R['ez'] < hi)
    if m.any():
        print(f'   |e2.z| {lo:.1f}-{hi:.1f}: {m.sum():4d} ends ({100 * m.mean():3.0f}%, straight {100 * R["st"][m].mean():3.0f}%)  before {um(R["b2"][m]):5.0f}  '
              f'raw {um(r2[m]):6.0f}  latency-corrected {um(r2c[m]):6.0f} um; median se {1e4 * np.median(R["se2"][m]):.0f}')
