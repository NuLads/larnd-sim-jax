"""Across runs: coherent fitted-minus-true drift offset (um, + = farther from anode) vs lifetime error."""
import sys, json, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/GEOFRZ')
import drift_residuals as dr
ARMS = {'DSREP': [1, 2, 3, 4, 7, 8], 'HITSEED': [1, 2, 4], 'GTRU': [1, 2, 4, 8], 'K6FREE': [1, 2, 4], 'PINSTRESS': [3, 7, 8],
        'LDPIN': [3, 7, 8], 'DXTRU': [3, 7, 8], 'DRIFTFRZ': [3, 7, 8],
        'DRIFTTRU': [3, 7, 8, 1, 2, 4], 'DRIFTTRUK6': [1, 2, 4], 'DRIFTNZ50': [1, 2, 4], 'DRIFTNZ150D': [1, 2, 4], 'GEOFRZ': [3, 7, 8], 'BOTHTRU': [3, 7, 8]}
TAG = lambda a, s: f'sci_fullGEOFRZ{s}' if a == 'GEOFRZ' else f'sci_full_{a}{s}'
rows = []
for a, ss in ARMS.items():
    for s in ss:
        try: tracks, errs = dr.run_residuals(TAG(a, s))
        except Exception as e: print('skip', a, s, e); continue
        sm = dr.summarise(tracks)
        near = np.nanmean(sm['mean'][:3])                       # 0-6 cm drift
        rows.append(dict(arm=a, s=s, off=sm['overall_mean'], near=near, slope=sm['slope_um_per_cm'], **errs))
        print(f"{a:12s} s{s}: coherent offset {sm['overall_mean']:+6.1f} um (0-6cm {near:+6.1f}, trend {sm['slope_um_per_cm']:+5.2f} um/cm) | lifetime {errs['lifetime']:+6.2f}  long_diff {errs['long_diff']:+6.2f}  Ab {errs['Ab']:+5.2f}", flush=True)
json.dump(rows, open('/sdf/scratch/neutrino/pgranger/review_2026_09/DRIFTHITS/coherent_vs_lifetime.json', 'w'), indent=1)
free = [r for r in rows if r['arm'] in ('DSREP', 'HITSEED', 'GTRU', 'K6FREE', 'PINSTRESS', 'LDPIN', 'DXTRU')]
x = np.array([r['off'] for r in free]); y = np.array([r['lifetime'] for r in free])
if len(x) > 3:
    p = np.polyfit(x, y, 1); r = np.corrcoef(x, y)[0, 1]
    print(f"\nfree-geometry runs (n={len(x)}): lifetime_err = {p[1]:+.2f} + {p[0]:+.3f} * offset_um   r = {r:+.2f}")
    m = np.array([r['arm'] not in ('PINSTRESS', 'LDPIN') for r in free])
    p = np.polyfit(x[m], y[m], 1); r = np.corrcoef(x[m], y[m])[0, 1]
    print(f"  excluding pinned (lifetime/long_diff pinned) arms (n={m.sum()}): lifetime_err = {p[1]:+.2f} + {p[0]:+.3f} * offset_um   r = {r:+.2f}")
