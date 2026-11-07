"""Exploratory (no rule; (n) is complete): per-track position residual vs truth at 20000, arm X (CFH, B2H) vs H,
same sample, tracks matched by order (checked by length). Splits the change of the segment-weighted mean |r| into
tracks that are outliers in H (|r| > 200 um, the seed-aliasing tail of e1_hit_seed_outliers) and the rest.
usage: python3 read_n_tracks.py [runs prefix list]  (needs POLISH/joint/per_track_<run>_20000.json)"""
import json, numpy as np
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint'
def T(run):
    a = json.load(open(f'{D}/per_track_{run}_20000.json'))
    return {k: np.array([o[k] for o in a]) for k in ('r_mean', 'n', 'L', 'r_z', 'r_xy')}
for b in ('KN10N', 'EXP'):
    print(f'\n{b}  (weighted mean |r| um; outliers = tracks > 200 / > 500 um; d = X - H split by the H tail / bulk)')
    for s in (1, 2, 4):
        h = T(f'{b}{s}H'); W = h['n'].sum(); tail = h['r_mean'] > 200
        line = f"  s{s} H {np.average(h['r_mean'], weights=h['n']):6.1f} ({tail.sum()}/{(h['r_mean'] > 500).sum()}, med {np.median(h['r_mean']):.1f})"
        for a in ('CFH', 'B2H'):
            x = T(f'{b}{s}{a}'); assert np.allclose(x['L'], h['L']), (b, s, a)
            dr = (x['r_mean'] - h['r_mean']) * h['n'] / W
            fixed = int(np.sum(tail & (x['r_mean'] <= 200))); made = int(np.sum(~tail & (x['r_mean'] > 200)))
            line += (f" | {a} {np.average(x['r_mean'], weights=x['n']):6.1f} ({(x['r_mean'] > 200).sum()}/{(x['r_mean'] > 500).sum()}, med {np.median(x['r_mean']):.1f})"
                     f" d {dr.sum():+5.1f} = tail {dr[tail].sum():+5.1f} + bulk {dr[~tail].sum():+5.1f}; fixed {fixed} made {made}")
        print(line)
