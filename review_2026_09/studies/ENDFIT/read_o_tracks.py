"""ENDFIT (o), descriptive follow-up of read_o.py (no rule): per-track position residual vs truth at 20000, arms CF and
SD vs P (same sample, tracks matched by order, checked by length). Splits the change of the segment-weighted mean |r|
into tracks that are outliers in P (|r| > 200 um) and the rest. Needs POLISH/joint/per_track_KN10N<s><arm>H_20000.json
(per_track_n.sh). usage (container): python3 read_o_tracks.py"""
import json, numpy as np
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint'
S = [3, 5, 6, 7, 8, 9]
def T(run):
    a = json.load(open(f'{D}/per_track_{run}_20000.json'))
    return {k: np.array([o[k] for o in a]) for k in ('r_mean', 'n', 'L')}
agg = {a: [] for a in ('CF', 'SD')}
print('KN10N (weighted mean |r| um; outliers > 200 / > 500 um; d = X - P split by the P tail / bulk)')
for s in S:
    p = T(f'KN10N{s}PH'); W = p['n'].sum(); tail = p['r_mean'] > 200
    line = f"  s{s} P {np.average(p['r_mean'], weights=p['n']):6.1f} ({tail.sum()}/{(p['r_mean'] > 500).sum()}, med {np.median(p['r_mean']):.1f})"
    for a in ('CF', 'SD'):
        x = T(f'KN10N{s}{a}H'); assert np.allclose(x['L'], p['L']), (s, a)
        dr = (x['r_mean'] - p['r_mean']) * p['n'] / W
        fixed = int(np.sum(tail & (x['r_mean'] <= 200))); made = int(np.sum(~tail & (x['r_mean'] > 200)))
        agg[a].append((dr.sum(), dr[tail].sum(), dr[~tail].sum(), fixed, made, np.median(x['r_mean']) - np.median(p['r_mean'])))
        line += (f" | {a} {np.average(x['r_mean'], weights=x['n']):6.1f} ({(x['r_mean'] > 200).sum()}/{(x['r_mean'] > 500).sum()}, med {np.median(x['r_mean']):.1f})"
                 f" d {dr.sum():+5.1f} = tail {dr[tail].sum():+5.1f} + bulk {dr[~tail].sum():+5.1f}; fixed {fixed} made {made}")
    print(line)
for a in agg:
    v = np.array(agg[a]); m = v.mean(0); se = v.std(0, ddof=1) / np.sqrt(len(v))
    print(f'{a} mean over 6: d {m[0]:+.1f} +- {se[0]:.1f} = tail {m[1]:+.1f} +- {se[1]:.1f} + bulk {m[2]:+.1f} +- {se[2]:.1f} um; '
          f'fixed {m[3]:.1f} made {m[4]:.1f} per sample; median shift {m[5]:+.1f} +- {se[5]:.1f} um')
