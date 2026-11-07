"""FB readout (rule: REVIEW.md 23 Oct, stated before any result): freeze release in the continuation, PRA<s> vs PRF<s>.
Needs per_track_{PRA,PRF}<s>_14000.json (per_track_res.py). usage: python3 read_fb_block.py"""
import sys, json, numpy as np
sys.argv = [sys.argv[0]]; exec(open('/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint/read_pr.py').read().split('\nfor s in sys.argv')[0])
J = '/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint'
def trk(run):
    a = json.load(open(f'{J}/per_track_{run}_14000.json')); return {k: np.array([o[k] for o in a]) for k in a[0]}
rep, sh = [], {p: [] for p in P}
for s in (1, 2, 3, 4, 5, 6):
    A, F = load(f'PRA{s}'), load(f'PRF{s}'); ta, tf = trk(f'PRA{s}'), trk(f'PRF{s}')
    assert len(ta['L']) == len(tf['L']) and np.allclose(ta['L'], tf['L']) and np.all(ta['batch'] == tf['batch']), s
    m = ta['frozen']; wa = np.average(ta['r_mean'][m], weights=ta['n'][m]); wf = np.average(tf['r_mean'][m], weights=tf['n'][m])
    ok = wf <= 0.7 * wa and F['pr_end'] <= 1.05 * A['pr_end']
    if s not in (1, 3): rep.append(ok)
    for p in P: sh[p].append(F['err'][p] - A['err'][p])
    print(f's{s}: frozen-in-PRA tracks n {m.sum()}: |r| {wa:.1f} -> {wf:.1f} um ({wf / wa:.3f}); pos_residual 13-14k '
          f'{1e4 * A["pr_end"]:.1f} -> {1e4 * F["pr_end"]:.1f} um ({F["pr_end"] / A["pr_end"]:.3f}) -> {"REPLICATES" if ok else "FAILS"}'
          + ('' if s not in (1, 3) else ' (old tree, not counted)') + ' | end err PRA->PRF ' + ' '.join(f'{p} {A["err"][p]:+.2f}->{F["err"][p]:+.2f}' for p in P))
print(f'\nPOSITIONS: replicates on {sum(rep)}/4 new samples -> {"CONFIRMED" if sum(rep) >= 3 else "NOT CONFIRMED"}')
for p in P:
    d = np.array(sh[p]); se = d.std(ddof=1) / np.sqrt(len(d))
    print(f'CALIBRATION {p:9s}: paired PRF-PRA {d.mean():+.2f} +- {se:.2f} ({" ".join(f"{x:+.2f}" for x in d)})'
          + (f' -> {"no gross change" if abs(d.mean()) < 2 * se else "SHIFT"}' if p in ('lifetime', 'long_diff') else ''))
