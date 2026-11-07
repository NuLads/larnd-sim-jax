"""KN10V full-length validation readout (rule: REVIEW.md 20 Oct, stated before any result).
Per sample s=1..3: D = KN10V_s - KN10S_s for the 5 calibration errors (%) and position (um), same readout as
KNOT/readout_block.py, against the KN10S 16-sample SD of each quantity. NEUTRAL if every |D| < 0.5 SD, else INCONCLUSIVE.
Speed: median step_time over iterations 1000-1e4 and total wall (sum of step_time) vs the same KN10S sample."""
import sys, glob, pickle, numpy as np
FR = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
ld = lambda arm: pickle.load(open(sorted(glob.glob(f'{FR}/sci_full_{arm}/history_iter10000_*.pkl'))[0], 'rb'))


def read(d):
    v = {p: 100 * (np.ravel(d[p + '_iter'])[-1000:].mean() / np.ravel(d[p + '_target'])[0] - 1) for p in P}
    v['pos'] = 1e4 * float(np.median(np.asarray(d['pos_residual_iter'], float).ravel()[-1000:]))
    v['loss'] = float(np.mean(np.ravel(d['losses_iter'])[-1000:]))
    st = np.ravel(d['step_time']).astype(float)
    v['step'] = float(np.median(st[1000:10000])); v['wall_h'] = st.sum() / 3600; v['e1_min'] = st[:100].sum() / 60
    v['n'] = len(np.ravel(d['losses_iter']))
    return v


S = {s: read(ld(f'KN10S{s}')) for s in range(1, 17)}
V = {s: read(ld(f'KN10V{s}')) for s in (1, 2, 3)}
Q = P + ['pos']
SD = {q: float(np.std([S[s][q] for s in S], ddof=1)) for q in Q}
print('KN10S 16-sample mean / SD: ' + '  '.join(f'{q} {np.mean([S[s][q] for s in S]):+.3f}/{SD[q]:.3f}' for q in Q))
worst = 0; allok = True
for s in V:
    v, k = V[s], S[s]
    row = []
    for q in Q:
        d = v[q] - k[q]; z = abs(d) / SD[q]; worst = max(worst, z); allok &= z < 0.5
        row.append(f'{q} {v[q]:+.3f}/{k[q]:+.3f} (D {d:+.3f} = {z:.2f} SD)')
    print(f's{s} n {v["n"]}/{k["n"]}: ' + '  '.join(row) + f'  loss {v["loss"]:.3f}/{k["loss"]:.3f}')
print(f'max |D|/SD = {worst:.2f} -> {"NEUTRAL" if allok else "INCONCLUSIVE"}')
for s in V:
    v, k = V[s], S[s]
    print(f's{s} speed: median step 1000-1e4 {v["step"]:.2f} vs {k["step"]:.2f} s = {v["step"] / k["step"]:.2f}x; '
          f'wall {v["wall_h"]:.2f} vs {k["wall_h"]:.2f} h = {v["wall_h"] / k["wall_h"]:.2f}x; epoch 1 {v["e1_min"]:.0f} vs {k["e1_min"]:.0f} min')
