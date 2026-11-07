"""UQ (31 Oct, exploratory, no rule): end-point training loss of different fits of the SAME data (production H, replica P,
C75) -- do they differ beyond evaluation noise, and does the lower loss go with the smaller calibration error?
Loss = sum of losses_iter over the last epoch (100 iterations = each batch once); also the last-1000 mean error vs truth
and the mean pos_residual. usage (container): python3 loss_vs_err.py"""
import glob, pickle, re, numpy as np
S = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: float(np.mean(x))


def read(run):
    fs = sorted(glob.glob(f'{S}/sci_full_{run}/history_iter20000_*.pkl'))
    if not fs:
        return None
    h = pickle.load(open(fs[0], 'rb'))
    argv = ' '.join(h['provenance'].get('argv', [])) if isinstance(h.get('provenance'), dict) else ''
    m = re.search(r'--data_seed (\d+)', argv)
    r = {p: 100 * (np.mean([f1(x) for x in h[f'{p}_iter'][-1000:]]) / f1(h[f'{p}_target'][0]) - 1) for p in P}
    r['L'] = float(np.sum([f1(x) for x in h['losses_iter'][-100:]]))
    r['Lprev'] = float(np.sum([f1(x) for x in h['losses_iter'][-200:-100]]))
    r['pos'] = 1e4 * float(np.nanmean([f1(x) for x in h['pos_residual_iter'][-1000:]]))
    r['seed'] = int(m.group(1)) if m else None
    return r


R = {}
for s in range(1, 17):
    for arm, run in (('O', f'KN10N{s}H'), ('P', f'KN10N{s}PH'), ('C', f'KN10N{s}C75H')):
        x = read(run)
        if x is not None:
            R[(s, arm)] = x
print('per sample: arm  data_seed  L_end (last epoch)  L_end - L_prev_epoch  pos(um)  errors(%) ' + ' '.join(P))
for s in range(1, 17):
    for arm in 'OPC':
        if (s, arm) in R:
            x = R[(s, arm)]
            print(f"  s{s:<2d} {arm} {x['seed']} {x['L']:12.1f} {x['L'] - x['Lprev']:+8.1f} {x['pos']:6.1f}  "
                  + ' '.join(f'{x[p]:+6.2f}' for p in P))
print('\npairs within a sample (same data): dL = L_b - L_a, and d|err| = |err_b| - |err_a| per parameter')
rows = []
for s in range(1, 17):
    for a, b in (('O', 'P'), ('O', 'C'), ('P', 'C')):
        if (s, a) in R and (s, b) in R:
            xa, xb = R[(s, a)], R[(s, b)]
            assert xa['seed'] == xb['seed'], (s, a, b, xa['seed'], xb['seed'])
            rows.append((s, a + b, xb['L'] - xa['L'], xb['pos'] - xa['pos'], [abs(xb[p]) - abs(xa[p]) for p in P]))
for s, ab, dL, dp, de in rows:
    print(f"  s{s:<2d} {ab}: dL {dL:+9.1f}  dpos {dp:+6.1f}  d|err| " + ' '.join(f'{v:+6.2f}' for v in de))
dL = np.array([r[2] for r in rows]); dp = np.array([r[3] for r in rows]); de = np.array([r[4] for r in rows])
print(f"\n  n pairs {len(rows)}; |dL| median {np.median(np.abs(dL)):.0f}; corr(dL, dpos) {np.corrcoef(dL, dp)[0, 1]:+.2f}")
for i, p in enumerate(P):
    agree = np.mean(np.sign(dL) == np.sign(de[:, i]))
    print(f"  {p:10s}: corr(dL, d|err|) {np.corrcoef(dL, de[:, i])[0, 1]:+.2f}; lower-loss fit also smaller |err| in {agree:.2f} of pairs")
