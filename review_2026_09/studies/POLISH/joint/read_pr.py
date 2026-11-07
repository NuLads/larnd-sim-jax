"""Readout of the chain-LR re-heat test (REVIEW.md 21-22 Oct, rules stated before any result): PRA<s> (chain LR 2e-3
re-started at 1e4) vs PRC<s> (production schedule, today's tree) vs KN10C<s> (13 Oct production continuation).
usage (in the container): python3 read_pr.py 1 [3 ...]"""
import sys, glob, pickle, numpy as np
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
TRUTH = dict(Ab=0.83488135, eField=0.52151894, tran_diff=1.00276338e-05, long_diff=5.81418228e-06, lifetime=2406.44659703)


def load(run):
    ck = pickle.load(open(sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter14000_*.pkl'))[0], 'rb'))
    r = np.array([np.nan if x is None else float(np.mean(x)) for x in ck['pos_residual_iter']])
    L = np.array([float(np.ravel(x)[0]) for x in ck['losses_iter']])
    off = int(ck['total_iter']) - len(r)                      # histories hold all iterations from 0
    w = lambda a, lo, hi: float(np.nanmean(a[lo - off:hi - off]))
    err = {k: 100 * (float(np.ravel(ck[k + '_iter'][-1])[0]) / TRUTH[k] - 1) for k in P}
    return dict(pr_end=w(r, 13000, 14000), pr_10k=w(r, 10000, 11000), pr_9k=w(r, 9000, 10000), L_end=w(L, 13000, 14000),
                L_10k=w(L, 10000, 11000), err=err, n=len(r))


for s in sys.argv[1:]:
    R = {k: load(f'{k}{s}') for k in ('KN10C', 'PRC', 'PRA')}
    for k, x in R.items():
        print(f'{k}{s}: pos_residual 9-10k {1e4 * x["pr_9k"]:.1f} um, 10-11k {1e4 * x["pr_10k"]:.1f}, 13-14k {1e4 * x["pr_end"]:.1f} | '
              f'loss 10-11k {x["L_10k"]:.1f}, 13-14k {x["L_end"]:.1f} | end error % ' + ' '.join(f'{p} {x["err"][p]:+.2f}' for p in P))
    c, a, k0 = R['PRC']['pr_end'], R['PRA']['pr_end'], R['KN10C']['pr_end']
    tree = abs(c / k0 - 1) <= 0.05
    print(f'  tree check PRC{s} / KN10C{s} = {c / k0:.3f} -> {"PASS" if tree else "FAIL (compare PRA with PRC only)"}')
    q = a / c
    print(f'  positions PRA{s} / PRC{s} = {q:.3f} -> re-heat ' + ('HELPS' if q <= 0.85 else 'HURTS' if q >= 1.05 else 'NEUTRAL'))
