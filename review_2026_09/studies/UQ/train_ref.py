"""UQ test (3 Nov): the TRAINING reference = frozen predictions for the out-of-sample test, from the existing production-C75
ensembles at H @ 20000 (errors = last-1000 mean / truth - 1, %): realistic ends EXP<s>C75H and true ends KN10N<s>C75H,
s = 1..12. Prints mean (bias), SD (ddof 1), n. usage (container): python3 train_ref.py"""
import glob, pickle, numpy as np
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: float(np.mean(x))


def err(run, it=20000):
    fs = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))
    if not fs:
        return None
    d = pickle.load(open(fs[0], 'rb'))
    return np.array([100 * (np.mean([f1(x) for x in d[f'{p}_iter'][-1000:]]) / f1(d[f'{p}_target'][0]) - 1) for p in P])


if __name__ == '__main__':
    for lab, fmt in (('realistic EXP<s>C75H', 'EXP{}C75H'), ('true KN10N<s>C75H', 'KN10N{}C75H')):
        E = [err(fmt.format(s)) for s in range(1, 13)]
        miss = [s for s, e in zip(range(1, 13), E) if e is None]
        E = np.array([e for e in E if e is not None])
        print(f'{lab}: n {len(E)}' + (f' (missing {miss})' if miss else ''))
        print('   bias ' + ' '.join(f'{p} {v:+.3f}' for p, v in zip(P, E.mean(0))))
        print('   SD   ' + ' '.join(f'{p} {v:.3f}' for p, v in zip(P, E.std(0, ddof=1))))
