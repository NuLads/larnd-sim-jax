"""Pull per-iteration fit traces of the production C75 runs into one small file.
EXP<s>C75H (realistic ends, n=36) and KN10N<s>C75H (true ends, n=12), all at iteration 20000."""
import pickle, sys, glob, numpy as np, multiprocessing as mp
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
S = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
EXP = [f'EXP{i}C75H' for i in list(range(1, 13)) + list(range(17, 41))]
KN = [f'KN10N{i}C75H' for i in range(1, 13)]
def one(n):
    f = glob.glob(f'{S}/sci_full_{n}/history_iter20000*.pkl')[0]
    h = pickle.load(open(f, 'rb'))
    o = dict(name=n, data_seed=int(h['config'].data_seed))
    for p in P:
        o[p] = np.asarray(h[p + '_iter'], float); o[p + '_target'] = float(h[p + '_target'][0]); o[p + '_init'] = float(h[p + '_iter'][0])
    for k in ['losses_iter', 'pos_residual_iter', 'pos_residual_batch', 'dedx_mae_iter', 'dedx_mae_truth_iter', 'mean_dedx_iter', 'step_time']:
        o[k] = np.asarray(h[k], float)
    o['argv'] = ' '.join(h['provenance']['argv'])
    return o
if __name__ == '__main__':
    with mp.Pool(16) as pool: R = pool.map(one, EXP + KN)
    out = dict(names=np.array([r['name'] for r in R]), data_seed=np.array([r['data_seed'] for r in R]),
               argv=np.array([r['argv'] for r in R]))
    for k in P + ['losses_iter', 'pos_residual_iter', 'pos_residual_batch', 'dedx_mae_iter', 'dedx_mae_truth_iter', 'mean_dedx_iter', 'step_time']:
        L = max(len(r[k]) for r in R)
        a = np.full((len(R), L), np.nan, dtype=np.float32 if k not in P else float)
        for i, r in enumerate(R): a[i, :len(r[k])] = r[k]
        out[k] = a
    for p in P:
        out[p + '_target'] = np.array([r[p + '_target'] for r in R]); out[p + '_init'] = np.array([r[p + '_init'] for r in R])
    np.savez_compressed('/sdf/scratch/neutrino/pgranger/talk_animations/data/traces.npz', **out)
    print('saved', len(R), 'runs', {k: v.shape for k, v in out.items() if k in ('Ab', 'losses_iter', 'pos_residual_iter')})
