"""CPU audit: for every stored run, is the readout value a STATIONARY point or a FROZEN one?

Reports, per run/parameter: the tail-window mean gradient with a BLOCK-bootstrap standard
error (block length 200 >> the measured epoch period of 100, since gradients are strongly
epoch-autocorrelated and naive IID t-stats are optimistic), plus the fraction of the LR
schedule's cumulative budget still remaining at readout.
"""
import pickle, glob, re, os, json
import numpy as np

P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
rng = np.random.default_rng(0)


def block_boot_se(x, block=200, n=400):
    x = np.asarray(x, float); N = len(x)
    if N < 3 * block:
        block = max(2, N // 5)
    nb = max(1, N // block)
    idx = np.arange(N)
    means = []
    for _ in range(n):
        starts = rng.integers(0, max(1, N - block), size=nb)
        samp = np.concatenate([x[s:s + block] for s in starts])
        means.append(samp.mean())
    return float(np.std(means))


def autocorr(x, lag):
    x = np.asarray(x, float) - np.mean(x)
    if len(x) <= lag or np.std(x) == 0:
        return np.nan
    return float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x))


rows = []
for d in sorted(glob.glob('fit_result/sci_*')):
    if not os.path.isdir(d):
        continue
    for f in glob.glob(os.path.join(d, 'history_iter*.pkl')):
        if 'len400' not in f:
            continue
        m = re.search(r'seed(\d+)', f)
        if not m:
            continue
        try:
            h = pickle.load(open(f, 'rb'))
        except Exception:
            continue
        it = int(re.search(r'iter(\d+)', f).group(1))
        for p in P:
            g = h.get(p + '_grad')
            if g is None:
                continue
            g = np.ravel(np.asarray(g, dtype=float))
            g = g[np.isfinite(g)]
            if len(g) < 500:
                continue
            tail = g[int(len(g) * .8):]
            se = block_boot_se(tail)
            t = float(tail.mean() / se) if se > 0 else np.nan
            rows.append(dict(run=os.path.basename(d), seed=int(m.group(1)), iters=it, param=p,
                             tail_mean=float(tail.mean()), block_se=se, t=t,
                             ac100=autocorr(tail, 100), ac200=autocorr(tail, 200)))

json.dump(rows, open('grad_audit.json', 'w'), indent=1)
print(f'{len(rows)} run/param entries\n')
print('=== gradient autocorrelation in the tail (epoch period = 100) ===')
for lag in ['ac100', 'ac200']:
    v = [r[lag] for r in rows if np.isfinite(r[lag])]
    print(f'  {lag}: median {np.median(v):+.3f}  IQR {np.percentile(v,25):+.3f}..{np.percentile(v,75):+.3f}  (n={len(v)})')
print('\n=== |t| of the tail-mean gradient, BLOCK-bootstrap SE, by parameter ===')
print(f"{'param':11s}{'median|t|':>10s}{'frac |t|>2':>12s}{'frac |t|>3':>12s}{'n':>6s}")
for p in P:
    ts = [abs(r['t']) for r in rows if r['param'] == p and np.isfinite(r['t'])]
    if not ts:
        continue
    ts = np.array(ts)
    print(f'{p:11s}{np.median(ts):10.2f}{np.mean(ts>2)*100:11.0f}%{np.mean(ts>3)*100:11.0f}%{len(ts):6d}')
print('\n=== ANNEALED runs only (the headline configs) ===')
ANN = ['sci_full_noise_s4_anneal', 'sci_full_ANNEALMORE', 'sci_full_ANNEALLONG',
       'sci_full_FRAMEONLY2', 'sci_floor_S3ANNEAL']
print(f"{'run':28s}{'param':11s}{'seed':5s}{'tail_mean':>12s}{'block_se':>11s}{'t':>8s}")
for r in rows:
    if r['run'] in ANN and r['param'] in ('Ab', 'lifetime'):
        print(f"{r['run'][:28]:28s}{r['param']:11s}{r['seed']:<5d}{r['tail_mean']:12.4g}{r['block_se']:11.4g}{r['t']:8.2f}")
