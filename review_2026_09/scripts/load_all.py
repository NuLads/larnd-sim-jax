"""Load every relevant fit_result pickle, extract endpoints + config, dump to CSV.
Read-only on the repo.
"""
import glob, pickle, os, json, csv
import numpy as np

REPO = '/sdf/group/neutrino/pgranger/larnd-sim-jax'
OUT = '/lscratch/pgranger/tmp/claude-45184/-sdf-group-neutrino-pgranger-larnd-sim-jax/e77eab56-46d4-44e7-b115-2992a0b283ba/scratchpad/C_curve'

PARAMS = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']

def f0(x):
    return float(np.asarray(x).ravel()[0])

def tail_mean(v, frac=0.1):
    v = np.asarray(v).ravel()
    n = len(v)
    k = max(1, int(n * frac))
    return float(np.mean(v[-k:]))

def tail_median(v, n_last=500):
    v = np.asarray(v).ravel()
    k = min(len(v), n_last)
    return float(np.median(v[-k:]))

def find_pkl(tag):
    for base in (os.path.join(REPO, 'fit_result'), '/sdf/scratch/neutrino/pgranger/fit_result'):
        fs = glob.glob(f'{base}/sci_full_{tag}/history_iter10000_*.pkl')
        if fs:
            return fs[0]
    return None

def load_one(tag):
    path = find_pkl(tag)
    if path is None:
        return None
    real = os.path.realpath(path)
    with open(path, 'rb') as fh:
        h = pickle.load(fh)
    c = h['config']
    row = {
        'tag': tag,
        'path': path,
        'realpath': real,
        'data_seed': getattr(c, 'data_seed', None),
        'seed': getattr(c, 'seed', None),
        'set_target_vals': json.dumps(getattr(c, 'set_target_vals', [])),
        'dedx_prior_weight': getattr(c, 'dedx_prior_weight', None),
        'dedx_student_scale_l': getattr(c, 'dedx_student_scale_l', None),
        'dedx_student_scale_r': getattr(c, 'dedx_student_scale_r', None),
        'dedx_student_nu_l': getattr(c, 'dedx_student_nu_l', None),
        'dedx_student_nu_r': getattr(c, 'dedx_student_nu_r', None),
        'dedx_mean_constraint_weight': getattr(c, 'dedx_mean_constraint_weight', None),
        'dedx_freeze_iter': getattr(c, 'dedx_freeze_iter', None),
        'iterations': getattr(c, 'iterations', None),
        'git_sha': h.get('provenance', {}).get('git_sha'),
        'git_dirty': h.get('provenance', {}).get('git_dirty'),
        'slurm_job_id': h.get('provenance', {}).get('slurm_job_id'),
        'timestamp': h.get('provenance', {}).get('timestamp'),
    }
    for p in PARAMS:
        v = np.asarray(h[p + '_iter']).ravel()
        t = f0(h[p + '_target'])
        init = f0(h[p + '_init'])
        end = tail_mean(v, 0.1)
        n = len(v)
        k = max(1, int(n * 0.1))
        tail_vals = v[-k:]
        row[f'{p}_target'] = t
        row[f'{p}_init'] = init
        row[f'{p}_endpoint'] = end
        row[f'{p}_endpoint_sd'] = float(np.std(tail_vals, ddof=1)) if len(tail_vals) > 1 else 0.0
        row[f'{p}_n_iter'] = n
        row[f'{p}_err_pct'] = 100.0 * (end / t - 1.0)
    # position residual: tail median of last 500 (per-batch, noisy) in cm -> um (x1e4 per authors' scripts)
    if 'pos_residual_iter' in h:
        pr = np.asarray(h['pos_residual_iter']).ravel()
        row['pos_residual_um'] = tail_median(pr, 500) * 1e4
        row['pos_residual_n'] = len(pr)
    else:
        row['pos_residual_um'] = None
    # dedx MAE
    for key in ('dedx_mae_iter', 'dedx_mae_truth_iter'):
        if key in h:
            row[key.replace('_iter', '') + '_tail'] = tail_median(h[key], 500)
        else:
            row[key.replace('_iter', '') + '_tail'] = None
    # likelihood-only metric (as authors defined it)
    try:
        A = h['aux_iter'][-2000:]
        row['lik_tail'] = float(np.median([abs(a['log_likelihood_tick']) + abs(a['no_match_penalty'])
                                            + abs(a['expected_total_hits']) for a in A]))
    except Exception as e:
        row['lik_tail'] = None
    # settledness: peak-to-peak of tail / |endpoint| for lifetime and long_diff
    for p in ('lifetime', 'long_diff'):
        v = np.asarray(h[p + '_iter']).ravel()
        n = len(v)
        k = max(1, int(n * 0.2))
        tail = v[-k:]
        row[f'{p}_settledness_pct'] = 100.0 * (tail.max() - tail.min()) / abs(np.mean(tail))
    return row


def main():
    tags = []
    curve_taus = [800, 1200, 1800, 2400, 3200, 4400]
    curve_ds = [1, 3, 7, 8]
    for tau in curve_taus:
        for ds in curve_ds:
            tags.append(f'CURVE{tau}D{ds}')
    for i in range(1, 17):
        tags.append(f'DSREP{i}')
    for pre in ('NARR015D', 'NARR025D', 'NARR040D', 'NARR06D', 'WIDE16D', 'WIDE25D'):
        seeds = range(1, 9) if pre == 'NARR040D' else (3, 7, 8)
        for s in seeds:
            tags.append(f'{pre}{s}')
    for s in (1, 3, 7):
        tags.append(f'LDPIN2400D{s}')

    rows = []
    missing = []
    for tag in tags:
        try:
            r = load_one(tag)
        except Exception as e:
            print('ERROR loading', tag, e)
            r = None
        if r is None:
            missing.append(tag)
            continue
        rows.append(r)
        print('loaded', tag)

    print('MISSING:', missing)

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(os.path.join(OUT, 'all_runs.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {len(rows)} rows to all_runs.csv')

if __name__ == '__main__':
    main()
