import pandas as pd, numpy as np
from scipy import stats

OUT = '/lscratch/pgranger/tmp/claude-45184/-sdf-group-neutrino-pgranger-larnd-sim-jax/e77eab56-46d4-44e7-b115-2992a0b283ba/scratchpad/C_curve'
df = pd.read_csv(f'{OUT}/all_runs.csv')
PARAMS = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']

def mean_abs_err(row):
    return np.mean([abs(row[f'{p}_err_pct']) for p in PARAMS])

df['mean_abs_err'] = df.apply(mean_abs_err, axis=1)

def tag_ds(tag, prefix):
    return int(tag.replace(prefix, ''))

# ---- baseline: DSREP{k} <-> data_seed k ----
baseline = df[df.tag.str.startswith('DSREP')].copy()
baseline['dseed'] = baseline.tag.str.extract(r'DSREP(\d+)').astype(int)
baseline = baseline.set_index('dseed')

TUNED = [3, 7, 8]
FRESH = [1, 2, 4, 5, 6]

print('=== K2: baseline (DSREP, x1.0) mean|err| per seed ===')
print(baseline[['tag', 'mean_abs_err'] + [f'{p}_err_pct' for p in PARAMS]].sort_index())

for pre, label in [('NARR040D', 'x0.40'), ('NARR06D', 'x0.60'), ('NARR025D', 'x0.25'), ('NARR015D', 'x0.15'),
                    ('WIDE16D', 'x1.6'), ('WIDE25D', 'x2.5')]:
    arm = df[df.tag.str.startswith(pre)].copy()
    arm['dseed'] = arm.tag.apply(lambda t: tag_ds(t, pre))
    arm = arm.set_index('dseed')
    print(f'\n=== {label} ({pre}) seeds available: {sorted(arm.index)} ===')

    for grp_name, grp in [('TUNED 3/7/8', TUNED), ('FRESH', FRESH), ('ALL avail', sorted(arm.index))]:
        common = [s for s in grp if s in arm.index and s in baseline.index]
        if not common:
            continue
        b = baseline.loc[common, 'mean_abs_err'].values
        x = arm.loc[common, 'mean_abs_err'].values
        d = x - b
        n = len(d)
        mean_d = d.mean()
        se_d = d.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
        tstat, pval = (stats.ttest_1samp(d, 0) if n > 1 else (np.nan, np.nan))
        n_improved = int((d < 0).sum())
        print(f'  {grp_name:14s} n={n} seeds={common}: base={b.round(2)} arm={x.round(2)} '
              f'paired Delta={mean_d:+.2f} +- {se_d:.2f}  (t={tstat:.2f}, p={pval:.3f})  {n_improved}/{n} improved')
        # lifetime alone
        bl = baseline.loc[common, 'lifetime_err_pct'].values
        xl = arm.loc[common, 'lifetime_err_pct'].values
        dl_abs = np.abs(xl) - np.abs(bl)
        print(f'    lifetime |err| paired Delta = {dl_abs.mean():+.2f} +- '
              f'{(dl_abs.std(ddof=1)/np.sqrt(n) if n>1 else float("nan")):.2f}  '
              f'({int((dl_abs<0).sum())}/{n} improved)')

print('\n=== K2 config-diff check: baseline vs x0.40 arm, ALL fields, for each matched seed ===')
import pickle, glob
def load_config(tag):
    base = '/sdf/group/neutrino/pgranger/larnd-sim-jax'
    for b in (f'{base}/fit_result', '/sdf/scratch/neutrino/pgranger/fit_result'):
        fs = glob.glob(f'{b}/sci_full_{tag}/history_iter10000_*.pkl')
        if fs:
            with open(fs[0], 'rb') as fh:
                h = pickle.load(fh)
            return vars(h['config']), h['env_overrides'], h['provenance']
    return None

for s in TUNED + FRESH:
    if s not in baseline.index:
        continue
    arm_tag = f'NARR040D{s}'
    if arm_tag not in df.tag.values:
        continue
    c1, e1, pr1 = load_config(f'DSREP{s}')
    c2, e2, pr2 = load_config(arm_tag)
    ks = set(c1) | set(c2)
    cdiff = {k: (c1.get(k), c2.get(k)) for k in ks if c1.get(k) != c2.get(k)}
    ks = set(e1) | set(e2)
    ediff = {k: (e1.get(k), e2.get(k)) for k in ks if e1.get(k) != e2.get(k)}
    same_sha = pr1['git_sha'] == pr2['git_sha']
    print(f'D{s}: config diffs (excl test_name) = '
          f'{ {k:v for k,v in cdiff.items() if k!="test_name"} } | env diffs = {ediff} | '
          f'same_git_sha={same_sha} (base {pr1["git_sha"][:8]} dirty={pr1["git_dirty"]}, '
          f'arm {pr2["git_sha"][:8]} dirty={pr2["git_dirty"]})')

print('\n=== K4: dedx_mae_tail across the constraint ladder (seeds 3/7/8), tests Neyman-Scott claim ===')
LADDER = [('WIDE25D', 'x2.5'), ('WIDE16D', 'x1.6'), ('DSREP', 'x1.0'), ('NARR06D', 'x0.60'),
          ('NARR040D', 'x0.40'), ('NARR025D', 'x0.25'), ('NARR015D', 'x0.15')]
for pre, lab in LADDER:
    arm = df[df.tag.str.startswith(pre)].copy()
    arm['dseed'] = arm.tag.apply(lambda t: tag_ds(t, pre) if pre != 'DSREP' else int(t.replace('DSREP', '')))
    common = arm[arm.dseed.isin(TUNED)]
    if len(common) == 0:
        continue
    print(f'  {lab:6s} n={len(common)}  dedx_mae_tail(dedx_mae_truth) mean={common.dedx_mae_truth_tail.mean():.4f}  '
          f'dedx_mae_tail(matched-truth as fit sees)={common.dedx_mae_tail.mean():.4f}  '
          f'mean_abs_err(calib)={common.mean_abs_err.mean():.2f}%')
