import pandas as pd, numpy as np, json
from scipy import stats

OUT = '/lscratch/pgranger/tmp/claude-45184/-sdf-group-neutrino-pgranger-larnd-sim-jax/e77eab56-46d4-44e7-b115-2992a0b283ba/scratchpad/C_curve'
df = pd.read_csv(f'{OUT}/all_runs.csv')

curve = df[df.tag.str.startswith('CURVE')].copy()
curve['tau'] = curve.tag.str.extract(r'CURVE(\d+)D').astype(int)
curve['dseed'] = curve.tag.str.extract(r'D(\d+)$').astype(int)
curve = curve.sort_values(['tau', 'dseed'])

print('=== CURVE runs: data_seed / seed / other-4-param targets ===')
for _, r in curve.iterrows():
    print(r.tag, 'data_seed=', r.data_seed, 'seed=', r.seed,
          'Ab_t=%.6f eF_t=%.6f trd_t=%.6e lgd_t=%.6e lt_t=%.2f' % (
              r.Ab_target, r.eField_target, r.tran_diff_target, r.long_diff_target, r.lifetime_target))

# Check: are the other 4 params' targets identical across ALL 24 runs?
for p in ['Ab_target', 'eField_target', 'tran_diff_target', 'long_diff_target']:
    u = curve[p].round(9).unique()
    print(p, 'unique target values across all 24 CURVE runs:', u)

print()
print('=== What differs between D1/D3/D7/D8 at fixed tau? ===')
for tau, g in curve.groupby('tau'):
    print(f'tau={tau}')
    print(g[['tag', 'data_seed', 'seed', 'lifetime_endpoint', 'lifetime_err_pct', 'long_diff_err_pct', 'Ab_err_pct', 'eField_err_pct', 'tran_diff_err_pct']].to_string(index=False))

curve.to_csv(f'{OUT}/curve_runs.csv', index=False)
print('wrote curve_runs.csv')
