import pandas as pd, numpy as np
from scipy import stats

OUT = '/lscratch/pgranger/tmp/claude-45184/-sdf-group-neutrino-pgranger-larnd-sim-jax/e77eab56-46d4-44e7-b115-2992a0b283ba/scratchpad/C_curve'
curve = pd.read_csv(f'{OUT}/curve_runs.csv')
curve = curve.sort_values(['tau', 'dseed']).reset_index(drop=True)

TAUS = sorted(curve.tau.unique())
DSS = sorted(curve.dseed.unique())
VALID_TAUS = [t for t in TAUS if t <= 3200]

print('=== Per-tau mean/SD/SE of tau_fit (n=4 D-seeds) ===')
per_tau = curve.groupby('tau').lifetime_endpoint.agg(['mean', 'std', 'count'])
per_tau['se'] = per_tau['std'] / np.sqrt(per_tau['count'])
print(per_tau)

# ---------- (1) In-sample OLS fit on means (reproduce authors) ----------
T = np.array(VALID_TAUS, float)
F = per_tau.loc[VALID_TAUS, 'mean'].values
SE = per_tau.loc[VALID_TAUS, 'se'].values
a_unw, b_unw = np.polyfit(T, F, 1)
print(f'\nUnweighted OLS on 6-pt means (authors\' method, VALID<3500): slope={a_unw:.4f} intercept={b_unw:.1f}')

# Weighted least squares using proper SE (authors used SE=std/2 which for n=4 equals std/sqrt(4) -- same thing, OK)
X = np.vstack([T, np.ones_like(T)]).T
W = np.diag(1.0 / SE**2)
cov = np.linalg.inv(X.T @ W @ X)
beta = cov @ X.T @ W @ F
a_w, b_w = beta
resid_w = F - (a_w * T + b_w)
chi2 = float(resid_w @ W @ resid_w)
dof = len(T) - 2
print(f'Weighted LS on 6-pt means: slope={a_w:.4f}+-{np.sqrt(cov[0,0]):.4f}  intercept={b_w:.1f}+-{np.sqrt(cov[1,1]):.1f}  chi2/dof={chi2/dof:.2f}')
print(f'  residual scatter (unweighted a,b): {np.std(F - (a_unw*T+b_unw), ddof=2):.1f} us')
print(f'  residual scatter (weighted a,b):   {np.std(resid_w, ddof=2):.1f} us')

# ---------- (2) Fit using ALL individual (per-D-seed) points pooled, not means ----------
print('\n=== Fit using ALL 16 individual points (4 D-seeds x 4 taus 800-3200), OLS ===')
sub = curve[curve.tau.isin(VALID_TAUS)]
Ti = sub.tau.values.astype(float)
Fi = sub.lifetime_endpoint.values.astype(float)
a_i, b_i = np.polyfit(Ti, Fi, 1)
resid_i = Fi - (a_i*Ti+b_i)
print(f'  slope={a_i:.4f} intercept={b_i:.1f}  residual RMS = {np.std(resid_i, ddof=2):.1f} us = {100*np.std(resid_i,ddof=2)/np.mean(Ti):.2f}% of mean tau')
print(f'  (this is what a REAL fit -- one draw, unknown target -- would see; author\'s 35us/2.9% comes from averaging 4 D-seeds together first)')

# ---------- (3) LEAVE-ONE-DATA-SEED-OUT validation ----------
print('\n=== Leave-one-data-seed-out (LODO) validation ===')
lodo_resid = []
for held in DSS:
    train = sub[sub.dseed != held]
    test = sub[sub.dseed == held]
    a_h, b_h = np.polyfit(train.tau.values.astype(float), train.lifetime_endpoint.values.astype(float), 1)
    pred = a_h * test.tau.values.astype(float) + b_h
    r = test.lifetime_endpoint.values.astype(float) - pred
    lodo_resid.extend(r.tolist())
    print(f'  held-out D{held}: slope={a_h:.4f} intercept={b_h:.1f}  residuals(us)={np.round(r,1).tolist()}  RMS={np.std(r,ddof=0):.1f}')
lodo_resid = np.array(lodo_resid)
print(f'  POOLED LODO out-of-sample RMS residual: {np.sqrt(np.mean(lodo_resid**2)):.1f} us '
      f'= {100*np.sqrt(np.mean(lodo_resid**2))/np.mean(Ti):.2f}% of mean tau_true')
print(f'  compare: in-sample (all 4 seeds, pooled) RMS = {np.std(resid_i,ddof=2):.1f} us; '
      f'authors\' quoted (fit+predict on SAME 4-seed means) = 35 us (2.9%)')

# ---------- (4) Δ(1/tau) constant-offset model vs linear-in-tau model ----------
print('\n=== Δ(1/tau) = 1/tau_fit - 1/tau_true : constant-offset model ===')
sub2 = sub.copy()
sub2['inv_true'] = 1.0/sub2.tau
sub2['inv_fit'] = 1.0/sub2.lifetime_endpoint
sub2['d_inv'] = sub2['inv_fit'] - sub2['inv_true']
per_tau_dinv = sub2.groupby('tau').d_inv.agg(['mean','std','count'])
per_tau_dinv['se'] = per_tau_dinv['std']/np.sqrt(per_tau_dinv['count'])
print(per_tau_dinv)
grand_mean = sub2.d_inv.mean()
grand_se = sub2.groupby('tau').d_inv.mean().std(ddof=1)/np.sqrt(len(VALID_TAUS))
print(f'grand mean Delta(1/tau) (per-tau-mean average) = {grand_mean*1e5:.3f}e-5 /us,  '
      f'across-tau SE = {grand_se*1e5:.3f}e-5 /us')
# chi2 for constant model vs per-tau means
means = per_tau_dinv['mean'].values
ses = per_tau_dinv['se'].values
w = 1/ses**2
const = np.sum(means*w)/np.sum(w)
const_err = 1/np.sqrt(np.sum(w))
chi2_const = np.sum(w*(means-const)**2)
dof_const = len(means)-1
print(f'weighted-mean constant = {const*1e5:.3f} +- {const_err*1e5:.3f} e-5/us ; chi2/dof (const model) = {chi2_const/dof_const:.2f} (dof={dof_const})')
print(f'  claim in docstring: Delta(1/tau) = -4.42 +- 0.75e-5, "constant to 17%"')
pctspread = 100*(means.max()-means.min())/abs(means.mean())
print(f'  actual spread of per-tau means: {100*means.std(ddof=1)/abs(means.mean()):.1f}% (std/mean), '
      f'{pctspread:.1f}% (max-min)/mean')

# is const model preferred over a tau-dependent (linear in tau) model for d_inv?
# fit d_inv vs tau linearly
Xd = np.vstack([per_tau_dinv.index.values.astype(float), np.ones(len(per_tau_dinv))]).T
Wd = np.diag(1/ses**2)
covd = np.linalg.inv(Xd.T@Wd@Xd)
betad = covd @ Xd.T @ Wd @ means
slope_d, intercept_d = betad
resid_lin = means - (Xd@betad)
chi2_lin = np.sum(w*resid_lin**2)
dof_lin = len(means)-2
print(f'linear-in-tau fit to Delta(1/tau): slope={slope_d:.3e}+-{np.sqrt(covd[0,0]):.3e}  chi2/dof={chi2_lin/dof_lin:.2f} (dof={dof_lin})')
Fstat = ((chi2_const-chi2_lin)/(dof_const-dof_lin)) / (chi2_lin/dof_lin) if dof_lin>0 else np.nan
pval = 1-stats.f.cdf(Fstat, dof_const-dof_lin, dof_lin) if dof_lin>0 else np.nan
print(f'F-test const vs linear: F={Fstat:.2f}, p={pval:.3f}  (slope significant if p<0.05, i.e. non-constant preferred)')
tstat_slope = slope_d/np.sqrt(covd[0,0])
print(f'slope t-stat = {tstat_slope:.2f} (>2 => significantly nonzero => NOT constant)')

# ---------- (5) does long_diff co-vary with tau_fit / lifetime error along the curve? ----------
print('\n=== long_diff error vs lifetime error correlation across CURVE runs ===')
r, p = stats.pearsonr(curve.long_diff_err_pct, curve.lifetime_err_pct)
print(f'Pearson r(long_diff_err%, lifetime_err%) across all 24 CURVE runs = {r:.3f}, p={p:.4f}')
r2, p2 = stats.pearsonr(curve[curve.tau.isin(VALID_TAUS)].long_diff_err_pct, curve[curve.tau.isin(VALID_TAUS)].lifetime_err_pct)
print(f'  restricted to valid tau<=3200: r={r2:.3f}, p={p2:.4f}, n={len(curve[curve.tau.isin(VALID_TAUS)])}')
# also vs tau itself
r3, p3 = stats.pearsonr(curve.tau, curve.long_diff_err_pct)
print(f'Pearson r(tau_true, long_diff_err%) = {r3:.3f}, p={p3:.4f}')

# ---------- (6) scatter across D-seeds relative to the fitted correction -> is the bias generalizable? ----------
print('\n=== Per-D-seed overall bias level (mean lifetime_err_pct across all 6 taus) -- tests "bias independent of unknowns" ===')
print(curve.groupby('dseed').lifetime_err_pct.agg(['mean','std','count']))
# ANOVA: is D-seed a significant factor beyond tau?
from scipy.stats import f_oneway
groups = [g.lifetime_err_pct.values for _, g in curve.groupby('dseed')]
F_ds, p_ds = f_oneway(*groups)
print(f'One-way ANOVA of lifetime_err_pct across D-seed (marginalizing tau): F={F_ds:.2f}, p={p_ds:.4f}')
