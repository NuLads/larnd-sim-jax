"""Analysis of truthdiag.py outputs: E1 charge/hit closure vs drift, E2 NLL profiles."""
import sys, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '/sdf/scratch/neutrino/pgranger/review_2026_09/E_truthdiag'
files = sys.argv[1:] or sorted(glob.glob(f'{D}/out/main_*.npz'))
parts = [dict(np.load(f, allow_pickle=True)) for f in files]
cat = lambda k: np.concatenate([p[k] for p in parts], axis=0)
lt_grid, ld_grid = parts[0]['lt_grid'], parts[0]['ld_grid']
truth = json.loads(str(parts[0]['truth']))
TAU = truth['lifetime']
batches = cat('batch')
nb = len(batches)
print(f'files={len(files)} batches={nb} unique={len(np.unique(batches))}')

T_SAMP = 0.1          # us / tick
TICK0 = 22.4          # tick offset of drift t=0 (drift check: argmax tick - d/v/ts = 22.1..22.7)
exp_n, exp_q = cat('exp_n'), cat('exp_q')                   # (nb, ntk)
exp_n_t, exp_q_t = cat('exp_n_tiny'), cat('exp_q_tiny')
on_n, on_q = cat('obs_n_on'), cat('obs_q_on')              # (nb, ns, ntk)
off_n, off_q = cat('obs_n_off'), cat('obs_q_off')          # (nb, ntk)
ns = on_n.shape[1]
ntk = exp_n.shape[1]
print('pixfrac min', cat('frac_tgt_pix_in_guess').min(), ' npix total', cat('npix').sum())

# drift variable
tick = np.arange(ntk)
tdrift = (tick - TICK0) * T_SAMP
occ = exp_n.sum(0)
nz = np.where(occ > 0)[0]
print(f'occupied ticks {nz.min()}..{nz.max()}  -> drift {tdrift[nz.min()]:.1f}..{tdrift[nz.max()]:.1f} us')
edges_us = np.linspace(0, 190, 20)                          # 19 bins of 10 us
edges_tk = np.round(edges_us / T_SAMP + TICK0).astype(int)
nbin = len(edges_us) - 1
def binsum(a):   # sum last axis into drift bins
    return np.stack([a[..., edges_tk[k]:edges_tk[k + 1]].sum(-1) for k in range(nbin)], -1)
tc = 0.5 * (edges_us[1:] + edges_us[:-1])
# charge-weighted bin centres
tcw = binsum(exp_q.sum(0) * tdrift) / binsum(exp_q.sum(0))

def fit_exp(t, r, e):
    """weighted LS of ln r = a - k t ; returns k, sigma_k, a, chi2/ndf"""
    y, s = np.log(r), e / r
    w = 1 / s ** 2
    A = np.vstack([np.ones_like(t), -t]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    a, k = cov @ (A.T @ (w * y))
    chi2 = np.sum(w * (y - (a - k * t)) ** 2)
    return k, np.sqrt(cov[1, 1]), a, chi2 / (len(t) - 2)

def tau_equiv(k):
    return 1 / (1 / TAU + k)

res = {}
def closure(label, obs, exp):
    """obs: (nb, ns, ntk) or (nb, ntk); exp: (nb, ntk). Error from realisation (seed) scatter when
    ns>1, else from batch bootstrap (jackknife over batches)."""
    E = binsum(exp.sum(0))
    if obs.ndim == 3:
        O_s = binsum(obs.sum(0))                     # (ns, nbin)
        R_s = O_s / E
        R = R_s.mean(0); eR = R_s.std(0, ddof=1) / np.sqrt(R_s.shape[0])
        tot_s = O_s.sum(1) / E.sum(); tot, etot = tot_s.mean(), tot_s.std(ddof=1) / np.sqrt(len(tot_s))
        ks = np.array([fit_exp(tcw, R_s[s], np.full(nbin, 1.0) * R_s.std(0, ddof=1))[0] for s in range(R_s.shape[0])])
    else:
        O = binsum(obs.sum(0)); R = O / E
        # jackknife over batches
        Ob, Eb = binsum(obs), binsum(exp)
        jk = np.array([(O - Ob[b]) / (E - Eb[b]) for b in range(obs.shape[0])])
        eR = np.sqrt((obs.shape[0] - 1) * jk.var(0))
        tot = O.sum() / E.sum()
        jkt = np.array([(O.sum() - Ob[b].sum()) / (E.sum() - Eb[b].sum()) for b in range(obs.shape[0])])
        etot = np.sqrt((obs.shape[0] - 1) * jkt.var())
    eR = np.maximum(eR, 1e-7)
    k, ek, a, chi = fit_exp(tcw, R, eR)
    res[label] = dict(R=R, eR=eR, k=k, ek=ek, a=a, chi2ndf=chi, tot=tot, etot=etot,
                      drop_full=1 - np.exp(-k * 190.), tau_eq=tau_equiv(k),
                      tau_bias_pct=100 * (tau_equiv(k) / TAU - 1),
                      tau_bias_err_pct=100 * ek * TAU)
    r = res[label]
    print(f'{label:34s} total={tot:.5f}+-{etot:.5f}  k={k:+.3e}+-{ek:.1e}/us  '
          f'ratio change over 190us={-(np.exp(-k*190)-1)*100:+.3f}%  '
          f'-> tau_eq bias {r["tau_bias_pct"]:+.2f}+-{r["tau_bias_err_pct"]:.2f}%  chi2/ndf={chi:.2f}')

print('\n=== E1 closure (obs/exp), exp fit ln R = a - k t_drift ; positive k = target loses MORE charge than model ===')
closure('Q  noise-ON tgt / prod model', on_q, exp_q)
closure('N  noise-ON tgt / prod model', on_n, exp_n)
closure('Q  noise-OFF tgt / tiny-noise model', off_q, exp_q_t)
closure('N  noise-OFF tgt / tiny-noise model', off_n, exp_n_t)
closure('Q  noise-OFF tgt / prod model', off_q, exp_q)
closure('N  noise-OFF tgt / prod model', off_n, exp_n)
# mean charge per hit
closure('Q/hit noise-ON', on_q, exp_q) if False else None

# lifetime sensitivity of the expected charge: dlnQ/dln tau per bin -> slope calibration
eq_lt = cat('exp_q_lt').sum(0)       # (nlt, ntk)
i1 = int(np.argmin(abs(lt_grid - 1)))
Rlt = binsum(eq_lt) / binsum(eq_lt[i1])[None]
k_lt = np.array([fit_exp(tcw, Rlt[j], np.full(nbin, 1e-4))[0] for j in range(len(lt_grid))])
print('\nmodel self-consistency: slope of E[Q](tau)/E[Q](tau_true) vs drift, compared with 1/tau_true-1/tau:')
for j, f in enumerate(lt_grid):
    print(f'   tau x{f:.4f}: k={k_lt[j]:+.3e}  analytic 1/tau-1/tau_true = {1/(TAU*f)-1/TAU:+.3e}')

# ------------------------------------------------------------------ E2 NLL profiles
def parab_min(x, y, xs=None):
    # LOCAL parabola through the lowest grid point and its two neighbours: the profiles are
    # strongly asymmetric over +-15%, so a global quadratic is biased (gave +2.8% vs +1.5% local).
    j = int(np.clip(np.argmin(y), 1, len(x) - 2)); x, y = x[j-1:j+2], y[j-1:j+2]
    c = np.polyfit(x, y, 2)
    xm = -c[1] / (2 * c[0])
    return xm, 1 / np.sqrt(2 * c[0]) if c[0] > 0 else np.nan, c

print('\n=== E2 NLL profiles (sum over batches) ===')
prof = {}
nll_lt_on = cat('nll_lt_on').sum(0)        # (nlt, ns)
nll_lt_off = cat('nll_lt_off').sum(0)      # (nlt,)
nll_tiny = cat('nll_lt_on_tinymodel_off').sum(0)
nll_ld_on = cat('nll_ld_on').sum(0)
nll_ld_off = cat('nll_ld_off').sum(0)
def report(name, grid, y2d, tag):
    mins = []
    for s in range(y2d.shape[1]):
        xm, ex, c = parab_min(grid, y2d[:, s]); mins.append((xm, ex))
    mins = np.array(mins)
    m, sd = mins[:, 0].mean(), mins[:, 0].std(ddof=1) if len(mins) > 1 else np.nan
    prof[tag] = dict(mins=mins, mean=m, sd=sd, sem=sd / np.sqrt(len(mins)) if len(mins) > 1 else np.nan)
    print(f'{name:40s} minima (x truth): ' + ' '.join(f'{v:.4f}' for v in mins[:, 0]) +
          f'\n{"":40s} mean {m:.4f}  seed-sd {sd:.4f}  sem {prof[tag]["sem"]:.4f}  '
          f'curvature 1-sigma(dNLL=0.5) {np.nanmean(mins[:,1]):.4f}')
report('lifetime | noise-ON target, prod model', lt_grid, nll_lt_on, 'lt_on')
report('lifetime | noise-OFF target, prod model', lt_grid, nll_lt_off[:, None], 'lt_off')
report('lifetime | noise-OFF target, tiny-noise model', lt_grid, nll_tiny[:, None], 'lt_off_tiny')
report('long_diff | noise-ON target, prod model', ld_grid, nll_ld_on, 'ld_on')
report('long_diff | noise-OFF target, prod model', ld_grid, nll_ld_off[:, None], 'ld_off')

# decomposition of the lifetime profile (keys sorted: expected_total_hits, log_likelihood_tick,
# matched_hits, no_match_penalty)
aux_on = cat('aux_lt_on').sum(0)      # (nlt, ns, 4)
aux_off = cat('aux_lt_off').sum(0)    # (nlt, 4)
keys = ['expected_total_hits', 'log_likelihood_tick', 'matched_hits', 'no_match_penalty']
print('\nNLL term decomposition vs lifetime (seed-mean, relative to truth point):')
for kk, key in enumerate(keys):
    v = aux_on[:, :, kk].mean(1); v = v - v[i1]
    print(f'   ON  {key:22s} ' + ' '.join(f'{x:+9.2f}' for x in v))
for kk, key in enumerate(keys):
    v = aux_off[:, kk] - aux_off[i1, kk]
    print(f'   OFF {key:22s} ' + ' '.join(f'{x:+9.2f}' for x in v))
# per-term minima (hit-term only and expected-hits-only slope at truth)
g_hits = np.polyfit(lt_grid, aux_on[:, :, 1].mean(1), 2); g_exp = np.polyfit(lt_grid, aux_on[:, :, 0].mean(1), 2)
print(f'   ON slope at truth: d(hit term)/dx = {2*g_hits[0]+g_hits[1]:+.2f}, d(expected hits)/dx = {2*g_exp[0]+g_exp[1]:+.2f}')

# per-batch jackknife of lifetime minimum (noise-ON seed-mean profile) -- sample (track) variance
pb = cat('nll_lt_on').mean(2)   # (nb, nlt)
tot = pb.sum(0)
jk = np.array([parab_min(lt_grid, tot - pb[b])[0] for b in range(nb)])
print(f'lifetime minimum (seed-mean profile) {parab_min(lt_grid, tot)[0]:.4f}; batch-jackknife sd {np.sqrt((nb-1)*jk.var()):.4f}')
pboff = cat('nll_lt_off'); totoff = pboff.sum(0)
jk2 = np.array([parab_min(lt_grid, totoff - pboff[b])[0] for b in range(nb)])
print(f'lifetime minimum noise-OFF {parab_min(lt_grid, totoff)[0]:.4f}; batch-jackknife sd {np.sqrt((nb-1)*jk2.var()):.4f}')
print(f'NLL curvature (ON, seed mean): {2*np.polyfit(lt_grid, nll_lt_on.mean(1), 2)[0]:.1f} per (dx)^2')

# ------------------------------------------------------------------ figures
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, q in zip(axs, ['Q', 'N']):
    for lab, col, mk in [(f'{q}  noise-ON tgt / prod model', 'C0', 'o'),
                         (f'{q}  noise-OFF tgt / tiny-noise model', 'C1', 's'),
                         (f'{q}  noise-OFF tgt / prod model', 'C2', '^')]:
        r = res[lab]
        ax.errorbar(tcw, (r['R'] - 1) * 100, r['eR'] * 100, fmt=mk, color=col, ms=4, capsize=2,
                    label=f'{lab.split("  ")[1]}: k={r["k"]:+.2e}±{r["ek"]:.1e}/µs')
        tt = np.linspace(0, 190, 50)
        ax.plot(tt, (np.exp(r['a'] - r['k'] * tt) - 1) * 100, color=col, lw=1)
    # reference: what +8% lifetime in the model would look like
    kref = 1 / TAU - 1 / (1.08 * TAU)
    ax.plot(tt, (np.exp(-(-kref) * (tt - 95)) - 1) * 100 + (res[f'{q}  noise-ON tgt / prod model']['R'].mean() - 1) * 100,
            'k--', lw=1, label='slope a +8% τ bias would need')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('drift time [µs]'); ax.set_ylabel(f'obs/exp − 1  [%]  ({"charge" if q=="Q" else "hit count"})')
    ax.legend(fontsize=7); ax.set_title(f'{"Charge" if q=="Q" else "Hit-count"} closure at truth ({nb} batches × 400 cm)')
plt.tight_layout(); plt.savefig(f'{D}/fig_E1_closure_vs_drift.png', dpi=130)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axs[0]
for s in range(nll_lt_on.shape[1]):
    y = nll_lt_on[:, s] - nll_lt_on[:, s].min()
    ax.plot((lt_grid - 1) * 100, y, '-', color='C0', alpha=0.35, lw=1, label='noise-ON target (each seed)' if s == 0 else None)
y = nll_lt_off - nll_lt_off.min(); ax.plot((lt_grid - 1) * 100, y, 'o-', color='C2', label='noise-OFF target, prod model')
y = nll_tiny - nll_tiny.min(); ax.plot((lt_grid - 1) * 100, y, 's-', color='C1', label='noise-OFF target, tiny-noise model')
for tag, col in [('lt_on', 'C0'), ('lt_off', 'C2'), ('lt_off_tiny', 'C1')]:
    ax.axvline((prof[tag]['mean'] - 1) * 100, color=col, ls=':', lw=1)
ax.axvline(8, color='k', ls='--', lw=1, label='production fit bias (+8%)'); ax.axvline(-3.7, color='gray', ls='--', lw=1, label='ceiling fit (−3.7%)')
ax.set_xlabel('lifetime − truth [%]'); ax.set_ylabel('ΔNLL (sum over batches)'); ax.legend(fontsize=7)
ax.set_title(f'Lifetime profile at truth (τ_true={TAU:.0f} µs)')
ax = axs[1]
for s in range(nll_ld_on.shape[1]):
    y = nll_ld_on[:, s] - nll_ld_on[:, s].min()
    ax.plot((ld_grid - 1) * 100, y, '-', color='C0', alpha=0.35, lw=1, label='noise-ON target (each seed)' if s == 0 else None)
y = nll_ld_off - nll_ld_off.min(); ax.plot((ld_grid - 1) * 100, y, 'o-', color='C2', label='noise-OFF target, prod model')
for tag, col in [('ld_on', 'C0'), ('ld_off', 'C2')]:
    ax.axvline((prof[tag]['mean'] - 1) * 100, color=col, ls=':', lw=1)
ax.set_xlabel('long_diff − truth [%]'); ax.set_ylabel('ΔNLL'); ax.legend(fontsize=7)
ax.set_title('long_diff profile at truth')
plt.tight_layout(); plt.savefig(f'{D}/fig_E2_nll_profiles.png', dpi=130)

np.savez(f'{D}/analysis_summary.npz', tc=tcw, edges_us=edges_us,
         **{f'E1_{k}_{f}': v[f] for k, v in [(k.replace(' ', '_').replace('/', '_'), v) for k, v in res.items()] for f in ('R', 'eR', 'k', 'ek')},
         **{f'E2_{k}_mins': v['mins'] for k, v in prof.items()})
print('bin table (noise-ON charge): t[us], R-1 [%], err [%]; hits R-1, err')
for j in range(nbin):
    a, b = res['Q  noise-ON tgt / prod model'], res['N  noise-ON tgt / prod model']
    c, d = res['Q  noise-OFF tgt / tiny-noise model'], res['Q  noise-OFF tgt / prod model']
    print(f'  {tcw[j]:6.1f}  {100*(a["R"][j]-1):+7.3f} {100*a["eR"][j]:6.3f} | {100*(b["R"][j]-1):+7.3f} {100*b["eR"][j]:6.3f} '
          f'| offtiny {100*(c["R"][j]-1):+7.3f} {100*c["eR"][j]:6.3f} | offprod {100*(d["R"][j]-1):+7.3f} {100*d["eR"][j]:6.3f}')
