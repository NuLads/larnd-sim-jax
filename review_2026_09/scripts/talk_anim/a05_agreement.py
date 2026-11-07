"""Measured agreement between the analytic P(t|n) (log_prob_pixel) and the 10,000 real stochastic throws of the same waveform.
Prints the numbers quoted on screen in a05 and in the report, and saves them to a05_agreement.json. Nothing here is plotted."""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
d = dict(np.load('review_2026_09/scripts/paperfigs/fig3_probabilistic_method_cache.npz'))
P = np.exp(np.clip(d['log_prob_pixel'], -40, 0)); ts = d['ticks_s'].astype(int); N = ts.shape[0]; Nt = d['wfs_pixel'].shape[0]; Nb = Nt - 1
SENT = Nt - 2                                       # production sentinel: no hit in this slot
rng = np.random.default_rng(0)
LO, HI = 150, 500                                   # first strike; nonzero current ends 320, next strike starts ~1400
out = {}
def stats(n, lo, hi, label):
    t = ts[:, n]; ok = (t != SENT) & (t >= lo) & (t < hi)
    c = np.bincount(t[ok] - lo, minlength=hi - lo).astype(float)
    p = P[n, lo:hi].astype(float)
    lam_m, lam_e = p.sum(), c.sum() / N
    se = np.sqrt(max(lam_m * (1 - lam_m), 1e-12) / N)
    tt = np.arange(lo, hi)
    mean_m = (tt * p).sum() / lam_m; mean_e = (tt * c).sum() / c.sum()
    sd_m = np.sqrt((tt ** 2 * p).sum() / lam_m - mean_m ** 2); sd_e = np.sqrt((tt ** 2 * c).sum() / c.sum() - mean_e ** 2)
    se_mean = sd_e / np.sqrt(c.sum())
    tv = 0.5 * np.abs(p - c / N).sum()
    # null: same TV if the throws really were drawn from the analytic P(t|n) (+ no-hit bin)
    pfull = np.append(p, max(1 - p.sum(), 0)); pfull /= pfull.sum()
    sim = rng.multinomial(N, pfull, size=4000)[:, :-1] / N
    tvn = 0.5 * np.abs(sim - p).sum(1)
    # chi2 on bins merged to expected count >= 10, Poisson-like (multinomial without the no-hit bin)
    e = p * N; edges = []; acc = 0; start = 0
    groups = []
    for i in range(len(e)):
        acc += e[i]
        if acc >= 10:
            groups.append((start, i + 1)); start = i + 1; acc = 0
    if start < len(e):
        s0, _ = groups[-1]; groups[-1] = (s0, len(e))
    E = np.array([e[a:b].sum() for a, b in groups]); O = np.array([c[a:b].sum() for a, b in groups])
    chi2 = ((O - E) ** 2 / E).sum(); nd = len(E)
    from scipy.stats import chi2 as ch, ks_1samp
    pv = float(ch.sf(chi2, nd))
    # KS on the conditional distribution
    cdf_m = np.cumsum(p) / lam_m; cdf_e = np.cumsum(c) / c.sum(); ks = np.abs(cdf_m - cdf_e).max()
    sim2 = rng.multinomial(int(c.sum()), p / lam_m, size=4000); ksn = np.abs(np.cumsum(sim2, 1) / c.sum() - cdf_m).max(1)
    r = dict(n=n, window=[lo, hi], lam_model=lam_m, lam_throws=lam_e, lam_se=se, mean_model=mean_m, mean_throws=mean_e, mean_se=se_mean, sd_model=sd_m, sd_throws=sd_e,
             TV=tv, TV_null_mean=float(tvn.mean()), TV_null_95=float(np.quantile(tvn, .95)), chi2=float(chi2), nbins=nd, p_value=pv, KS=float(ks), KS_null_95=float(np.quantile(ksn, .95)))
    print(label, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True); return r
out['n0_window'] = stats(0, LO, HI, 'n=0 first strike')
out['n1_window'] = stats(1, LO, HI, 'n=1 first strike')
out['n2_window'] = {'model': float(P[2, LO:HI].sum()), 'throws': float(((ts[:, 2] != SENT) & (ts[:, 2] >= LO) & (ts[:, 2] < HI)).mean())}
out['n1_late'] = {'model': float(P[1, 1000:].sum()), 'throws': float(((ts[:, 1] != SENT) & (ts[:, 1] >= 1000)).mean())}
out['n0_late'] = {'model': float(P[0, 1000:].sum()), 'throws': float(((ts[:, 0] != SENT) & (ts[:, 0] >= 1000)).mean())}
ok = ts != SENT
out['lam_all'] = {f'n{n}': dict(model=float(P[n].sum()), cached_lam=float(d['lam_pixel'][n]), throws=float(ok[:, n].mean())) for n in range(5)}
print(out['n2_window'], out['n1_late'], out['n0_late']); print(out['lam_all'])
pat = {}
for row in ok.astype(int)[:, :6]:
    k = ''.join(map(str, row)); pat[k] = pat.get(k, 0) + 1
print('slot patterns', pat); out['patterns'] = pat
# per-tick pulls on the two peaks
for n in (0, 1):
    t = ts[:, n]; c = np.bincount(t[t != SENT], minlength=Nb)[:Nb]; e = P[n] * N
    m = e > 25; pull = (c[m] - e[m]) / np.sqrt(e[m])
    print('n', n, 'bins', m.sum(), 'pull mean', pull.mean().round(3), 'rms', pull.std().round(3), 'max', np.abs(pull).max().round(2), 'bin rel diff max', np.abs(c[m] / e[m] - 1).max().round(3))
    out[f'pulls_n{n}'] = dict(bins=int(m.sum()), mean=float(pull.mean()), rms=float(pull.std()), max=float(np.abs(pull).max()))
# all hits in the first strike, slots merged
lo, hi = LO, HI; allc = np.zeros(hi - lo)
for n in range(5):
    t = ts[:, n]; w = (t != SENT) & (t >= lo) & (t < hi); allc += np.bincount(t[w] - lo, minlength=hi - lo)
p_all = P[:5, lo:hi].sum(0); print('all slots: TV (per-throw hit density)', 0.5 * np.abs(p_all - allc / N).sum(), 'total model', p_all.sum(), 'throws', allc.sum() / N)
out['all_slots'] = dict(TV=float(0.5 * np.abs(p_all - allc / N).sum()), model=float(p_all.sum()), throws=float(allc.sum() / N))
json.dump(out, open(f'{HERE}/a05_agreement.json', 'w'), indent=1, default=float)
