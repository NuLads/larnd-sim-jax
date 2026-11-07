"""Independent-seed check of the a05 agreement: same waveform, production stochastic code, 12 other keys of 10,000 throws."""
import sys, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS')
import jax, jax.numpy as jnp, paper_common as pc
from jax import random
from larndsim.fee_jax import get_adc_values
from scipy.stats import chi2 as ch
d = dict(np.load('review_2026_09/scripts/paperfigs/fig3_probabilistic_method_cache.npz'))
params, _ = pc.params(); N = 10000
wfs = jnp.tile(jnp.asarray(d['wfs_pixel'])[None, :], (N, 1)); P = np.exp(np.clip(d['log_prob_pixel'], -40, 0)); LO, HI = 150, 500
def groups(e):
    g = []; acc = 0; st = 0
    for i in range(len(e)):
        acc += e[i]
        if acc >= 10:
            g.append((st, i + 1)); st = i + 1; acc = 0
    if st < len(e):
        g[-1] = (g[-1][0], len(e))
    return g
res = {0: [], 1: []}; pool = {0: np.zeros(HI - LO), 1: np.zeros(HI - LO)}
SEEDS = [12345] + list(range(1, 13))
for seed in SEEDS:
    _, ts = get_adc_values(params, wfs, random.key(seed)); ts = np.asarray(ts).astype(int)
    row = []
    for n in (0, 1):
        t = ts[:, n]; ok = (t >= LO) & (t < HI); c = np.bincount(t[ok] - LO, minlength=HI - LO).astype(float); p = P[n, LO:HI].astype(float); e = p * N
        g = groups(e); E = np.array([e[a:b].sum() for a, b in g]); O = np.array([c[a:b].sum() for a, b in g])
        chi = ((O - E) ** 2 / E).sum(); tv = .5 * np.abs(p - c / N).sum(); tt = np.arange(LO, HI); mean_e = (tt * c).sum() / c.sum(); mean_m = (tt * p).sum() / p.sum()
        lam_e = c.sum() / N
        res[n].append((chi, len(E), tv, mean_e - mean_m, lam_e - p.sum())); pool[n] += c
        row += [round(chi, 1), len(E), round(tv, 4), round(mean_e - mean_m, 3), round(lam_e - p.sum(), 5)]
    print(seed, row, flush=True)
for n in (0, 1):
    a = np.array(res[n][1:]); print('n', n, 'seeds', len(a), 'mean chi2', a[:, 0].mean().round(1), 'nb', a[0, 1], 'mean TV', a[:, 2].mean().round(4), 'mean dt', a[:, 3].mean().round(3), 'sd dt', a[:, 3].std(ddof=1).round(3), 'mean dlam', a[:, 4].mean().round(5), 'sd', a[:, 4].std(ddof=1).round(5))
# pooled: all 13 keys together
Ntot = N * len(SEEDS)
for n in (0, 1):
    p = P[n, LO:HI].astype(float); e = p * Ntot; g = groups(e); E = np.array([e[a:b].sum() for a, b in g]); O = np.array([pool[n][a:b].sum() for a, b in g])
    chi = ((O - E) ** 2 / E).sum(); tt = np.arange(LO, HI); mean_e = (tt * pool[n]).sum() / pool[n].sum(); mean_m = (tt * p).sum() / p.sum(); sd = np.sqrt((tt ** 2 * pool[n]).sum() / pool[n].sum() - mean_e ** 2)
    lam_e = pool[n].sum() / Ntot; se = np.sqrt(p.sum() * (1 - p.sum()) / Ntot)
    pu = (O - E) / np.sqrt(E)
    print('POOLED n', n, 'N', Ntot, 'chi2', round(chi, 1), 'bins', len(E), 'p', round(float(ch.sf(chi, len(E))), 4), 'lam model', round(p.sum(), 5), 'lam throws', round(lam_e, 5), '+-', round(se, 5), 'dmean', round(mean_e - mean_m, 4), '+-', round(sd / np.sqrt(pool[n].sum()), 4), 'worst pulls', np.round(np.sort(np.abs(pu))[-3:], 2), 'at groups', [g[i] for i in np.argsort(np.abs(pu))[-3:]], 'TV', round(.5 * np.abs(p - pool[n] / Ntot).sum(), 4))
    if n == 1:
        print('TAIL n=1 ticks>=315: throws', O[-1], 'model', round(E[-1], 1), 'of', Ntot, '; ticks',  [(int(LO + a), int(LO + b), int(o), round(float(x), 1)) for (a, b), o, x in zip(g[-6:], O[-6:], E[-6:])])
        print('n=1 per-tick tail throws', [(int(LO + i), int(pool[1][i]), round(float(e[i]), 1)) for i in range(160, 180)])
