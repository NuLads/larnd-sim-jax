"""Length-binned split-Student-t population prior for per-segment true dE/dx (target file).
Same density as optimize/dedx_utils.split_t_logpdf_jax, in numpy. One count per segment."""
import h5py, numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t as tdist
def logc(nu): return gammaln((nu+1)/2)-gammaln(nu/2)-0.5*np.log(nu*np.pi)
def logpdf(x, nuL, nuR, loc, sL, sR):
    lKL = tdist.logpdf((x-loc)/sL, nuL)-logc(nuL); lKR = tdist.logpdf((x-loc)/sR, nuR)-logc(nuR)
    lZ = np.log(0.5*sL*np.exp(-logc(nuL))+0.5*sR*np.exp(-logc(nuR)))
    return np.where(x < loc, lKL, lKR) - lZ
f = h5py.File('/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5','r')
s = f['segments']; dx = s['dx'][:]; de = s['dEdx'][:]
ok = (dx > 0) & (de > 0) & np.isfinite(de)
dx, de = dx[ok], de[ok]
edges = np.array([0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.25, 3.25, 4.5, 6.5, 9.0, 1e9])
out = []
rng = np.random.default_rng(0)
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (dx >= lo) & (dx < hi); x = de[m]
    if len(x) > 60000: x = rng.choice(x, 60000, replace=False)
    x = x[x < 30]
    med = np.median(x); iqr = np.subtract(*np.percentile(x, [75, 25]))
    p0 = [np.log(5), np.log(2.5), med, np.log(iqr/1.5), np.log(iqr/1.2)]
    nll = lambda p: -np.mean(logpdf(x, np.exp(p[0]), np.exp(p[1]), p[2], np.exp(p[3]), np.exp(p[4])))
    r = minimize(nll, p0, method="L-BFGS-B")
    p = r.x; par = [np.exp(p[0]), np.exp(p[1]), p[2], np.exp(p[3]), np.exp(p[4])]
    print(f'dx[{lo:.2f},{hi:.2f}) n={m.sum():7d} nuL {par[0]:6.2f} nuR {par[1]:5.2f} loc {par[2]:.4f} sL {par[3]:.4f} sR {par[4]:.4f}  nll {r.fun:.4f} {r.success}')
    out.append(par)
np.savez('prior_bins.npz', edges=edges, params=np.array(out))
