"""Did the fitted per-segment dE/dx develop a DRIFT-DEPENDENT profile?

THE DEGENERACY. Every segment sits at a definite drift coordinate, so a drift-correlated pattern in
the fitted dE/dx is observationally almost identical to a change in electron lifetime. The ~4000
per-segment nuisances are free to produce exactly that, and the base likelihood does not forbid it.
Measured previously at 400 cm with true positions: lifetime -1.5 +/- 1.9% with TRUE dE/dx versus
+18.5 +/- 18.9% with FITTED dE/dx, restored to -1.69 +/- 0.68% by the drift-profile penalty
(`--dedx_drift_profile_weight`, weight 1e6). That penalty is 0.0 in the production config, so if the
degeneracy is being exercised it should be visible as a non-flat fitted dE/dx profile versus depth.

THIS IS RECONSTRUCTABLE OFFLINE, unlike the truth-matched MAE. It needs only within-batch
information -- no global event ids -- because both the geometry and the dE/dx values live in the
checkpoint:
  * chain_contexts[b][t] : x0_fixed, theta0_i/phi0_i, step_len, n_chain  -> the track's nominal path
  * batch_parent_ids[b]  : sub-step row -> parent segment, giving each parent's span along the track
  * dedx_cache[b]        : the fitted log dE/dx per parent
  * true_dedx_cache[b]   : the SIM INPUT dE/dx per parent (the values the fit started from)

Sub-steps are uniform in length, so a parent's fractional span is just its sub-step count.
The nominal (unwarped) path is used for z: the fitted geometry differs by ~168 um, four orders
below the ~30 cm drift range, so it cannot matter here.

STATISTIC. The same one the penalty minimises, so the numbers are directly comparable to the
weight that would act on them:

    trend = cov_w(|z|, log dEdx) / sqrt(var_w(|z|))        [log units per 1 sigma of |z|]

reported for the FITTED values and for the INPUT values. **The difference is what the fit added** --
the input file has its own profile, and only the change is attributable to the degeneracy.

INTERPRETATION. A slope m in log(dEdx) per cm of drift absorbs an attenuation exp(m z), so the
lifetime left to explain the rest satisfies 1/tau_fit = 1/tau_true - v_drift * m. That converts a
measured dE/dx drift trend into the lifetime bias it can account for.
"""
import argparse, glob, os, pickle
import numpy as np

VDRIFT = 0.1587           # cm/us
TAU_TRUE = 2200.0         # us


def dir_from_angles(t, p):
    st = np.sin(t)
    return np.array([st * np.cos(p), st * np.sin(p), np.cos(t)], float)


def collect(h):
    """-> z (cm), log dEdx fitted, log dEdx input, weight (segment length in sub-steps)."""
    Z, YF, YI, W = [], [], [], []
    cc, pidm, dcm = h.get('chain_contexts'), h.get('batch_parent_ids'), h.get('dedx_cache')
    sim_ref = h.get('true_dedx_cache', {})
    if not cc or not pidm or not dcm:
        return None
    for b, ctxs in cc.items():
        pids = np.asarray(pidm.get(b, []))
        dc = dcm.get(b)
        if dc is None or pids.size == 0:
            continue
        fit = np.exp(np.asarray(dc['log_dedx'] if isinstance(dc, dict) else dc, float))
        inp = np.asarray(sim_ref.get(b, []), float)
        for c in ctxs:
            idx = np.asarray(c['idxs']).ravel()
            idx = idx[(idx >= 0) & (idx < len(pids))]
            p = pids[idx]; p = p[p >= 0]
            if p.size == 0:
                continue
            parents, counts = np.unique(p, return_counts=True)
            if parents.max() >= len(fit):
                continue
            edges = np.concatenate([[0.0], np.cumsum(counts)]) / counts.sum()
            mid = 0.5 * (edges[:-1] + edges[1:])
            nc = int(c['n_chain'])
            u0 = dir_from_angles(float(c['theta0_i']), float(c['phi0_i']))
            z0 = float(np.asarray(c['x0_fixed'])[2])
            zn = z0 + np.arange(nc + 1) * float(c['step_len']) * u0[2]
            z = np.interp(mid * nc, np.arange(nc + 1), zn)
            Z.append(z); W.append(counts.astype(float))
            YF.append(np.log(np.maximum(fit[parents], 1e-3)))
            YI.append(np.log(np.maximum(inp[parents], 1e-3)) if len(inp) > parents.max()
                      else np.full(len(parents), np.nan))
    if not Z:
        return None
    return (np.concatenate(Z), np.concatenate(YF), np.concatenate(YI), np.concatenate(W))


def trend(zc, y, w):
    """Weighted cov(|z|, y)/sqrt(var(|z|)) — the penalty's statistic — plus the raw slope per cm."""
    ok = np.isfinite(y) & np.isfinite(zc) & (w > 0)
    if ok.sum() < 10:
        return np.nan, np.nan, np.nan
    zc, y, w = np.abs(zc[ok]), y[ok], w[ok]
    W = w.sum()
    zb, yb = (w * zc).sum() / W, (w * y).sum() / W
    cov = (w * (zc - zb) * (y - yb)).sum() / W
    var = (w * (zc - zb) ** 2).sum() / W
    if var <= 0:
        return np.nan, np.nan, np.nan
    return cov / np.sqrt(var), cov / var, np.sqrt(var)      # trend, slope per cm, sigma_z


def tau_equiv(slope_per_cm):
    """Lifetime the dE/dx trend alone could fake: 1/tau_fit = 1/tau_true - v*m."""
    inv = 1.0 / TAU_TRUE - VDRIFT * slope_per_cm
    return 1.0 / inv if inv > 0 else np.inf


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    a = ap.parse_args()
    hdr = (f"{'run':24s}{'seed':>5s}{'n_seg':>7s}{'trend FIT':>11s}{'trend IN':>10s}"
           f"{'Δtrend':>9s}{'slope/cm':>10s}{'σ_z':>7s}{'τ faked':>10s}{'→ bias':>9s}")
    print(hdr); print('-' * len(hdr))
    for d in a.dirs:
        for f in sorted(glob.glob(os.path.join(d, 'history_iter*.pkl'))):
            try:
                h = pickle.load(open(f, 'rb'))
            except Exception:
                continue
            r = collect(h)
            if r is None:
                continue
            z, yf, yi, w = r
            tf, sf, sz = trend(z, yf, w)
            ti, si, _ = trend(z, yi, w)
            dslope = sf - si if np.isfinite(si) else np.nan
            te = tau_equiv(dslope) if np.isfinite(dslope) else np.nan
            bias = 100 * (te / TAU_TRUE - 1) if np.isfinite(te) else np.nan
            sd = f.split('seed')[1].split('.')[0]
            print(f"{os.path.basename(d):24s}{sd:>5s}{len(z):>7d}{tf:>11.4f}{ti:>10.4f}"
                  f"{tf - ti:>9.4f}{dslope:>10.5f}{sz:>7.2f}{te:>10.0f}{bias:>+8.1f}%")
