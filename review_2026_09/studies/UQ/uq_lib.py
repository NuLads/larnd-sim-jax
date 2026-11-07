"""UQ (31 Oct): estimators from a LARND_JOINT_NEWTON_DUMP file (fit_params._jn_dump). All outputs in RELATIVE units
(fraction of the parameter value), via jac = d ln(phys) / d(norm) at the dumped point.
  E1 conditional  V = (sum_b H_cc,b)^-1
  E2 dE/dx-prof.  V = (sum_b Sd_b)^-1                      (geometry conditional)
  E3 profile      V = (sum_b S_b)^-1                       (dE/dx + geometry profiled; kn = nuisance damping index)
  E4 sandwich     V = S^-1 J S^-1, J = B/(B-1) sum_b (r_b - rbar)(r_b - rbar)^T  (per-batch profile scores)
  Newton offset   d = -S^-1 r (r = sum_b r_b), z = d / sigma; EDM = r^T S^-1 r / 2."""
import numpy as np


def load(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def estimators(d, kn=0):
    jac = d['jac'] if 'jac' in d else np.ones(5)
    J_ = np.diag(jac)
    Bn = d['S'].shape[1]
    pieces = {'E1': (d['Hcc'], d['gc']), 'E2': (d['Sd'], d['rd']), 'E3': (d['S'][kn], d['r'][kn])}
    out = {}
    for name, (Sb, rb) in pieces.items():
        S = Sb.sum(0); r = rb.sum(0)
        w = np.linalg.eigvalsh(0.5 * (S + S.T))
        V = np.linalg.inv(S)
        rc = rb - rb.mean(0); Jm = Bn / (Bn - 1) * rc.T @ rc
        Vs = V @ Jm @ V
        dn = -V @ r
        Vr, Vsr = J_ @ V @ J_, J_ @ Vs @ J_
        out[name] = dict(S_mineig=float(w.min()), S_cond=float(w.max() / w.min()) if w.min() > 0 else np.inf,
                         sig=np.sqrt(np.abs(np.diag(Vr))), sig_sw=np.sqrt(np.abs(np.diag(Vsr))),
                         corr=Vr / np.sqrt(np.outer(np.abs(np.diag(Vr)), np.abs(np.diag(Vr)))),
                         d=jac * dn, z=(jac * dn) / np.sqrt(np.abs(np.diag(Vr))), z_sw=(jac * dn) / np.sqrt(np.abs(np.diag(Vsr))),
                         edm=float(0.5 * r @ V @ r), V=Vr, Vsw=Vsr)
    out['dropped_frac'] = float(d['n_neg'].sum() / d['n'].sum())
    out['dropped_dedx_frac'] = float(d['nd_neg'].sum() / d['Nd'].sum())
    out['B'] = int(Bn)
    return out


def truth_errors(ckpt_pickle, params):
    """(relative error of the checkpoint's CURRENT calibration vs the target, last-1000 mean error) per parameter."""
    import pickle
    f1 = lambda x: float(np.mean(x))
    h = pickle.load(open(ckpt_pickle, 'rb'))
    cur = np.array([f1(h[f'{p}_iter'][-1]) / f1(h[f'{p}_target'][0]) - 1 for p in params])
    m1k = np.array([np.mean([f1(x) for x in h[f'{p}_iter'][-1000:]]) / f1(h[f'{p}_target'][0]) - 1 for p in params])
    return cur, m1k
