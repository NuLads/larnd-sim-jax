"""UQ pilot follow-up (31 Oct): per-batch anatomy of the profile pieces of a dump -- where does the non-PD sum come from?
Relative units: S_rel = J^-1 S J^-1 (J = diag(jac)), so 1/sqrt(S_rel) is a relative sigma.
usage: python3 diag_batches.py <dump.npz> [<dump2.npz> ...]"""
import sys, numpy as np
from uq_lib import load

P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']


def rel(S, jac):
    Ji = np.diag(1.0 / jac)
    return Ji @ S @ Ji


for path in sys.argv[1:]:
    d = load(path)
    jac = d['jac']; B = d['S'].shape[1]
    print(f"\n=== {path.split('/')[-1]}: {B} batches, eps {d['eps']}, nuisances {int(d['n'].sum())}")
    Hcc = np.array([rel(h, jac) for h in d['Hcc']])
    Sd = np.array([rel(h, jac) for h in d['Sd']])
    S = np.array([[rel(h, jac) for h in d['S'][k]] for k in range(len(d['kn']))])
    mineig = lambda A: np.array([np.linalg.eigvalsh(0.5 * (a + a.T)).min() for a in A])
    tot = lambda A: np.linalg.eigvalsh(0.5 * (A.sum(0) + A.sum(0).T))
    print('  Hcc_b min eig: neg in', int((mineig(Hcc) < 0).sum()), 'batches; total eig', np.array2string(tot(Hcc), precision=3))
    for name, A in [('Sd', Sd)] + [(f"S kn{d['kn'][k]:g}", S[k]) for k in range(len(d['kn']))]:
        m = mineig(A)
        # diagonal per-parameter: fraction of the summed profile curvature removed, i.e. 1 - S_pp / Hcc_pp
        frac = 1 - A.sum(0).diagonal() / Hcc.sum(0).diagonal()
        # the batches whose S_b diagonal is the most negative, per parameter
        neg_diag = (np.array([a.diagonal() for a in A]) < 0).sum(0)
        print(f"  {name:8s}: S_b non-PD in {int((m < 0).sum()):3d}/{B} batches (worst {m.min():+.3g}, median min eig {np.median(m):+.3g})"
              f" | total eig {np.array2string(tot(A), precision=3)}")
        print(f"            curvature removed by profiling (1 - S_pp/Hcc_pp): " + ' '.join(f'{p} {f:+.3f}' for p, f in zip(P, frac))
              + f" | negative S_b,pp in batches: {neg_diag}")
        # robustness: total with the worst 5 batches (by min eig) removed
        keep = np.argsort(m)[5:]
        print(f"            total eig without the 5 worst batches: {np.array2string(tot(A[keep]), precision=3)}")
    # nuisance block spectrum
    print('  Hnn kept eig min / max per batch (median):', f"{np.median(d['eig_min']):.3g} / {np.median(d['eig_max']):.3g}",
          '| min over batches', f"{np.min(d['eig_min']):.3g}", '| dropped per batch median', int(np.median(d['n_neg'])),
          'max', int(d['n_neg'].max()), '| dE/dx dropped', int(d['nd_neg'].sum()))
