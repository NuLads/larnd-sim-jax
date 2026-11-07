"""UQ pilot 2, offline (31 Oct): anatomy of the profile from a LARND_JOINT_NEWTON_DUMP_FULL dump (every batch's assembled
blocks). Relative units throughout (J = diag(jac)).
  (a) is each batch's joint matrix M_b = [[H_cc,b, H_nc^T], [H_nc, H_nn]] PSD? (min eig / max eig)
  (b) which nuisance modes remove the calibration curvature: R_pp = sum_k (v_k^T H_nc,p)^2 / w_k binned in w_k / max w
  (c) alternative profiles: spectral cut of H_nn at rel = 1e-6 (as dumped), 1e-5, 1e-4, 1e-3; and the per-batch PSD
      projection of M_b (negative eigenvalues of M_b set to 0, then the Schur complement of the projected M_b).
usage: python3 diag_full.py <full_dump.npz> [...]"""
import sys, numpy as np
from uq_lib import load

P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']


def blocks(d):
    oh = on = 0
    for j in range(len(d['n'])):
        n = int(d['n'][j])
        Hn = d['Hnn_flat'][oh:oh + n * n].reshape(n, n); oh += n * n
        Hc = d['Hnc_flat'][on * 5:(on + n) * 5].reshape(n, 5); g = d['gn_flat'][on:on + n]; on += n
        yield j, int(d['Nd'][j]), 0.5 * (Hn + Hn.T), Hc, g


def summ(Sb, rb, jac):
    S = Sb.sum(0); r = rb.sum(0); w = np.linalg.eigvalsh(S)
    if w.min() <= 0:
        return f'not PD (min eig {w.min():+.3g})', None
    V = np.linalg.inv(S); Jd = np.diag(jac); Vr = Jd @ V @ Jd; sig = np.sqrt(np.diag(Vr)); dd = -jac * (V @ r)
    rc = rb - rb.mean(0); Jm = len(rb) / (len(rb) - 1) * rc.T @ rc; sw = np.sqrt(np.diag(Jd @ V @ Jm @ V @ Jd))
    return ('sigma(%) ' + ' '.join(f'{100 * v:.3g}' for v in sig) + ' | sandwich(%) ' + ' '.join(f'{100 * v:.3g}' for v in sw)
            + ' | z ' + ' '.join(f'{v:+.2f}' for v in dd / sig)), sig


def main(paths):
  for path in paths:
      d = load(path)
      jac = d['jac'] if 'jac' in d else np.ones(5); Ji = np.diag(1 / jac)
      B = len(d['n'])
      print(f"\n=== {path.split('/')[-1]}: {B} batches, eps {d['eps']}, calib eps {d.get('eps_calib')}")
      cuts = (1e-6, 1e-5, 1e-4, 1e-3)
      bins = (0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1.01)
      Rbin = np.zeros((len(bins) - 1, 5))
      Sc = {c: (np.zeros((B, 5, 5)), np.zeros((B, 5))) for c in cuts}
      Sp = (np.zeros((B, 5, 5)), np.zeros((B, 5))); Mneg = []; Hcc_tot = np.zeros((5, 5))
      for j, Nd, Hn, Hc, g in blocks(d):
          Hcc = d['Hcc'][j]; gc = d['gc'][j]; Hcc_tot += Ji @ Hcc @ Ji
          w, V = np.linalg.eigh(Hn); wmax = np.abs(w).max()
          a = V.T @ (Hc @ Ji)                       # (n, 5) mode couplings in relative units
          for k in range(len(w)):
              if w[k] <= 0:
                  continue                                 # dropped by every profile here
              Rbin[np.searchsorted(bins, w[k] / wmax, side='right') - 1] += a[k] ** 2 / w[k]
          for c in cuts:
              keep = w > c * wmax
              Vk, wk = V[:, keep], w[keep]
              Z = Vk @ ((Vk.T @ Hc) / wk[:, None]); z = Vk @ ((Vk.T @ g) / wk)
              Sc[c][0][j] = Hcc - Hc.T @ Z; Sc[c][1][j] = gc - Hc.T @ z
          # (a) + PSD projection of the joint matrix
          M = np.block([[Hcc, Hc.T], [Hc, Hn]]); Ds = np.concatenate([1 / jac, np.ones(len(Hn))])
          wm, Vm = np.linalg.eigh(M)
          Mneg.append(wm.min() / wm.max())
          Mp = (Vm * np.clip(wm, 0, None)) @ Vm.T
          Hn_p, Hc_p, Hcc_p = Mp[5:, 5:], Mp[5:, :5], Mp[:5, :5]
          wp, Vp = np.linalg.eigh(0.5 * (Hn_p + Hn_p.T)); keep = wp > 1e-6 * wp.max()
          Vk, wk = Vp[:, keep], wp[keep]
          Sp[0][j] = Hcc_p - Hc_p.T @ (Vk @ ((Vk.T @ Hc_p) / wk[:, None]))
          Sp[1][j] = gc - Hc_p.T @ (Vk @ ((Vk.T @ g) / wk))
      Mneg = np.array(Mneg)
      print(f"  (a) joint M_b min/max eig: negative in {int((Mneg < -1e-9).sum())}/{B} batches; median {np.median(Mneg):+.2e}, "
            f"worst {Mneg.min():+.2e}")
      hd = np.diag(Hcc_tot)
      print('  (b) curvature removed, as a fraction of H_cc,pp, by nuisance modes binned in w / max w:')
      for b in range(len(bins) - 1):
          print(f"      [{bins[b]:.0e}, {bins[b + 1]:.0e}): " + ' '.join(f'{p} {Rbin[b, i] / hd[i]:8.3f}' for i, p in enumerate(P)))
      print('  (c) profile estimators:')
      for c in cuts:
          print(f"      spectral cut {c:.0e}: " + summ(*Sc[c], jac)[0])
      print('      per-batch PSD projection: ' + summ(*Sp, jac)[0])
      print('      conditional E1: ' + summ(d['Hcc'], d['gc'], jac)[0])


if __name__ == '__main__':
    main(sys.argv[1:])
