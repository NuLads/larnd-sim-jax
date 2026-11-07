"""UQ (31 Oct, exploratory): composition of the most negative eigenvector of each batch's joint matrix M_b (relative
calibration units): weight on calibration / dE/dx / geometry, and the 2x2 Cauchy-Schwarz test on the calibration
couplings: c_pk^2 / (H_cc,pp w_k) > 1 for a nuisance mode k means the assembled blocks are not jointly PSD on (p, k).
usage: python3 neg_modes.py <full_dump.npz>"""
import sys, numpy as np
from uq_lib import load
from diag_full import blocks
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
d = load(sys.argv[1]); jac = d['jac']; Ji = np.diag(1 / jac)
comp, cs_worst, cs_cnt, wneg_geom, wneg_dedx = [], [], np.zeros(5), [], []
for j, Nd, Hn, Hc, g in blocks(d):
    Hcc = Ji @ d['Hcc'][j] @ Ji; Hcr = Hc @ Ji
    M = np.block([[Hcc, Hcr.T], [Hcr, Hn]])
    # scale-free: symmetric diagonal scaling (Jacobi) so the eigenvectors are not dominated by units
    dg = np.sqrt(np.abs(np.diag(M))) + 1e-300; Ms = M / np.outer(dg, dg)
    w, V = np.linalg.eigh(Ms); v = V[:, 0] ** 2
    comp.append((w[0], v[:5].sum(), v[5:5 + Nd].sum(), v[5 + Nd:].sum(), int(np.argmax(v[:5]))))
    wn, Vn = np.linalg.eigh(Hn); keep = wn > 1e-6 * np.abs(wn).max()
    a = Vn[:, keep].T @ Hcr; ratio = a ** 2 / (np.diag(Hcc)[None, :] * wn[keep][:, None])
    cs_worst.append(ratio.max(0)); cs_cnt += (ratio > 1).any(0)
    # nuisance blocks alone: negative eigenvalues of the dE/dx block and of the geometry part
    wneg_dedx.append(np.linalg.eigvalsh(Hn[:Nd, :Nd]).min() / np.abs(wn).max())
    wneg_geom.append(np.linalg.eigvalsh(Hn[Nd:, Nd:]).min() / np.abs(wn).max() if Hn.shape[0] > Nd else 0)
c = np.array([x[:4] for x in comp]); arg = np.array([x[4] for x in comp])
print(f"{sys.argv[1].split('/')[-1]}: Jacobi-scaled M_b min eig: median {np.median(c[:, 0]):+.3f}, {int((c[:, 0] < -0.05).sum())}/{len(c)} below -0.05")
print(f"  most negative eigenvector weight (median over batches): calibration {np.median(c[:, 1]):.2f}, dE/dx {np.median(c[:, 2]):.2f}, geometry {np.median(c[:, 3]):.2f}")
print('  dominant calibration parameter in it: ' + ', '.join(f'{P[i]} {int((arg == i).sum())}' for i in range(5)))
cw = np.array(cs_worst)
print('  Cauchy-Schwarz c^2/(H_pp w_k) > 1 for some kept nuisance mode, batches: ' + ' '.join(f'{p} {int(n)}' for p, n in zip(P, cs_cnt))
      + ' | median of the max ratio: ' + ' '.join(f'{v:.2f}' for v in np.median(cw, 0)))
print(f"  nuisance-only min eig / max|eig|: dE/dx block median {np.median(wneg_dedx):+.2e}; geometry part median {np.median(wneg_geom):+.2e}, "
      f"negative in {int((np.array(wneg_geom) < 0).sum())}/{len(c)}")
