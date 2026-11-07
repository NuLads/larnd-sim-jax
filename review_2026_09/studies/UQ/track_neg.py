"""UQ (31 Oct, exploratory): per-track geometry curvature blocks of FULL dumps -- which tracks have negative curvature,
and are they the same tracks at different FD steps (real) or not (artefact)? usage: python3 track_neg.py a.npz b.npz ..."""
import sys, numpy as np
from uq_lib import load
from diag_full import blocks
res = {}
for path in sys.argv[1:]:
    d = load(path); neg = []; ratio = []
    ot = 0
    for j, Nd, Hn, Hc, g in blocks(d):
        nt = d['n_t_flat'][ot:ot + int(d['n_tracks'][j])]; ot += int(d['n_tracks'][j]); a = Nd
        for t, m in enumerate(nt):
            blk = Hn[a:a + m, a:a + m]; a += m
            w = np.linalg.eigvalsh(blk)
            neg.append(w.min() < -1e-3 * np.abs(w).max()); ratio.append(w.min() / np.abs(w).max())
    res[path.split('/')[-1]] = (np.array(neg), np.array(ratio))
    print(f"{path.split('/')[-1]}: {len(neg)} tracks; per-track geometry block with min eig < -1e-3 max|eig|: {np.mean(neg):.3f} "
          f"(median min/max {np.median(ratio):+.2e})")
ks = list(res)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        a, b = res[ks[i]][0], res[ks[j]][0]
        if len(a) == len(b):
            print(f"  {ks[i]} vs {ks[j]}: both negative {int((a & b).sum())}, only one {int((a ^ b).sum())}, "
                  f"overlap / union {(a & b).sum() / max((a | b).sum(), 1):.2f}")
