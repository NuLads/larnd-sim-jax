"""Ground-truth test (9 Oct): which simulation puts the charge where it physically belongs?
Truth = time-integrated charge collected per pixel, computed exactly from the segments the reference simulated
(its stored 'tracks': positions in the common swapped frame, n_electrons after recombination and attenuation,
tran_diff), each segment integrated in 1 um steps with a 2D Gaussian of width tran_diff over the pixel squares
(pitch 0.4434 cm, grid origin = TPC border, same for both codes after the geometry alignment).
Each code's per-pixel time-integrated induced charge (reference pixel_signals, ours wfs; sum x 0.1 us) is compared with it.
usage: python truth_comparison.py STAGE   (uses out_stage{STAGE}_{ref,jax}_geo.h5)"""
import sys, numpy as np, h5py
from scipy.special import erf
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP'
stage = int(sys.argv[1]) if len(sys.argv) > 1 else 1
sys.argv = sys.argv[:1]; sys.path.insert(0, D)
import analyze_comparison as ac
P = 0.4434; X0 = -31.038; Y0 = -62.076; DT = 0.1

ref = ac.load_reference_flat(f'{D}/out_stage{stage}_ref_geo.h5')
tr = h5py.File(f'{D}/out_stage{stage}_ref_geo.h5', 'r')['tracks'][:]
jx = ac.load_jax_direct(f'{D}/out_stage{stage}_jax_geo.h5')

# ---- truth ----
truth = {}
for t in tr:
    a = np.array([t['x_start'], t['y_start'], t['z_start']]); b = np.array([t['x_end'], t['y_end'], t['z_end']])
    L = np.linalg.norm(b - a); n = max(int(np.ceil(L / 1e-4)), 1)
    f = (np.arange(n) + 0.5) / n; pts = a[None, :] + f[:, None] * (b - a)[None, :]
    plane = 0 if 0.5 * (a[2] + b[2]) > 0 else 1
    q = float(t['n_electrons']) / n; s = float(t['tran_diff'])
    ix0 = np.floor((pts[:, 0] - X0) / P).astype(int); iy0 = np.floor((pts[:, 1] - Y0) / P).astype(int)
    if s < 1e-6:
        keys, cnt = np.unique(np.stack([ix0, iy0], 1), axis=0, return_counts=True)
        for (i, j), c in zip(keys, cnt): truth[(i, j, plane)] = truth.get((i, j, plane), 0.0) + q * c
    else:
        for di in range(-3, 4):
            ex_lo = X0 + (ix0 + di) * P; fx = 0.5 * (erf((ex_lo + P - pts[:, 0]) / (np.sqrt(2) * s)) - erf((ex_lo - pts[:, 0]) / (np.sqrt(2) * s)))
            for dj in range(-3, 4):
                ey_lo = Y0 + (iy0 + dj) * P; fy = 0.5 * (erf((ey_lo + P - pts[:, 1]) / (np.sqrt(2) * s)) - erf((ey_lo - pts[:, 1]) / (np.sqrt(2) * s)))
                w = fx * fy * q
                keys, inv = np.unique(np.stack([ix0 + di, iy0 + dj], 1), axis=0, return_inverse=True)
                sums = np.bincount(inv.ravel(), weights=w)
                for (i, j), v in zip(keys, sums): truth[(i, j, plane)] = truth.get((i, j, plane), 0.0) + v
# ---- reference ----
rx, ry, rp = ac.ref_pixel_physical_coords(ref['pixels'])
qr = ref['pixel_signals'].sum(1) * DT
R = {}
for x, y, p, q in zip(rx, ry, rp, qr):
    k = (int(round((x - X0) / P)), int(round((y - Y0) / P)), int(p)); R[k] = R.get(k, 0.0) + q
# ---- ours ----
up = np.asarray(jx['unique_pixels']); keep = up >= 0
jxx = np.asarray(jx['pix_x_decoded'])[keep]; jyy = np.asarray(jx['pix_y_decoded'])[keep]; jp = np.asarray(jx['pix_plane_decoded'])[keep]
qj = np.asarray(jx['wfs']).sum(1)[keep] * DT
J = {}
for x, y, p, q in zip(jxx, jyy, jp, qj):
    k = (int(np.floor((x - X0) / P)), int(np.floor((y - Y0) / P)), int(p)); J[k] = J.get(k, 0.0) + q

tot_t = sum(truth.values())
print(f'stage {stage}: segments {len(tr)}, track length {tr["dx"].sum():.1f} cm | total charge truth {tot_t:.4g}, ref {sum(R.values()):.4g} ({100*(sum(R.values())/tot_t-1):+.2f}%), ours {sum(J.values()):.4g} ({100*(sum(J.values())/tot_t-1):+.2f}%)')
keys = [k for k, v in truth.items() if v > 2e3]
T = np.array([truth[k] for k in keys]); Rv = np.array([R.get(k, 0.0) for k in keys]); Jv = np.array([J.get(k, 0.0) for k in keys])
for name, V in (('reference', Rv), ('ours', Jv)):
    r = V / T
    for lo, hi in ((2e3, 1e4), (1e4, 2e4), (2e4, 1e9)):
        m = (T >= lo) & (T < hi)
        print(f'  {name:9s} truth in [{lo:.0e},{hi:.0e}): n {m.sum():4d}  ratio median {np.median(r[m]):.3f}  IQR [{np.percentile(r[m],25):.3f},{np.percentile(r[m],75):.3f}]  |dev| median {np.median(np.abs(r[m]-1)):.3f}')
    print(f'  {name:9s} all pixels with truth > 2e3: sum |code - truth| / sum truth = {np.abs(V-T).sum()/T.sum():.4f}')
# charge put where truth has (almost) none
for name, M in (('reference', R), ('ours', J)):
    spur = sum(v for k, v in M.items() if truth.get(k, 0.0) < 200 and v > 0)
    neg = sum(v for k, v in M.items() if v < 0)
    print(f'  {name:9s} charge on pixels with truth < 200 e-: {spur:.4g} ({100*spur/tot_t:.2f}% of total); net negative pixels: {neg:.4g}')
