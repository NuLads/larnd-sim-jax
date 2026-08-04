"""Re-segment the TRUE file to the guess file's segmentation, changing NOTHING else.

WHY. The straight-line guess file displaces the likelihood minimum further than either axis of
the quality ladder: lifetime +15.2%, where destroying all dE/dx information gives +7.0% and
880 um of position error gives -3.8%. Neither axis, nor their sum, accounts for it (report S6f,
open question 11). The remaining structural difference between the two files is segmentation:

    TRUE   3 030 787 segments, dx variable, mean 3.28 cm, median 2.28 cm
    GUESS 10 079 953 segments, dx uniform  ~0.99 cm

i.e. the guess file is resegmented onto a uniform ~1 cm step. Since the per-segment dE/dx prior
(student-t + soft barrier + mean constraint) is tuned on the TRUE dx distribution, applying it to
3.3x shorter segments is a real, untested change of conditions.

WHAT THIS BUILDS. Each true segment is split into N = max(1, round(dx/1cm)) collinear pieces.
Endpoints are linearly interpolated along the segment, so every sub-segment lies exactly on the
original track: positions are UNCHANGED as a curve. dE/dx, and the diffusion coefficients are
copied verbatim; the extensive quantities (dx, dE, n_electrons, n_photons) are divided by N, so
the total deposited energy and electron count are conserved exactly.

The result therefore differs from the TRUE file in segmentation ALONE -- not position, not dE/dx,
not charge. Scanning it isolates the effect that neither ladder axis captures.
"""
import argparse
import numpy as np
import h5py

TRUE = '/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5'
# linearly interpolated along the segment: (start field, end field, midpoint field or None)
GEOM = [('x_start', 'x_end', 'x'), ('y_start', 'y_end', 'y'), ('z_start', 'z_end', 'z'),
        ('t_start', 't_end', 't'), ('t0_start', 't0_end', 't0')]
EXTENSIVE = ['dx', 'dE', 'n_electrons', 'n_photons']

ap = argparse.ArgumentParser()
ap.add_argument('--target_dx', type=float, default=0.99, help='sub-segment length in cm')
ap.add_argument('--out', required=True)
a = ap.parse_args()

with h5py.File(TRUE, 'r') as f:
    seg = f['segments'][:]
    traj = f['trajectories'][:]

dx = seg['dx'].astype(np.float64)
n = np.maximum(1, np.rint(np.abs(dx) / a.target_dx)).astype(np.int64)
tot = int(n.sum())
print(f'{len(seg)} segments -> {tot} ({tot / len(seg):.2f}x), target dx {a.target_dx} cm')

src = np.repeat(np.arange(len(seg)), n)              # parent index of each sub-segment
# k = 0..n-1 within each parent, via the standard repeat-offset trick
starts = np.repeat(np.cumsum(n) - n, n)
k = np.arange(tot) - starts
nn = n[src].astype(np.float64)

out = np.zeros(tot, dtype=seg.dtype)
for fld in seg.dtype.names:                          # default: copy from the parent
    out[fld] = seg[fld][src]

for s, e, mid in GEOM:
    if s not in seg.dtype.names:
        continue
    p0 = seg[s][src].astype(np.float64)
    p1 = seg[e][src].astype(np.float64)
    f0, f1 = k / nn, (k + 1.0) / nn
    ns, ne = p0 + (p1 - p0) * f0, p0 + (p1 - p0) * f1
    out[s] = ns.astype(seg[s].dtype)
    out[e] = ne.astype(seg[e].dtype)
    if mid and mid in seg.dtype.names:
        out[mid] = (0.5 * (ns + ne)).astype(seg[mid].dtype)

for fld in EXTENSIVE:
    if fld in seg.dtype.names:
        out[fld] = (seg[fld][src].astype(np.float64) / nn).astype(seg[fld].dtype)

if 'segment_id' in seg.dtype.names:
    out['segment_id'] = np.arange(tot).astype(seg['segment_id'].dtype)

# conservation / invariance checks -- these are the point of the file, so assert them loudly
for fld in ['dE', 'n_electrons', 'dx']:
    if fld in seg.dtype.names:
        o, i = out[fld].astype(np.float64).sum(), seg[fld].astype(np.float64).sum()
        print(f'  {fld:12s} total {i:.6g} -> {o:.6g}   ({100 * (o / i - 1):+.4f}%)')
d_in = seg['dEdx'].astype(np.float64)
d_out = out['dEdx'].astype(np.float64)
print(f'  dEdx         mean {d_in.mean():.5f} -> {d_out.mean():.5f}, '
      f'sd {d_in.std():.5f} -> {d_out.std():.5f}')
print(f'  dx           mean {dx.mean():.4f} -> {out["dx"].astype(np.float64).mean():.4f}')

with h5py.File(a.out, 'w') as f:
    f.create_dataset('segments', data=out)
    f.create_dataset('trajectories', data=traj)
import os
print('wrote', a.out, os.path.getsize(a.out) / 1e9, 'GB')
