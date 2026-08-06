"""Build controlled degradations of the TRUE segment file, for the likelihood-scan
quality studies.

We cannot interpolate between the true file and linear_guess_segments.h5: they have different
segmentations (3.03M vs 10.08M segments), so there is no row correspondence. Instead we perturb
the true file itself, which keeps everything except the axis under study identical.

POSITION ladder (--mode pos --rms <um>): each TRAJECTORY is displaced by a single random 3-D
offset with the requested RMS magnitude. A whole-trajectory rigid shift (rather than per-segment
jitter) is the right model here because the spline geometry basis represents smooth per-track
displacement, and it is what the fitted geometry residual (~168 um) actually measures.

DEDX ladder (--mode dedx --frac f): dEdx is blended toward the global mean,
    dEdx' = mean + f*(dEdx - mean)
so f=1 is the truth and f=0 is constant-at-the-mean (the CONSTDEDX condition). dE is rescaled
consistently (dE = dEdx*dx) and n_electrons is scaled by the same ratio, since the number of
ionisation electrons is proportional to the deposited energy.

CHORD ladder (--mode chord --chord c): each trajectory is pulled toward the straight line joining
its first and last point, by fraction c (0 = truth, 1 = perfectly straight). This is a SYSTEMATIC,
along-track-correlated geometry error -- the kind the straight-line guess file actually has --
as opposed to the `pos` mode's random rigid offsets, which are uncorrelated between trajectories
and preserve each track's shape. The `pos` ladder showed random offsets cost variance but not
bias, while the guess file is badly biased, so the error STRUCTURE is the remaining suspect.
dEdx is held fixed and dE/n_electrons are rescaled to the new (shorter) dx, so this is a pure
geometry change and does not leak into the dEdx axis.
"""
import argparse, os
import numpy as np
import h5py

TRUE = '/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5'
POSF = [('x', 'y', 'z'), ('x_start', 'y_start', 'z_start'), ('x_end', 'y_end', 'z_end')]

ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['pos', 'dedx', 'chord'], required=True)
ap.add_argument('--rms', type=float, default=0.0, help='position RMS in micrometres')
ap.add_argument('--frac', type=float, default=1.0, help='dEdx retention fraction (1=truth, 0=mean)')
ap.add_argument('--chord', type=float, default=1.0, help='straightening fraction (0=truth, 1=straight)')
ap.add_argument('--post_frac', type=float, default=1.0,
                help='dEdx spread blend applied AFTER the geometry mode (composes with pos/chord)')
ap.add_argument('--mean_shift', type=float, default=0.0,
                help='multiplicative dEdx mean shift, e.g. 0.02 = +2%%; spread shape preserved')
ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=12345)
a = ap.parse_args()

with h5py.File(TRUE, 'r') as f:
    seg = f['segments'][:]
    traj = f['trajectories'][:]

if a.mode == 'pos':
    rms_cm = a.rms * 1e-4                       # micrometres -> cm
    # one rigid offset per (event_id, traj_id); sigma per axis so that |offset| has the target RMS
    key = seg['event_id'].astype(np.int64) * 1000000 + seg['traj_id'].astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    rng = np.random.default_rng(a.seed)
    off = rng.normal(0.0, rms_cm / np.sqrt(3.0), size=(len(uniq), 3))
    got = np.sqrt((off ** 2).sum(1).mean()) * 1e4
    print(f'[pos] {len(uniq)} trajectories, requested RMS {a.rms:.1f} um, realised {got:.1f} um')
    for trip in POSF:
        for j, fld in enumerate(trip):
            if fld in seg.dtype.names:
                seg[fld] = seg[fld] + off[inv, j]
elif a.mode == 'chord':
    c = a.chord
    key = seg['event_id'].astype(np.int64) * 1000000 + seg['traj_id'].astype(np.int64)
    # order segments along each track so "first start" and "last end" are the track's endpoints
    order = np.lexsort((seg['segment_id'], key))
    ks = key[order]
    grp_start = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    grp_end = np.r_[grp_start[1:] - 1, len(ks) - 1]
    gid = np.zeros(len(ks), np.int64)
    gid[grp_start[1:]] = 1
    gid = np.cumsum(gid)                                   # group index per (sorted) row

    P0 = np.stack([seg[f][order][grp_start].astype(np.float64) for f in ('x_start', 'y_start', 'z_start')], 1)
    P1 = np.stack([seg[f][order][grp_end].astype(np.float64) for f in ('x_end', 'y_end', 'z_end')], 1)
    D = P1 - P0
    L = np.linalg.norm(D, axis=1)
    u = D / np.where(L[:, None] > 0, L[:, None], 1.0)      # unit chord direction per track
    print(f'[chord] {len(P0)} trajectories, straightening fraction c={c}')

    dev = []
    for trip in POSF:
        if not all(f in seg.dtype.names for f in trip):
            continue
        p = np.stack([seg[f][order].astype(np.float64) for f in trip], 1)
        rel = p - P0[gid]
        proj = P0[gid] + (np.einsum('ij,ij->i', rel, u[gid])[:, None]) * u[gid]
        dev.append(np.linalg.norm(p - proj, axis=1))
        newp = p + c * (proj - p)                          # pull toward the chord
        for j, fld in enumerate(trip):
            out = seg[fld].copy()
            out[order] = newp[:, j].astype(seg[fld].dtype)
            seg[fld] = out
    print(f'[chord] transverse deviation from the chord, before: mean {np.mean(dev[0]) * 1e4:.0f} um, '
          f'95th pct {np.percentile(dev[0], 95) * 1e4:.0f} um  -> after: x{1 - c:.2f}')

    # dEdx is the physics we are NOT changing: recompute dx from the new endpoints and rescale
    # the extensive quantities so dEdx stays exactly as it was.
    ns = np.stack([seg[f].astype(np.float64) for f in ('x_start', 'y_start', 'z_start')], 1)
    ne = np.stack([seg[f].astype(np.float64) for f in ('x_end', 'y_end', 'z_end')], 1)
    new_dx = np.linalg.norm(ne - ns, axis=1)
    old_dx = seg['dx'].astype(np.float64)
    ratio = np.where(old_dx > 0, new_dx / np.where(old_dx > 0, old_dx, 1.0), 1.0)
    seg['dx'] = new_dx.astype(seg['dx'].dtype)
    for fld in ('dE', 'n_electrons', 'n_photons'):
        if fld in seg.dtype.names:
            seg[fld] = (seg[fld].astype(np.float64) * ratio).astype(seg[fld].dtype)
    print(f'[chord] dx mean {old_dx.mean():.4f} -> {new_dx.mean():.4f} '
          f'(total track length x{new_dx.sum() / old_dx.sum():.4f}); dEdx unchanged by construction')
else:
    dedx = seg['dEdx'].astype(np.float64)
    dx = seg['dx'].astype(np.float64)
    m = dedx.mean()
    new = m + a.frac * (dedx - m)
    ratio = np.where(dedx > 0, new / np.where(dedx > 0, dedx, 1.0), 1.0)
    seg['dEdx'] = new.astype(seg['dEdx'].dtype)
    if 'dE' in seg.dtype.names:
        seg['dE'] = (new * dx).astype(seg['dE'].dtype)
    if 'n_electrons' in seg.dtype.names:
        seg['n_electrons'] = (seg['n_electrons'].astype(np.float64) * ratio).astype(seg['n_electrons'].dtype)
    print(f'[dedx] frac={a.frac}  mean={m:.5f}  spread {dedx.std():.5f} -> {new.std():.5f}')

# ---------------------------------------------------------------- composable dE/dx post-transforms
# These run AFTER the geometry mode, so a single file can carry BOTH a geometry defect and a dE/dx
# defect -- which is what the straight-line guess file actually has, and what no single-axis rung
# has ever tested. Applied to whatever `seg` the mode above produced.
if a.post_frac != 1.0:
    dedx = seg['dEdx'].astype(np.float64)
    m = dedx.mean()
    new = m + a.post_frac * (dedx - m)
    ratio = np.where(dedx > 0, new / np.where(dedx > 0, dedx, 1.0), 1.0)
    seg['dEdx'] = new.astype(seg['dEdx'].dtype)
    if 'dE' in seg.dtype.names:
        seg['dE'] = (new * seg['dx'].astype(np.float64)).astype(seg['dE'].dtype)
    if 'n_electrons' in seg.dtype.names:
        seg['n_electrons'] = (seg['n_electrons'].astype(np.float64) * ratio).astype(seg['n_electrons'].dtype)
    print(f'[post-dedx] frac={a.post_frac}  spread {dedx.std():.5f} -> {new.std():.5f}')

# MEAN SHIFT: dEdx -> dEdx*(1+s), i.e. the whole distribution scaled, spread SHAPE preserved.
# This is the error an over-stiff dEdx mean constraint imposes: the fitted mean is pinned by the
# prior (target 1.887 at w=1e5) rather than determined by the data, so if the data's true mean
# differs, every segment carries a common multiplicative offset. The `dedx` ladder shrinks the
# SPREAD and leaves the mean alone; this is the orthogonal axis and had never been measured.
if a.mean_shift != 0.0:
    dedx = seg['dEdx'].astype(np.float64)
    new = dedx * (1.0 + a.mean_shift)
    seg['dEdx'] = new.astype(seg['dEdx'].dtype)
    if 'dE' in seg.dtype.names:
        seg['dE'] = (new * seg['dx'].astype(np.float64)).astype(seg['dE'].dtype)
    if 'n_electrons' in seg.dtype.names:
        seg['n_electrons'] = (seg['n_electrons'].astype(np.float64) * (1.0 + a.mean_shift)).astype(seg['n_electrons'].dtype)
    print(f'[mean-shift] {a.mean_shift:+.4f}  mean {dedx.mean():.5f} -> {new.mean():.5f}')

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with h5py.File(a.out, 'w') as f:
    f.create_dataset('segments', data=seg)
    f.create_dataset('trajectories', data=traj)
print('wrote', a.out, os.path.getsize(a.out) / 1e9, 'GB')
