"""Load the a06 sweep npz files and the data, and align everything on one pixel index. Helper, not an animation."""
import glob, os, numpy as np
T = '/sdf/scratch/neutrino/pgranger/talk_animations'
SW = f'{T}/data/a06_sweep'
PARAMS = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']


def data():
    z = np.load(f'{T}/data/a06_hero_small.npz')
    f = [str(s) for s in z['fields']]; tr = z['true_rows']
    D = dict(pid=z['data_pixel_id'], px=z['data_pixel_x'], py=z['data_pixel_y'], t=z['data_ticks'], q=z['data_q'], f=f, tr=tr, z=z)
    return D


def track_geom(D, xs, ys):
    """For pixel centres: nearest true segment (xy) -> track id, s (cm along that track), d (signed cm off-axis in xy), z (drift coordinate)."""
    f, tr = D['f'], D['tr']; ix = f.index
    Tid = tr[:, ix('trackID')]; X, Y, Z = tr[:, ix('x')], tr[:, ix('y')], tr[:, ix('z')]; dx = tr[:, ix('dx')]
    o = dict(track=np.zeros(len(xs)), s=np.zeros(len(xs)), d=np.zeros(len(xs)), z=np.zeros(len(xs)))
    tracks = np.unique(Tid)
    best = np.full(len(xs), np.inf)
    for t in tracks:
        m = Tid == t
        order = np.argsort(X[m])                          # all three tracks run mostly along x
        xt, yt, zt = X[m][order], Y[m][order], Z[m][order]
        # smooth centre line: track direction from local differences
        idx = np.linspace(0, len(xt) - 1, 200).astype(int); xk, yk, zk = xt[idx], yt[idx], zt[idx]
        ds = np.r_[0, np.cumsum(np.hypot(np.diff(xk), np.diff(yk)))]
        for j, (a, b) in enumerate(zip(xs, ys)):
            r = np.hypot(xk - a, yk - b); k = int(r.argmin())
            if r[k] < best[j]:
                best[j] = r[k]; k1 = min(k + 1, len(xk) - 1); k0 = max(k - 1, 0)
                ux, uy = xk[k1] - xk[k0], yk[k1] - yk[k0]; n = np.hypot(ux, uy); ux, uy = ux / n, uy / n
                o['track'][j] = t; o['s'][j] = ds[k] + (a - xk[k]) * ux + (b - yk[k]) * uy
                o['d'][j] = -(a - xk[k]) * uy + (b - yk[k]) * ux; o['z'][j] = zk[k]
    return o


def sweep(p, tag='main'):
    fs = glob.glob(f'{SW}/{tag}_{p}_*.npz')
    R = [dict(np.load(f)) for f in fs if not f.endswith('.tmp.npz')]
    R.sort(key=lambda r: float(r['val']))
    return R


def align(R, key, fill=0.0):
    """Pixel-indexed array (Npoints, Nunion, ...) on the union of simulated pixels."""
    u = np.unique(np.concatenate([r['pid'] for r in R]))
    sh = R[0][key].shape[1:]
    a = np.full((len(R), len(u)) + sh, fill, dtype=np.float32)
    for i, r in enumerate(R):
        a[i, np.searchsorted(u, r['pid'])] = r[key]
    return u, a
