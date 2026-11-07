"""Per-piece contributions to one real pixel of the main track: the production simulate_signals, evaluated with the charge of all but one 0.5 mm piece
of track set to zero (it is linear in the electron counts). Saves data/a03_pixel.npz. Prints the closure against the full-event waveform."""
import os, sys, pickle, time, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS')
import paper_common as pc
import jax, jax.numpy as jnp
from larndsim.sim_jax import simulate_wfs, _signal_inputs, simulate_signals
from larndsim.detsim_jax import id2pixel
OUT = '/sdf/scratch/neutrino/pgranger/talk_animations/data/a03_pixel.npz'
z = np.load('/sdf/scratch/neutrino/pgranger/talk_animations/data/a01_03_sim.npz')
h = pickle.load(open('/sdf/scratch/neutrino/pgranger/talk_animations/data/hero.pkl', 'rb'))
truth, response = pc.params(); P = truth.replace(**h['calib_true'])
f = tuple(h['tfields']); tr = np.asarray(h['true_rows'])
seg = z['seg']; sx, sy = z['sx'], z['sy']; X, Y, Z, TID = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 12]
# candidates: offsets of the track from the pixel centre
pitch = float(z['pitch']); b = z['tpc_borders'][0]
for cand in [(21, 215), (22, 215), (20, 215), (19, 215), (12, 216), (13, 216), (11, 216), (73, 210), (74, 210)]:
    m = (sx == cand[0]) & (sy == cand[1]) & (TID == 4)
    cx = b[0, 0] + (cand[0] + .5) * pitch; cy = b[1, 0] + (cand[1] + .5) * pitch
    print(cand, m.sum(), 'dy from centre mm', 10 * (Y[m].mean() - cy), 'dx span', 10 * (X[m].min() - cx), 10 * (X[m].max() - cx), 'other tracks rows', ((sx == cand[0]) & (sy == cand[1]) & (TID != 4)).sum())
PX = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (21, 215)
near = (np.abs(sx - PX[0]) <= 5) & (np.abs(sy - PX[1]) <= 5)
print('PX', PX, 'rows near', near.sum(), 'tracks', np.unique(TID[near], return_counts=True))
idx = np.where(near)[0]
idx = idx[np.argsort(z['tdrift'][idx])]               # arrival order
sub = jnp.asarray(tr[idx])
args, kw = _signal_inputs(P, response, sub, f)
args = list(args)
up = np.asarray(args[1]); px_, py_, _, _ = [np.asarray(a) for a in id2pixel(P, jnp.asarray(up))]
iP = np.where((px_ == PX[0]) & (py_ == PX[1]) & (up >= 0))[0][0]
ne0, nn0 = args[5], args[8]
t0 = time.time(); W = np.asarray(simulate_signals(*args, **kw))[:, 1:]; print('compile+run', time.time() - t0, flush=True)
full = z['wfs'][np.where((z['upx'] == PX[0]) & (z['upy'] == PX[1]))[0][0]]
print('closure restricted vs full event on this pixel: max |diff| e/us', np.abs(W[iP] - full).max(), 'peak', full.max())
GR = 5   # rows per piece (0.5 mm)
n = len(idx); grp = np.arange(n) // GR; ng = grp.max() + 1
wg = np.zeros((ng, W.shape[1]), np.float32)
t0 = time.time()
gj = jnp.asarray(grp)
for g in range(ng):
    m = (gj == g)
    a2 = list(args); a2[5] = ne0 * jnp.repeat(m, 25).astype(ne0.dtype); a2[8] = nn0 * m.astype(nn0.dtype)
    wg[g] = np.asarray(simulate_signals(*a2, **kw))[iP, 1:]
print('groups', ng, 'time', time.time() - t0, 'sum closure max', np.abs(wg.sum(0) - W[iP]).max(), flush=True)
gi = lambda a: np.array([a[idx[grp == g]].mean() for g in range(ng)])
gs = lambda a: np.array([a[idx[grp == g]].sum() for g in range(ng)])
np.savez_compressed(OUT, px=np.array(PX), wg=wg, full=full, wfull_restricted=W[iP], gx=gi(X), gy=gi(Y), gz=gi(Z), gt=gi(z['tdrift']), gn=gs(z['nd']), gsx=gi(sx.astype(float)), gsy=gi(sy.astype(float)), gtid=gi(TID), idx=idx, grp=grp, GR=GR)
print('saved', os.path.getsize(OUT) / 1e6)
