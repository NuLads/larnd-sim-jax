"""a08 helper: extract what a08_track_converging needs from hero.pkl, a08_seed.pkl and traces.npz into $T/data/a08_pack.npz.
Definitions of the per-pixel data and first-hit residual follow event_display.py exactly."""
import pickle, numpy as np
T = '/sdf/scratch/neutrino/pgranger/talk_animations'
h = pickle.load(open(f'{T}/data/hero.pkl', 'rb')); sd = pickle.load(open(f'{T}/data/a08_seed.pkl', 'rb'))
m = h['main_ctx']; D = h['data']; ST = h['states']
up, inv = np.unique(D['pixel_id'], return_inverse=True)
Dp = dict(pid=up, q=np.bincount(inv, D['q']), nhit=np.bincount(inv), x=np.zeros(len(up)), y=np.zeros(len(up)), t0=np.full(len(up), np.inf))
for j in range(len(up)):
    s = inv == j; Dp['x'][j], Dp['y'][j] = D['pixel_x'][s][0], D['pixel_y'][s][0]; Dp['t0'][j] = D['ticks'][s].min()
u0 = np.asarray(m['u0']); uxy = u0[:2] / np.linalg.norm(u0[:2]); nxy = np.array([-uxy[1], uxy[0]]); o = np.asarray(m['x0'])[:2]
rot = lambda P: np.stack([(P - o) @ uxy, (P - o) @ nxy], -1)
Lxy = m['total_len'] * np.linalg.norm(u0[:2])
lookup = lambda src, key, pids: np.where(np.isin(pids, src['pid']), src[key][np.clip(np.searchsorted(src['pid'], pids), 0, len(src['pid']) - 1)], np.nan)
out = {}
for k in ('pid', 'q', 'x', 'y', 't0', 'nhit'): out['D_' + k] = Dp[k]
s_cm = np.asarray(m['s_cm']); o_ = np.argsort(s_cm)
out['s_cm'] = s_cm[o_]; out['dev_true'] = np.asarray(m['dev_true_um'])[o_] / 1e4 * 10; out['dedx_true'] = np.asarray(m['dedx_true'])[o_]
out['total_len'] = m['total_len']; out['Lxy'] = Lxy; out['uxy'] = uxy; out['nxy'] = nxy; out['o'] = o; out['e2z'] = np.asarray(m['e2'])[2]
tp = np.asarray(m['true_pos'])[o_]; out['true_xy'] = rot(tp[:, :2]); out['true_z'] = tp[:, 2]
names = ['straight', 'seed', 'iter10000', 'iter14000', 'iter20000']
out['names'] = np.array(names)
sa = rot(np.stack([Dp['x'], Dp['y']], 1)); near = np.abs(sa[:, 1]) < 1.0; out['sa'] = sa[:, 0]; out['across'] = sa[:, 1]
Mfit = ST['iter20000']['pred']; rf = Dp['t0'] - lookup(Mfit, 't0', Dp['pid'])
k = near & np.isfinite(rf) & (lookup(Mfit, 'lam0', Dp['pid']) > 0.5)        # event_display.py: selection from the fitted model
out['tmask'] = k; out['sdf'] = lookup(Mfit, 't0sd', Dp['pid'])
print('timing pixels', k.sum(), 'of', len(k))
for nm in names:
    S = sd if nm == 'seed' else ST['start' if nm == 'straight' else nm]
    p = S['pred']; dev = np.asarray(S['dev_um'])[o_] / 1e4 * 10
    for kk in ('pid', 'x', 'y', 'q', 'nhit', 'lam0', 't0', 't0sd'): out[f'{nm}_{kk}'] = np.asarray(p[kk])
    out[f'{nm}_dev'] = dev; out[f'{nm}_dedx'] = np.asarray(S['dedx_main'])[o_]
    out[f'{nm}_rs'] = Dp['t0'] - lookup(p, 't0', Dp['pid'])
    out[f'{nm}_calib'] = np.array([S['calib'][q] for q in ('Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime')])
    tr = out['dev_true']; rs = out[f'{nm}_rs'][k]
    print(f"{nm:10s} ke {p['q'].sum():7.0f} hits {p['nhit'].sum():6.1f} | e2 rms {np.sqrt(np.mean((dev[:,1]-tr[:,1])**2)):.3f} mm e1 rms {np.sqrt(np.mean((dev[:,0]-tr[:,0])**2)):.3f} max e2 {dev[:,1].max():.2f}"
          f" | dEdx MAE {np.mean(np.abs(out[f'{nm}_dedx']-out['dedx_true'])):.4f} mean {out[f'{nm}_dedx'].mean():.4f} rms {100*np.sqrt(np.mean((out[f'{nm}_dedx']/out['dedx_true']-1)**2)):.2f}% "
          f"| t res mean {np.nanmean(rs):.2f} sd {np.nanstd(rs):.2f} n {np.isfinite(rs).sum()}")
print('data ke', Dp['q'].sum(), 'hits', len(D['q']), 'true dEdx mean', out['dedx_true'].mean(), 'median sdf', np.nanmedian(out['sdf'][k]))
z = np.load(f'{T}/data/traces.npz', allow_pickle=True); nm = [str(n) for n in z['names']]; i = nm.index('KN10N8C75H')
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
for p in P:
    out['tr_' + p] = z[p][i].astype(float); out['true_' + p] = float(z[p + '_target'][i]); print(p, 'target trace', z[p + '_target'][i], 'hero true', h['calib_true'][p], 'init', z[p + '_init'][i], z[p][i][0])
out['tr_loss'] = z['losses_iter'][i].astype(float); out['tr_pos'] = z['pos_residual_iter'][i].astype(float); out['tr_posb'] = z['pos_residual_batch'][i].astype(float)
out['tr_dedx_mae'] = z['dedx_mae_truth_iter'][i].astype(float); out['tr_mean_dedx'] = z['mean_dedx_iter'][i].astype(float)
np.savez_compressed(f'{T}/data/a08_pack.npz', **out)
cyc = lambda a: np.convolve(a, np.ones(100) / 100, mode='valid')             # running full-cycle mean, label = last iteration of the cycle
L = cyc(out['tr_loss']); Pp = cyc(out['tr_pos']) * 1e4
for it in (99, 500, 1000, 2000, 5000, 7400, 7600, 9900, 10100, 10300, 11000, 13900, 14100, 15000, 19999):
    print(it, 'loss', round(L[it - 99], 1), 'pos um', round(Pp[it - 99], 1), {p: round(100 * (out['tr_' + p][it] / out['true_' + p] - 1), 2) for p in P}, 'dedx mae', out['tr_dedx_mae'][min(it, len(out['tr_dedx_mae']) - 1)], out['tr_mean_dedx'][min(it, len(out['tr_mean_dedx']) - 1)])
for it, name in ((10000, 'iter10000'), (14000, 'iter14000'), (20000, 'iter20000')):
    print(name, 'cal state vs trace last-1000 mean', [round(100 * (float(out['tr_' + p][it - 999:it + 1].mean()) / out[f'{name}_calib'][j] - 1), 3) for j, p in enumerate(P)])
print('batch order check', (out['tr_posb'][:200] == np.arange(200) % 100).all(), (out['tr_posb'][-100:] == np.arange(100)).all())
print(len(out['tr_loss']), len(out['tr_dedx_mae']), len(out['tr_mean_dedx']))
