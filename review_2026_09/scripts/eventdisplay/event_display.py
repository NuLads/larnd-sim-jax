"""Event display for the paper: one event of a finished joint fit (recipe R2), comparing the fit's own target hits
("data": the stochastic simulation at the true parameters and true segments) with the probabilistic model evaluated
at the fit's endpoint -- fitted calibration, fitted trajectories, fitted per-segment dE/dx -- rebuilt from the
checkpoint (rebuild.py). Also shows, for the event's longest track, the trajectory and the energy loss: true,
fitted, and at the start of the fit (straight line, initial dE/dx). CPU is enough: one event.
usage: python event_display.py --batch 35 --event 2 [--track 4] --out fig.png"""
import argparse, json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuild import FitRebuild
CK = '/sdf/scratch/neutrino/pgranger/fit_result_staging/sci_full_STRFRZ508/history_iter10000_sci_full_b100_len400_mw2500.0_refit_seed0.pkl'
ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', default=CK); ap.add_argument('--batch', type=int, required=True)
ap.add_argument('--event', type=float, required=True); ap.add_argument('--track', type=float, default=None)
ap.add_argument('--out', required=True); ap.add_argument('--cache', default=None); a = ap.parse_args()

R = FitRebuild(a.ckpt); b = R.batch(a.batch)
import jax.numpy as jnp
from larndsim.losses_jax import adc2charge
from optimize.strategies import LUTProbabilisticSimulation
f, col = R.sfields, R.col
evi, dxi, dEdxi = f.index('eventID'), col['dx'], f.index('dEdx')
rows = np.where((b['guess'][:, evi] == a.event) & (b['guess'][:, dxi] > 0))[0]
ctxs = [c for c in b['ctxs'] if c.track_id[0] == a.event]
main = max(ctxs, key=lambda c: c.total_len) if a.track is None else [c for c in ctxs if c.track_id[1] == a.track][0]
print(f'batch {a.batch} event {a.event}: {len(ctxs)} tracks ' + ', '.join(f'{c.track_id[1]:.0f}:{c.total_len:.0f}cm' for c in ctxs)
      + f'; main track {main.track_id} ({main.total_len:.1f} cm)', flush=True)
P_fit = R.truth_params.replace(**R.calib_fit); P_ini = R.truth_params.replace(**R.calib_init)

def summarise(pred, P):
    hp = np.exp(np.asarray(pred['hit_prob']))                       # (Npix, Nslot, Ntick): P(hit n at tick t)
    q = np.asarray(adc2charge(pred['adcs_distrib'], P))              # ke, expected charge given hit n at tick t
    upx = np.asarray(pred['unique_pixels']); ok = upx >= 0
    tt = np.arange(hp.shape[2], dtype=float)
    lam = hp.sum(2); p0 = hp[:, 0, :]; l0 = np.maximum(lam[:, 0], 1e-12)
    m0 = (p0 * tt).sum(1) / l0; s0 = np.sqrt(np.maximum((p0 * tt ** 2).sum(1) / l0 - m0 ** 2, 0))
    return dict(pid=upx[ok], x=np.asarray(pred['pixel_x'])[ok], y=np.asarray(pred['pixel_y'])[ok], nhit=lam.sum(1)[ok],
                q=(hp * q).sum((1, 2))[ok], lam0=lam[ok, 0], t0=m0[ok], t0sd=s0[ok])

cache = a.cache or a.out.rsplit('.', 1)[0] + '_cache.npz'
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True); M = z['M'].item(); S = z['S'].item()
else:
    sim = LUTProbabilisticSimulation(R.response)
    M = summarise(sim.predict(P_fit, jnp.asarray(b['fitted'][rows]), f, None), P_fit)
    sim = LUTProbabilisticSimulation(R.response)
    S = summarise(sim.predict(P_ini, jnp.asarray(b['guess'][rows]), f, None), P_ini)
    np.savez(cache, M=M, S=S)
tg = R.target(a.batch); m = tg['event'] == int(a.event)
D = {k: v[m] for k, v in tg.items()}; D['q'] = np.asarray(adc2charge(jnp.asarray(D['adcs']), R.truth_params))   # ke per hit
# per-pixel data: total charge, number of hits, first-hit tick
up, inv = np.unique(D['pixel_id'], return_inverse=True)
Dp = dict(pid=up, q=np.bincount(inv, D['q']), nhit=np.bincount(inv), x=np.zeros(len(up)), y=np.zeros(len(up)), t0=np.full(len(up), np.inf))
for j in range(len(up)):
    s = inv == j; Dp['x'][j], Dp['y'][j] = D['pixel_x'][s][0], D['pixel_y'][s][0]; Dp['t0'][j] = D['ticks'][s].min()
print(f"data: {m.sum()} hits on {len(up)} pixels, {Dp['q'].sum():.0f} ke | fitted model: {M['nhit'].sum():.1f} hits, {M['q'].sum():.0f} ke"
      f" | start model: {S['nhit'].sum():.1f} hits, {S['q'].sum():.0f} ke", flush=True)

# ---------- main-track geometry and energy loss, per segment row ----------
fp = R.fp; u0 = fp._dir_from_angles(main.theta0_i, main.phi0_i); e1, e2 = fp._transverse_frame(u0)
idx = np.asarray(main.idxs); s_cm = main.alpha_mid * main.total_len
nom = main.x0_fixed[None, :] + s_cm[:, None] * u0[None, :]
xyz = lambda arr: arr[idx][:, [col['x'], col['y'], col['z']]]
tp = R.ck['true_positions'][a.batch]; tid = tuple(map(float, main.track_id))
true_pos = np.asarray(tp.get(tid, tp.get(str(tid))))
dev = {k: 1e4 * np.stack([(p - nom) @ e1, (p - nom) @ e2], 1) for k, p in
       (('true', true_pos), ('fit', xyz(b['fitted'])), ('start', xyz(b['guess'])))}          # micrometres
pid = np.asarray(R.ck['batch_parent_ids'][a.batch])[idx]
tde = np.asarray(R.ck['true_dedx_matched_cache'][a.batch])
ded = dict(true=tde[pid], fit=b['fitted'][idx, dEdxi], start=b['guess'][idx, dEdxi])
res = {k: float(np.linalg.norm(xyz(b[kk]) - true_pos, axis=1).mean() * 1e4) for k, kk in (('fit', 'fitted'), ('start', 'guess'))}
print(f"main track mean distance to truth: fitted {res['fit']:.0f} um, straight start {res['start']:.0f} um; |e2_z| = {abs(e2[2]):.2f}", flush=True)
json.dump(dict(batch=a.batch, event=a.event, main=list(map(float, main.track_id)), L=main.total_len, res=res,
               data_hits=int(m.sum()), data_q=float(Dp['q'].sum()), fit_hits=float(M['nhit'].sum()), fit_q=float(M['q'].sum()),
               start_hits=float(S['nhit'].sum()), start_q=float(S['q'].sum()), calib_fit=R.calib_fit, calib_true=R.calib_true),
          open(a.out.rsplit('.', 1)[0] + '_numbers.json', 'w'), indent=1)

# ---------------------------------- figure ----------------------------------
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.collections import PolyCollection
plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'legend.fontsize': 6.5})
pitch = 0.4434
uxy = u0[:2] / np.linalg.norm(u0[:2]); nxy = np.array([-uxy[1], uxy[0]]); o = main.x0_fixed[:2]
rot = lambda P: np.stack([(P - o) @ uxy, (P - o) @ nxy], -1)                 # pixel plane -> (along, across) of the main track
corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * pitch / 2
def pixels(ax, x, y, v, norm, cmap):
    polys = rot(np.stack([x, y], 1)[:, None, :] + corners[None, :, :])
    pc = PolyCollection(polys, cmap=cmap, norm=norm, lw=0.25, edgecolors='0.9'); pc.set_array(np.asarray(v)); ax.add_collection(pc); return pc
lookup = lambda src, key, pids: np.where(np.isin(pids, src['pid']), src[key][np.clip(np.searchsorted(src['pid'], pids), 0, len(src['pid']) - 1)], np.nan)
Lxy = main.total_len * np.linalg.norm(u0[:2]); w0, w1, hw = Lxy / 2 - 13, Lxy / 2 + 13, 1.6     # 26 cm strip around the middle
tl = rot(true_pos[np.argsort(main.alpha_mid)][:, :2])
fig = plt.figure(figsize=(7.2, 5.4))
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.38)
top = outer[0].subgridspec(3, 2, width_ratios=[1, 3.1], hspace=0.65, wspace=0.18)
bot = outer[1].subgridspec(1, 3, wspace=0.42)
qmax = np.percentile(np.concatenate([Dp['q'], M['q']]), 99)
ax = fig.add_subplot(top[:, 0])
ax.scatter(Dp['x'], Dp['y'], c=Dp['q'], s=1.2, marker='s', lw=0, cmap='viridis', norm=Normalize(0, qmax))
box = o + np.array([[w0, -hw], [w1, -hw], [w1, hw], [w0, hw], [w0, -hw]]) @ np.stack([uxy, nxy])
ax.plot(box[:, 0], box[:, 1], 'r-', lw=0.8); ax.set_aspect('equal'); ax.set_xlabel('$x$ (cm)'); ax.set_ylabel('$y$ (cm)')
ax.set_title('(a) event, data'); ax.margins(0.08)
strips = [(Dp, '(b) data: charge per pixel', Normalize(0, qmax), 'viridis', 'q')]
strips.append((M, '(c) fitted model: expected charge per pixel', Normalize(0, qmax), 'viridis', 'q'))
allp = np.union1d(Dp['pid'], M['pid']); dq = np.zeros(len(allp)); X = np.zeros(len(allp)); Y = np.zeros(len(allp))
for src, sgn in ((Dp, 1.), (M, -1.)):
    k = np.searchsorted(allp, src['pid']); dq[k] += sgn * src['q']; X[k] = src['x']; Y[k] = src['y']
r = max(np.percentile(np.abs(dq), 98), 1e-3)
strips.append((dict(x=X, y=Y, q=dq), '(d) data $-$ model', TwoSlopeNorm(0, -r, r), 'RdBu_r', 'q'))
for k, (src, title, norm, cmap, key) in enumerate(strips):
    ax = fig.add_subplot(top[k, 1]); pc = pixels(ax, src['x'], src['y'], src[key], norm, cmap)
    if k < 2: ax.plot(tl[:, 0], tl[:, 1], color='w', lw=0.5, alpha=0.9)
    ax.set_xlim(w0, w1); ax.set_ylim(-hw, hw); ax.set_aspect('equal'); ax.set_title(title, loc='left')
    ax.set_yticks([-1, 0, 1]); ax.set_ylabel('across (cm)', fontsize=6.5)
    if k == 2: ax.set_xlabel('position along the track, pixel plane (cm)')
    else: ax.set_xticklabels([])
    cax = ax.inset_axes([1.015, 0.0, 0.012, 1.0]); cb = fig.colorbar(pc, cax=cax); cb.ax.tick_params(labelsize=6)
    if k != 1: cb.set_label('ke$^-$', fontsize=6.5)
# (e) first-hit time residuals along the main track: data - model, for the fitted model and the start of the fit
sa = rot(np.stack([Dp['x'], Dp['y']], 1)); near = np.abs(sa[:, 1]) < 1.0; sa = sa[:, 0]
rf = Dp['t0'] - lookup(M, 't0', Dp['pid']); rs = Dp['t0'] - lookup(S, 't0', Dp['pid']); sdf = lookup(M, 't0sd', Dp['pid'])
k = near & np.isfinite(rf) & (lookup(M, 'lam0', Dp['pid']) > 0.5); o_ = np.argsort(sa[k])
ax = fig.add_subplot(bot[0, 0])
ax.fill_between(sa[k][o_], -sdf[k][o_], sdf[k][o_], color='C3', alpha=0.15, lw=0, label=r'fitted model $\pm1\sigma$')
ax.scatter(sa[k], rs[k], s=2.5, c='0.55', lw=0, label=r'data $-$ nominal start')
ax.scatter(sa[k], rf[k], s=2.5, c='C3', lw=0, label=r'data $-$ fitted model')
ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('position along the track (cm)'); ax.set_ylabel('first-hit time residual (ticks)')
ax.set_title('(e) hit timing'); ax.legend(loc='upper left', frameon=False, markerscale=2.5); ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
# (f) trajectory: deviation from the straight line between the endpoints, along the drift-carrying direction
ax = fig.add_subplot(bot[0, 1])
ax.plot(s_cm, dev['true'][:, 1] / 1e3, 'k-', lw=1.0, label='true')
ax.plot(s_cm, dev['fit'][:, 1] / 1e3, color='C3', lw=1.0, label='fitted')
ax.plot(s_cm, dev['start'][:, 1] / 1e3, color='0.55', lw=1.0, ls='--', label='straight line')
ax.set_xlabel('arc length (cm)'); ax.set_ylabel(r'displacement along $\hat e_2$ (mm)'); ax.set_title('(f) trajectory'); ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.06))
# (g) energy loss along the main track
ax = fig.add_subplot(bot[0, 2]); o_ = np.argsort(s_cm)
ax.plot(s_cm[o_], ded['true'][o_], color='k', lw=0.7, label='true')
ax.plot(s_cm[o_], ded['fit'][o_], color='C3', lw=0.8, label='fitted')
ax.plot(s_cm[o_], ded['start'][o_], color='0.55', lw=1.0, ls='--', label='initial')
lo, hi = np.percentile(np.concatenate([ded['true'], ded['fit']]), [0.5, 99.5]); ax.set_ylim(lo - 0.3, hi + 0.08)
ax.set_xlabel('arc length (cm)'); ax.set_ylabel('$dE/dx$ (MeV/cm)'); ax.set_title('(g) energy loss'); ax.legend(frameon=False, ncol=3, loc='lower center', columnspacing=0.5, handlelength=1.1, borderaxespad=0.3, fontsize=6)
stats = dict(timing_resid_fit_mean=float(np.nanmean(rf[k])), timing_resid_fit_sd=float(np.nanstd(rf[k])),
             timing_resid_start_mean=float(np.nanmean(rs[k])), timing_resid_start_sd=float(np.nanstd(rs[k])),
             model_t0sd_median=float(np.nanmedian(sdf[k])), n_pix_timing=int(k.sum()),
             dedx_mae_fit=float(np.mean(np.abs(ded['fit'] - ded['true']))), dedx_mae_init=float(np.mean(np.abs(ded['start'] - ded['true']))),
             dedx_rel_rms_fit=float(np.sqrt(np.mean((ded['fit'] / ded['true'] - 1) ** 2))), dedx_rel_rms_init=float(np.sqrt(np.mean((ded['start'] / ded['true'] - 1) ** 2))),
             e2_true_maxabs_um=float(np.abs(dev['true'][:, 1]).max()), e2_fit_rms_um=float(np.sqrt(np.mean((dev['fit'][:, 1] - dev['true'][:, 1]) ** 2))),
             e1_fit_rms_um=float(np.sqrt(np.mean((dev['fit'][:, 0] - dev['true'][:, 0]) ** 2))))
print('timing residuals (ticks):', {kk: round(v, 2) for kk, v in stats.items()})
js = a.out.rsplit('.', 1)[0] + '_numbers.json'; d0 = json.load(open(js)); d0.update(stats); json.dump(d0, open(js, 'w'), indent=1)
fig.savefig(a.out, dpi=220, bbox_inches='tight'); print('wrote', a.out)
