"""Report figures 25 (per-track drift-shape error distributions by configuration) and 26 (freeze threshold trade-off)."""
import sys, glob, pickle, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import chi2
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
from optimize.fit_params import _dir_from_angles, _transverse_frame, _DRIFT_AXIS
R = '/sdf/group/neutrino/pgranger/larnd-sim-jax/'; OUT = R + 'review_2026_09/figs/'
ag = np.linspace(0, 1, 101)
def errs(tag):
    fs = sorted(glob.glob(f'{R}fit_result/{tag}/history_iter*.pkl'), key=lambda x: int(x.split('iter')[-1].split('_')[0]))
    fs = [x for x in fs if int(x.split('iter')[-1].split('_')[0]) <= 10000] or fs
    ck = pickle.load(open(fs[-1], 'rb')); E = []
    for b, ctxs in ck['chain_contexts'].items():
        for c, st in zip(ctxs, ck['chain_cache'][b]):
            u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0)
            if not (abs(e1[2]) < 1e-6 and np.linalg.norm(np.cross(_DRIFT_AXIS, u0)) > 1e-3): continue
            a = np.asarray(st['angles'], float); K = len(a) // 2
            tr = np.asarray(ck['true_positions'][b][str(tuple(c['track_id']))], float); al = np.asarray(c['alpha_mid'], float)[:len(tr)]; tr = tr[:len(al)]
            x0 = np.asarray(c['x0_fixed'], float); L = c['total_len']
            Bt = np.sin(np.outer(al, np.pi * np.arange(1, K + 1))); bt2 = np.linalg.lstsq(Bt, (tr - x0 - al[:, None] * L * u0) @ e2, rcond=None)[0]
            Bg = np.sin(np.outer(ag, np.pi * np.arange(1, K + 1))); E.append(np.sqrt(np.mean((Bg @ (a[K:] - bt2)) ** 2)) * 1e4)
    return np.array(E)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for arm, col, lab in (('DSREP', 'C3', 'production (straight start)'), ('GTRU', 'C1', 'true start, free'), ('HITSEED', 'C4', 'hit-seeded start, free'), ('STRFRZ50', 'C2', 'recipe')):
    E = np.concatenate([errs(f'sci_full_{arm}{s}') for s in (1, 2, 4)])
    xs = np.sort(E); ax[0].loglog(xs, 1 - np.arange(len(xs)) / len(xs), color=col, lw=1.8, label=f'{lab}: median {np.median(E):.0f}, mean {E.mean():.0f} µm')
    print(arm, len(E), np.median(E).round(1), E.mean().round(1), np.percentile(E, 95).round(0), flush=True)
ax[0].set_xlabel('per-track drift-shape error vs truth [µm]'); ax[0].set_ylabel('fraction of tracks with a larger error')
ax[0].set_title('drift-shape error distribution (samples 1/2/4)'); ax[0].legend(fontsize=7.5); ax[0].set_xlim(5, 2e4)
# freeze-threshold trade-off (sample 8 offline: production contexts + GEOFRZ8 targets)
import subprocess
out = subprocess.run([sys.executable, R + 'review_2026_09/scripts/drift_hits/straight_select.py', glob.glob(R + 'fit_result/sci_full_DSREP8/history_iter10000*.pkl')[0],
                      R + 'target_sci_fullGEOFRZ8_sci_full_b100_len400_mw2500.0_refit_seed0_Ab_eField_tran_diff_long_diff_lifetime'], capture_output=True, text=True).stdout
rows = [l for l in out.splitlines() if 'freeze if not significant' in l]
cl = [float(l.split('CL')[1].split(':')[0]) for l in rows]; fr = [float(l.split('frozen ')[1].split(' ')[0]) for l in rows]
mean = [float(l.split('mean')[1].split()[0]) for l in rows]; p95 = [float(l.split('p95')[1].split()[0]) for l in rows]
ax2 = ax[1]; ax2.plot(cl, mean, 'o-', color='C0', label='mean straight-line error of frozen tracks'); ax2.plot(cl, p95, 's--', color='C0', alpha=0.6, label='95th percentile')
ax2.set_xlabel('significance threshold for calling a track curved (CL)'); ax2.set_ylabel('error of the frozen straight line [µm]'); ax2.set_ylim(0, 650)
tw = ax2.twinx(); tw.plot(cl, np.array(fr) * 100, 'D-', color='C2', label='fraction of tracks frozen'); tw.set_ylabel('tracks frozen [%]', color='C2'); tw.set_ylim(0, 65)
for x_, lab in ((0.5, 'recipe'), (0.9, 'over-corrects')): ax2.axvline(x_, color='k', lw=0.6, ls=':'); ax2.text(x_ + 0.005, 610, lab, fontsize=8)
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = tw.get_legend_handles_labels(); ax2.legend(h1 + h2, l1 + l2, fontsize=7.5, loc='center left')
ax2.set_title('which tracks get a fixed drift coordinate (sample 8)')
fig.tight_layout(); fig.savefig(OUT + 'fig25_shape_tail_and_threshold.png', dpi=140)
print(rows)
