"""Figure: what the joint fit reconstructs, over the nine track samples of recipe R2 (byproduct.py arrays).
(a) per-track mean distance between reconstructed and true trajectory: straight line between the endpoints, hit-derived
    starting trajectory (if byproduct.py ran with BYPRODUCT_HITSEED=1) and fitted;
(b) per-segment energy loss, relative deviation from the true deposit: fitted, starting value (true track average),
    constant; (c) track-averaged energy loss, relative deviation from the true track average: fitted vs constant."""
import json, os, numpy as np, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = os.environ.get('BYPRODUCT_OUT', '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/figs/byproduct')
out = sys.argv[1] if len(sys.argv) > 1 else f'{D}/byproduct.png'
z = np.load(f'{D}/byproduct_arrays.npz'); T = {k[2:]: z[k] for k in z.files if k.startswith('T_')}; S = {k[2:]: z[k] for k in z.files if k.startswith('S_')}
plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'legend.fontsize': 5.9})
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.35)); fig.subplots_adjust(wspace=0.40)
nums = {}
# (a) trajectory accuracy
ax = axs[0]; bins = np.logspace(0.5, 4.3, 46)
CURVES = (('d_str', 'straight line', '0.55', '--'),) + ((('d_hit', 'hit-derived start', '#009E73', '-.'),) if 'd_hit' in T else ()) \
         + (('d_fit', 'fitted', 'C3', '-'),)
for key, lab, c, ls in CURVES:
    v = T[key]; ax.hist(v, bins=bins, histtype='step', color=c, ls=ls, lw=1.1, label=f'{lab} ({np.median(v):.0f} $\\mu$m)')
    nums[key] = dict(median=float(np.median(v)), mean=float(np.mean(v)), p90=float(np.percentile(v, 90)))
ax.set_xscale('log'); ax.set_xlabel(r'distance to true trajectory ($\mu$m)'); ax.set_ylabel('tracks')
ax.set_title('(a) trajectory'); ax.legend(frameon=False, loc='upper left', title='median', title_fontsize=5.9)
ax.set_ylim(top=ax.get_ylim()[1] * (1.5 if 'd_hit' in T else 1.3))
# (b) per-segment energy loss
ax = axs[1]; w = S['dx']; bins = np.linspace(-20, 20, 81)
est = (('constant', np.full_like(S['true'], np.average(S['true'], weights=w)), 'k', ':'),
       ('track average (start)', S['init'], '0.55', '--'), ('fitted', S['fit'], 'C3', '-'))
for lab, e, c, ls in est:
    r = 100 * (e / S['true'] - 1); sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    ax.hist(r, bins=bins, weights=w / w.sum(), histtype='step', color=c, ls=ls, lw=1.1, label=f'{lab} ($\\sigma$ {sig:.1f}%)')
    nums['seg_' + lab.split()[0]] = dict(robust_sigma=float(sig), rms=float(np.sqrt(np.average(r ** 2, weights=w))), bias=float(np.average(r, weights=w)))
ax.set_xlabel(r'segment $dE/dx$ error (%)'); ax.set_ylabel('fraction of track length'); ax.set_title('(b) energy loss per segment')
ax.legend(frameon=False, loc='upper left'); ax.set_ylim(top=ax.get_ylim()[1] * 1.45)
# (c) track-averaged energy loss
ax = axs[2]; tr = S['trk']; nt = len(T['L'])
num = lambda v: np.bincount(tr, weights=v * w, minlength=nt); den = np.bincount(tr, weights=w, minlength=nt); ok = den > 0
tt, tf = num(S['true'])[ok] / den[ok], num(S['fit'])[ok] / den[ok]; C = np.average(S['true'], weights=w)
bins = np.linspace(-10, 10, 61)
for lab, e, c, ls in (('constant', np.full_like(tt, C), 'k', ':'), ('fitted', tf, 'C3', '-')):
    r = 100 * (e / tt - 1); sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    ax.hist(r, bins=bins, histtype='step', color=c, ls=ls, lw=1.1, label=f'{lab} ($\\sigma$ {sig:.1f}%, mean {np.mean(r):+.2f}%)')
    nums['trk_' + lab] = dict(robust_sigma=float(sig), rms=float(np.sqrt(np.mean(r ** 2))), bias=float(np.mean(r)))
ax.set_xlabel(r'track-averaged $dE/dx$ error (%)'); ax.set_ylabel('tracks'); ax.set_title('(c) energy loss per track')
ax.legend(frameon=False, loc='upper left'); ax.set_ylim(top=ax.get_ylim()[1] * 1.4)
nums.update(n_tracks=int(nt), n_segments=int(len(w)), samples=sorted(set(T['sample'].tolist())))
json.dump(nums, open(f'{D}/fig_byproduct_numbers.json', 'w'), indent=1); print(json.dumps(nums, indent=1))
fig.savefig(out, dpi=220, bbox_inches='tight'); print('wrote', out)
