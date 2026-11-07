"""Paper figure: larnd-sim-jax against the reference larnd-sim (CUDA, DUNE develop Sep 2024) on the same 66 segments
(216 cm of through-going muon track, 3 events), same response table and detector parameters (9 Oct 2026, SLURM 40319804
for JAX, 40321613 for the reference with the overrides re-applied after importlib.reload(fee)).
Pixels are matched by physical position (JAX -> reference frame: x - pitch/2, y - 0.318697 cm), hit pixels only.
(a) time-integrated induced charge per pixel (stage 2: diffusion on, noise off); (b) hits per pixel (stage 3, noise on);
(c) ADC per hit (stage 3); (d) first-hit time difference per pixel (stage 2), relative to its median (time-origin convention).
Inputs/outputs: /sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP/."""
import sys, json, numpy as np
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP'
sys.argv = sys.argv[:1]; sys.path.insert(0, D)
import analyze_comparison as ac
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = '/sdf/group/neutrino/pgranger/NuLads_draft/figures/larndsim_comparison'
DX, DY = -0.2217, -0.318697
C_REF, C_JAX = '#0072B2', '#D55E00'
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.8, 'xtick.labelsize': 7, 'ytick.labelsize': 7})

def load(stage):
    ref = ac.load_reference_flat(f'{D}/out_stage{stage}_ref.h5'); jx = ac.load_jax_direct(f'{D}/out_stage{stage}_jax_direct.h5')
    rx, ry = ac.ref_pixel_physical_coords(ref['pixels'])[:2]
    up = np.asarray(jx['unique_pixels']); keep = up >= 0
    jxx = np.asarray(jx['pix_x_decoded'])[keep] + DX; jyy = np.asarray(jx['pix_y_decoded'])[keep] + DY
    # map each reference row to the JAX row at the same position (-1 if none)
    jrow = np.full(len(rx), -1)
    idx = np.where(keep)[0]
    for i in range(len(rx)):
        m = np.where((np.abs(jxx - rx[i]) < 0.01) & (np.abs(jyy - ry[i]) < 0.01))[0]
        if len(m): jrow[i] = idx[m[0]]
    return ref, jx, jrow

def per_pixel_hits(ref, jx, jrow):
    valid = ref['ticks'] != 0
    hp = np.asarray(jx['hit_pixels']); up = np.asarray(jx['unique_pixels'])
    J = {}
    for p, a, t in zip(hp, np.asarray(jx['adc']), np.asarray(jx['ticks'])):
        if p < 0: continue
        d = J.setdefault(int(p), [0, 1e9]); d[0] += 1; d[1] = min(d[1], t)
    nref = valid.sum(1); njax = np.array([J.get(int(up[j]), [0])[0] if j >= 0 else 0 for j in jrow])
    tref = np.where(nref > 0, np.where(valid, ref['ticks'], np.inf).min(1), np.nan)
    tjax = np.array([J.get(int(up[j]), [0, np.nan])[1] * 0.1 if j >= 0 and int(up[j]) in J else np.nan for j in jrow])
    return nref, njax, tref, tjax, J

nums = {}
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.15)); fig.subplots_adjust(wspace=0.55)
# (a) integrated induced charge per pixel, stage 2
ref, jx, jrow = load(2)
dt = 0.1
qref = ref['pixel_signals'].sum(1) * dt; wfs = np.asarray(jx['wfs']); qjax = np.array([wfs[j].sum() * dt if j >= 0 else np.nan for j in jrow])
nref, njax, *_ = per_pixel_hits(ref, jx, jrow)
sel = (jrow >= 0) & (nref > 0) & (njax > 0)
ratio = qjax[sel] / qref[sel]
nums['a_charge_ratio_median'] = float(np.median(ratio)); nums['a_charge_ratio_iqr'] = [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))]
nums['a_n'] = int(sel.sum())
ax = axs[0]; ax.scatter(qref[sel] / 1e3, qjax[sel] / 1e3, s=3, color='k', alpha=0.5, lw=0)
lim = [0, 1.05 * max(qref[sel].max(), qjax[sel].max()) / 1e3]; ax.plot(lim, lim, color=C_JAX, lw=0.8, ls='--')
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel(r'reference (10$^3$ e$^-$)'); ax.set_ylabel(r'larnd-sim-jax (10$^3$ e$^-$)')
ax.set_title('(a) charge per pixel', fontsize=8)
# (b) hits per pixel, stage 3 ; (c) ADC per hit ; (d) first-hit time, stage 2
ref3, jx3, jrow3 = load(3)
n3r, n3j, *_ = per_pixel_hits(ref3, jx3, jrow3)
ax = axs[1]; bins = np.arange(0.5, 6.5)
ax.hist(n3r[n3r > 0], bins=bins, histtype='step', color=C_REF, lw=1.2, label=f'reference ({int(n3r.sum())} hits)')
hpj = np.asarray(jx3['hit_pixels']); cj = np.bincount(np.unique(hpj[hpj >= 0], return_inverse=True)[1])
ax.hist(cj, bins=bins, histtype='step', color=C_JAX, lw=1.2, ls='--', label=f'larnd-sim-jax ({int(cj.sum())})')
ax.set_xlabel('hits per pixel'); ax.set_ylabel('pixels'); ax.set_ylim(top=ax.get_ylim()[1]*1.45); ax.legend(frameon=False, loc='upper right', fontsize=6, handlelength=1.6); ax.set_title('(b) hits, noise on', fontsize=8)
nums['b_hits'] = [int(n3r.sum()), int(cj.sum())]
ax = axs[2]
ra = ref3['adc'][ref3['ticks'] != 0].ravel(); ja = np.asarray(jx3['adc'])[hpj >= 0]
b2 = np.arange(np.floor(min(ra.min(), ja.min())) - 0.5, np.ceil(max(ra.max(), ja.max())) + 1.5, 2)
ax.hist(ra, bins=b2, histtype='step', color=C_REF, lw=1.2, density=True, label=f'mean {ra.mean():.1f}, sd {ra.std():.1f}')
ax.hist(ja, bins=b2, histtype='step', color=C_JAX, lw=1.2, ls='--', density=True, label=f'mean {ja.mean():.1f}, sd {ja.std():.1f}')
ax.set_xlabel('ADC counts per hit'); ax.set_ylabel('density'); ax.set_ylim(top=ax.get_ylim()[1]*1.45); ax.legend(frameon=False, loc='upper right', fontsize=6, handlelength=1.6); ax.set_title('(c) charge per hit, noise on', fontsize=8)
nums['c_adc'] = dict(ref_mean=float(ra.mean()), ref_sd=float(ra.std()), jax_mean=float(ja.mean()), jax_sd=float(ja.std()))
ax = axs[3]
n2r, n2j, t2r, t2j, _ = per_pixel_hits(ref, jx, jrow)
s2 = (n2r > 0) & (n2j > 0)
d = t2r[s2] - t2j[s2]; off = np.median(d); d0 = d - off
nums['d_offset_us'] = float(off); nums['d_iqr_us'] = [float(np.percentile(d0, 25)), float(np.percentile(d0, 75))]; nums['d_frac_within_0p2us'] = float(np.mean(np.abs(d0) <= 0.2))
ax.hist(d0, bins=np.arange(-1.5, 1.55, 0.1), color='0.4')
ax.set_xlabel(r'$\Delta t$, first hit ($\mu$s)'); ax.set_ylabel('pixels'); ax.set_title('(d) timing', fontsize=8)
for a, l in zip(axs, 'abcd'): pass
fig.savefig(OUT + '.pdf', bbox_inches='tight'); fig.savefig(OUT + '.png', dpi=250, bbox_inches='tight')
json.dump(nums, open(f'{D}/analysis_output/fig_larndsim_comparison_numbers.json', 'w'), indent=1)
print(json.dumps(nums, indent=1))
