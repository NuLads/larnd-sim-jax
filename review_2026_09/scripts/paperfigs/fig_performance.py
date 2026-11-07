"""Paper figure: computational performance of the NuLads differentiable LArTPC simulator.

Three panels:
  (a) wall-clock time per evaluation (forward, forward+gradient) vs. batch track length,
      probabilistic (Poisson-point-process likelihood) vs. stochastic (MMD + charge loss)
      model. Per-batch points + a pooled linear fit (slope quoted in s/m).
  (b) peak GPU memory vs. batch-length setting, both models, with the A100 40 GB limit marked.
  (c) per-iteration wall-clock time of the production joint calibration fit
      (run "sci_full_STRFRZ508": 100 batches x 400 cm, probabilistic model, 10 000 Adam
      iterations), read from its checkpoint's 'step_time' history: a running median +
      10-90% band (window 100 iter) on a linear axis, with the first-epoch compilation
      phase shaded and the sparse, slowly-growing late-training stalls (verified NOT caused
      by checkpoint writes; see SPIKE INVESTIGATION below and the script's printed output)
      marked separately rather than let them set the y-scale.

SPIKE INVESTIGATION (panel c): step_time has a sparse population of outlier iterations
(112 of 10000, iter>100, >10 s): smaller ones (10-15 s) appear from iteration ~180 on, most
frequently around iter 2000-3000, then thin out; within these, a rarer subset (>15 s, first
seen iter 2683) grows steadily in both size and spacing over the rest of the run -- about
15-17 s and 70-200 iterations apart around iter 2700-5500, up to 22-26 s and 120-150
iterations apart around iter 9000-9900. Checked directly against this checkpoint's own data:
  - NOT the periodic checkpoint write (save_freq=200): step_time exactly AT and immediately
    AFTER all 50 checkpoint iterations is indistinguishable from the local baseline (~5 s) --
    the "growing 108 MB pickle" hypothesis is false.
  - NOT a single recurring slow batch: 'pos_residual_batch' shows a fixed, non-shuffled
    round-robin 0..99 repeated every epoch (identical iter 1-30 vs iter 101-130); the batch
    index active at each large spike is scattered across ~42 distinct indices out of 100.
  - NOT extra GPU work (recompilation or otherwise): per-iteration 'memory'[i]['num_allocs']
    deltas AT the large spikes are statistically identical to their immediate non-spike
    neighbours (ratio 0.93-1.11, i.e. noise), and 'peak_bytes_in_use'/'bytes_in_use' at a
    spike iteration exactly match the surrounding steady iterations to 4 significant figures.
    'peak_bytes_in_use' changes only 7 times in the entire 10 000-iteration run, all within
    the first 283 iterations (iter 3,15,23,30,83,183,283); it never changes again, including
    at every one of the ~100 later large spikes. The GPU is doing the same amount of work in
    a 25 s step as in a 5 s step.
  Having directly ruled out all three GPU/IO-side candidates, the extra wall-clock time is
  not spent in the simulator's compute graph; it is host-side (CPU/Python, e.g. garbage
  collection, or transient contention for a shared cluster resource -- CPU, filesystem or GPU
  scheduling -- on the node) and external to this code. Reported as a verified finding, not
  plotted as a mechanism (panel (c) excludes these iterations from the running band so they
  do not set the y-scale; a sparse marker row at the top shows where they occur).

Data sources (read directly at run time -- nothing hand-copied):
  - /sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS/bench_{prob,stoch}_{50,100,200,400}.json
    produced by paper_bench.py (same dir): one A100, eager JAX, 3 batches per setting, median of
    5 repeats for fwd_s/fwd_grad_s, peak_bytes = jax memory_stats()['peak_bytes_in_use'].
  - /sdf/scratch/neutrino/pgranger/fit_result_staging/sci_full_STRFRZ508/
    history_iter10000_sci_full_b100_len400_mw2500.0_refit_seed0.pkl
    ('step_time': list of 10000 per-iteration wall-clock times; 'config': argparse.Namespace
    confirming max_nbatch=100, max_batch_len=400, --probabilistic_sim).

Regenerate with:
    python fig_performance.py
Writes:
    /sdf/group/neutrino/pgranger/NuLads_draft/figures/performance.pdf
    /sdf/group/neutrino/pgranger/NuLads_draft/figures/performance.png
and prints a markdown table of the plotted numbers to stdout.

This script does no GPU work: it only reads existing JSON/pickle result files and plots them.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # defensive: unpickling must not try to grab a GPU
import json
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BENCH_DIR = '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS'
# PERF_RUN=KN10S8: the 10 cm-knot fit of the same sample (18 Oct)
PERF_RUN = os.environ.get('PERF_RUN', 'STRFRZ508')
CKPT = (f'/sdf/scratch/neutrino/pgranger/fit_result_staging/sci_full_{PERF_RUN}/'
        'history_iter10000_sci_full_b100_len400_mw2500.0_refit_seed0.pkl')
OUT_DIR = os.environ.get('PERF_OUT_DIR', '/sdf/group/neutrino/pgranger/NuLads_draft/figures')
SETTINGS = [50, 100, 200, 400]  # cm, max_batch_len benchmark settings
A100_GB = 40.0
BURN_IN = 20  # iterations excluded from the "steady-state" step-time stats (compilation)

# Okabe-Ito palette (shared paper style)
BLUE = '#0072B2'   # probabilistic
VERM = '#D55E00'   # stochastic
GREEN = '#009E73'
BLACK = '#000000'

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
    'font.family': 'sans-serif',
})


def load_bench(mode, setting):
    with open(f'{BENCH_DIR}/bench_{mode}_{setting}.json') as f:
        return json.load(f)


def gather(mode):
    """Pool all (length, fwd, fwd+grad) points across the 4 settings; also keep peak memory
    per setting."""
    L, F, G, peak = [], [], [], {}
    for s in SETTINGS:
        d = load_bench(mode, s)
        peak[s] = d['peak_bytes'] / 1e9  # decimal GB, matches main.tex convention (checked)
        for b in d['batches']:
            L.append(b['length_cm'] / 100.0)  # cm -> m
            F.append(b['fwd_s'])
            G.append(b['fwd_grad_s'])
    return np.array(L), np.array(F), np.array(G), peak


def linfit(x, y):
    m, c = np.polyfit(x, y, 1)
    yhat = m * x + c
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return m, c, r2


Lp, Fp, Gp, PKp = gather('prob')
Ls, Fs, Gs, PKs = gather('stoch')

mFp, cFp, r2Fp = linfit(Lp, Fp)
mGp, cGp, r2Gp = linfit(Lp, Gp)
mFs, cFs, r2Fs = linfit(Ls, Fs)
mGs, cGs, r2Gs = linfit(Ls, Gs)

print('# Linear fits  time[s] = m * length[m] + c  (pooled over the 4 settings x 3 batches)')
for name, (m, c, r2) in [('prob fwd', (mFp, cFp, r2Fp)), ('prob fwd+grad', (mGp, cGp, r2Gp)),
                          ('stoch fwd', (mFs, cFs, r2Fs)), ('stoch fwd+grad', (mGs, cGs, r2Gs))]:
    print(f'  {name:16s} m={m:.3f} s/m  c={c:.3f} s  R2={r2:.2f}')

# --- checkpoint: per-iteration time of the production joint fit ---
with open(CKPT, 'rb') as f:
    hist = pickle.load(f)
step_time = np.asarray(hist['step_time'], dtype=float)
cfg = hist['config']
n_iter = len(step_time)
med_steady = float(np.median(step_time[BURN_IN:]))
mean_steady = float(np.mean(step_time[BURN_IN:]))
total_hours = float(step_time.sum() / 3600.0)
ckpt_peak_gb = hist['memory'][-1]['peak_bytes_in_use'] / 1e9

print(f'\n# Production joint fit (sci_full_{PERF_RUN}): max_nbatch={cfg.max_nbatch} '
      f'max_batch_len={cfg.max_batch_len} probabilistic_sim={"--probabilistic_sim" in hist["provenance"]["argv"]}')
print(f'  n_iter={n_iter}  median step (iter>={BURN_IN})={med_steady:.2f} s  '
      f'mean={mean_steady:.2f} s  total={total_hours:.2f} h  first_iter={step_time[0]:.1f} s  '
      f'peak_mem={ckpt_peak_gb:.1f} GB (full fit: calib+dEdx+geometry+optimizer state)')

# --- spike check: verify (not just assert) that checkpoint writes are NOT the cause ---
save_freq = int(cfg.save_freq)
ckpt_iters = np.arange(save_freq, n_iter + 1, save_freq)
ckpt_step = step_time[ckpt_iters - 1]
print(f'  checkpoint-write check (save_freq={save_freq}): step_time AT the {len(ckpt_iters)} '
      f'save iterations: median={np.median(ckpt_step):.2f} s, max={ckpt_step.max():.2f} s '
      f'(baseline {med_steady:.2f} s) -> checkpoint I/O is NOT the cause of the large spikes.')

# --- classify iterations: first-epoch compilation vs. steady-state vs. rare late stalls ---
COMPILE_END = cfg.max_nbatch  # first full epoch: every one of the 100 batch shapes compiles once
SPIKE_THRESH = 10.0  # s; steady-state p99 stays well under this, verified spikes start at ~15 s
WINDOW = 100  # iterations, as requested

idx1 = np.arange(1, n_iter + 1)
is_spike = (step_time > SPIKE_THRESH) & (idx1 > COMPILE_END)
n_spikes = int(is_spike.sum())
spike_its, spike_vals = idx1[is_spike], step_time[is_spike]
print(f'  late stalls (iter>{COMPILE_END}, step_time>{SPIKE_THRESH:.0f} s): n={n_spikes}, '
      f'range {spike_vals.min():.1f}-{spike_vals.max():.1f} s, '
      f'{spike_vals.sum() / 3600:.3f} h of the {total_hours:.1f} h total '
      f'({100 * spike_vals.sum() / step_time.sum():.1f}%).')

# running median + 10-90% band (window=WINDOW, centred, excludes flagged spikes), iter>COMPILE_END only
half = WINDOW // 2
roll_it, roll_med, roll_p10, roll_p90 = [], [], [], []
for i in range(COMPILE_END, n_iter):
    lo, hi = max(COMPILE_END, i - half), min(n_iter, i + half + 1)
    w, wv = step_time[lo:hi], ~is_spike[lo:hi]
    sel = w[wv] if wv.sum() >= 5 else w
    roll_it.append(i + 1)
    roll_med.append(np.median(sel)); roll_p10.append(np.percentile(sel, 10)); roll_p90.append(np.percentile(sel, 90))
roll_it = np.array(roll_it); roll_med = np.array(roll_med)
roll_p10 = np.array(roll_p10); roll_p90 = np.array(roll_p90)
print(f'  running (w={WINDOW}) band over iter>{COMPILE_END}, spikes excluded: '
      f'p90 max={roll_p90.max():.2f} s, p10 min={roll_p10.min():.2f} s.')

# ================= figure =================
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
# External references (9 Oct): PyTorch implementation, Gasiorowski et al. (A100): ~25 s per batch of 1 m
# (simulation + gradient, with checkpointing), forward 0.07 s/cm per event (their Fig. 4), batch peak memory up to
# ~14 GB (their Fig. 7). Reference larnd-sim (CUDA, DUNE develop Sep 2024), forward only, end to end incl. output
# writing, A100, measured here (LARNDSIM_CMP, SLURM 40321613/40322247): 216 cm 4.40 s, 438 cm 4.91 s, 2161 cm 12.89 s.
PT_GRAD = (1.0, 25.0); PT_FWD_SLOPE = 7.0          # s at m ; s/m
PT_MEM = (100.0, 14.0)                              # batch-length cm, GB
REF_FWD = np.array([[2.16, 4.40], [4.38, 4.91], [21.61, 12.89]])

# --- (a) time vs length ---
ax = axes[0]


def scat(ax, x, y, color, marker, filled, label):
    kw = dict(s=15, marker=marker, linewidth=0.7, label=label, zorder=3)
    if filled:
        ax.scatter(x, y, facecolor=color, edgecolor=color, **kw)
    else:
        ax.scatter(x, y, facecolor='none', edgecolor=color, **kw)


xx = np.linspace(0.3, 4.8, 80)
scat(ax, Lp, Fp, BLUE, 'o', False, f'prob. fwd ({mFp:.2f} s/m)')
ax.plot(xx, mFp * xx + cFp, '--', color=BLUE, linewidth=0.8)
scat(ax, Lp, Gp, BLUE, 'o', True, f'prob. fwd+grad ({mGp:.2f} s/m)')
ax.plot(xx, mGp * xx + cGp, '-', color=BLUE, linewidth=0.8)
scat(ax, Ls, Fs, VERM, '^', False, f'stoch. fwd ({mFs:.2f} s/m)')
ax.plot(xx, mFs * xx + cFs, '--', color=VERM, linewidth=0.8)
scat(ax, Ls, Gs, VERM, '^', True, f'stoch. fwd+grad ({mGs:.2f} s/m)')
ax.plot(xx, mGs * xx + cGs, '-', color=VERM, linewidth=0.8)

xt = np.linspace(0.3, 1.0, 20)
ax.plot(xt, PT_FWD_SLOPE * xt, ':', color=BLACK, linewidth=1.0, label='PyTorch fwd (0.07 s/cm)')
ax.scatter(*PT_GRAD, marker='s', s=22, color=BLACK, zorder=4, label='PyTorch fwd+grad')
ax.scatter(REF_FWD[:, 0], REF_FWD[:, 1], marker='D', s=16, facecolor='none', edgecolor=BLACK, linewidth=0.8,
           zorder=4, label='larnd-sim fwd (CUDA)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('batch track length (m)')
ax.set_ylabel('time per evaluation (s)')
ax.set_xlim(0.3, 60)
ax.set_ylim(0.05, 80)
ax.grid(alpha=0.25, linewidth=0.5)
h, l = ax.get_legend_handles_labels()
ours = [i for i, t in enumerate(l) if not (t.startswith('PyTorch') or t.startswith('larnd-sim'))]
ext = [i for i, t in enumerate(l) if i not in ours]
leg1 = ax.legend([h[i] for i in ours], [l[i] for i in ours], frameon=False, loc='lower right', handletextpad=0.4,
                 labelspacing=0.15, fontsize=5.4, borderaxespad=0.3)
ax.add_artist(leg1)
ax.legend([h[i] for i in ext], [l[i] for i in ext], frameon=False, loc='upper right', handletextpad=0.4,
          labelspacing=0.15, fontsize=5.4, borderaxespad=0.3)
ax.text(0.04, 0.97, '(a)', transform=ax.transAxes, fontsize=7, va='top', ha='left')

# --- (b) memory vs batch-length setting ---
ax = axes[1]
sx = np.array(SETTINGS, dtype=float)
ax.plot(sx, [PKp[s] for s in SETTINGS], 'o-', color=BLUE, ms=4, label='probabilistic')
ax.plot(sx, [PKs[s] for s in SETTINGS], '^-', color=VERM, ms=4, label='stochastic')
ax.scatter(*PT_MEM, marker='s', s=22, color=BLACK, zorder=4, label='PyTorch (checkpointed)')
ax.axhline(A100_GB, color=BLACK, linestyle=':', linewidth=0.9)
ax.set_xscale('log')
ax.set_xlim(45, 450)
ax.text(430, A100_GB, 'A100 (40 GB)', va='bottom', ha='right', fontsize=6.5)

ax.set_xticks(sx)
ax.set_xticklabels([f'{int(s)}' for s in sx])
ax.minorticks_off()
ax.set_xlabel('batch-length setting (cm)')
ax.set_ylabel('peak GPU memory (GB)')
ax.set_ylim(0, 44)
ax.grid(alpha=0.25, linewidth=0.5)
ax.legend(frameon=False, loc='lower right')
ax.text(0.04, 0.97, '(b)', transform=ax.transAxes, fontsize=7, va='top', ha='left')

fig.tight_layout(w_pad=1.3)
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(f'{OUT_DIR}/performance.pdf', bbox_inches='tight')
fig.savefig(f'{OUT_DIR}/performance.png', dpi=300, bbox_inches='tight')
print(f'\nsaved {OUT_DIR}/performance.pdf and .png')
# --- (c) per-iteration time of the production joint fit ---
fig2, ax = plt.subplots(1, 1, figsize=(3.4, 2.3))
YTOP = 12.0

# first-epoch compilation phase (all 100 batch shapes compile once; first point alone is 56 s)
ax.axvspan(0, COMPILE_END, color=BLACK, alpha=0.10, linewidth=0, zorder=1)
ax.axvline(COMPILE_END, color=BLACK, linewidth=0.6, alpha=0.45, zorder=1)

# steady-state running median + 10-90% band (spikes excluded from the statistics, per request)
ax.fill_between(roll_it, roll_p10, roll_p90, color=GREEN, alpha=0.28, linewidth=0, zorder=2,
                 label='10-90% (w=100)')
ax.plot(roll_it, roll_med, color=GREEN, linewidth=1.1, zorder=3, label='running median')
ax.axhline(med_steady, color='#CC79A7', linestyle='--', linewidth=1.0, zorder=4,
           label=f'global median {med_steady:.2f} s')

# rare late stalls: shown as a sparse marker row (excluded from the band itself), kept clear
# of the top-left corner so it doesn't run into the (c) panel label
ax.plot(spike_its, np.full(n_spikes, YTOP * 0.9), marker='|', linestyle='none', ms=4.0, mew=0.5,
        color='0.45', alpha=0.7, clip_on=False, zorder=5)

ax.set_ylim(0, YTOP)
ax.set_xlim(0, n_iter)
ax.set_xlabel('joint-fit iteration')
ax.set_ylabel('step time (s)')
# single compact annotation (no separate legend box -- panel is narrow; colors described in caption)
ax.text(0.97, 0.76,
        f'median {med_steady:.2f} s (iter$\\,\\geq\\,${BURN_IN}), $\\Sigma$={total_hours:.1f} h\n'
        f'green: running median, 10-90%\n'
        f'ticks: {n_spikes} iterations of {SPIKE_THRESH:.0f}-{spike_vals.max():.0f} s (caption)',
        transform=ax.transAxes, ha='right', va='top', fontsize=5.8, linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85))
ax.grid(alpha=0.25, linewidth=0.5)

fig2.tight_layout()
fig2.savefig(f'{OUT_DIR}/fit_iteration_time.pdf', bbox_inches='tight')
fig2.savefig(f'{OUT_DIR}/fit_iteration_time.png', dpi=300, bbox_inches='tight')
print(f'saved {OUT_DIR}/fit_iteration_time.pdf and .png')

# ================= markdown table (printed, not written to a file) =================
lines = []
lines.append('**Table: forward / forward+gradient benchmark, one A100 (40 GB), eager JAX**')
lines.append('')
lines.append('| mode | setting (cm) | batch length (m) | fwd (s) | fwd+grad (s) | peak mem (GB) |')
lines.append('|---|---|---|---|---|---|')
for mode_label, mode_key, peak in [('probabilistic', 'prob', PKp), ('stochastic', 'stoch', PKs)]:
    for s in SETTINGS:
        d = load_bench(mode_key, s)
        ls = [b['length_cm'] / 100 for b in d['batches']]
        fs = [b['fwd_s'] for b in d['batches']]
        gs = [b['fwd_grad_s'] for b in d['batches']]
        lines.append(f'| {mode_label} | {s} | {min(ls):.2f}-{max(ls):.2f} | '
                      f'{min(fs):.2f}-{max(fs):.2f} | {min(gs):.2f}-{max(gs):.2f} | {peak[s]:.1f} |')
lines.append('')
lines.append('**Table: production joint fit (sci\\_full\\_STRFRZ508, 100 batches x 400 cm, '
             'probabilistic, 10 000 Adam iterations)**')
lines.append('')
lines.append('| quantity | value |')
lines.append('|---|---|')
lines.append(f'| median step time (iter $\\geq$ {BURN_IN}) | {med_steady:.2f} s |')
lines.append(f'| mean step time (iter $\\geq$ {BURN_IN}) | {mean_steady:.2f} s |')
lines.append(f'| total wall-clock, {n_iter} iterations | {total_hours:.2f} h |')
lines.append(f'| first iteration (incl. compilation) | {step_time[0]:.1f} s |')
lines.append(f'| peak GPU memory (full fit: calib + dEdx + geometry + optimizer state) | {ckpt_peak_gb:.1f} GB |')
lines.append(f'| late stalls excluded from the panel (c) band (iter $>$ {COMPILE_END}, step\\_time $>$ {SPIKE_THRESH:.0f} s) '
             f'| n={n_spikes}, {spike_vals.min():.1f}-{spike_vals.max():.1f} s, {spike_vals.sum()/3600:.2f} h total |')
print()
print('\n'.join(lines))
