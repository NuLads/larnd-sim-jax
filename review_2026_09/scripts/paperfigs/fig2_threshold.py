"""
Figure 2 (fig:tick_distrib_vs_threshold) for the NuLads paper, electronics section.

Regenerates a single, two-panel figure replacing the old
figures/tick_distrib_vs_threshold.png + figures/trigger_prob_vs_threshold.png:

  (a) Distribution over independent noise throws of the *first-hit* trigger
      tick, for one representative pixel, at each of the 8 discrimination
      thresholds swept (3-10 ke-). Small triangles mark the noise-free
      (deterministic) crossing tick for the same pixel and threshold.
  (b) Probability that the 1st/2nd/3rd/4th hit of a pixel exists at all
      (i.e. the tick array does not hit the "no trigger" sentinel),
      averaged over every active pixel of the waveform set, vs threshold.

Data: stochastic front-end mode, 10 000 independent noise throws per
threshold, from
  /sdf/group/neutrino/pgranger/larnd-sim-jax/adc_output_thresh{3000..10000}.h5
  /sdf/group/neutrino/pgranger/larnd-sim-jax/adc_output_no_noise_thresh{..}.h5
  /sdf/group/neutrino/pgranger/larnd-sim-jax-bak/old_wfs.h5   (pixel map, for
      picking + describing the representative pixel of panel a)
This reproduces, with real data, the computation of noise_new.ipynb cells 7-9
(the notebook that made the two original images), but:
  * panel (a) shows the FIRST hit (hit index 0) rather than the second
    (hit index 1): hit 0 triggers at every threshold tested (9963-10000/10000
    throws), while hit 1 has zero surviving throws for th >= 8000 e- and
    cannot be histogrammed there. Hit 0 also cleanly shows the
    later+sharper-with-threshold trend the text motivates.
  * panel (b) averages only over the 2014/2064 *active* pixels of the
    waveform set (unique_pixels >= 0); the original notebook cell averaged
    over all 2064 slots including ~50 padding slots, which dilutes the
    curves by about 2% (padding slots are not real detector pixels).

No jax import is needed: everything here is read directly from the MC
output files with h5py/numpy. Run with: python fig2_threshold.py
"""
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ----------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------
DATADIR = "/sdf/group/neutrino/pgranger/larnd-sim-jax"
WFS_FILE = "/sdf/group/neutrino/pgranger/larnd-sim-jax-bak/old_wfs.h5"
OUTDIR = "/sdf/group/neutrino/pgranger/NuLads_draft/figures"
OUT_BASENAME = "threshold_dependence"

THLIST = list(range(3000, 10001, 1000))  # e-, the 8 swept thresholds
PIX = 50          # representative pixel index (array position, not pixel ID)
SENTINEL = 1998   # results_ticks >= this value means "hit did not occur"
N_THROWS = 10000  # noise throws per threshold (verified against file shape)
DT = 0.1          # tick length in microseconds (t_sampling = CLOCK_CYCLE, Table 1)
N_HIT_SHOW = 4    # show hit indices 0..3 in panel (b), as in the original figure

# Okabe-Ito colorblind-safe categorical palette (shared style guide)
C_BLUE, C_VERM, C_GREEN, C_ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
HIT_COLORS = [C_BLUE, C_VERM, C_GREEN, C_ORANGE]
HIT_LABELS = ["first hit", "second hit", "third hit", "fourth hit"]

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "font.family": "sans-serif",
})

CMAP = plt.get_cmap("viridis")
NORM = Normalize(vmin=THLIST[0] / 1000.0, vmax=THLIST[-1] / 1000.0)


def th_color(th):
    return CMAP(NORM(th / 1000.0))


# ----------------------------------------------------------------------
# Load the pixel map used to make adc_output_thresh*.h5 (for panel a
# bookkeeping only: which pixel index 50 actually is, and its peak charge).
# ----------------------------------------------------------------------
with h5py.File(WFS_FILE, "r") as f:
    unique_pixels = f["unique_pixels"][:]
    wf_pix = f["wfs"][PIX, :]

valid_mask = unique_pixels >= 0
n_valid = int(valid_mask.sum())
n_total = len(unique_pixels)
pixel_id = int(unique_pixels[PIX])
q_cum_pix = np.cumsum(wf_pix[1:]) * DT
q_max_pix = float(q_cum_pix.max())

print(f"[info] representative pixel: array index {PIX}, pixel ID {pixel_id}, "
      f"peak cumulative charge {q_max_pix:.0f} e-")
print(f"[info] active pixels in waveform set: {n_valid}/{n_total}")

# ----------------------------------------------------------------------
# Pass 1: per-threshold histogram of the hit-0 tick for the representative
# pixel (panel a), plus the pixel-averaged existence probability of hits
# 0..3 over all ACTIVE pixels (panel b). One threshold file (826 MB) is
# read at a time and freed immediately.
# ----------------------------------------------------------------------
BIN_LO, BIN_HI = 615, 693  # ticks; covers the 0.5-99.9 percentile range of
                           # every threshold's hit-0 distribution (checked
                           # separately: min 0.5%ile 622 @3 ke-, max 99.9%ile
                           # 690 @10 ke-), plus margin. A handful of throws
                           # (<0.5% at the lowest threshold) with very early,
                           # noise-driven crossings fall outside this window.
tick_edges = np.arange(BIN_LO, BIN_HI + 1, 1)   # 1-tick-wide bins
tick_centers_us = 0.5 * (tick_edges[:-1] + tick_edges[1:]) * DT

hist_hit0 = {}         # th -> probability-per-tick array
prob_valid = {}        # th -> array of length >=4, P(hit n exists), active-pixel average
n_trig_hit0 = {}        # th -> number of throws (out of N_THROWS) where hit 0 triggered at all

for th in THLIST:
    fname = os.path.join(DATADIR, f"adc_output_thresh{th}.h5")
    with h5py.File(fname, "r") as f:
        ticks_pix0 = f["results_ticks"][:, PIX, 0]          # cheap hyperslab read
        ticks_all = f["results_ticks"][:, :, :N_HIT_SHOW]    # full read, freed below

    assert ticks_all.shape[0] == N_THROWS

    counts, _ = np.histogram(ticks_pix0, bins=tick_edges)
    hist_hit0[th] = counts / N_THROWS
    n_trig_hit0[th] = int(np.sum(ticks_pix0 < SENTINEL))

    triggered = ticks_all < SENTINEL
    frac_per_pixel = triggered.mean(axis=0)          # (n_pixels, N_HIT_SHOW)
    prob_valid[th] = frac_per_pixel[valid_mask].mean(axis=0)

    del ticks_all, triggered
    print(f"[info] threshold {th} e-: hit0 triggered {n_trig_hit0[th]}/{N_THROWS} throws "
          f"for pixel {PIX}; <lambda_n> over active pixels = "
          + ", ".join(f"{p:.3f}" for p in prob_valid[th]))

# ----------------------------------------------------------------------
# Pass 2: deterministic (noise-free) hit-0 crossing tick for the same
# pixel, at each threshold (tiny reads: a single scalar per file).
# ----------------------------------------------------------------------
det_tick0 = {}
for th in THLIST:
    fname = os.path.join(DATADIR, f"adc_output_no_noise_thresh{th}.h5")
    with h5py.File(fname, "r") as f:
        t0 = f["results_ticks"][0, PIX, 0]
    det_tick0[th] = float(t0)

print("[info] deterministic hit-0 tick vs threshold:", det_tick0)

# A couple of summary numbers for the caption / sanity.
mean0 = {th: np.sum(tick_centers_us * hist_hit0[th]) / max(hist_hit0[th].sum(), 1e-12)
         for th in THLIST}
# proper weighted std in ticks
std0_ticks = {}
for th in THLIST:
    centers_tick = 0.5 * (tick_edges[:-1] + tick_edges[1:])
    w = hist_hit0[th]
    m = np.sum(centers_tick * w) / max(w.sum(), 1e-12)
    v = np.sum(w * (centers_tick - m) ** 2) / max(w.sum(), 1e-12)
    std0_ticks[th] = np.sqrt(v)

print("[info] hit-0 tick mean (ticks):", {k: round(v / DT, 2) for k, v in mean0.items()})
print("[info] hit-0 tick std  (ticks):", {k: round(v, 2) for k, v in std0_ticks.items()})

# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.0, 2.85))

# --- panel (a): hit-0 tick distribution vs threshold -------------------
ymax = max(h.max() for h in hist_hit0.values())
ax0.set_ylim(-0.055 * ymax, 1.12 * ymax)

for th in THLIST:
    c = th_color(th)
    ax0.stairs(hist_hit0[th], tick_edges * DT, color=c, linewidth=1.2, baseline=None)
    ax0.plot(det_tick0[th] * DT, -0.028 * ymax, marker="^", color=c,
              markersize=3.2, clip_on=False, zorder=5)

ax0.set_xlim(BIN_LO * DT, BIN_HI * DT)
ax0.axhline(0, color="0.3", linewidth=0.5, zorder=0)
ax0.set_xlabel(r"First-hit trigger time $t_0$ [$\mu$s]")
ax0.set_ylabel("Probability per tick")
ax0.grid(alpha=0.2, linewidth=0.5)
ax0.text(0.03, 0.95, "(a)", transform=ax0.transAxes, ha="left", va="top")

secax = ax0.secondary_xaxis("top", functions=(lambda us: us / DT, lambda tck: tck * DT))
secax.set_xlabel("Tick index $t_i$", labelpad=4)
secax.tick_params(labelsize=6.5)

sm = ScalarMappable(norm=NORM, cmap=CMAP)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax0, pad=0.025, fraction=0.05, aspect=22)
cbar.set_label(r"Threshold $V_{th}$ [$10^3\,e^-$]", fontsize=8)
cbar.set_ticks([t / 1000 for t in THLIST])
cbar.ax.tick_params(labelsize=6.5)

# small annotation of which pixel this is (the left part of the panel is
# empty of curves; the triangle marker itself is explained in the caption)
ax0.text(0.03, 0.80, f"pixel {pixel_id}", transform=ax0.transAxes,
          ha="left", va="top", fontsize=6.5, color="0.25")

# --- panel (b): per-hit existence probability vs threshold -------------
th_arr_ke = np.array(THLIST) / 1000.0
for n in range(N_HIT_SHOW):
    y = [prob_valid[th][n] for th in THLIST]
    ax1.plot(th_arr_ke, y, marker="o", markersize=3.2, color=HIT_COLORS[n],
              label=HIT_LABELS[n])

ax1.set_xlabel(r"Discrimination threshold $V_{th}$ [$10^3\,e^-$]")
ax1.set_ylabel("Average hit probability")
ax1.set_ylim(-0.03, 1.16)
ax1.set_xlim(th_arr_ke[0] - 0.3, th_arr_ke[-1] + 0.3)
ax1.grid(alpha=0.2, linewidth=0.5)
ax1.legend(frameon=False, loc="center right", handlelength=1.6, borderaxespad=0.3,
            bbox_to_anchor=(1.0, 0.54))
ax1.text(0.03, 0.95, "(b)", transform=ax1.transAxes, ha="left", va="top")

fig.tight_layout(w_pad=1.4)

os.makedirs(OUTDIR, exist_ok=True)
pdf_path = os.path.join(OUTDIR, OUT_BASENAME + ".pdf")
png_path = os.path.join(OUTDIR, OUT_BASENAME + ".png")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"[info] wrote {pdf_path}")
print(f"[info] wrote {png_path}")
