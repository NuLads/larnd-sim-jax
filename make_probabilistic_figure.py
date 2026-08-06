#!/usr/bin/env python3
"""Paper figure: does the analytic (probabilistic) FEE model reproduce the
distribution that the stochastic simulation produces by sampling?

This replaces figures/probabilistic_distributions.png, which showed the right
comparison but carried no axis labels and no legend. The content is the same
comparison performed in compare_stochastic_probabilistic.ipynb, extracted into a
script so the figure is reproducible.

For every pixel the probabilistic model returns P(t | pixel, first hit); the
stochastic model is run N times with independent noise throws and the empirical
distribution of its first-hit tick is histogrammed. Agreement of the two is the
claim the probabilistic treatment rests on.

Run on a GPU node:  python make_probabilistic_figure.py --runs 10000 --events 5
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import jax
import jax.numpy as jnp

from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from larndsim.fee_jax import get_adc_values
from larndsim.losses_jax import adc2charge
from optimize.strategies import LUTProbabilisticSimulation
from optimize.dataio import TracksDataset, DataLoader
from optimize.ranges import ranges

INPUT_FILE = "/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5"
LUT_FILE = "src/larndsim/detector_properties/response_44_v2a_full_tick.npz"
DET_PROPS = "src/larndsim/detector_properties/module0.yaml"
PIXEL_LAYOUTS = "src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml"
RELEVANT_PARAMS = ["Ab", "kb", "lifetime", "tran_diff", "long_diff", "eField",
                   "shift_x", "shift_y", "shift_z"]

# Validated categorical palette (slots 1 and 2), light surface.
C_MC = "#2a78d6"     # stochastic / Monte Carlo
C_AN = "#eb6834"     # analytic / probabilistic
INK = "#0b0b0b"
INK_MUTE = "#52514e"
GRID = "#d8d7d2"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_MUTE, labelsize=8, width=0.8)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for lbl in (ax.xaxis.label, ax.yaxis.label):
        lbl.set_color(INK_MUTE)
        lbl.set_fontsize(8.5)
    ax.title.set_color(INK)
    ax.title.set_fontsize(9)


def build(args):
    ParamsClass = build_params_class(RELEVANT_PARAMS)
    params = load_detector_properties(ParamsClass, DET_PROPS, PIXEL_LAYOUTS)
    params = params.replace(
        electron_sampling_resolution=0.01,
        number_pix_neighbors=4,
        signal_length=150,
        time_window=150,
    )
    try:
        response, params = load_lut(LUT_FILE, params)
    except Exception:
        response, params = load_lut("src/larndsim/detector_properties/response_44.npy", params)
    params = params.replace(**{p: ranges[p]["nom"] for p in RELEVANT_PARAMS if p in ranges})

    dataset = TracksDataset(filename=INPUT_FILE, nevents=args.events,
                            electron_sampling_resolution=0.01)
    fields = dataset.get_track_fields()
    # A dataset item is one trajectory group, finely subdivided -- a single item
    # lights only ~100 pixels regardless of how many events were loaded. Stack
    # several so the pixel population spans a useful range of hit probability.
    nb = min(args.batches, len(dataset))
    chunks = [np.asarray(dataset[i]).reshape(-1, len(fields)) for i in range(nb)]
    tracks = jax.device_put(np.concatenate(chunks, axis=0))
    print(f"batches used: {nb} of {len(dataset)}", flush=True)
    return params, response, tracks, fields


def run(args):
    params, response, tracks, fields = build(args)
    print(f"segments: {tracks.shape[0]}", flush=True)

    prob = LUTProbabilisticSimulation(response).predict(params, tracks, fields, 42)
    wfs = prob["wfs"]
    upix = np.array(prob["unique_pixels"])
    # P(t | pixel, first hit) and the expected charge at each tick
    t_prob = np.exp(np.array(prob["hit_prob"][:, 0, :]))
    a_exp = np.array(prob["adcs_distrib"][:, 0, :])
    print(f"pixels: {(upix >= 0).sum()}  ticks: {t_prob.shape[1]}", flush=True)

    # --- stochastic: N independent noise throws over the same waveforms -------
    @jax.jit
    def chunk(keys):
        f = jax.vmap(get_adc_values, in_axes=(None, None, 0))
        adc, tick = f(params, wfs, keys)
        return adc[:, :, 0], tick[:, :, 0]        # first hit slot only

    keys = jax.random.split(jax.random.PRNGKey(42), args.runs)
    adcs, ticks = [], []
    for i in range(0, args.runs, args.chunk):
        a, t = chunk(keys[i:i + args.chunk])
        adcs.append(np.array(a))
        ticks.append(np.array(t))
        print(f"  {min(i + args.chunk, args.runs)}/{args.runs}", flush=True)
    adcs = np.concatenate(adcs)      # (runs, Npix)
    ticks = np.concatenate(ticks)

    np.savez_compressed(args.cache, upix=upix, t_prob=t_prob, a_exp=a_exp,
                        adcs=adcs, ticks=ticks,
                        gain=float(params.GAIN), vref=float(params.V_REF),
                        vcm=float(params.V_CM), vped=float(params.V_PEDESTAL),
                        adc_counts=float(params.ADC_COUNTS))
    print("cached ->", args.cache, flush=True)
    return upix, t_prob, a_exp, adcs, ticks


def plot(upix, t_prob, a_exp, adcs, ticks, out):
    valid = upix >= 0
    fired = adcs > 0                                   # (runs, Npix)
    emp_rate = fired.mean(axis=0)                      # empirical P(hit)
    lam = t_prob.sum(axis=1)                           # analytic P(hit)

    # pixels with enough statistics to compare a shape
    good = valid & (emp_rate > 0.05)
    idx_good = np.where(good)[0]
    # Show four pixels spanning the hit-probability range. Target specific values
    # rather than percentiles: most pixels saturate at P=1, so percentiles would
    # pick four nearly identical saturated pixels and hide the marginal regime,
    # which is exactly where the noise treatment has to work.
    show = []
    for target in (0.35, 0.65, 0.90, 0.999):
        cand = idx_good[np.argmin(np.abs(emp_rate[idx_good] - target))]
        show.append(cand)

    fig = plt.figure(figsize=(7.1, 5.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.45)
    fig.patch.set_facecolor("#fcfcfb")

    # --- top row: four example pixels ---------------------------------------
    for k, p in enumerate(show):
        ax = fig.add_subplot(gs[0, k])
        tk = ticks[:, p][fired[:, p]]
        if len(tk) == 0:
            continue
        # Zoom to the bulk: the tails are long and would squash the shape.
        lo = int(np.percentile(tk, 0.2)) - 6
        hi = int(np.percentile(tk, 99.8)) + 6
        bins = np.arange(lo, hi + 1) - 0.5
        # density over the fired subset, so both curves are conditional on a hit
        ax.hist(tk, bins=bins, density=True, color=C_MC, alpha=0.85,
                label="stochastic" if k == 0 else None)
        tt = np.arange(t_prob.shape[1])
        pn = t_prob[p] / max(t_prob[p].sum(), 1e-12)
        m = (tt >= lo) & (tt <= hi)
        ax.plot(tt[m], pn[m], color=C_AN, lw=2,
                label="probabilistic" if k == 0 else None)
        ax.set_title(f"$P_{{hit}}$ = {emp_rate[p]:.2f}", loc="left")
        ax.set_xlabel("trigger tick")
        ax.margins(y=0.30)
        if k == 0:
            ax.set_ylabel("probability / tick")
            leg = ax.legend(frameon=False, fontsize=7, loc="upper left",
                            handlelength=1.2, borderpad=0.1, labelspacing=0.25)
            for t in leg.get_texts():
                t.set_color(INK_MUTE)
        ax.tick_params(labelsize=7)
        style(ax)

    # --- bottom left: hit probability, analytic vs empirical -----------------
    # Residual rather than lambda-vs-empirical: most pixels sit at P=1, so a y=x
    # scatter collapses to a single blob and shows nothing. The residual keeps the
    # full range on the x axis and puts the quantity of interest on y.
    ax = fig.add_subplot(gs[1, :2])
    d_all = lam[valid] - emp_rate[valid]
    ax.axhline(0, color=INK_MUTE, lw=1, ls="--", zorder=1)
    ax.scatter(emp_rate[valid], d_all, s=7, color=C_MC, alpha=0.45,
               linewidths=0, zorder=2)
    ax.set_xlabel("stochastic hit probability  (fraction of throws)")
    ax.set_ylabel("probabilistic $\\lambda$ $-$ stochastic")
    ax.set_title("(a)  Hit probability, every pixel", loc="left")
    ax.text(0.97, 0.06, f"bias {d_all.mean():+.4f}\nMAE {np.abs(d_all).mean():.4f}\n"
                        f"{valid.sum()} pixels",
            transform=ax.transAxes, fontsize=7.5, color=INK_MUTE,
            va="bottom", ha="right")
    d = d_all
    style(ax)

    # --- bottom right: per-pixel mean tick ----------------------------------
    ax = fig.add_subplot(gs[1, 2:])
    mt_mc, mt_an = [], []
    for p in idx_good:
        tk = ticks[:, p][fired[:, p]]
        w = t_prob[p]
        if len(tk) == 0 or w.sum() <= 0:
            continue
        mt_mc.append(tk.mean())
        mt_an.append((np.arange(len(w)) * w).sum() / w.sum())
    mt_mc, mt_an = np.array(mt_mc), np.array(mt_an)
    resid = mt_an - mt_mc
    # A handful of pixels sit far out in the tail; plotting the full range squashes
    # the core into one bin. Clip the axis and state how many fell outside.
    lim = max(1.0, float(np.percentile(np.abs(resid), 99)))
    n_outside = int((np.abs(resid) > lim).sum())
    ax.hist(np.clip(resid, -lim, lim), bins=np.linspace(-lim, lim, 45),
            color=C_MC, alpha=0.9)
    ax.axvline(0, color=INK_MUTE, lw=1, ls="--")
    ax.set_xlabel("mean trigger tick: probabilistic $-$ stochastic")
    ax.set_ylabel("pixels")
    ax.set_title("(b)  Timing agreement", loc="left")
    ax.text(0.97, 0.95, f"median {np.median(resid):+.3f} ticks\n"
                        f"68% within {np.percentile(np.abs(resid), 68):.2f}\n"
                        f"{n_outside} pixels beyond $\\pm${lim:.1f}",
            transform=ax.transAxes, fontsize=7.5, color=INK_MUTE,
            va="top", ha="right")
    style(ax)

    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", out)
    print(f"  hit-prob bias {d.mean():+.5f}  MAE {np.abs(d).mean():.5f}")
    print(f"  mean-tick residual median {np.median(resid):+.3f} ticks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--events", type=int, default=60)
    ap.add_argument("--batches", type=int, default=12,
                    help="number of dataset items (trajectory groups) to stack")
    ap.add_argument("--cache", default="plots/prob_vs_stoch.npz")
    ap.add_argument("--out", default="plots/probabilistic_vs_stochastic.png")
    ap.add_argument("--replot", action="store_true",
                    help="skip simulation, plot from the cached npz")
    args = ap.parse_args()

    if args.replot and os.path.exists(args.cache):
        z = np.load(args.cache)
        plot(z["upix"], z["t_prob"], z["a_exp"], z["adcs"], z["ticks"], args.out)
    else:
        upix, t_prob, a_exp, adcs, ticks = run(args)
        plot(upix, t_prob, a_exp, adcs, ticks, args.out)


if __name__ == "__main__":
    main()
