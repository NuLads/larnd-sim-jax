#!/usr/bin/env python3
"""Figure for the paper: what electronics noise does to the hit distribution.

Reads the threshold scan produced earlier (adc_output[_no_noise]_thresh*.h5),
each holding (Nthrows, Npixels, Nslots) ADC values and trigger ticks. The
no-noise files are deterministic, so throw 0 is the whole content.

The message of the figure is that noise is not a widening of the noiseless
answer: it moves the expectation, creates hits that have no noiseless
counterpart, and destroys some that do. That is what makes a noiseless forward
model a biased estimator of the noisy data, and hence why the probabilistic
treatment is needed.

Usage:  python make_noise_figure.py [--throws 2000] [--out plots/noise_effect.png]
"""
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1 and 2 of the validated palette (CVD-checked: worst
# adjacent pair dE 9.2 deutan / 27.6 normal against the light surface).
C_NOISY = "#2a78d6"
C_CLEAN = "#eb6834"
INK      = "#0b0b0b"
INK_MUTE = "#52514e"
GRID     = "#d8d7d2"

THRESHOLDS = [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
NOMINAL = 5000


def yields(thresholds, nthrows):
    """Hit yield per throw, with and without noise, at each threshold.

    Also returns the noiseless hits-per-firing-pixel, which explains the shape of
    the bias curve: the multiplicity collapses between 6 and 7 ke- as the
    threshold rises above the charge left on a pixel after a reset, so the
    secondary hits stop being produced.
    """
    clean, noisy_mean, noisy_sd, mult = [], [], [], []
    for th in thresholds:
        with h5py.File(f"adc_output_no_noise_thresh{th}.h5", "r") as h:
            a = h["results"][0]
        clean.append(int((a > 0).sum()))
        mult.append((a > 0).sum() / max(1, (a > 0).any(axis=1).sum()))
        with h5py.File(f"adc_output_thresh{th}.h5", "r") as h:
            n = (h["results"][:nthrows] > 0).sum(axis=(1, 2))
        noisy_mean.append(n.mean())
        noisy_sd.append(n.std())
    return (np.array(clean), np.array(noisy_mean),
            np.array(noisy_sd), np.array(mult))


def per_hit_stats(th, nthrows):
    """Occupancy, tick spread and ADC spread per (pixel, slot) at one threshold."""
    with h5py.File(f"adc_output_thresh{th}.h5", "r") as h:
        adc = h["results"][:nthrows]
        tick = h["results_ticks"][:nthrows]
    with h5py.File(f"adc_output_no_noise_thresh{th}.h5", "r") as h:
        adc0 = h["results"][0]

    present = adc > 0
    occ = present.mean(axis=0)
    has_clean = adc0 > 0

    # Only slots that fire often enough for a spread to mean anything.
    sel = has_clean & (occ > 0.05)
    masked_t = np.where(present, tick, np.nan)
    masked_a = np.where(present, adc, np.nan)
    with np.errstate(invalid="ignore"):
        tick_sd = np.nanstd(masked_t, axis=0)[sel]
        adc_sd = np.nanstd(masked_a, axis=0)[sel]
        adc_mean = np.nanmean(masked_a, axis=0)[sel]
    adc_rel = adc_sd / np.maximum(adc_mean, 1.0)

    return {
        "occ_clean": occ[has_clean],          # occupancy of hits the noiseless sim makes
        "n_clean": int(has_clean.sum()),
        "n_created": int(((~has_clean) & (occ > 0.05)).sum()),
        "tick_sd": tick_sd[np.isfinite(tick_sd)],
        "adc_rel": adc_rel[np.isfinite(adc_rel)],
    }


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_MUTE, labelsize=8, width=0.8)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for lbl in (ax.xaxis.label, ax.yaxis.label):
        lbl.set_color(INK_MUTE)
        lbl.set_fontsize(9)
    ax.title.set_color(INK)
    ax.title.set_fontsize(9.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--throws", type=int, default=2000)
    ap.add_argument("--out", default="plots/noise_effect.png")
    args = ap.parse_args()

    clean, nmean, nsd, mult = yields(THRESHOLDS, args.throws)
    st = per_hit_stats(NOMINAL, args.throws)
    kt = np.array(THRESHOLDS) / 1e3

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.4))
    fig.patch.set_facecolor("#fcfcfb")

    # (a) yield vs threshold -------------------------------------------------
    ax = axes[0, 0]
    ax.plot(kt, clean, "-o", color=C_CLEAN, lw=2, ms=5, label="no noise")
    ax.plot(kt, nmean, "-o", color=C_NOISY, lw=2, ms=5, label="with noise")
    ax.fill_between(kt, nmean - nsd, nmean + nsd, color=C_NOISY, alpha=0.22, lw=0)
    ax.set_xlabel(r"threshold  [$10^3\,e^-$]")
    ax.set_ylabel("hits per event")
    ax.set_title("(a) Hit yield", loc="left")
    leg = ax.legend(frameon=False, fontsize=8, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK_MUTE)
    style(ax)

    # (b) relative bias ------------------------------------------------------
    ax = axes[0, 1]
    bias = 100 * (nmean - clean) / clean
    ax.axhline(0, color=INK_MUTE, lw=1, ls="--", zorder=1)
    ax.plot(kt, bias, "-o", color=C_NOISY, lw=2, ms=5, zorder=2)
    ax.set_xlabel(r"threshold  [$10^3\,e^-$]")
    ax.set_ylabel("bias of the noiseless\nmodel  [%]")
    ax.set_title("(b) The noiseless model is biased", loc="left")
    i = int(np.argmax(bias[2:]) + 2)
    ax.annotate("secondary hits\ndie out here", (kt[i], bias[i]),
                textcoords="offset points", xytext=(-4, -26), fontsize=7.5,
                color=INK_MUTE, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK_MUTE, lw=0.8))
    ax.margins(y=0.28)
    style(ax)

    # (c) occupancy of noiseless hits ---------------------------------------
    ax = axes[1, 0]
    ax.hist(st["occ_clean"], bins=np.linspace(0, 1, 41), color=C_NOISY, alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylim(top=2e4)
    ax.set_xlabel("fraction of noise throws containing the hit")
    ax.set_ylabel("noiseless hits")
    ax.set_title("(c) Hits appear and disappear", loc="left")
    frac = 100 * (st["occ_clean"] < 0.99).mean()
    ax.text(0.04, 0.95,
            f"{frac:.0f}% lost in $\\geq$1% of throws\n"
            f"{st['n_created']} extra hits have no\nnoiseless counterpart",
            transform=ax.transAxes, fontsize=7.5, color=INK_MUTE, va="top")
    style(ax)

    # (d) per-hit charge spread ---------------------------------------------
    ax = axes[1, 1]
    ax.hist(st["adc_rel"], bins=np.linspace(0, 0.6, 45), color=C_NOISY, alpha=0.9)
    ax.set_xlabel("relative spread of the recorded charge")
    ax.set_ylabel("noiseless hits")
    ax.set_title("(d) Charge of a surviving hit", loc="left")
    med = np.median(st["adc_rel"])
    ax.margins(y=0.22)
    ax.axvline(med, color=C_CLEAN, lw=2)
    ax.annotate(f"median {med:.0%}", (med, ax.get_ylim()[1] * 0.92),
                textcoords="offset points", xytext=(7, 0), fontsize=8,
                color=INK_MUTE, va="top")
    style(ax)

    fig.tight_layout(pad=1.1, w_pad=2.0, h_pad=1.6)
    fig.savefig(args.out, dpi=200, facecolor=fig.get_facecolor())
    print("wrote", args.out)
    print(f"  tick spread: median {np.median(st['tick_sd']):.2f} ticks, "
          f"p90 {np.percentile(st['tick_sd'], 90):.2f}")
    print(f"  charge spread: median {med:.3f}")
    print(f"  noiseless hits {st['n_clean']}, noise-created {st['n_created']}")
    print("  hits per firing pixel (noiseless):", np.round(mult,2))


if __name__ == "__main__":
    main()
