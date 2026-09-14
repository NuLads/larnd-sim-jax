"""Auto-generate one comprehensive markdown report per A2-B5-C2-D10 run
(scan or fit). Reports cover config, per-parameter results, α diagnostics,
and an auto-analysis of fit quality with actionable suggestions.

Usage:
    # Cover every A2-B5-C2-D10 run under the default result dirs:
    python -m optimize.analysis_scripts.make_run_report

    # Filter to a subset (matches inside the label):
    python -m optimize.analysis_scripts.make_run_report --filter mllhd_tQ_sQ5p0

    # Copy each report dir into your Obsidian vault:
    python -m optimize.analysis_scripts.make_run_report --vault ~/Obsidian/larndsim_loss_dQdx

    # Only regenerate scans (or only fits):
    python -m optimize.analysis_scripts.make_run_report --skip-fit
    python -m optimize.analysis_scripts.make_run_report --skip-scan
"""

import argparse
import glob
import json
import os
import pickle
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path("/sdf/group/neutrino/cyifan/diff_sim_playground/larndsim_loss_dQdx")
CONFIG_FILTER_DEFAULT = "A2-B5-C2-D10"

# Canonical physics-param order for panel layouts and tables. Matches the
# 2×3 arrangement Ab, kb, eField / long_diff, tran_diff, lifetime.
PARAM_ORDER = ["Ab", "kb", "eField", "long_diff", "tran_diff", "lifetime",
               "shift_x", "shift_y", "shift_z"]
PARAM_RANK  = {p: i for i, p in enumerate(PARAM_ORDER)}
# Configuration tag pattern used to normalize labels. Any prefix before the
# first `A\d+-B...-C...-D...` sub-string is discarded (older pickles doubled
# the param name in front, e.g. `Ab_A2-B5-C2-D10_...` — they are the same
# physical scan as `A2-B5-C2-D10_...`).
CONFIG_TAG_RE = re.compile(r"A\d+-B\d+-C\d+-D\d+")


def _ordered_params(params):
    """Return params in canonical PARAM_ORDER, appending unknown names last."""
    return sorted(params, key=lambda p: (PARAM_RANK.get(p, 999), p))

# --------------------------------------------------------------------------
# Grouping helpers
# --------------------------------------------------------------------------

# Scans:   history_{param}_batch{N}_{LABEL}.pkl
# Fits:    history_iter{N}_{LABEL}.pkl   (label ends with _tgtsd{S})
SCAN_RE = re.compile(r"^history_([A-Za-z_]+)_batch(\d+)_(.+)\.pkl$")
FIT_RE  = re.compile(r"^history_iter(\d+)_(.+)\.pkl$")


SHZ_RE = re.compile(r"_shz(m?\d+p\d+)")
AB_RE  = re.compile(r"_Ab(m?\d+p\d+)")
KB_RE  = re.compile(r"_kb(m?\d+p\d+)")
SQ_RE  = re.compile(r"_sQ(\d+p\d+)")
LS_RE  = re.compile(r"_ls(\d+p\d+)")

# Registry of supported 2-D outer axes. Keys are the marker string that the
# scan script bakes into the label (via `_{marker}_`); values describe how to
# parse the outer-axis value out of a filename and how to label it in plots.
OUTER_AXIS_SPECS = {
    "2Dsz": dict(name="shift_z",  unit="cm", regex=SHZ_RE, truth=0.0),
    "2Dab": dict(name="Ab",       unit="",   regex=AB_RE,  truth=0.8),
    "2Dkb": dict(name="kb",       unit="",   regex=KB_RE,  truth=0.0486),
}


def _outer_marker(label):
    for marker in OUTER_AXIS_SPECS:
        if f"_{marker}_" in label:
            return marker
    return None


def _strip_outer_axis_tags(s):
    """Strip every registered 2-D outer-axis tag from a label. Adding a new
    OUTER_AXIS_SPECS entry automatically extends this."""
    for spec in OUTER_AXIS_SPECS.values():
        s = spec["regex"].sub("", s)
    return s


def scan_group_key(label):
    """A scan's group key is its LABEL with any 2-D outer-axis tag stripped.
    Different scan params share a common LABEL (the varying part is which
    param is fit — captured by `{param}_batch{N}_`), and each 2-D scan script
    injects a `_{axis}{tag}` suffix per outer-axis point. Stripping both lets
    all pickles for one physical condition collapse into one group.
    """
    return _strip_outer_axis_tags(label)


def sweep_group_key(label):
    """Sweep-family key: strip σ_Q, α_prior, and 2-D outer-axis tags. All
    hyperparameter-sweep points for the same physical run collapse together.
    """
    s = _strip_outer_axis_tags(label)
    s = SQ_RE.sub("_sQXXX", s)
    s = LS_RE.sub("_lsXXX", s)
    return s


def _extract_sq_ls(label):
    sq = SQ_RE.search(label)
    ls = LS_RE.search(label)
    return (sq.group(1) if sq else None, ls.group(1) if ls else None)


def is_2d_scan(label):
    return _outer_marker(label) is not None


def fit_group_key(label):
    """A fit's group key is its LABEL with the tgtsd tail stripped, so all
    seeds of the same fit configuration collapse into one report.
    """
    return re.sub(r"_tgtsd\d+$", "", label)


def _normalize_label(label):
    """Drop any noise before the configuration tag. Legacy scan pickles were
    named `history_Ab_batch49_Ab_A2-B5-C2-D10_...` so the captured label was
    `Ab_A2-B5-C2-D10_...`. The `Ab_` prefix has no effect on the physical
    scan and just fractures the grouping — strip it."""
    m = CONFIG_TAG_RE.search(label)
    if m:
        return label[m.start():]
    return label


def discover(pickle_dirs, config_filter):
    scan_groups = defaultdict(list)
    fit_groups = defaultdict(list)
    for d in pickle_dirs:
        for f in glob.glob(str(Path(d) / "history_*.pkl")):
            bn = os.path.basename(f)
            if config_filter not in bn:
                continue
            m = SCAN_RE.match(bn)
            if m:
                param, batch, label = m.group(1), int(m.group(2)), m.group(3)
                label = _normalize_label(label)
                scan_groups[scan_group_key(label)].append(
                    dict(path=f, kind="scan", param=param, batch=batch, label=label))
                continue
            m = FIT_RE.match(bn)
            if m:
                it, label = int(m.group(1)), m.group(2)
                label = _normalize_label(label)
                fit_groups[fit_group_key(label)].append(
                    dict(path=f, kind="fit", iter=it, label=label))
    return scan_groups, fit_groups


# --------------------------------------------------------------------------
# Pickle parsing
# --------------------------------------------------------------------------

def _series(aux, key, nbt=None):
    if not aux or aux[0] is None or key not in aux[0]:
        return None
    raw = np.array([
        float(a.get(key, np.nan)) if a is not None else np.nan for a in aux
    ])
    if nbt is None:
        return raw
    return np.nanmean(raw.reshape(nbt, -1).T, axis=1)


def load_scan_pickle(path):
    with open(path, "rb") as f:
        r = pickle.load(f)
    params = [k.replace("_grad", "") for k in r.keys() if k.endswith("_grad")]
    if not params:
        return None
    param = params[0]
    piter = np.asarray(r[f"{param}_iter"])
    pval = np.sort(np.unique(piter[1:]))
    nbt = int(np.count_nonzero(piter == pval[0]))
    if nbt == 0:
        return None
    max_i = (len(r.get("aux_iter") or [])) // nbt * nbt
    loss = np.array(r["losses_iter"][:max_i]).reshape(nbt, -1).T
    loss_mean = np.nanmean(loss, axis=1)
    loss_std  = np.nanstd(loss, axis=1) / max(np.sqrt(max(nbt, 1)), 1)  # SE of the mean
    aux = r.get("aux_iter", [])[:max_i]

    # Autodiff gradient of the loss w.r.t. the scanned parameter. Shape
    # matches losses_iter (nbt · npoints); reshape and reduce like the loss.
    grad_arr = r.get(f"{param}_grad")
    grad_mean = grad_se = None
    if grad_arr is not None and len(grad_arr) >= max_i and max_i > 0:
        grad = np.array(grad_arr[:max_i]).reshape(nbt, -1).T
        grad_mean = np.nanmean(grad, axis=1)
        grad_se   = np.nanstd(grad, axis=1) / max(np.sqrt(max(nbt, 1)), 1)

    return dict(
        param=param, target=float(r[f"{param}_target"][0]),
        pval=pval, loss=loss_mean, loss_se=loss_std,
        grad=grad_mean, grad_se=grad_se,
        aux=aux, nbt=nbt,
        config=vars(r["config"]) if r.get("config") is not None else {},
        path=path,
    )


def load_fit_pickle(path):
    """A fit pickle contains trajectory history (per-iter values + aux).
    Return the trajectory for every fit param plus loss.
    """
    with open(path, "rb") as f:
        r = pickle.load(f)
    # Restrict to keys that also have a matching `<param>_target` — that filters
    # out bookkeeping arrays like `aux_iter` while keeping every real fit param.
    params = [k.replace("_iter", "") for k in r.keys()
              if k.endswith("_iter") and k != "losses_iter"
              and not k.endswith("_target_iter")
              and (k.replace("_iter", "_target") in r or k.replace("_iter", "_grad") in r)]
    trajectories = {}
    targets = {}
    for p in params:
        traj = np.asarray(r[f"{p}_iter"])
        trajectories[p] = traj
        # target for fit: some fits log <param>_target
        if f"{p}_target" in r:
            v = r[f"{p}_target"]
            targets[p] = float(v[0]) if hasattr(v, "__len__") else float(v)
    losses = np.asarray(r.get("losses_iter", [])).ravel()
    aux = r.get("aux_iter", [])
    return dict(
        params=params, trajectories=trajectories, targets=targets,
        losses=losses, aux=aux,
        config=vars(r["config"]) if r.get("config") is not None else {},
        path=path,
    )


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------

def _grad_zero_crossing(pval, grad):
    """Linear-interpolate the first sign change of `grad` along `pval`.
    Returns None if `grad` is missing or never changes sign in the range."""
    if grad is None or not np.any(np.isfinite(grad)):
        return None
    idx = np.where(np.diff(np.sign(grad)) != 0)[0]
    if len(idx) == 0:
        return None
    j = int(idx[0])
    dj = grad[j + 1] - grad[j]
    if abs(dj) < 1e-30:
        return None
    return float(pval[j] - grad[j] * (pval[j + 1] - pval[j]) / dj)


def plot_scan_all(parsed, out_png):
    """Loss + autodiff gradient on the SAME panel per parameter, using a
    dual y-axis (loss on left, gradient on right). Panel order follows the
    canonical Ab, kb, eField / long_diff, tran_diff, lifetime layout.

    Overlays:
      - red dashed vertical: parameter target
      - grey dashed vertical: autodiff dL/dθ zero-crossing (only when the
        sign change exists inside the scan range).
      - C0 solid + shaded SE: batch-mean loss ± SE
      - C1 dotted line: batch-mean autodiff gradient dL/dθ
      - C1 horizontal grey at y=0: reference for zero-crossing

    Interpretation:
      - Where the gradient crosses zero should coincide with the loss
        minimum. A visible gap between the target-red and grad-zero-grey
        vertical lines is the scan-level bias.
      - The gradient curve makes a stochastic-sim autodiff bias visible
        (autodiff should agree with the finite-difference of the loss
        curve; a mismatch signals a non-differentiable op in the sim path).
    """
    by_param = defaultdict(list)
    for p in parsed:
        by_param[p["param"]].append(p)
    params = _ordered_params(by_param.keys())
    n_cols = 3
    n_rows = int(np.ceil(len(params) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                            squeeze=False)
    for i, prm in enumerate(params):
        ax_loss = axs[i // n_cols, i % n_cols]
        ax_grad = ax_loss.twinx()
        gz = None
        for p in by_param[prm]:
            # --- loss curve (left axis) ---
            ax_loss.plot(p["pval"], p["loss"], lw=1.2, color="C0",
                         label="loss")
            se = p.get("loss_se")
            if se is not None and np.any(np.isfinite(se)):
                ax_loss.fill_between(p["pval"], p["loss"] - se, p["loss"] + se,
                                     alpha=0.25, color="C0", linewidth=0)
            # --- gradient curve (right axis) ---
            g = p.get("grad")
            if g is not None and np.any(np.isfinite(g)):
                ax_grad.plot(p["pval"], g, lw=1.0, color="C1", linestyle=":",
                             label="dL/dθ (autodiff)")
                gse = p.get("grad_se")
                if gse is not None and np.any(np.isfinite(gse)):
                    ax_grad.fill_between(p["pval"], g - gse, g + gse,
                                         alpha=0.15, color="C1", linewidth=0)
                if gz is None:
                    gz = _grad_zero_crossing(p["pval"], g)
            ax_loss.axvline(p["target"], color="red", linestyle="--",
                            linewidth=1, label=f"target = {p['target']:.4g}")
        if gz is not None:
            ax_loss.axvline(gz, color="0.35", linestyle="--", linewidth=1,
                            label=f"grad zero = {gz:.4g}")
        ax_grad.axhline(0.0, color="0.6", linewidth=0.6)

        p0 = by_param[prm][0]
        bias, snr, _ = _scan_stats(p0)
        snr_s = f"{snr:.1f}×" if np.isfinite(snr) else "∞"
        ax_loss.set_title(f"{prm}  (bias {bias:+.1f}%, SNR {snr_s})")
        ax_loss.set_xlabel(prm)
        ax_loss.set_ylabel("mean loss ± SE", color="C0")
        ax_loss.tick_params(axis="y", labelcolor="C0")
        ax_grad.set_ylabel("dL/dθ", color="C1")
        ax_grad.tick_params(axis="y", labelcolor="C1")
        ax_loss.grid(alpha=0.3)
        # Merge legends from both axes.
        h1, l1 = ax_loss.get_legend_handles_labels()
        h2, l2 = ax_grad.get_legend_handles_labels()
        ax_loss.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")
    for j in range(len(params), n_rows * n_cols):
        axs[j // n_cols, j % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_scan_gradients(parsed, out_png):
    """Autodiff gradient dL/dθ vs θ, one panel per param.

    Truth-consistency: the gradient should cross zero at the loss minimum.
    Reads the batch-mean grad + SE saved in `load_scan_pickle`. Also
    overlays a finite-difference gradient of the batch-mean loss so an
    autodiff-vs-FD disagreement (e.g. under a stochastic sim's hard
    discriminator) is visible at a glance.

    Returns True if at least one panel had gradient data.
    """
    by_param = defaultdict(list)
    for p in parsed:
        by_param[p["param"]].append(p)
    params = _ordered_params(by_param.keys())
    n_cols = 3
    n_rows = int(np.ceil(len(params) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                            squeeze=False)
    any_grad = False
    for i, prm in enumerate(params):
        ax = axs[i // n_cols, i % n_cols]
        p0 = by_param[prm][0]
        pval = p0["pval"]
        # Autodiff gradient with SE band.
        g = p0.get("grad")
        if g is not None and np.any(np.isfinite(g)):
            any_grad = True
            ax.plot(pval, g, lw=1.2, color="C0", label="autodiff")
            gse = p0.get("grad_se")
            if gse is not None and np.any(np.isfinite(gse)):
                ax.fill_between(pval, g - gse, g + gse,
                                alpha=0.25, color="C0", linewidth=0)
            gz = _grad_zero_crossing(pval, g)
            if gz is not None:
                ax.axvline(gz, color="C0", linestyle="--", lw=1,
                           label=f"grad zero: {gz:.4g}")
        else:
            ax.text(0.5, 0.5, "no gradient recorded",
                    ha="center", va="center", transform=ax.transAxes,
                    color="0.5")

        # Finite-difference gradient from the batch-mean loss curve.
        loss = p0.get("loss")
        if loss is not None and len(pval) >= 3:
            fd = np.gradient(loss, pval)
            ax.plot(pval, fd, lw=1.0, color="C3", linestyle="--",
                    label="finite-diff (from loss curve)")

        ax.axvline(p0["target"], color="red", linestyle="--", linewidth=1,
                   label=f"target: {p0['target']:.4g}")
        ax.axhline(0.0, color="0.6", linewidth=0.8)
        ax.set_title(f"{prm}  gradient")
        ax.set_xlabel(prm)
        ax.set_ylabel("dL/dθ")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    for j in range(len(params), n_rows * n_cols):
        axs[j // n_cols, j % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return any_grad


def plot_scan_alpha(parsed, out_png):
    """α diagnostics per param: mean±std log α, prior_term, frac_underflow."""
    by_param = defaultdict(list)
    for p in parsed:
        by_param[p["param"]].append(p)
    params = _ordered_params(by_param.keys())
    n_cols = 3
    n_rows = int(np.ceil(len(params) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                            squeeze=False)
    any_alpha = False
    for i, prm in enumerate(params):
        ax = axs[i // n_cols, i % n_cols]
        p = by_param[prm][0]
        mla = _series(p["aux"], "mean_log_alpha_track", p["nbt"])
        sla = _series(p["aux"], "std_log_alpha_track", p["nbt"])
        if mla is None:
            ax.text(0.5, 0.5, "no α diagnostics",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        any_alpha = True
        ax.plot(p["pval"], mla, color="C0", label="mean log α")
        if sla is not None:
            ax.fill_between(p["pval"], mla - sla, mla + sla,
                            alpha=0.2, color="C0")
        ax.axhline(0, color="gray", linestyle=":", lw=0.8)
        ax.axvline(p["target"], color="red", linestyle="--", lw=1)
        ax.set_title(f"{prm}: log α_track")
        ax.set_xlabel(prm)
        ax.grid(alpha=0.3)
    for j in range(len(params), n_rows * n_cols):
        axs[j // n_cols, j % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return any_alpha


def _fit_color(seed):
    """Match plot.ipynb — seed 1..N maps to C1..CN (not C0..C{N-1})."""
    return f"C{seed % 10}"


def plot_fit_trajectories(fits, out_png):
    """One panel per param; overlay per-seed trajectories, each with its own
    dashed target line in the matching color (targets differ across seeds when
    --scan_tgt_nom is on). Colors and legend text follow plot.ipynb: `C{seed}`,
    and the legend shows `target: T | fitted: F ({bias%})` per seed."""
    all_params = _ordered_params({p for f in fits for p in f["params"]})
    if not all_params:
        return False
    n_cols = 3
    n_rows = int(np.ceil(len(all_params) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                            squeeze=False)
    fits_sorted = sorted(fits, key=lambda f: f["seed"])
    for i, prm in enumerate(all_params):
        ax = axs[i // n_cols, i % n_cols]
        for f in fits_sorted:
            if prm not in f["trajectories"]:
                continue
            traj = np.asarray(f["trajectories"][prm]).ravel()
            c = _fit_color(f["seed"])
            tgt = f["targets"].get(prm)
            final = _fit_final(traj)
            if tgt is not None:
                bias = 100 * (final - tgt) / max(abs(tgt), 1e-30)
                lbl = f"target: {tgt:.2E} | fitted: {final:.2E} ({bias:+.2f}%)"
            else:
                lbl = f"seed {f['seed']}: {final:.2E}"
            ax.plot(traj, lw=1, alpha=0.9, color=c, label=lbl)
            if tgt is not None:
                ax.axhline(tgt, color=c, linestyle="--", linewidth=1, alpha=0.7)
        ax.legend(fontsize=7, loc="lower left")
        ax.set_title(f"{prm}  (dashed = per-seed target)")
        ax.set_xlabel("training iterations")
        ax.set_ylabel(prm)
        ax.grid(alpha=0.3)
        if prm == "lifetime":
            ax.set_yscale("log")
    for j in range(len(all_params), n_rows * n_cols):
        axs[j // n_cols, j % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_fit_loss(fits, out_png):
    fig, ax = plt.subplots(figsize=(8, 4))
    for f in sorted(fits, key=lambda f: f["seed"]):
        L = np.asarray(f["losses"]).ravel()
        if len(L) == 0:
            continue
        tail = float(np.nanmean(L[-min(len(L), 500):]))
        ax.plot(L, ".", ms=2, alpha=0.7, color=_fit_color(f["seed"]),
                label=f"seed {f['seed']}: last-500 loss = {tail:.2E}")
    ax.set_xlabel("training iterations")
    ax.set_ylabel("loss")
    ax.set_title("Fit loss trajectory (per seed)")
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Auto-analysis
# --------------------------------------------------------------------------

def _scan_stats(p):
    """Return (bias_pct, depth_snr, i_min).
    depth_snr = (loss range across scan) / (typical SE of the batch-mean).
    SNR is the honest "is this minimum resolvable above stochastic noise" number —
    unlike (max-min)/|min|, it does not depend on the absolute loss magnitude
    (which is misleading for LLHD losses whose mean is O(10) but the interesting
    variation is O(1e-3)).
    """
    loss = p["loss"]
    se = p.get("loss_se")
    i_min = int(np.nanargmin(loss))
    bias = 100 * (p["pval"][i_min] - p["target"]) / max(p["target"], 1e-30)
    rng = float(np.nanmax(loss) - np.nanmin(loss))
    noise = float(np.nanmedian(se)) if se is not None and np.any(np.isfinite(se)) else np.nan
    snr = rng / noise if noise and noise > 0 else np.inf
    return bias, snr, i_min


def analyze_scan(parsed):
    """Return list of (severity, message) tuples."""
    obs = []
    for p in parsed:
        bias, snr, _ = _scan_stats(p)
        prm = p["param"]
        # depth resolved above stochastic noise
        if not np.isfinite(snr):
            pass
        elif snr < 3:
            obs.append(("HIGH", f"`{prm}`: SNR {snr:.1f}× — minimum not resolved above batch noise. "
                                f"Bias {bias:+.1f}% is dominated by stochastic fluctuation, not a real preference."))
        elif snr < 10:
            obs.append(("MED", f"`{prm}`: modest SNR {snr:.1f}× — bias {bias:+.1f}% only marginally resolved above noise."))
        # bias — only flag when there is a real minimum
        if snr >= 3 or not np.isfinite(snr):
            if abs(bias) > 20:
                obs.append(("HIGH", f"`{prm}`: large bias {bias:+.1f}% (SNR {snr:.1f}×) — "
                                    "check for degeneracy or missing systematic (shift_z?)."))
            elif abs(bias) > 5:
                obs.append(("MED", f"`{prm}`: moderate bias {bias:+.1f}% (SNR {snr:.1f}×) — could be residual systematic."))
        # underflow
        fu = _series(p["aux"], "frac_underflow", p["nbt"])
        if fu is not None:
            i_t = int(np.argmin(np.abs(p["pval"] - p["target"])))
            if fu[i_t] > 0.05:
                obs.append(("MED",
                    f"`{prm}`: frac_underflow @ target = {fu[i_t]:.2%}. "
                    "Some target hits get density-floored. Consider wider σ_Q or mllhdP."))
        # prior dominance
        pri = _series(p["aux"], "alpha_prior_term", p["nbt"])
        mmo = _series(p["aux"], "mmd_only", p["nbt"])
        if pri is not None and mmo is not None:
            i_t = int(np.argmin(np.abs(p["pval"] - p["target"])))
            if mmo[i_t] > 0 and pri[i_t] / mmo[i_t] > 0.5:
                obs.append(("MED",
                    f"`{prm}`: α prior / mmd_only @ target = {pri[i_t] / mmo[i_t]:.2f} — "
                    "prior contributes ≳50% of the loss; consider loosening α prior."))
    return obs


def _fit_final(traj, tail_iters=500):
    """Mean of the last `tail_iters` iterations — matches plot.ipynb which
    reports `np.mean(history[param + '_iter'][-500:])` for the bias %.
    Falls back to the last 10 % when the trajectory is shorter than
    tail_iters (short scans / early-stopped fits)."""
    traj = np.asarray(traj).ravel()
    if len(traj) == 0:
        return np.nan
    n = min(len(traj), max(tail_iters, int(len(traj) * 0.1)))
    return float(np.nanmean(traj[-n:]))


def analyze_fit(fits):
    obs = []
    # Per-param, per-seed bias against that seed's own target.
    per_param_biases = defaultdict(list)
    for f in fits:
        L = np.asarray(f["losses"]).ravel()
        if len(L) > 100:
            tail = L[-int(len(L) * 0.1):]
            slope = np.polyfit(np.arange(len(tail)), tail, 1)[0]
            rel_slope = slope * len(tail) / (tail.mean() + 1e-30)
            if abs(rel_slope) > 0.02:
                obs.append(("MED",
                    f"seed {f['seed']}: loss still {'decreasing' if slope < 0 else 'increasing'} "
                    f"at end (rel change over last 10%: {rel_slope:+.2%}) — needs more iterations."))
        for prm, target in f["targets"].items():
            traj = f["trajectories"].get(prm)
            if traj is None or len(traj) == 0:
                continue
            final = _fit_final(traj)
            if not np.isfinite(final):
                continue
            bias = 100 * (final - target) / max(abs(target), 1e-30)
            per_param_biases[prm].append((f["seed"], final, target, bias))
    # Aggregate per-param biases across seeds.
    for prm, rows in per_param_biases.items():
        biases = np.array([r[3] for r in rows])
        mean_b = float(np.nanmean(biases))
        # Same-sign across seeds → systematic
        signs = np.sign(biases[np.isfinite(biases)])
        if len(signs) >= 2 and (np.all(signs > 0) or np.all(signs < 0)) and abs(mean_b) > 5:
            obs.append(("HIGH",
                f"`{prm}`: same-sign bias across all {len(signs)} seeds "
                f"(mean {mean_b:+.1f}%) — systematic, not stochastic. "
                "Consider adding a correlated d.o.f. (e.g. shift_z for lifetime)."))
        elif abs(mean_b) > 20:
            obs.append(("HIGH",
                f"`{prm}`: large mean bias {mean_b:+.1f}% across seeds — degeneracy or missing d.o.f."))
        elif abs(mean_b) > 5:
            obs.append(("MED", f"`{prm}`: moderate mean bias {mean_b:+.1f}% across seeds."))
    return obs


def _outer_from_label(label):
    """Return (marker_key, outer_value) or (None, None) if the label has no
    recognized outer-axis tag."""
    for marker, spec in OUTER_AXIS_SPECS.items():
        if f"_{marker}_" not in label:
            continue
        m = spec["regex"].search(label)
        if not m:
            continue
        tag = m.group(1)
        sign = -1 if tag.startswith("m") else 1
        body = tag.lstrip("m")
        return marker, sign * float(body.replace("p", "."))
    return None, None


def plot_2d_scan(entries, out_png, outer_spec):
    """Assemble the 2-D (lifetime × outer-axis) surface from per-outer-value
    pickles and render it as heatmap + per-slice lifetime curves. Works for
    both shift_z (`_2Dsz_`) and Ab (`_2Dab_`) outer axes; the outer axis is
    labeled per `outer_spec`."""
    ax_name = outer_spec["name"]
    ax_unit = outer_spec["unit"]
    ax_truth = outer_spec["truth"]
    ax_regex = outer_spec["regex"]

    def _parse(label):
        m = ax_regex.search(label)
        if not m:
            return None
        tag = m.group(1)
        sign = -1 if tag.startswith("m") else 1
        body = tag.lstrip("m")
        return sign * float(body.replace("p", "."))

    curves = {}
    lifetime_target = None
    for e in entries:
        if e["param"] != "lifetime":
            continue
        outer = _parse(e["label"])
        if outer is None:
            continue
        with open(e["path"], "rb") as f:
            r = pickle.load(f)
        piter = np.asarray(r["lifetime_iter"])
        pval = np.sort(np.unique(piter[1:]))
        nbt = int(np.count_nonzero(piter == pval[0]))
        if nbt == 0:
            continue
        max_i = (len(r["losses_iter"]) // nbt) * nbt
        loss = np.array(r["losses_iter"][:max_i]).reshape(nbt, -1).T
        loss_mean = np.nanmean(loss, axis=1)
        curves.setdefault(outer, []).append((pval, loss_mean))
        lifetime_target = float(r["lifetime_target"][0])
    if not curves:
        return None

    outers = sorted(curves.keys())
    pval_ref = curves[outers[0]][0][0]
    grid = np.full((len(outers), len(pval_ref)), np.nan)
    for i, o in enumerate(outers):
        losses = np.stack([lm for (_, lm) in curves[o]], axis=0)
        grid[i, :] = np.nanmean(losses, axis=0)
    i_min, j_min = np.unravel_index(np.nanargmin(grid), grid.shape)
    outer_at_min = outers[i_min]
    lifetime_min = pval_ref[j_min]
    per_slice_lifetime_min = np.array([
        pval_ref[int(np.nanargmin(grid[i, :]))] for i in range(len(outers))
    ])

    unit_suffix = f" ({ax_unit})" if ax_unit else ""

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    ax = axs[0]
    pcm = ax.pcolormesh(pval_ref, outers, grid, shading="nearest", cmap="viridis")
    plt.colorbar(pcm, ax=ax, label="mean loss")
    try:
        ax.contour(pval_ref, outers, grid, levels=10,
                   colors="white", linewidths=0.5, alpha=0.5)
    except Exception:
        pass
    ax.axvline(lifetime_target, color="red", linestyle="--", linewidth=1.2,
               label=f"lifetime target = {lifetime_target:.3g}")
    ax.axhline(ax_truth, color="red", linestyle="--", linewidth=1.2,
               label=f"{ax_name} target = {ax_truth}")
    ax.plot(lifetime_target, ax_truth, "r*", markersize=14, label="truth")
    ax.plot(lifetime_min, outer_at_min, "yD", markersize=10,
            label=f"joint min: ({lifetime_min:.3g}, {outer_at_min:+.3g})")
    ax.plot(per_slice_lifetime_min, outers, "w-", linewidth=1.5, alpha=0.8,
            label="per-slice τ min")
    ax.plot(per_slice_lifetime_min, outers, "wx", markersize=8, alpha=0.9)
    ax.set_xlabel("lifetime (µs)")
    ax.set_ylabel(f"{ax_name}{unit_suffix}")
    ax.set_title("2-D loss surface")
    ax.legend(fontsize=8, loc="upper right")

    ax2 = axs[1]
    cmap_s = plt.cm.coolwarm
    for i, o in enumerate(outers):
        color = cmap_s(i / max(len(outers) - 1, 1))
        ax2.plot(pval_ref, grid[i, :], color=color, alpha=0.9,
                 label=f"{ax_name}={o:+.3g}")
    ax2.axvline(lifetime_target, color="red", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("lifetime (µs)")
    ax2.set_ylabel("mean loss")
    ax2.set_title(f"Lifetime slices, colored by {ax_name}")
    ax2.legend(fontsize=7, loc="best", ncol=2)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    slope, intercept = np.polyfit(outers, per_slice_lifetime_min, 1)
    r = float(np.corrcoef(outers, per_slice_lifetime_min)[0, 1])
    return dict(
        outers=outers, outer_name=ax_name, outer_unit=ax_unit, outer_truth=ax_truth,
        pval=pval_ref, grid=grid,
        lifetime_target=lifetime_target,
        outer_at_min=outer_at_min, lifetime_min=lifetime_min,
        per_slice_lifetime_min=per_slice_lifetime_min,
        valley_slope=float(slope), valley_r=r,
        n_outer=len(outers),
    )


def write_2d_scan_report(label, entries, out_dir):
    marker = _outer_marker(label)
    if marker is None:
        return None
    spec = OUTER_AXIS_SPECS[marker]

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    surface_png = plots_dir / f"2d_lifetime_{spec['name']}.png"
    stats = plot_2d_scan(entries, surface_png, spec)
    if stats is None:
        return None

    ax_name = stats["outer_name"]
    ax_unit = stats["outer_unit"]
    unit_suffix = f" {ax_unit}" if ax_unit else ""
    ax_truth = stats["outer_truth"]

    with open(entries[0]["path"], "rb") as f:
        r0 = pickle.load(f)
    cfg = vars(r0["config"]) if r0.get("config") is not None else {}
    hp = _hyperparams_from_kw(cfg.get("loss_fn_kw", {}))

    lifetime_bias = 100 * (stats["lifetime_min"] - stats["lifetime_target"]) / max(abs(stats["lifetime_target"]), 1e-30)

    data_mt = _pickle_mtime(e["path"] for e in entries)
    lines = []
    lines.append("---")
    lines.append(f"date: {datetime.now().isoformat(timespec='seconds')}")
    if data_mt:
        lines.append(f"data_mtime: {data_mt}")
    lines.append(f"run_type: 2d_scan")
    lines.append(f"outer_axis: {ax_name}")
    lines.append(f"metric: {hp.get('metric')}")
    lines.append(f"n_outer_points: {stats['n_outer']}")
    lines.append("tags:")
    lines.append("  - larndsim")
    lines.append("  - scan_2d")
    lines.append(f"  - {ax_name}")
    lines.append(f"  - {hp.get('metric', 'unknown')}")
    lines.append("---\n")

    lines.append(f"# 2-D scan (lifetime × {ax_name}) — `{label[:100]}`\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}  ")
    lines.append(f"{ax_name} points sampled: {stats['n_outer']}  (values: "
                 f"{', '.join(f'{v:+.3g}' for v in stats['outers'])})\n")

    lines.append("## Run condition\n")
    lines.append("| key | value |")
    lines.append("|---|---|")
    lines.append(f"| **run type** | 2-D scan (lifetime × {ax_name}) |")
    lines.append(f"| outer axis | {ax_name} ({stats['n_outer']} pts) |")
    lines.append(f"| inner axis | lifetime ({len(stats['pval'])} pts along `param_list` range) |")
    lines.append(f"| loss family | `{cfg.get('loss_fn', '?')}` / `{hp.get('metric')}` |")
    lines.append(f"| probabilistic sim | {cfg.get('probabilistic_sim', '?')} |")
    lines.append(f"| batch (bt × nbt × iter) | {cfg.get('max_batch_len','?')} × "
                 f"{cfg.get('max_nbatch','?')} × {cfg.get('iterations','?')} |")
    lines.append(f"| dedx density mode | `{cfg.get('dedx_density_mode','?')}` / "
                 f"flow_expectation=`{cfg.get('flow_expectation_mode','?')}` |")
    lines.append(f"| normalization | `{cfg.get('normalization_scheme','?')}` |")
    lines.append("")

    lines.append("### D10 loss hyperparameters\n")
    lines.append("| knob | value |")
    lines.append("|---|---|")
    for k, v in hp.items():
        lines.append(f"| {k} | `{v}` |")
    lines.append("")

    lines.append("## 2-D loss surface\n")
    lines.append(f"![]({surface_png.relative_to(out_dir).as_posix()})\n")

    slope_unit = "µs" + (f" / {ax_unit}" if ax_unit else " / unit")
    lines.append("## Joint-minimum diagnostic\n")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append(f"| truth lifetime | {stats['lifetime_target']:.4g} µs |")
    lines.append(f"| joint-min lifetime | {stats['lifetime_min']:.4g} µs "
                 f"(bias {lifetime_bias:+.2f}%) |")
    lines.append(f"| joint-min {ax_name} | {stats['outer_at_min']:+.4g}{unit_suffix} "
                 f"(truth = {ax_truth}) |")
    lines.append(f"| valley slope | {stats['valley_slope']:+.3g} {slope_unit} |")
    lines.append(f"| valley correlation r | {stats['valley_r']:+.2f} |")
    lines.append("")

    lines.append("## Analysis & suggestions\n")
    if abs(stats["valley_r"]) > 0.5:
        direction = "positive" if stats["valley_slope"] > 0 else "negative"
        lines.append(f"⚠️ **HIGH** — strong {direction} correlation between the per-slice lifetime "
                     f"minimum and `{ax_name}` (r = {stats['valley_r']:+.2f}). "
                     f"(lifetime, {ax_name}) are degenerate: Adam in a 4/6-par fit will slide along "
                     f"this valley. If `{ax_name}` drifts in the fit, lifetime will absorb the drift "
                     f"in the opposite direction along this slope.")
    else:
        lines.append(f"ℹ️ Weak correlation between lifetime-min-per-slice and `{ax_name}` "
                     f"(r = {stats['valley_r']:+.2f}). Lifetime bias in the fit is *not* explained "
                     f"by (lifetime, {ax_name}) degeneracy — look elsewhere.")
    lines.append("")
    outer_offset = stats["outer_at_min"] - ax_truth
    if abs(outer_offset) > 1e-3:
        lines.append(f"- Joint minimum sits at {ax_name} = {stats['outer_at_min']:+.3g}"
                     f"{unit_suffix} (offset {outer_offset:+.3g} from truth {ax_truth}).")
    if abs(lifetime_bias) > 5:
        lines.append(f"- Lifetime bias at joint min is {lifetime_bias:+.2f}% — this is what a fit sees "
                     f"before it can free `{ax_name}`. Compare to the fit report.")
    lines.append("")

    lines.append("## Source pickles\n")
    for e in sorted(entries, key=lambda x: (_outer_from_label(x["label"])[1] or 0)):
        _, v = _outer_from_label(e["label"])
        lines.append(f"- ({ax_name} = {v:+.3g}) `{os.path.basename(e['path'])}`")

    (out_dir / "report.md").write_text("\n".join(lines))

    with open(out_dir / "metrics.csv", "w") as f:
        f.write(f"{ax_name},lifetime_min,loss_min\n")
        for i, o in enumerate(stats["outers"]):
            j = int(np.nanargmin(stats["grid"][i, :]))
            f.write(f"{o},{stats['pval'][j]},{stats['grid'][i, j]}\n")

    return out_dir / "report.md"


def _hyperparams_from_kw(kw):
    """Human-readable D10 hyperparameters summary.

    The slope-penalty knobs are only included when the term was actually
    configured (weight > 0 or diag mode on) so vanilla-D10 tables don't
    fill with unused rows.
    """
    if not isinstance(kw, dict):
        return {}
    hp = {
        "metric": kw.get("distance_metric", "?"),
        "include_Q": kw.get("marginal_llhd_include_Q", "?"),
        "σ_Q (ke)": kw.get("marginal_llhd_sigma_Q_ke", "?"),
        "α mode": kw.get("per_track_scale_mode", "?"),
        "α prior log-σ": kw.get("per_track_scale_prior_log_sigma", "?"),
        "ε (floor)": kw.get("marginal_llhd_eps", "?"),
    }
    sp_kind   = kw.get("slope_penalty_kind")
    sp_weight = kw.get("slope_penalty_weight", 0)
    sp_diag   = kw.get("slope_penalty_diag", False)
    try:
        sp_weight_f = float(sp_weight) if sp_weight is not None else 0.0
    except (TypeError, ValueError):
        sp_weight_f = 0.0
    sp_active = sp_weight_f > 0 or bool(sp_diag)
    if sp_kind is not None and sp_active:
        # Encode mode compactly: production if weight > 0, diag if only diag.
        if sp_weight_f > 0:
            mode = "production"
        else:
            mode = "diag (no gradient effect)"
        hp["slope kind"]    = sp_kind
        hp["slope mode"]    = mode
        hp["slope weight"]  = sp_weight
        hp["slope σ"]       = kw.get("slope_penalty_sigma", "?")
    elif sp_kind is not None and sp_kind != "within_track":
        # Inert-configured — the pickle carried a non-default kind but the
        # term was off. Flag it so the reader knows the label suffix
        # (_sp0<kind>) reflects intent, not an actual penalty in the loss.
        hp["slope kind"]    = f"{sp_kind} (inert)"
    return hp


# --------------------------------------------------------------------------
# Report writers
# --------------------------------------------------------------------------

def _load_sweep_point(path):
    """Load one pickle for the sweep summary: returns dict with pval, loss,
    target, and a few aux time-series at the resolution of `nbt` averages."""
    with open(path, "rb") as f:
        r = pickle.load(f)
    params = [k.replace("_grad", "") for k in r.keys() if k.endswith("_grad")]
    if not params:
        return None
    prm = params[0]
    piter = np.asarray(r[f"{prm}_iter"])
    pval = np.sort(np.unique(piter[1:]))
    nbt = int(np.count_nonzero(piter == pval[0]))
    if nbt == 0:
        return None
    max_i = (len(r.get("aux_iter") or [])) // nbt * nbt
    loss = np.array(r["losses_iter"][:max_i]).reshape(nbt, -1).T
    aux = r.get("aux_iter", [])[:max_i]
    d = dict(
        param=prm,
        pval=pval,
        loss=np.nanmean(loss, axis=1),
        loss_se=np.nanstd(loss, axis=1) / max(np.sqrt(max(nbt, 1)), 1),
        target=float(r[f"{prm}_target"][0]),
        frac_underflow=_series(aux, "frac_underflow", nbt),
        mean_log_alpha=_series(aux, "mean_log_alpha_track", nbt),
        alpha_prior=_series(aux, "alpha_prior_term", nbt),
    )
    return d


def plot_sweep_curves(curves, sq_tags, ls_tags, out_png, title):
    """Overlaid loss curves: one panel per param, one colored line per
    (σ_Q, α_prior). Mirrors Fig 1 in loss_scan.ipynb Cell 9."""
    params = _ordered_params({k[0] for k in curves})
    n_cols = 3
    n_rows = int(np.ceil(len(params) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                            squeeze=False)
    cmap = plt.cm.viridis
    combos = [(sq, ls) for sq in sq_tags for ls in ls_tags]
    color_of = {c: cmap(i / max(len(combos) - 1, 1)) for i, c in enumerate(combos)}
    for i_par, prm in enumerate(params):
        ax = axs[i_par // n_cols, i_par % n_cols]
        target_val = None
        for sq in sq_tags:
            for ls in ls_tags:
                d = curves.get((prm, sq, ls))
                if d is None:
                    continue
                target_val = d["target"]
                lbl = f"σ_Q={sq.replace('p','.')} ls={ls.replace('p','.')}"
                ax.plot(d["pval"], d["loss"], color=color_of[(sq, ls)],
                        lw=1.5, label=lbl, alpha=0.85)
        if target_val is not None:
            ax.axvline(target_val, color="red", linestyle="--", linewidth=1.2)
        ax.set_xlabel(prm)
        ax.set_ylabel("mean loss")
        ax.set_title(prm)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="best", ncol=2)
    for j in range(len(params), n_rows * n_cols):
        axs[j // n_cols, j % n_cols].axis("off")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_sweep_heatmap(curves, sq_tags, ls_tags, out_png, title):
    """Heatmaps of bias/SNR/frac_uf/mean_log_α vs (σ_Q, α_prior),
    one row per param. Adapted from Fig 2 in loss_scan.ipynb Cell 9,
    replacing (max-min)/min depth with SNR (fixes the sQ5p0 mis-flag)."""
    heatmaps = {
        "bias_pct":       "loss-min bias (% of truth)",
        "snr":            "SNR = range / SE(mean)",
        "frac_uf":        "frac_underflow @ truth",
        "mean_log_alpha": "mean_log_α @ truth",
    }
    params = _ordered_params({k[0] for k in curves})
    n_metrics = len(heatmaps)
    fig, axs = plt.subplots(len(params), n_metrics,
                            figsize=(3.5 * n_metrics, 3.2 * len(params)),
                            squeeze=False)
    best_by_param = {}

    def _at_target(pval, y, target):
        if y is None or np.all(np.isnan(y)):
            return np.nan
        i = int(np.argmin(np.abs(pval - target)))
        return float(y[i])

    for i_par, prm in enumerate(params):
        for i_m, key in enumerate(heatmaps):
            grid = np.full((len(sq_tags), len(ls_tags)), np.nan)
            for i_sq, sq in enumerate(sq_tags):
                for i_ls, ls in enumerate(ls_tags):
                    d = curves.get((prm, sq, ls))
                    if d is None:
                        continue
                    if key == "bias_pct":
                        j = int(np.nanargmin(d["loss"]))
                        grid[i_sq, i_ls] = 100 * (d["pval"][j] - d["target"]) / max(abs(d["target"]), 1e-30)
                    elif key == "snr":
                        rng = float(np.nanmax(d["loss"]) - np.nanmin(d["loss"]))
                        se = float(np.nanmedian(d["loss_se"])) if d["loss_se"] is not None else np.nan
                        grid[i_sq, i_ls] = rng / se if se and se > 0 else np.inf
                    elif key == "frac_uf":
                        grid[i_sq, i_ls] = _at_target(d["pval"], d["frac_underflow"], d["target"])
                    elif key == "mean_log_alpha":
                        grid[i_sq, i_ls] = _at_target(d["pval"], d["mean_log_alpha"], d["target"])
            ax = axs[i_par, i_m]
            im = ax.imshow(
                grid,
                cmap="RdBu_r" if key in ("bias_pct", "mean_log_alpha") else "viridis",
                aspect="auto", origin="lower",
            )
            if key == "bias_pct" and not np.all(np.isnan(grid)):
                m = np.nanmax(np.abs(grid))
                im.set_clim(-m, m)
            ax.set_xticks(range(len(ls_tags)))
            ax.set_xticklabels([s.replace("p", ".") for s in ls_tags])
            ax.set_yticks(range(len(sq_tags)))
            ax.set_yticklabels([s.replace("p", ".") for s in sq_tags])
            if i_par == 0:
                ax.set_title(heatmaps[key], fontsize=10)
            if i_m == 0:
                ax.set_ylabel(f"{prm}\n\nσ_Q (ke)")
            else:
                ax.set_ylabel("σ_Q (ke)")
            if i_par == len(params) - 1:
                ax.set_xlabel("α prior log-σ")
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    v = grid[i, j]
                    if np.isnan(v):
                        continue
                    if key == "bias_pct":
                        txt = f"{v:+.1f}"
                    elif key == "snr":
                        txt = f"{v:.1f}" if np.isfinite(v) else "∞"
                    elif key == "mean_log_alpha":
                        txt = f"{v:+.2f}"
                    else:
                        txt = f"{v:.3f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="black")
            plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
            # Track the best (sQ, ls) per param — minimum |bias| subject to SNR>=3.
            if key == "bias_pct":
                # We need SNR too; recompute below in one pass over grid.
                pass
        # After all metrics for this param, pick the sweep-best cell.
        bias_grid = np.full((len(sq_tags), len(ls_tags)), np.nan)
        snr_grid  = np.full_like(bias_grid, np.nan)
        for i_sq, sq in enumerate(sq_tags):
            for i_ls, ls in enumerate(ls_tags):
                d = curves.get((prm, sq, ls))
                if d is None:
                    continue
                j = int(np.nanargmin(d["loss"]))
                bias_grid[i_sq, i_ls] = 100 * (d["pval"][j] - d["target"]) / max(abs(d["target"]), 1e-30)
                rng = float(np.nanmax(d["loss"]) - np.nanmin(d["loss"]))
                se = float(np.nanmedian(d["loss_se"])) if d["loss_se"] is not None else np.nan
                snr_grid[i_sq, i_ls] = rng / se if se and se > 0 else np.inf
        # A cell is "trusted" if SNR >= 3; among trusted cells, pick smallest |bias|.
        trusted = snr_grid >= 3
        if np.any(trusted):
            b = np.where(trusted, np.abs(bias_grid), np.inf)
            k = int(np.nanargmin(b))
            i_sq, i_ls = np.unravel_index(k, b.shape)
            best_by_param[prm] = dict(
                sq=sq_tags[i_sq], ls=ls_tags[i_ls],
                bias=float(bias_grid[i_sq, i_ls]),
                snr=float(snr_grid[i_sq, i_ls]),
            )
    fig.suptitle(title, y=1.005)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return best_by_param


def write_sweep_report(sweep_label, points, out_dir):
    """`points` is a list of dicts, each with:
        - 'label'  (full scan_group_key)
        - 'sq_tag' / 'ls_tag'
        - 'entries' (list of per-param pickle entries)
    Renders the sweep-summary figures from loss_scan.ipynb Cell 9 into one
    report covering the whole (σ_Q × α_prior) grid, replacing the per-point
    scans."""
    curves = {}
    for pt in points:
        for e in pt["entries"]:
            d = _load_sweep_point(e["path"])
            if d is None:
                continue
            curves[(d["param"], pt["sq_tag"], pt["ls_tag"])] = d
    if not curves:
        return None

    sq_tags = sorted({k[1] for k in curves},
                     key=lambda s: float(s.replace("p", ".")))
    ls_tags = sorted({k[2] for k in curves},
                     key=lambda s: float(s.replace("p", ".")))

    # Sniff config from any one pickle.
    with open(points[0]["entries"][0]["path"], "rb") as f:
        r0 = pickle.load(f)
    cfg = vars(r0["config"]) if r0.get("config") is not None else {}
    hp = _hyperparams_from_kw(cfg.get("loss_fn_kw", {}))

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    curves_png = plots_dir / "sweep_loss_curves.png"
    heat_png = plots_dir / "sweep_heatmap.png"
    plot_sweep_curves(curves, sq_tags, ls_tags, curves_png,
                      f"σ_Q × α_prior sweep — loss curves ({hp.get('metric')})")
    best = plot_sweep_heatmap(curves, sq_tags, ls_tags, heat_png,
                              f"σ_Q × α_prior sweep — heatmaps ({hp.get('metric')})")

    data_mt = _pickle_mtime(
        e["path"] for p in points for e in p["entries"]
    )
    lines = []
    lines.append("---")
    lines.append(f"date: {datetime.now().isoformat(timespec='seconds')}")
    if data_mt:
        lines.append(f"data_mtime: {data_mt}")
    lines.append(f"run_type: hp_sweep")
    lines.append(f"metric: {hp.get('metric')}")
    lines.append(f"n_sigma_Q: {len(sq_tags)}")
    lines.append(f"n_alpha_prior: {len(ls_tags)}")
    lines.append("tags:")
    lines.append("  - larndsim")
    lines.append("  - hp_sweep")
    lines.append(f"  - {hp.get('metric', 'unknown')}")
    lines.append("---\n")

    lines.append(f"# D10 σ_Q × α_prior sweep — `{sweep_label[:100]}`\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Grid: σ_Q ∈ {{{', '.join(s.replace('p','.') for s in sq_tags)}}} ke  ×  "
                 f"α_prior log-σ ∈ {{{', '.join(s.replace('p','.') for s in ls_tags)}}}\n")

    lines.append("## Run condition\n")
    lines.append("| key | value |")
    lines.append("|---|---|")
    lines.append(f"| **run type** | hyperparameter sweep over (σ_Q, α_prior) |")
    lines.append(f"| loss family | `{cfg.get('loss_fn', '?')}` / `{hp.get('metric')}` |")
    lines.append(f"| probabilistic sim | {cfg.get('probabilistic_sim', '?')} |")
    lines.append(f"| batch (bt × nbt × iter) | {cfg.get('max_batch_len','?')} × "
                 f"{cfg.get('max_nbatch','?')} × {cfg.get('iterations','?')} |")
    lines.append(f"| dedx density mode | `{cfg.get('dedx_density_mode','?')}` / "
                 f"flow_expectation=`{cfg.get('flow_expectation_mode','?')}` |")
    lines.append(f"| normalization | `{cfg.get('normalization_scheme','?')}` |")
    lines.append("")

    lines.append("## Loss curves — all (σ_Q, α_prior) overlaid\n")
    lines.append(f"![]({curves_png.relative_to(out_dir).as_posix()})\n")

    lines.append("## Heatmaps — bias / SNR / α diagnostics vs hyperparameters\n")
    lines.append("Rows = params; columns = metrics. SNR ≥ 3 is the trust threshold "
                 "for treating bias as a real signal rather than batch noise.\n")
    lines.append(f"![]({heat_png.relative_to(out_dir).as_posix()})\n")

    lines.append("## Sweep-best (σ_Q, α_prior) per param\n")
    lines.append("Chosen as: smallest |bias| among cells with SNR ≥ 3.\n")
    lines.append("| param | best σ_Q (ke) | best α_prior | bias % | SNR |")
    lines.append("|---|---|---|---|---|")
    for prm in sorted(best):
        b = best[prm]
        lines.append(f"| `{prm}` | {b['sq'].replace('p','.')} | {b['ls'].replace('p','.')} "
                     f"| {b['bias']:+.2f} | {b['snr']:.1f} |")
    lines.append("")

    lines.append("## Source pickles (one per (σ_Q, α_prior) point)\n")
    for pt in sorted(points, key=lambda p: (p["sq_tag"], p["ls_tag"])):
        lines.append(f"- σ_Q={pt['sq_tag'].replace('p','.')} ls={pt['ls_tag'].replace('p','.')}: "
                     f"{len(pt['entries'])} params")

    (out_dir / "report.md").write_text("\n".join(lines))

    with open(out_dir / "metrics.csv", "w") as f:
        f.write("param,sigma_Q,alpha_prior,target,loss_min_at,bias_pct,snr\n")
        for (prm, sq, ls), d in sorted(curves.items()):
            j = int(np.nanargmin(d["loss"]))
            bias = 100 * (d["pval"][j] - d["target"]) / max(abs(d["target"]), 1e-30)
            rng = float(np.nanmax(d["loss"]) - np.nanmin(d["loss"]))
            se = float(np.nanmedian(d["loss_se"])) if d["loss_se"] is not None else np.nan
            snr = rng / se if se and se > 0 else np.inf
            f.write(f"{prm},{sq.replace('p','.')},{ls.replace('p','.')},{d['target']},"
                    f"{d['pval'][j]},{bias:.4f},{snr if np.isfinite(snr) else ''}\n")

    return out_dir / "report.md"


def write_scan_report(label, entries, out_dir):
    """One scan run report. `entries` is a list of dicts (from discover())."""
    # Load one representative pickle per param.
    per_param = {}
    for e in entries:
        if e["param"] not in per_param:
            per_param[e["param"]] = e
    parsed = []
    for e in per_param.values():
        p = load_scan_pickle(e["path"])
        if p is not None:
            parsed.append(p)
    if not parsed:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    loss_png = plots_dir / "scan_loss_curves.png"
    alpha_png = plots_dir / "scan_alpha_diagnostics.png"
    plot_scan_all(parsed, loss_png)  # combined loss + gradient (dual y-axis)
    alpha_ok = plot_scan_alpha(parsed, alpha_png)

    cfg = parsed[0]["config"]
    hp = _hyperparams_from_kw(cfg.get("loss_fn_kw", {}))
    obs = analyze_scan(parsed)

    data_mt = _pickle_mtime(e["path"] for e in entries)
    lines = []
    lines.append("---")
    lines.append(f"date: {datetime.now().isoformat(timespec='seconds')}")
    if data_mt:
        lines.append(f"data_mtime: {data_mt}")
    lines.append(f"run_type: scan")
    lines.append(f"config: {cfg.get('input_file_sim', '?').split('/')[-1][:60]}")
    lines.append(f"metric: {hp.get('metric')}")
    lines.append("tags:")
    lines.append("  - larndsim")
    lines.append("  - scan")
    lines.append(f"  - {hp.get('metric', 'unknown')}")
    lines.append("---\n")

    lines.append(f"# Scan report — `{label[:80]}`\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Pickles read: {len(parsed)} params · {len(entries)} files\n")

    lines.append("## Run condition\n")
    lines.append("| key | value |")
    lines.append("|---|---|")
    lines.append(f"| **run type** | scan (`--fit_type scan`) |")
    lines.append(f"| loss family | `{cfg.get('loss_fn', '?')}` / `{hp.get('metric')}` |")
    lines.append(f"| probabilistic sim | {cfg.get('probabilistic_sim', '?')} |")
    lines.append(f"| sim seed strategy | `{cfg.get('sim_seed_strategy', '?')}` |")
    lines.append(f"| batch (bt × nbt × iter) | {cfg.get('max_batch_len','?')} × "
                 f"{cfg.get('max_nbatch','?')} × {cfg.get('iterations','?')} |")
    lines.append(f"| dedx density mode | `{cfg.get('dedx_density_mode','?')}` / "
                 f"flow_expectation=`{cfg.get('flow_expectation_mode','?')}` |")
    lines.append(f"| normalization | `{cfg.get('normalization_scheme','?')}` |")
    lines.append("")

    lines.append("### D10 loss hyperparameters\n")
    lines.append("| knob | value |")
    lines.append("|---|---|")
    for k, v in hp.items():
        lines.append(f"| {k} | `{v}` |")
    lines.append("")

    lines.append("## Scan results per parameter\n")
    lines.append(f"![]({loss_png.relative_to(out_dir).as_posix()})\n")
    lines.append("SNR = (loss range across scan) / (typical SE of the batch-mean loss). "
                 "SNR ≥ 10 → minimum is well-resolved; 3–10 → marginal; < 3 → within noise.  "
                 "`grad zero @` = linear-interp location where the autodiff gradient crosses "
                 "zero; should match the loss minimum. Disagreement points to a stochastic-sim "
                 "discriminator (or another non-differentiable operation) breaking autodiff.\n")
    lines.append("| param | target | loss min @ | grad zero @ | loss-min bias | grad-zero bias | range | SE | SNR | α_prior @ tgt | frac_uf @ tgt |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in parsed:
        bias, snr, i_min = _scan_stats(p)
        rng = float(np.nanmax(p["loss"]) - np.nanmin(p["loss"]))
        se = p.get("loss_se")
        se_med = float(np.nanmedian(se)) if se is not None else np.nan
        i_t = int(np.argmin(np.abs(p["pval"] - p["target"])))
        pri = _series(p["aux"], "alpha_prior_term", p["nbt"])
        fu  = _series(p["aux"], "frac_underflow",    p["nbt"])
        pri_s = f"{pri[i_t]:.4g}" if pri is not None else "-"
        fu_s  = f"{fu[i_t]:.3f}"  if fu is not None else "-"
        snr_s = f"{snr:.1f}" if np.isfinite(snr) else "∞"
        gz_val = _grad_zero_crossing(p["pval"], p.get("grad"))
        if gz_val is not None:
            gz_s = f"{gz_val:.4g}"
            gz_bias_s = f"{100 * (gz_val - p['target']) / max(abs(p['target']), 1e-30):+.2f}"
        else:
            gz_s = "-"
            gz_bias_s = "-"
        lines.append(f"| `{p['param']}` | {p['target']:.4g} | {p['pval'][i_min]:.4g} | {gz_s} | "
                     f"{bias:+.2f} | {gz_bias_s} | {rng:.4g} | {se_med:.4g} | {snr_s} | "
                     f"{pri_s} | {fu_s} |")
    lines.append("")

    if alpha_ok:
        lines.append("## Per-track α diagnostics\n")
        lines.append(f"![]({alpha_png.relative_to(out_dir).as_posix()})\n")

    lines.append("## Analysis & suggestions\n")
    if not obs:
        lines.append("All parameters look healthy at this configuration. Bias is within noise, "
                     "depth is meaningful, α is not straining, no underflow issues.\n")
    else:
        by_sev = defaultdict(list)
        for sev, msg in obs:
            by_sev[sev].append(msg)
        for sev in ("HIGH", "MED", "LOW"):
            if sev in by_sev:
                icon = {"HIGH": "⚠️", "MED": "🟡", "LOW": "ℹ️"}[sev]
                lines.append(f"### {icon} {sev}\n")
                for m in by_sev[sev]:
                    lines.append(f"- {m}")
                lines.append("")

    lines.append("## What could improve this scan\n")
    lines.append("- If depth < 0.01 on any parameter: raise statistics (`BATCH_SIZE` or `MAX_NBATCH`)")
    lines.append("- If frac_underflow > 5%: widen σ_Q or switch metric to `marginal_llhd_prod`")
    lines.append("- If α prior dominates the loss at truth: loosen `per_track_scale_prior_log_sigma`")
    lines.append("- If bias is >20% and depth is meaningful: check for degeneracy — run a 2-D scan or free the correlated parameter")
    lines.append("- If lifetime/eField biased: check `shift_z` — could be reco-z systematic")
    lines.append("")

    lines.append("## Source pickles\n")
    for e in entries:
        lines.append(f"- `{os.path.basename(e['path'])}`")

    (out_dir / "report.md").write_text("\n".join(lines))

    # Metrics CSV
    with open(out_dir / "metrics.csv", "w") as f:
        f.write("param,target,loss_min_at,grad_zero_at,bias_pct,grad_zero_bias_pct,"
                "range,se,snr,grad_at_tgt,grad_se_at_tgt,alpha_prior_at_tgt,frac_uf_at_tgt\n")
        for p in parsed:
            bias, snr, i_min = _scan_stats(p)
            rng = float(np.nanmax(p["loss"]) - np.nanmin(p["loss"]))
            se_arr = p.get("loss_se")
            se_med = float(np.nanmedian(se_arr)) if se_arr is not None else ""
            i_t = int(np.argmin(np.abs(p["pval"] - p["target"])))
            pri = _series(p["aux"], "alpha_prior_term", p["nbt"])
            fu  = _series(p["aux"], "frac_underflow",    p["nbt"])
            # Gradient zero-crossing (interpolated) and gradient at target.
            g   = p.get("grad")
            gse = p.get("grad_se")
            gz_val = _grad_zero_crossing(p["pval"], g)
            if gz_val is not None:
                gz = f"{gz_val:.6g}"
                gz_bias = f"{100 * (gz_val - p['target']) / max(abs(p['target']), 1e-30):.4f}"
            else:
                gz = ""
                gz_bias = ""
            g_at_t   = f"{g[i_t]:.6g}"   if g   is not None else ""
            gse_at_t = f"{gse[i_t]:.6g}" if gse is not None else ""
            f.write(f"{p['param']},{p['target']},{p['pval'][i_min]},{gz},{bias:.4f},{gz_bias},"
                    f"{rng:.6g},{se_med},{snr if np.isfinite(snr) else ''},"
                    f"{g_at_t},{gse_at_t},"
                    f"{pri[i_t] if pri is not None else ''},{fu[i_t] if fu is not None else ''}\n")

    return out_dir / "report.md"


def write_fit_report(label, entries, out_dir):
    # Load the latest-iter pickle per seed (highest iter for each tgtsd).
    by_seed = defaultdict(list)
    for e in entries:
        m = re.search(r"_tgtsd(\d+)", e["label"])
        seed = int(m.group(1)) if m else 0
        by_seed[seed].append(e)
    fits = []
    for seed, es in by_seed.items():
        es_sorted = sorted(es, key=lambda e: -e["iter"])
        latest = es_sorted[0]
        f = load_fit_pickle(latest["path"])
        if f is not None:
            f["seed"] = seed
            fits.append(f)
    if not fits:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    traj_png = plots_dir / "fit_trajectories.png"
    loss_png = plots_dir / "fit_loss_trajectory.png"
    plot_fit_trajectories(fits, traj_png)
    plot_fit_loss(fits, loss_png)

    cfg = fits[0]["config"]
    hp = _hyperparams_from_kw(cfg.get("loss_fn_kw", {}))
    obs = analyze_fit(fits)

    data_mt = _pickle_mtime(e["path"] for e in entries)
    lines = []
    lines.append("---")
    lines.append(f"date: {datetime.now().isoformat(timespec='seconds')}")
    if data_mt:
        lines.append(f"data_mtime: {data_mt}")
    lines.append(f"run_type: fit")
    lines.append(f"config: {cfg.get('input_file_sim', '?').split('/')[-1][:60]}")
    lines.append(f"metric: {hp.get('metric')}")
    lines.append(f"n_seeds: {len(fits)}")
    lines.append("tags:")
    lines.append("  - larndsim")
    lines.append("  - fit")
    lines.append(f"  - {hp.get('metric', 'unknown')}")
    lines.append("---\n")

    lines.append(f"# Fit report — `{label[:80]}`\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Fit seeds: {sorted(f['seed'] for f in fits)}")
    lines.append(f"Params fitted: {sorted({p for f in fits for p in f['params']})}\n")

    lines.append("## Run condition\n")
    lines.append("| key | value |")
    lines.append("|---|---|")
    lines.append(f"| **run type** | fit (`--fit_type chain`) |")
    lines.append(f"| loss family | `{cfg.get('loss_fn', '?')}` / `{hp.get('metric')}` |")
    lines.append(f"| probabilistic sim | {cfg.get('probabilistic_sim', '?')} |")
    lines.append(f"| sim seed strategy | `{cfg.get('sim_seed_strategy', '?')}` |")
    lines.append(f"| batch (bt × nbt × iter) | {cfg.get('max_batch_len','?')} × "
                 f"{cfg.get('max_nbatch','?')} × {cfg.get('iterations','?')} |")
    lines.append(f"| LR / scheduler | {cfg.get('lr','?')} / `{cfg.get('lr_scheduler','?')}` |")
    lines.append(f"| dedx density mode | `{cfg.get('dedx_density_mode','?')}` / "
                 f"flow_expectation=`{cfg.get('flow_expectation_mode','?')}` |")
    lines.append(f"| normalization | `{cfg.get('normalization_scheme','?')}` |")
    lines.append("")

    lines.append("### D10 loss hyperparameters\n")
    lines.append("| knob | value |")
    lines.append("|---|---|")
    for k, v in hp.items():
        lines.append(f"| {k} | `{v}` |")
    lines.append("")

    lines.append("## Final parameter values (per seed, per-seed target)\n")
    lines.append("Each seed has its own target (drawn per fit via `--scan_tgt_nom`), so "
                 "bias is computed seed-by-seed against that seed's own truth. "
                 "`fitted` = mean of the last 500 iterations (matches `plot.ipynb`'s "
                 "`np.mean(history[param + '_iter'][-500:])`; falls back to last 10 % "
                 "for shorter trajectories).\n")
    all_params = _ordered_params({p for f in fits for p in f["params"]})
    fits_sorted = sorted(fits, key=lambda f: f["seed"])
    for prm in all_params:
        lines.append(f"### `{prm}`\n")
        lines.append("| seed | target | fitted | bias % |")
        lines.append("|---|---|---|---|")
        biases = []
        for f in fits_sorted:
            if prm not in f["trajectories"]:
                continue
            traj = np.asarray(f["trajectories"][prm]).ravel()
            if len(traj) == 0:
                continue
            final = _fit_final(traj)
            tgt = f["targets"].get(prm)
            if tgt is None:
                lines.append(f"| {f['seed']} | - | {final:.4g} | - |")
            else:
                b = 100 * (final - tgt) / max(abs(tgt), 1e-30)
                biases.append(b)
                lines.append(f"| {f['seed']} | {tgt:.4g} | {final:.4g} | {b:+.2f}% |")
        if biases:
            mean_b = float(np.nanmean(biases))
            std_b  = float(np.nanstd(biases))
            lines.append(f"| **mean** | — | — | **{mean_b:+.2f}% ± {std_b:.2f}%** |")
        lines.append("")

    lines.append("## Trajectories\n")
    lines.append(f"![]({traj_png.relative_to(out_dir).as_posix()})\n")
    lines.append("## Loss trajectory\n")
    lines.append(f"![]({loss_png.relative_to(out_dir).as_posix()})\n")

    lines.append("## Analysis & suggestions\n")
    if not obs:
        lines.append("Fit converged with all parameters within acceptable bias tolerances.\n")
    else:
        by_sev = defaultdict(list)
        for sev, msg in obs:
            by_sev[sev].append(msg)
        for sev in ("HIGH", "MED", "LOW"):
            if sev in by_sev:
                icon = {"HIGH": "⚠️", "MED": "🟡", "LOW": "ℹ️"}[sev]
                lines.append(f"### {icon} {sev}\n")
                for m in by_sev[sev]:
                    lines.append(f"- {m}")
                lines.append("")

    lines.append("## What could improve this fit\n")
    lines.append("- **Loss still moving at end** → raise `iterations`, or loosen LR decay (`decay_rate=0.999`)")
    lines.append("- **Correlated param biases in opposite directions** (e.g. Ab↑ + lifetime↓) → degeneracy along ΣQ direction")
    lines.append("  - Freeze one, refit the other")
    lines.append("  - Add `shift_z` to the fit (7-par or 8-par) to give reco-z systematic a proper home")
    lines.append("  - Loosen α prior to let α absorb more ΣQ scatter")
    lines.append("- **Same-sign bias across seeds** → systematic, not stochastic; needs a loss/parameter change")
    lines.append("- **Compare with the corresponding scan report** — 1-D scans of same hyperparameters isolate per-axis biases")
    lines.append("")

    lines.append("## Source pickles\n")
    for e in entries:
        lines.append(f"- `{os.path.basename(e['path'])}`")

    (out_dir / "report.md").write_text("\n".join(lines))

    # Metrics CSV — per-seed target (each fit has its own truth).
    with open(out_dir / "metrics.csv", "w") as f:
        f.write("param,seed,target,final,bias_pct\n")
        for prm in all_params:
            for fi in fits:
                if prm not in fi["trajectories"]:
                    continue
                traj = np.asarray(fi["trajectories"][prm]).ravel()
                if len(traj) == 0:
                    continue
                final = _fit_final(traj)
                tgt = fi["targets"].get(prm)
                bias = (100 * (final - tgt) / max(abs(tgt), 1e-30)) if tgt is not None else ""
                f.write(f"{prm},{fi['seed']},{tgt if tgt is not None else ''},{final},{bias}\n")

    return out_dir / "report.md"


# --------------------------------------------------------------------------
# Vault copy
# --------------------------------------------------------------------------

def copy_to_vault(src, vault):
    dest = Path(vault) / src.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return dest


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _safe_dir_name(s, maxlen=180):
    """Sanitize + truncate. Append an 8-char hash so distinct labels can't
    collide when truncation lops off the distinguishing suffix (e.g. the
    shz0p06 vs shz0p12 tail of a 2-D scan label)."""
    import hashlib
    safe = re.sub(r"[^\w.-]", "_", s)
    if len(safe) <= maxlen:
        return safe
    h = hashlib.md5(s.encode()).hexdigest()[:8]
    return safe[: maxlen - 9] + "_" + h


def _pickle_mtime(source_paths):
    """Return the most-recent mtime across a set of source pickle paths as an
    ISO-8601 string (second precision, local time). Empty / all-missing input
    returns None so callers can decide how to render the absence."""
    mtimes = []
    for p in source_paths:
        try:
            mtimes.append(os.path.getmtime(p))
        except OSError:
            continue
    if not mtimes:
        return None
    return datetime.fromtimestamp(max(mtimes)).isoformat(timespec="seconds")


def _read_frontmatter_field(report_path, key):
    """Peek the first ~20 lines of report.md for `key: value` inside the
    YAML frontmatter. Returns the raw value string, or None on absence."""
    try:
        with open(report_path) as f:
            for i, line in enumerate(f):
                if i > 25:
                    break
                if line.startswith(f"{key}:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _is_up_to_date(out_dir, source_paths):
    """A report is up-to-date if:
      - its directory exists,
      - `report.md` is present,
      - every source pickle's mtime is ≤ report.md's mtime.
    Add a new pickle (or `touch` an existing one) and the mtime check will
    fire; otherwise the report is reused unchanged."""
    report = out_dir / "report.md"
    if not report.exists():
        return False
    try:
        report_mtime = report.stat().st_mtime
    except OSError:
        return False
    for p in source_paths:
        try:
            if os.path.getmtime(p) > report_mtime:
                return False
        except OSError:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG_FILTER_DEFAULT,
                    help="config-string filter (default: A2-B5-C2-D10)")
    ap.add_argument("--filter", default="",
                    help="additional substring filter on the label")
    ap.add_argument("--scan-dir",   default=str(BASE / "fit_result" / "scan"))
    ap.add_argument("--fit-dir",    default=str(BASE / "fit_result" / "fit_noise"))
    ap.add_argument("--out-dir",    default=str(BASE / "reports"))
    ap.add_argument("--vault",      default=None, help="Obsidian vault path")
    ap.add_argument("--skip-scan",  action="store_true")
    ap.add_argument("--skip-fit",   action="store_true")
    ap.add_argument("--force",      action="store_true",
                    help="Regenerate all reports even if they exist and are up-to-date.")
    args = ap.parse_args()

    scan_groups, fit_groups = discover(
        [args.scan_dir, args.fit_dir], args.config,
    )

    if args.filter:
        scan_groups = {k: v for k, v in scan_groups.items() if args.filter in k}
        fit_groups  = {k: v for k, v in fit_groups.items()  if args.filter in k}

    print(f"Discovered {len(scan_groups)} scan run(s), {len(fit_groups)} fit run(s) "
          f"matching config='{args.config}' filter='{args.filter}'")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    written = []          # dirs (re)written in THIS invocation
    skipped = []          # dirs left alone because they were up-to-date
    all_dirs = []         # every existing report dir after this run (for index/vault)

    # Group scan_groups into sweep families. A sweep family with >1
    # (σ_Q, α_prior) point collapses into a single sweep-summary report;
    # families with exactly one point stay as individual scan / 2D-scan
    # reports.
    sweep_families = defaultdict(list)
    if not args.skip_scan:
        for label, entries in scan_groups.items():
            sk = sweep_group_key(label)
            sq, ls = _extract_sq_ls(label)
            sweep_families[sk].append(dict(
                label=label, entries=entries, sq_tag=sq, ls_tag=ls,
            ))

        sweep_labels = set()   # scan-group labels absorbed into a sweep report
        for sk, points in sorted(sweep_families.items()):
            # A point contributes to the sweep only if it has both sQ and ls tags
            # AND is a 1-D scan (2-D scans have their own report).
            valid = [p for p in points
                     if p["sq_tag"] and p["ls_tag"] and not is_2d_scan(p["label"])]
            uniq = {(p["sq_tag"], p["ls_tag"]) for p in valid}
            if len(uniq) < 2:
                continue   # not a sweep — fall through to per-label rendering
            out_dir = out_root / f"sweep__{_safe_dir_name(sk)}"
            sources = [e["path"] for p in valid for e in p["entries"]]
            for p in valid:
                sweep_labels.add(p["label"])
            all_dirs.append(out_dir)
            if not args.force and _is_up_to_date(out_dir, sources):
                skipped.append(out_dir)
                continue
            try:
                path = write_sweep_report(sk, valid, out_dir)
                if path:
                    print(f"  sweep report → {path}")
                    written.append(out_dir)
            except Exception as e:
                print(f"  ⚠  sweep report failed for {sk[:60]}: {e}")

        for label, entries in sorted(scan_groups.items()):
            if label in sweep_labels:
                continue
            if is_2d_scan(label):
                out_dir = out_root / f"scan2d__{_safe_dir_name(label)}"
                writer = write_2d_scan_report
                kind = "2-D scan"
            else:
                out_dir = out_root / f"scan__{_safe_dir_name(label)}"
                writer = write_scan_report
                kind = "scan"
            sources = [e["path"] for e in entries]
            all_dirs.append(out_dir)
            if not args.force and _is_up_to_date(out_dir, sources):
                skipped.append(out_dir)
                continue
            try:
                path = writer(label, entries, out_dir)
                if path:
                    print(f"  {kind} report → {path}")
                    written.append(out_dir)
            except Exception as e:
                print(f"  ⚠  {kind} report failed for {label[:60]}: {e}")

    if not args.skip_fit:
        for label, entries in sorted(fit_groups.items()):
            out_dir = out_root / f"fit__{_safe_dir_name(label)}"
            sources = [e["path"] for e in entries]
            all_dirs.append(out_dir)
            if not args.force and _is_up_to_date(out_dir, sources):
                skipped.append(out_dir)
                continue
            try:
                path = write_fit_report(label, entries, out_dir)
                if path:
                    print(f"  fit report  → {path}")
                    written.append(out_dir)
            except Exception as e:
                print(f"  ⚠  fit report failed for {label[:60]}: {e}")

    # Existing report dirs on disk — used both here and by the index/vault
    # steps below.
    existing_dirs = [p for p in out_root.iterdir()
                     if p.is_dir() and (p / "report.md").exists()]
    print(f"\nUpdated {len(written)} report(s); skipped {len(skipped)} "
          f"already-up-to-date. Total on disk: {len(existing_dirs)}. "
          f"Use --force to rebuild everything.")

    if args.vault:
        vault = Path(args.vault).expanduser()
        vault.mkdir(parents=True, exist_ok=True)
        # Sync every report dir on disk, not just the ones (re)written this
        # run. Otherwise a fresh vault (or one that missed previous runs)
        # ends up with an index.md whose wiki-links point at report dirs
        # that don't exist locally. To keep the sync fast, only re-copy a
        # directory if the source `report.md` is newer than the vault copy
        # (or the vault copy is missing).
        n_copied = 0
        n_kept = 0
        for d in [p for p in out_root.iterdir()
                  if p.is_dir() and (p / "report.md").exists()]:
            dest = vault / d.name
            src_report = d / "report.md"
            dest_report = dest / "report.md"
            try:
                needs_sync = (
                    not dest_report.exists()
                    or src_report.stat().st_mtime > dest_report.stat().st_mtime
                )
            except OSError:
                needs_sync = True
            if needs_sync:
                copy_to_vault(d, vault)
                n_copied += 1
            else:
                n_kept += 1
        print(f"  vault sync  → {n_copied} copied, {n_kept} unchanged "
              f"(at {vault})")

    # Top-level index — enumerate every report directory currently on disk
    # (not just the ones (re)written this run) so incremental updates don't
    # accidentally drop old entries from the index. Each entry shows the
    # most-recent source pickle mtime (from the report's YAML frontmatter),
    # so you can see at a glance when a run was produced — sorted
    # newest-first within each section.
    idx = out_root / "index.md"
    existing = existing_dirs
    lines = [f"# larndsim run index — generated {datetime.now().isoformat(timespec='seconds')}\n"]
    def _mtime_key(d):
        mt = _read_frontmatter_field(d / "report.md", "data_mtime")
        # Fall back on report.md's own mtime for legacy reports that
        # predate the frontmatter field.
        if mt is None:
            try:
                return datetime.fromtimestamp(
                    (d / "report.md").stat().st_mtime
                ).isoformat(timespec="seconds")
            except OSError:
                return "0000-00-00T00:00:00"
        return mt
    def _section(title, prefix):
        matched = [d for d in existing if d.name.startswith(prefix)]
        if not matched:
            return
        lines.append(f"\n## {title}\n")
        # Newest data first — reversed lexicographic sort of ISO-8601 mtime.
        for d in sorted(matched, key=lambda dd: _mtime_key(dd), reverse=True):
            mt = _mtime_key(d)
            lines.append(f"- `{mt}` — [[{d.name}/report|{d.name[len(prefix):][:100]}]]")
    _section("σ_Q × α_prior sweeps", "sweep__")
    _section("2-D scans",           "scan2d__")
    _section("1-D scans",           "scan__")
    _section("Fits",                "fit__")
    idx.write_text("\n".join(lines))
    print(f"\nIndex → {idx}")
    if args.vault:
        vault_idx = Path(args.vault).expanduser() / "index.md"
        vault_idx.write_text("\n".join(lines))
        print(f"Vault index → {vault_idx}")


if __name__ == "__main__":
    main()
