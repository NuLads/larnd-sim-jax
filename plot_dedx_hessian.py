#!/usr/bin/env python3
"""Illustration plots for the dE/dx Hessian structure measurements.

Sources (measured 2026-07-30, turing; see optimize/scripts/start_dedx_hessian.sh):
  fit_result/dedx_hessian_b1_b1.pkl      batch 1, 13 segments, FULL Hessian, production weights
  fit_result/dedx_hessian_b0_meanw0.pkl  batch 0, 11 segments, FULL, mean-constraint weight = 0
  fit_result/dedx_hessian_b0_c1.pkl      batch 0, 11 segments, 6 of 11 columns, production weights

Decomposition under test:  H = H_data + diag(student-t + barrier) + w_mean * q q^T
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

# ---- palette (dataviz reference instance) ----------------------------------
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK2      = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
SEQ_BLUE  = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SERIES    = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
             "yellow": "#eda100", "violet": "#4a3aa7", "red": "#e34948"}
CMAP_SEQ  = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
CMAP_SEQ.set_under("#f0efec")      # exact zero -> neutral gray midpoint
CMAP_SEQ.set_bad("#d9d8d2")        # unmeasured -> darker neutral

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.titlecolor": INK,
})


def load(tag):
    d = pickle.load(open("fit_result/%s.pkl" % tag, "rb"))
    n, ci = int(d["n_segments"]), list(d["col_indices"])
    Ht = np.full((n, n), np.nan)
    Hp = np.full((n, n), np.nan)
    for k in ci:
        Ht[:, k] = np.asarray(d["cols_total"][k])
        Hp[:, k] = np.asarray(d["cols_prior"][k])
    return d, Ht, Hp, ci


def rank1_from_offdiag(Hp):
    """Extract q with Hp_offdiag = q q^T, using a well-conditioned reference row."""
    n = Hp.shape[0]
    off = Hp - np.diag(np.diag(Hp))
    r = int(np.nanargmax(np.abs(np.nansum(off, axis=1))))
    s, t = [j for j in range(n) if j != r][:2]
    qr = np.sqrt(off[r, s] * off[r, t] / off[s, t])
    q = off[:, r] / qr
    q[r] = qr
    return q


EPS32 = np.finfo(np.float32).eps  # 1.19e-7


def cancellation_floor(Ht, Hp):
    """Absolute float32 noise incurred by forming H_data = H_total - H_prior.

    Where the rank-1 mean term dominates, H_total and H_prior are both ~1e4 and
    their difference loses ~1e4*eps of absolute precision. With w_mean = 0 the
    prior off-diagonal is exactly 0, so no cancellation occurs at all.
    """
    return 2.0 * EPS32 * np.maximum(np.abs(Ht), np.abs(Hp))


def logmap(M, ref, floor=-8.0, unresolved=None):
    """log10(|M| / ref).

    Exact zeros are pushed below the colormap's 'under' bound; entries flagged
    `unresolved` (below the float32 cancellation floor) are masked as 'bad'.
    """
    out = np.full(M.shape, np.nan)
    ok = np.isfinite(M)
    with np.errstate(divide="ignore"):
        out[ok] = np.log10(np.abs(M[ok]) / ref)
    out[ok & (np.abs(M) == 0)] = floor - 1.0
    out = np.ma.masked_invalid(out)
    if unresolved is not None:
        out = np.ma.masked_where(unresolved & (np.abs(M) != 0), out)
    return out


def draw_matrix(ax, M, ref, title, vmin=-8, vmax=0, unresolved=None):
    im = ax.imshow(logmap(M, ref, vmin, unresolved), cmap=CMAP_SEQ,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9, pad=6)
    n = M.shape[0]
    ax.set_xticks(range(0, n, 4)); ax.set_yticks(range(0, n, 4))
    ax.tick_params(length=2)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    return im


# ============================================================================
# Figure 1 — the three-term decomposition (batch 1, full 13x13)
# ============================================================================
d1, Ht1, Hp1, _ = load("dedx_hessian_b1_b1")
n1 = Ht1.shape[0]
Hdata1 = Ht1 - Hp1                       # likelihood term
q1 = rank1_from_offdiag(Hp1)
Hmean1 = np.outer(q1, q1)                # w * q q^T  (w already folded in)
Hdiag1 = np.diag(np.diag(Hp1 - Hmean1))  # student-t + soft barrier
ref1 = np.nanmax(np.abs(Ht1))

unres1 = np.abs(Hdata1) < cancellation_floor(Ht1, Hp1)

fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.25))
panels = [
    (Ht1,     "Total  $H$\ndense — nothing obvious to exploit", None),
    (Hdata1,  "Data / likelihood\nstrong band $|i-j|\\leq2$", unres1),
    (Hdiag1,  "Student-t + barrier\nexactly diagonal", None),
    (Hmean1,  "Mean constraint\nexactly rank-1  $w\\,qq^{\\!\\top}$", None),
]
for ax, (M, t, u) in zip(axes, panels):
    im = draw_matrix(ax, M, ref1, t, unresolved=u)
    ax.set_xlabel("segment $j$")
axes[0].set_ylabel("segment $i$")

cb = fig.colorbar(im, ax=axes, fraction=0.021, pad=0.015, extend="min")
cb.set_label("$\\log_{10}\\,|H_{ij}| \\,/\\, \\max|H|$", color=INK2)
cb.outline.set_edgecolor(GRID)
cb.ax.tick_params(color=MUTED, labelcolor=MUTED, length=2)
cb.ax.text(0.5, -0.055, "exact 0", transform=cb.ax.transAxes, ha="center",
           va="top", fontsize=7.5, color=INK2)

fig.suptitle("dE/dx Hessian decomposes into banded + diagonal + rank-1  "
             "(batch 1, 13 segments, 43.1 cm, production weights)",
             fontsize=10.5, y=1.0, color=INK)
fig.text(0.5, 0.045,
         "$H = H_{\\rm data} + \\mathrm{diag}(\\text{student-}t+\\text{barrier}) "
         "+ w_{\\rm mean}\\, q q^{\\!\\top}$   "
         "$\\Rightarrow$  O(N) solve: bandwidth-2 banded LU + one Sherman–Morrison correction",
         ha="center", fontsize=9, color=INK2)
fig.text(0.5, -0.115,
         "Panel 2: darker gray = below the float32 cancellation floor of "
         "$H_{\\rm total}-H_{\\rm prior}$ (both $\\sim\\!10^4$ here, so $\\sim\\!10^{-3}$ absolute "
         "precision is lost).\nAt production weights this batch cannot resolve the band tail at all "
         "— see the $w_{\\rm mean}\\!=\\!0$ control for that.",
         ha="center", fontsize=7.5, color=INK2)
fig.savefig("plots/fig_dedx_hessian_decomposition.png", dpi=200,
            bbox_inches="tight")
plt.close(fig)

# ============================================================================
# Figure 2 — quantitative evidence: band decay + rank-1 exactness
# ============================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.2, 3.9))

# --- left: |H_data| vs |i-j| -------------------------------------------------
# Batch 0 with w_mean = 0 is the ONLY trustworthy probe of the band tail: with the
# rank-1 term switched off, H_prior's off-diagonal is exactly 0, so H_data = H_total
# involves no cancellation. Batch 1 at production weights subtracts two ~1e4 numbers.
ZLINE = 3e-11
for tag, lbl, col, mk in [("dedx_hessian_b0_meanw0",
                           "batch 0, $w_{\\rm mean}\\!=\\!0$  — no cancellation", SERIES["orange"], "s"),
                          ("dedx_hessian_b1_b1",
                           "batch 1, $w_{\\rm mean}\\!=\\!10^5$ — cancellation-limited", SERIES["blue"], "o")]:
    d, Ht, Hp, ci = load(tag)
    Hd = Ht - Hp
    n = Hd.shape[0]
    dref = np.nanmax(np.abs(np.diag(Hd)))
    flr = cancellation_floor(Ht, Hp) / dref
    seps, vals, res = [], [], []
    for j in ci:
        for i in range(n):
            if np.isfinite(Hd[i, j]):
                seps.append(abs(i - j)); vals.append(abs(Hd[i, j]) / dref)
                res.append(abs(Hd[i, j]) / dref > flr[i, j])
    seps, vals, res = np.array(seps), np.array(vals), np.array(res)
    jit = 0.10 * (1 if mk == "o" else -1)
    nz = vals > 0
    axL.scatter(seps[nz & res] + jit, vals[nz & res], s=17, color=col, alpha=0.8,
                marker=mk, linewidths=0, label=lbl, zorder=3)
    axL.scatter(seps[nz & ~res] + jit, vals[nz & ~res], s=34, facecolors="none",
                edgecolors=col, marker=mk, linewidths=1.2, zorder=4)
    axL.scatter(seps[~nz] + jit, np.full((~nz).sum(), ZLINE), s=24, marker="x",
                color=col, linewidths=1.1, zorder=3)

# batch-1 cancellation floor, as a relative band
d1b, Ht1b, Hp1b, ci1b = load("dedx_hessian_b1_b1")
f1 = np.nanmedian(cancellation_floor(Ht1b, Hp1b)[np.isfinite(Ht1b)]) / \
     np.nanmax(np.abs(np.diag(Ht1b - Hp1b)))
axL.axhspan(ZLINE / 3, f1, color="#f0efec", zorder=0)
axL.axhline(f1, color=SERIES["blue"], lw=1, ls=(0, (4, 3)), zorder=2, alpha=0.8)
axL.text(6.5, f1 * 1.35, "batch 1 cancellation floor", fontsize=7,
         color=SERIES["blue"], ha="right", va="bottom")
axL.text(-0.45, ZLINE * 1.5, "exact zero  ($\\times$)", fontsize=7.5, color=INK2,
         ha="left", va="bottom")
axL.annotate("real, symmetric\ncouplings", xy=(3.0, 3.1e-7), xytext=(4.3, 8e-5),
             fontsize=7.5, color=INK2, ha="left",
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
axL.text(6.5, 1.1e-6, "hollow = below that floor\n(unresolvable, not measured zero)",
         fontsize=7, color=INK2, ha="right", va="top")
axL.set_yscale("log")
axL.set_ylim(1.0e-11, 3)
axL.set_xlim(-0.6, 6.6)
axL.set_xticks(range(0, 7))
axL.set_xlabel("segment separation  $|i-j|$")
axL.set_ylabel("$|H^{\\rm data}_{ij}| \\,/\\, \\max_i H^{\\rm data}_{ii}$")
axL.set_title("Strong band to $|i-j|\\leq2$, sparse real tail to $|i-j|=4$",
              fontsize=9.5, pad=6)
axL.grid(axis="y", color=GRID, lw=0.6)
axL.set_axisbelow(True)
leg = axL.legend(frameon=False, fontsize=8, loc="upper right")
for txt in leg.get_texts():
    txt.set_color(INK2)
for sp in axL.spines.values():
    sp.set_color(GRID)
axL.spines["top"].set_visible(False); axL.spines["right"].set_visible(False)

# --- right: rank-1 exactness -------------------------------------------------
off1 = Hp1 - np.diag(np.diag(Hp1))
mask = ~np.eye(n1, dtype=bool)
pred = np.outer(q1, q1)[mask]
obs = off1[mask]
axR.scatter(pred, obs, s=18, color=SERIES["violet"], alpha=0.7, linewidths=0,
            zorder=3, label="prior off-diagonal  $H^{\\rm prior}_{ij},\\ i\\neq j$")
lim = [min(pred.min(), obs.min()) * 0.7, max(pred.max(), obs.max()) * 1.4]
axR.plot(lim, lim, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2,
         label="$q_i q_j$  (exact rank-1)")
axR.set_xscale("log"); axR.set_yscale("log")
axR.set_xlim(lim); axR.set_ylim(lim)
axR.set_xlabel("rank-1 prediction  $q_i q_j$")
axR.set_ylabel("measured  $H^{\\rm prior}_{ij}$")
rel = np.abs(pred - obs) / np.abs(obs).max()
axR.set_title("Mean constraint is exactly rank-1\n"
              "max rel. residual %.1e (float32 noise)" % rel.max(),
              fontsize=9.5, pad=6)
axR.grid(color=GRID, lw=0.6); axR.set_axisbelow(True)
leg = axR.legend(frameon=False, fontsize=8, loc="upper left")
for txt in leg.get_texts():
    txt.set_color(INK2)
for sp in axR.spines.values():
    sp.set_color(GRID)
axR.spines["top"].set_visible(False); axR.spines["right"].set_visible(False)

axR.text(0.97, 0.06,
         "156 off-diagonal entries reproduced by 13 numbers.\n"
         "Batch-0 $w_{\\rm mean}\\!:\\,0\\!\\to\\!10^5$ control confirms the full form\n"
         "$2w\\,gg^{\\!\\top} + 2w(m\\!-\\!t)\\,\\mathrm{diag}(g)$ — mean dE/dx recovered\n"
         "two independent ways: 1.873995 vs 1.874004",
         transform=axR.transAxes, ha="right", va="bottom", fontsize=7,
         color=INK2)

fig.tight_layout()
fig.savefig("plots/fig_dedx_hessian_structure.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ============================================================================
# Figure 3 — the mean constraint is the sole source of dense coupling
# ============================================================================
d0w, Ht0w, Hp0w, ci0w = load("dedx_hessian_b0_meanw0")
d0p, Ht0p, Hp0p, ci0p = load("dedx_hessian_b0_c1")

fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5),
                         gridspec_kw={"width_ratios": [1, 1, 1.45]})

ref0 = np.nanmax(np.abs(Ht0p))
draw_matrix(axes[0], Hp0w, ref0,
            "$w_{\\rm mean}=0$\nprior off-diagonal exactly 0")
im = draw_matrix(axes[1], Hp0p, ref0,
                 "$w_{\\rm mean}=10^{5}$ (production)\ndense")
for ax in axes[:2]:
    ax.set_xlabel("segment $j$")
axes[0].set_ylabel("segment $i$")
axes[1].text(0.5, -0.30, "gray columns = not measured (6 of 11 computed)",
             transform=axes[1].transAxes, ha="center", fontsize=7.5, color=INK2)

# eigenvalue spectra
axE = axes[2]
for M, lbl, col, mk in [(Ht0w, "batch 0, $w_{\\rm mean}=0$", SERIES["aqua"], "s"),
                        (Ht1, "batch 1, $w_{\\rm mean}=10^{5}$", SERIES["blue"], "o")]:
    Hs = 0.5 * (M + M.T)
    ev = np.linalg.eigvalsh(Hs)[::-1]
    axE.plot(range(1, len(ev) + 1), ev, marker=mk, ms=4.5, lw=1.6, color=col,
             label="%s   $\\kappa$=%.0f" % (lbl, np.linalg.cond(Hs)), zorder=3)
axE.set_yscale("log")
axE.set_xlabel("eigenvalue index (descending)")
axE.set_ylabel("eigenvalue")
axE.set_title("The rank-1 term sets the conditioning", fontsize=9.5, pad=6)
axE.grid(color=GRID, lw=0.6); axE.set_axisbelow(True)
leg = axE.legend(frameon=False, fontsize=8, loc="lower left")
for txt in leg.get_texts():
    txt.set_color(INK2)
for sp in axE.spines.values():
    sp.set_color(GRID)
axE.spines["top"].set_visible(False); axE.spines["right"].set_visible(False)
ev1 = np.linalg.eigvalsh(0.5 * (Ht1 + Ht1.T))[::-1]
axE.annotate("rank-1 mean\nconstraint", xy=(1, ev1[0]), xytext=(3.4, ev1[0] * 0.9),
             fontsize=7.5, color=INK2, va="center",
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))

fig.suptitle("The mean-dE/dx constraint is the sole source of dense coupling — "
             "and of the ill-conditioning", fontsize=10.5, y=1.02, color=INK)
fig.tight_layout()
fig.savefig("plots/fig_dedx_hessian_meanconstraint.png", dpi=200,
            bbox_inches="tight")
plt.close(fig)

print("wrote plots/fig_dedx_hessian_decomposition.png")
print("wrote plots/fig_dedx_hessian_structure.png")
print("wrote plots/fig_dedx_hessian_meanconstraint.png")
