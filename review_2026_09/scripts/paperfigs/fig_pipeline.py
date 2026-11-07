"""Paper figure: block diagram of the realistic-calibration pipeline (subsec:probabilistic_loss).

Replaces fig:flow_chart (figures/flow_chart_lads.png), which is a project-status slide
("Establish the data-like calibration pipeline", work-package numbers and a Done / In
progress / To be started colour code) rather than a paper figure. This script draws the
actual pipeline described in Sec. IV of main.tex instead, with no status colour code.

It is deliberately the *concrete* companion to the *conceptual* fig:reco_calibration_flow
(the small TikZ figure in subsec:problem_setup, main.tex lines 794-813): that figure shows
the abstract roles of a generic simulator f(chi, theta) in a closure test (chi = chi_true)
versus a realistic fit (chi = chi_reco); this one names the actual boxes -- stochastic vs.
probabilistic simulation, the PPP likelihood, the three annealed Adam blocks -- and reuses
its theta_target / theta_i notation so the two figures read as a pair. It does not repeat
the closure-vs-realistic framing.

Content, grounded in main.tex (section "Realistic calibration with reconstructed inputs",
sec:results):
  - data-equivalent branch (tier 1, top row): true energy-deposit segments ("edep-sim", line
    229: "The simulation chain takes energy-deposit segments...") at theta_target -> STOCHASTIC
    simulation with electronics noise thrown explicitly (subsec:sim_setup, line 826) ->
    target hits d (pixel, tick, charge).
  - fit-input branch (tier 1, bottom row): selection (line 857, item iv: "the track selection
    is applied to true positions") -> fit input, initialised not measured: nominal axis = the
    straight line between the (still-true, line 857 item i / abstract line 147: "The track
    endpoints are still taken from the simulation") endpoints; per-segment dE/dx initialised
    to the track average (also a truth input, line 840 and the fig:event_display caption);
    trajectory shape seeded from the track's own target hits, drift coordinate held fixed
    for tracks whose hits look straight (subsec:muon_reco_input, lines 838-848). The dagger
    marks the still-truth-derived quantities explicitly listed as open points at line 857.
  - fit loop (tier 2, dashed container): the probabilistic (differentiable) simulation of
    Sec. sec:electronics evaluated at the current theta/dE/dx/trajectory estimate returns,
    per pixel/hit-slot/tick, a hit-time distribution and expected charge (subsec:reduction:
    "the beam search returns ... a log-probability P(t|j,n) ... together with the expected
    charges"); compared against the target hits via the Poisson point-process negative
    log-likelihood of Eq. (ppp_nll_full) plus the nuisance priors of subsec:nuisance
    (L_tot = L_hit + L_dE/dx + L_bar + L_mean + L_drift + L_MCS); the gradient drives Adam
    updates of the three parameter blocks -- calibration theta, per-segment dE/dx, track
    shape -- each with its own annealed learning rate (subsec:optimization, lines 1001-1007);
    iterate over batches (of order 4e3 segments/batch, sim_setup line 822).
  - outputs (right): the calibration parameters theta as the primary product, plus the
    reconstructed trajectories and per-segment energy losses as a by-product (abstract,
    line 147: "As a by-product, the fit reconstructs every trajectory...").
  - optional comparison mode (small dashed box inside the loop): replacing the probabilistic
    simulation + PPP likelihood by the STOCHASTIC simulation (the same generative model used
    for the target branch) plus a distance-based loss (MMD) -- subsec:nuisance, paragraph
    "The hit term in stochastic mode", lines 996-997.

Style: matches the house convention of the sibling scripts in this directory (fig1_noise_effect.py,
fig2_threshold.py, fig_convergence.py, fig_initialisation.py, ...): font.size 8, sans-serif,
Okabe-Ito colorblind-safe palette, figsize width 7.0 in for a full-width (figure*) figure.

Geometry is not hand-tuned. A helper measures each label's actual rendered extent (through
matplotlib's own mathtext layout, so $...$ spans are measured correctly rather than as
literal characters) and sizes every box to fit its own text, enforcing a body/title font
floor of 7.3/7.9 pt (the draft's ">=7 pt at print size" requirement). The pipeline's natural
single-chain width (truth -> selection -> fit input -> current estimate -> sim -> likelihood
-> update -> outputs) does not fit 7 in at that font floor, so the figure is split into two
tiers (one-time setup above, the per-batch loop below): the loop keeps the usual rectangle
(current-estimate/sim share a column, likelihood/update share the other), with the two
tier-1 -> tier-2 links drawn as a plain arrow where the geometry allows (target hits -> the
likelihood) and as an explicit 3-segment elbow where a straight line would cut through the
"probabilistic simulation" box (fit input -> current estimate).

Usage:
    python fig_pipeline.py
Writes:
    /sdf/group/neutrino/pgranger/NuLads_draft/figures/pipeline.pdf
    /sdf/group/neutrino/pgranger/NuLads_draft/figures/pipeline.png  (300 dpi)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path
from matplotlib.font_manager import FontProperties

OUT_DIR = "/sdf/group/neutrino/pgranger/NuLads_draft/figures"

# ----------------------------------------------------------------------
# House style (Okabe-Ito colorblind-safe palette; shared with sibling scripts)
# ----------------------------------------------------------------------
BLUE = "#0072B2"    # simulation / forward model
VERM = "#D55E00"    # data / targets
GREEN = "#009E73"   # fitted / optimised quantities
SKY = "#56B4E9"      # muted variant of blue; optional/alternative path only
GREY = "#595959"    # reconstruction inputs / bookkeeping steps
INK = "#1A1A1A"      # the objective itself (operator node, not a "quantity")

plt.rcParams.update({
    "font.size": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "font.family": "sans-serif",
})

FS_BODY, FS_TITLE = 7.3, 7.9     # >= 7 pt floor required for the draft
FS_EDGE = 7.0                     # arrow / loop-container labels
FIGW = 7.0                        # full-width (figure*) column
GAP = 0.18

# ----------------------------------------------------------------------
# Text measurement (mathtext-aware: widths/heights match what's actually drawn)
# ----------------------------------------------------------------------
_meas_fig = plt.figure()
_renderer = _meas_fig.canvas.get_renderer()


def text_size_in(s, fontsize, weight="normal"):
    fp = FontProperties(family="sans-serif", weight=weight, size=fontsize)
    w, h, d = _renderer.get_text_width_height_descent(s, fp, ismath=True)
    return w / _meas_fig.dpi, h / _meas_fig.dpi


def box_size(lines, min_w=0.0, min_h=0.0, pad_x=0.13, pad_y=0.08, line_gap=0.045):
    """lines: [(text, bold), ...]. Returns the (w, h) in inches needed to fit them."""
    w, h = 0.0, 0.0
    for text, bold in lines:
        fs = FS_TITLE if bold else FS_BODY
        tw, th = text_size_in(text, fs, "bold" if bold else "normal")
        w = max(w, tw)
        h += th + line_gap
    h -= line_gap
    return max(w + 2 * pad_x, min_w), max(h + 2 * pad_y, min_h)


# ----------------------------------------------------------------------
# Node content (edit here; geometry below is derived, not hand-picked)
# ----------------------------------------------------------------------
NODES = {
    "T1": dict(lines=[("True energy-deposit", True), ("segments (edep-sim)", False)],
               color=GREY, min_w=1.35),
    "T2": dict(lines=[("Stochastic", True), ("simulation", True),
                       ("at $\\theta_{\\rm target}$, + noise", False)],
               color=BLUE),
    "T3": dict(lines=[("Target hits $d$", True), ("(pixel, tick, charge)", False)],
               color=VERM),
    "B2": dict(lines=[("Fit input (initial)", True),
                       ("axis: straight, endpoints$^\\dagger$", False),
                       ("$dE/dx$: track mean$^\\dagger$", False),
                       ("shape: hit-seeded", False)],
               color=GREY),
    "L1": dict(lines=[("Current estimate", True),
                       ("$\\theta,\\ dE/dx_k,\\ $shape$_k$", False)],
               color=GREEN),
    "L2": dict(lines=[("Probabilistic", True), ("simulation", True),
                       ("$\\Lambda_j(t;\\theta),\\ \\langle Q\\rangle$", False)],
               color=BLUE),
    "L4": dict(lines=[("PPP likelihood", True), ("$\\mathcal{L}(\\theta)$ + priors", False)],
               color=INK),
    "L5": dict(lines=[("Adam update", True), ("$\\theta$, $dE/dx$, shape", False),
                       ("(annealed)", False)],
               color=GREEN),
    "O1": dict(lines=[("Calibration", True), ("$\\hat\\theta = \\arg\\min\\,\\mathcal{L}$", False)],
               color=GREEN, lw=1.8),
    "O2": dict(lines=[("By-product", True), ("trajectories, $dE/dx$", False)],
               color=GREEN, lw=1.0),
    "ALT": dict(lines=[("(alt.) stochastic sim. + MMD distance", False),
                        ("comparison mode", False)],
                color=SKY, ls=(0, (3, 2)), lw=1.1),
}

for _spec in NODES.values():
    _spec["w"], _spec["h"] = box_size(_spec["lines"], min_w=_spec.get("min_w", 0.0),
                                       min_h=_spec.get("min_h", 0.0))


def set_x(name, cx):
    NODES[name]["x"] = cx


def right_of(name):
    return NODES[name]["x"] + NODES[name]["w"] / 2


def left_of(name):
    return NODES[name]["x"] - NODES[name]["w"] / 2


# -- tier 2 (the loop): colL={L1,L2}, colR={L4,L5}, colO={O1,O2}, chained from x=0 --
# colL-colR uses a wider gap than the house GAP: the B2 -> L1 elbow (drawn below)
# needs a clear vertical corridor between the two columns to detour around L2.
GAP_LOOP = 0.34
colL_w = max(NODES["L1"]["w"], NODES["L2"]["w"])
colR_w = max(NODES["L4"]["w"], NODES["L5"]["w"])
colO_w = max(NODES["O1"]["w"], NODES["O2"]["w"])
x_colL = colL_w / 2
x_colR = x_colL + colL_w / 2 + GAP_LOOP + colR_w / 2
x_colO = x_colR + colR_w / 2 + GAP + colO_w / 2
for k in ("L1", "L2"):
    set_x(k, x_colL)
for k in ("L4", "L5"):
    set_x(k, x_colR)
for k in ("O1", "O2"):
    set_x(k, x_colO)
set_x("ALT", (x_colL + x_colR) / 2)

# ----------------------------------------------------------------------
# x layout: hub T1 feeds row A (T2 -> T3, aligned above colR) and B2 directly
# (mirroring T2; "selection" is drawn as a label on that arrow rather than its
# own box -- the step is a single filter, not a transformation worth a node).
# ----------------------------------------------------------------------
set_x("T2", 0.0)
set_x("B2", 0.0)
# T3 aligned exactly above L4 (colR) for a clean vertical arrow; never left of its
# own natural minimum position in row A.
t3_natural = right_of("T2") + GAP + NODES["T3"]["w"] / 2
set_x("T3", max(t3_natural, x_colR))
# if alignment pulled T3 right, nudge T2 so the row doesn't look oddly gapped
if NODES["T3"]["x"] > t3_natural + 0.02:
    set_x("T2", NODES["T3"]["x"] - NODES["T3"]["w"] / 2 - GAP - NODES["T2"]["w"] / 2)

x_rowA_left = left_of("T2")
x_rowB_left = left_of("B2")
hub_target_right = min(x_rowA_left, x_rowB_left) - GAP
set_x("T1", hub_target_right - NODES["T1"]["w"] / 2)

# shift everything so the leftmost element sits at a small left margin
MARGIN = 0.10
x_min = min(left_of(k) for k in NODES)
shift = MARGIN - x_min
for n in NODES.values():
    n["x"] += shift

x_max = max(right_of(k) for k in NODES)
FIG_CONTENT_W = x_max + 0.10
assert FIG_CONTENT_W <= FIGW + 1e-6, f"content width {FIG_CONTENT_W:.3f} in exceeds {FIGW} in"

# B2 (tier 1) must not reach into L4's column (colR): the vertical arrow T3 -> L4
# passes right through B2's row on its way down, so if their x-spans overlapped
# it would cut through the B2 box.
assert right_of("B2") < left_of("L4") - 0.05, (
    f"B2 (r={right_of('B2'):.2f}) reaches into L4's column (l={left_of('L4'):.2f}); "
    "the T3 -> L4 arrow would cut through B2 -- shorten B2's text or widen the gap.")

# the B2 -> L1 elbow needs a clear vertical corridor between colL and colR to
# detour around L2 (see draw step below).
gap_x = (max(right_of("L1"), right_of("L2")) + min(left_of("L4"), left_of("L5"))) / 2
assert gap_x - 0.10 > max(right_of("L1"), right_of("L2")), "no clear corridor between colL and colR"

# ----------------------------------------------------------------------
# y layout: tier 1 (rows A, B; hub between them) stacked above tier 2
# (loop rows C, D; ALT below D). Tier 2 is built bottom-up so the dashed
# container and ALT box have a known, fixed footprint.
# ----------------------------------------------------------------------
Y_ALT = NODES["ALT"]["h"] / 2 + 0.28
Y_D = Y_ALT + NODES["ALT"]["h"] / 2 + 0.34 + max(NODES["L1"]["h"], NODES["L5"]["h"]) / 2
Y_C = Y_D + max(NODES["L1"]["h"], NODES["L5"]["h"]) / 2 + GAP + max(NODES["L2"]["h"], NODES["L4"]["h"]) / 2
TIER_GAP = 0.50  # room above L2/L4 for the B2->L1 elbow's jog, clear of L2/colL
Y_B = Y_C + max(NODES["L2"]["h"], NODES["L4"]["h"]) / 2 + TIER_GAP + NODES["B2"]["h"] / 2
Y_A = Y_B + NODES["B2"]["h"] / 2 + GAP + max(NODES["T2"]["h"], NODES["T3"]["h"]) / 2
Y_HUB = (Y_A + Y_B) / 2

for k in ("T2", "T3"):
    NODES[k]["y"] = Y_A
NODES["B2"]["y"] = Y_B
NODES["T1"]["y"] = Y_HUB
for k in ("L2", "L4", "O1"):
    NODES[k]["y"] = Y_C
for k in ("L1", "L5", "O2"):
    NODES[k]["y"] = Y_D
NODES["ALT"]["y"] = Y_ALT

y_max = max(n["y"] + n["h"] / 2 for n in NODES.values())
FIGH = y_max + 0.16  # small top margin (the loop-container label sits inside it, top-right)

# ----------------------------------------------------------------------
# Draw
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(FIGW, FIGH))
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_xlim(0, FIGW)
ax.set_ylim(0, FIGH)
ax.axis("off")


def draw_box(name):
    s = NODES[name]
    x, y, w, h = s["x"], s["y"], s["w"], s["h"]
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                        boxstyle="round,pad=0,rounding_size=0.08",
                        linewidth=s.get("lw", 1.4), linestyle=s.get("ls", "-"),
                        edgecolor=s["color"], facecolor=s.get("fill", "white"), zorder=3)
    ax.add_patch(p)
    n = len(s["lines"])
    dy = h / (n + 1)
    for i, (txt, bold) in enumerate(s["lines"]):
        ty = y + h / 2 - dy * (i + 1)
        ax.text(x, ty, txt, ha="center", va="center",
                 fontsize=FS_TITLE if bold else FS_BODY,
                 fontweight="bold" if bold else "normal", color="black", zorder=4)
    return {"x": x, "y": y, "w": w, "h": h,
            "l": x - w / 2, "r": x + w / 2, "t": y + h / 2, "b": y - h / 2}


def arrow(p0, p1, color="#333333", lw=1.15, ls="-", rad=0.0, shrinkA=2, shrinkB=2,
          mscale=9, zorder=2):
    a = FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                         mutation_scale=mscale, linewidth=lw, linestyle=ls,
                         color=color, shrinkA=shrinkA, shrinkB=shrinkB, zorder=zorder)
    ax.add_patch(a)


def elbow(pts, color="#333333", lw=1.15, ls="-", mscale=9, zorder=2):
    """A polyline through pts with a single arrowhead at the final vertex."""
    path = Path(pts, [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1))
    ax.add_patch(PathPatch(path, edgecolor=color, facecolor="none", linewidth=lw,
                            linestyle=ls, capstyle="round", joinstyle="round", zorder=zorder))
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=mscale,
                                  linewidth=lw, color=color, shrinkA=0, shrinkB=1, zorder=zorder))


def label(x, y, text, fontsize=FS_EDGE, color="#333333", style="italic", ha="center",
          va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color, style=style, zorder=4)


R = {name: draw_box(name) for name in NODES}

# -- fit-loop dashed container (tier 2 only: the per-batch cycle + the alt path) --
pad = 0.20
LX0 = R["L1"]["l"] - pad
LX1 = R["L5"]["r"] + pad
LY0 = R["ALT"]["b"] - pad
LY1 = R["L2"]["t"] + 0.36
loop_c = FancyBboxPatch((LX0, LY0), LX1 - LX0, LY1 - LY0,
                         boxstyle="round,pad=0,rounding_size=0.13",
                         linewidth=1.0, linestyle=(0, (5, 3)),
                         edgecolor="#8A8A8A", facecolor="none", zorder=1)
ax.add_patch(loop_c)
label(LX0 - 0.10, LY1 - 0.22, "fit loop,\niterated\nover batches", fontsize=FS_EDGE,
      color="#595959", ha="right", va="top")

# -- tier-1 internal arrows: hub fan-out, row A -------------------------------
arrow((R["T1"]["r"], R["T1"]["y"] + 0.13), (R["T2"]["l"], R["T2"]["y"] - 0.15), rad=-0.16)
arrow((R["T1"]["r"], R["T1"]["y"] - 0.13), (R["B2"]["l"], R["B2"]["y"] + 0.15), rad=0.16)
arrow((R["T2"]["r"], R["T2"]["y"]), (R["T3"]["l"], R["T3"]["y"]))
label((R["T1"]["r"] + R["B2"]["l"]) / 2 - 0.05, (R["T1"]["y"] + R["B2"]["y"]) / 2 - 0.10,
      "selection")

# -- tier1 -> tier2 links -------------------------------------------------------
# target hits: clean vertical drop straight into the likelihood (nothing else is
# in that column between tier 1 and L4).
arrow((R["T3"]["x"], R["T3"]["b"]), (R["L4"]["x"], R["L4"]["t"]), color=VERM)

# fit input -> current estimate: B2 sits directly above L1 (same column as L2),
# so a straight line down would cut through L2 ("probabilistic simulation").
# Detour around it: drop to just above L2, jog right into the open corridor
# between colL and colR, drop past L2's height in that corridor, jog back and
# enter L1's top edge offset from centre so the arrowhead doesn't collide with
# the L1 -> L2 arrow leaving from the same box.
jog_y = R["L2"]["t"] + 0.14
bend_y = (R["L1"]["t"] + R["L2"]["b"]) / 2
entry_x = R["L1"]["x"] + 0.30
elbow([(R["B2"]["x"], R["B2"]["b"]), (R["B2"]["x"], jog_y),
       (gap_x, jog_y), (gap_x, bend_y),
       (entry_x, bend_y), (entry_x, R["L1"]["t"])])
label((R["B2"]["x"] + gap_x) / 2, jog_y + 0.16, "first batch", fontsize=FS_EDGE)

# -- loop cycle: L1 -> L2 -> L4 -> L5 -> L1 -------------------------------------
arrow((R["L1"]["x"] - 0.15, R["L1"]["t"] - 0.01), (R["L2"]["x"] - 0.15, R["L2"]["b"] + 0.01))
arrow((R["L2"]["r"], R["L2"]["y"]), (R["L4"]["l"], R["L4"]["y"]))
arrow((R["L4"]["x"], R["L4"]["b"]), (R["L5"]["x"], R["L5"]["t"]))
label(R["L4"]["x"] + 0.40, (R["L4"]["b"] + R["L5"]["t"]) / 2, "$\\nabla\\mathcal{L}$")
arrow((R["L5"]["l"], R["L5"]["y"]), (R["L1"]["r"], R["L1"]["y"]), color="#1A1A1A")
label((R["L5"]["l"] + R["L1"]["r"]) / 2, R["L1"]["b"] - 0.17, "next batch")

# -- optional comparison-mode shortcut (dashed) --------------------------------
arrow((R["L1"]["x"] - 0.12, R["L1"]["b"]), (R["ALT"]["l"] + 0.35, R["ALT"]["t"]),
      color=SKY, ls=(0, (3, 2)), lw=1.0, rad=-0.18, mscale=7)
arrow((R["ALT"]["r"] - 0.35, R["ALT"]["t"]), (R["L5"]["x"] + 0.12, R["L5"]["b"]),
      color=SKY, ls=(0, (3, 2)), lw=1.0, rad=-0.18, mscale=7)

# -- outputs --------------------------------------------------------------------
arrow((R["L4"]["r"], R["L4"]["y"]), (R["O1"]["l"], R["O1"]["y"]))
arrow((R["L5"]["r"], R["L5"]["y"]), (R["O2"]["l"], R["O2"]["y"]))

for ext in ("pdf", "png"):
    kw = {"dpi": 300} if ext == "png" else {}
    fig.savefig(f"{OUT_DIR}/pipeline.{ext}", **kw)
print(f"figure size: {FIGW:.3f} x {FIGH:.3f} in (content width {FIG_CONTENT_W:.3f} in)")
print(f"wrote {OUT_DIR}/pipeline.pdf and {OUT_DIR}/pipeline.png")
