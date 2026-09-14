"""Diagnostic plots for the drift-attenuation penalty term.

Reads target hit-list caches from `target_<label>/batch*_target.npz` (populated
by any scan / fit run — the target is deterministic given nominal params) and
plots four views that reveal whether the current between_track / per_hit
observables can actually measure the exponential drift attenuation slope, or
whether they are dominated by the Bragg-peak dE/dx variation:

  Panel A — All hits: log(Q) vs tick, colored by within-track charge rank
            (bright = top-Q hit = probable Bragg endpoint; dim = MIP hits).
            The naive per_hit fit is drawn on top.
  Panel B — Per-track summary: one point per track at (⟨tick⟩, ⟨log Q⟩), color
            = track's Bragg fraction (fraction of track charge in its top-N%
            hits). The between_track fit is drawn on top.
  Panel C — Same as B but with a Bragg-endpoint mask: top-20% highest-Q hits
            per track dropped before averaging. If the fitted slope moves
            significantly closer to −1/τ, Bragg is the dominant contaminant
            and a MIP-only cut fixes the observable. If the slope stays put,
            dE/dx variation within the MIP region matters too and a normalized
            observable (option 2) is needed.
  Panel D — Slope-vs-Bragg-mask-fraction: sweep the fraction of top-Q hits
            dropped per track from 0 to 0.5 and plot the between_track and
            per_hit slopes vs the mask fraction. Convergence to a plateau ⇒
            MIP-only cut resolves the bias.
  Panel E — Per-track-demeaned residuals: for every hit, plot
              δlogQ_i = log Q_i − ⟨log Q⟩_track   vs   Δt_i = t_i − ⟨t⟩_track.
            This subtracts out per-track amplitude and per-track drift offset,
            so what remains is the pure within-track (log Q, t) SHAPE. The
            line the eye sees at the bottom of Panel A now becomes the
            central band, and the amplitude-nuisance scatter is gone.
            Attaching a linear fit here isolates −1/τ (up to the Bragg
            correlation with position-along-track).
  Panel F — Bragg-model-normalized: use the residual range from each track's
            stopping endpoint (approximated as its max-tick pixel) to look
            up an expected dE/dx from the Bethe-Bloch stopping-muon curve,
            then plot log(Q / dE_expected) vs tick. If the physics model is
            right this should be a cleaner −1/τ line. Note this uses an
            APPROXIMATE Bragg model and a rough endpoint assignment — use
            it as guidance, not a calibration.

Usage
-----
    python -m optimize.analysis_scripts.plot_input_logQ_spread \
        --target-dir target_A2-B5-C2-D10_stopp_dx0.01_..._sigmoid \
        --n-batches 3 \
        --tau-nominal 2200 \
        --out plots/logQ_spread_diagnostic.png

The `--target-dir` defaults to the newest `target_A2-B5-C2-D10_*` folder if
you omit it. `--tau-nominal` sets the reference exponential slope drawn as a
dashed line for eyeballing (in µs; converts to per-tick using ticks_per_us).
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path("/sdf/group/neutrino/cyifan/diff_sim_playground/larndsim_loss_dQdx")

# Default input-h5 for A2 targets. Overridable via --input-h5.
DEFAULT_INPUT_H5 = (
    "/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/"
    "production_884072/job_23771825_0000/"
    "output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_"
    "range_0.05cm.h5"
)


def _load_h5_segments(h5_path, event_ids_needed):
    """Load segment (x,y,z,dEdx,dE,range) fields from the input h5, for only
    the event IDs listed in `event_ids_needed`. Returns a dict of arrays."""
    import h5py
    needed = set(int(e) for e in event_ids_needed)
    with h5py.File(h5_path, "r") as f:
        seg = f["segments"]
        ev = np.asarray(seg["event_id"])
        # Mask up-front to avoid materialising the entire 30M-row table.
        mask = np.isin(ev, list(needed))
        idx = np.flatnonzero(mask)
        chunk = seg[idx]
    return dict(
        event=np.asarray(chunk["event_id"], dtype=np.int64),
        x=np.asarray(chunk["x"], dtype=np.float32),
        y=np.asarray(chunk["y"], dtype=np.float32),
        z=np.asarray(chunk["z"], dtype=np.float32),
        dEdx=np.asarray(chunk["dEdx"], dtype=np.float32),
        dE=np.asarray(chunk["dE"], dtype=np.float32),
        rng=np.asarray(chunk["range"], dtype=np.float32),
    )


def _match_hits_to_segments(hits, segs):
    """For each hit, find the closest segment in (x, y, z) sharing the same
    event. Returns (dEdx_per_hit, dE_per_hit, range_per_hit) as arrays
    aligned with the hit arrays. Hits with no segments in their event get
    NaN."""
    n = len(hits["event"])
    dEdx_h = np.full(n, np.nan, dtype=np.float32)
    dE_h   = np.full(n, np.nan, dtype=np.float32)
    rng_h  = np.full(n, np.nan, dtype=np.float32)
    ev_hit = hits["event"].astype(np.int64)
    ev_seg = segs["event"]
    px = np.asarray(hits.get("pixel_x", np.zeros(n)))
    py = np.asarray(hits.get("pixel_y", np.zeros(n)))
    pz = np.asarray(hits["pixel_z"])
    # `event` in the npz was made globally unique across batches (see
    # _load_target_hits: adds i*100_000). Reduce back to h5 event_id via
    # modulo, since the offset step is 100_000 and the real event ids are
    # much smaller than that.
    ev_hit_h5 = ev_hit % 100_000
    for ev in np.unique(ev_hit_h5):
        mh = ev_hit_h5 == ev
        ms = ev_seg == int(ev)
        if not mh.any() or not ms.any():
            continue
        # Nearest-neighbour by 3-D distance. Small enough (few×100 hits ×
        # few×1000 segments per event) to do naively.
        sx = segs["x"][ms]; sy = segs["y"][ms]; sz = segs["z"][ms]
        d2 = ((px[mh, None] - sx) ** 2
              + (py[mh, None] - sy) ** 2
              + (pz[mh, None] - sz) ** 2)
        j = np.argmin(d2, axis=1)
        dEdx_h[mh] = segs["dEdx"][ms][j]
        dE_h[mh]   = segs["dE"][ms][j]
        rng_h[mh]  = segs["rng"][ms][j]
    return dEdx_h, dE_h, rng_h


def _first_hit_per_pixel(hits):
    """Filter to one hit per (event, pixel_x, pixel_y) keeping the smallest
    tick — the leading-edge trigger of each pixel. Reduces multi-hit
    pileup and gives one representative time per pixel."""
    ev  = hits["event"]
    px  = hits.get("pixel_x", np.zeros_like(ev))
    py  = hits.get("pixel_y", np.zeros_like(ev))
    key = ev.astype(np.int64) * 1_000_000_000 + \
          (px.astype(np.int64) + 1_000_000) * 1_000_000 + \
          py.astype(np.int64) + 1_000_000
    # Argsort so ties break by tick ascending; then unique-first.
    order = np.lexsort((hits["tick"], key))
    key_sorted = key[order]
    keep_local = np.concatenate([[True], np.diff(key_sorted) != 0])
    keep_global = np.zeros_like(key, dtype=bool)
    keep_global[order[keep_local]] = True
    out = {k: v[keep_global] for k, v in hits.items()
           if isinstance(v, np.ndarray) and v.shape == ev.shape}
    out["n_batches_read"] = hits["n_batches_read"]
    return out


def _weighted_linreg_slope(x, y, w):
    """Weighted 1-D linear regression slope. Returns (slope, intercept)."""
    W = np.sum(w) + 1e-30
    xb = np.sum(w * x) / W
    yb = np.sum(w * y) / W
    num = np.sum(w * (x - xb) * (y - yb))
    den = np.sum(w * (x - xb) ** 2) + 1e-30
    slope = num / den
    intercept = yb - slope * xb
    return float(slope), float(intercept)


def _load_target_hits(target_dir, n_batches, adc_to_ke=1.0 / 4.0):
    """Concatenate hits from the first `n_batches` `batch*_target.npz` files.

    Returns dict of arrays: Q_ke, tick, event, pixel_z, plus a global event id
    that spans batches (batch*10000 + local event) so per-track segment_sum
    still gives per-track quantities across the concatenated arrays.

    adc_to_ke: converts adcs → Q in ke. The exact conversion depends on the
    simulator config; the default 1/4 is a rough visual-scale factor that
    doesn't affect the SLOPE calculation (slope of log(αQ) = slope of log(Q)),
    just the y-axis units. Override with --adc-scale if you want physical ke.
    """
    files = sorted(glob.glob(f"{target_dir}/batch*_target.npz"))
    if not files:
        raise FileNotFoundError(f"No batch*_target.npz in {target_dir}")
    n = min(n_batches, len(files))
    Q_list, tick_list, ev_list, pz_list = [], [], [], []
    for i, f in enumerate(files[:n]):
        d = np.load(f)
        adcs = np.asarray(d["adcs"])
        ticks = np.asarray(d["ticks"])
        event = np.asarray(d["event"])
        pixel_z = np.asarray(d.get("pixel_z", np.zeros_like(ticks)))
        # Keep only real hits (event >= 0).
        m = event >= 0
        Q_list.append(adcs[m] * adc_to_ke)
        tick_list.append(ticks[m].astype(np.float32))
        ev_list.append(event[m].astype(np.int64) + i * 100_000)  # disambiguate across batches
        pz_list.append(pixel_z[m])
    return dict(
        Q=np.concatenate(Q_list),
        tick=np.concatenate(tick_list),
        event=np.concatenate(ev_list),
        pixel_z=np.concatenate(pz_list),
        n_batches_read=n,
    )


def _per_track_stats(Q, tick, event, log_floor=0.05):
    """Compute per-track ⟨tick⟩, ⟨log Q⟩, ΣQ, n_hits, and Bragg fraction.

    Bragg fraction = ΣQ_top20 / ΣQ_track, where top20 = top 20 % Q hits within
    the track. High Bragg fraction (>0.4) marks tracks whose charge is
    heavily concentrated in a few hits — likely the Bragg endpoint.
    """
    ev_unique, ev_inv = np.unique(event, return_inverse=True)
    n_tracks = len(ev_unique)
    Q_safe = np.maximum(Q, log_floor)
    logQ = np.log(Q_safe)

    t_bar = np.zeros(n_tracks)
    l_bar = np.zeros(n_tracks)
    sumQ  = np.zeros(n_tracks)
    n_hits = np.zeros(n_tracks, dtype=np.int32)
    bragg_frac = np.zeros(n_tracks)
    hit_rank_norm = np.zeros_like(Q)  # 0=lowest Q, 1=highest Q within track

    for k in range(n_tracks):
        m = ev_inv == k
        n_hits[k] = np.sum(m)
        if n_hits[k] == 0:
            continue
        qs = Q[m]
        ts = tick[m]
        t_bar[k] = ts.mean()
        l_bar[k] = np.log(np.maximum(qs, log_floor)).mean()
        sumQ[k]  = qs.sum()
        # Bragg fraction: top-20 % of hits by Q.
        n_top = max(1, int(np.ceil(0.2 * n_hits[k])))
        q_sort_desc = np.sort(qs)[::-1]
        bragg_frac[k] = q_sort_desc[:n_top].sum() / (sumQ[k] + 1e-30)
        # Within-track normalized Q rank for coloring in Panel A.
        order = np.argsort(qs)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(qs))
        hit_rank_norm[m] = rank / max(len(qs) - 1, 1)

    return dict(
        ev_unique=ev_unique, ev_inv=ev_inv,
        t_bar=t_bar, l_bar=l_bar, sumQ=sumQ,
        n_hits=n_hits, bragg_frac=bragg_frac,
        hit_rank_norm=hit_rank_norm,
    )


def _mask_top_Q_per_track(Q, event, top_frac):
    """Return a boolean mask that KEEPS hits below the top `top_frac` fraction
    of Q within each track. Used to drop Bragg-endpoint hits."""
    if top_frac <= 0:
        return np.ones_like(Q, dtype=bool)
    keep = np.ones_like(Q, dtype=bool)
    ev_unique, ev_inv = np.unique(event, return_inverse=True)
    for k in range(len(ev_unique)):
        m = ev_inv == k
        if m.sum() == 0:
            continue
        qs = Q[m]
        thresh = np.quantile(qs, 1 - top_frac)
        idx_local = np.flatnonzero(m)
        keep[idx_local] = qs < thresh
    return keep


def _per_hit_track_means(Q, tick, event, log_floor=0.05):
    """Return per-hit arrays `⟨log Q⟩_track` and `⟨tick⟩_track` broadcast
    back to hit level, so we can subtract per-hit for the demeaned view."""
    ev_unique, ev_inv = np.unique(event, return_inverse=True)
    n_tracks = len(ev_unique)
    logQ = np.log(np.maximum(Q, log_floor))
    l_bar_track = np.zeros(n_tracks)
    t_bar_track = np.zeros(n_tracks)
    for k in range(n_tracks):
        m = ev_inv == k
        if m.sum() == 0:
            continue
        l_bar_track[k] = logQ[m].mean()
        t_bar_track[k] = tick[m].mean()
    return l_bar_track[ev_inv], t_bar_track[ev_inv]


def _residual_range_dedx(Q, tick, event, pixel_z, ticks_per_us, v_drift=0.16):
    """Approximate per-hit dE/dx from residual range to the track's endpoint.

    Endpoint chosen as the pixel_z that has the highest total Q in the
    track (typically the Bragg peak). Residual range R = |z_hit − z_endpoint|
    (in cm; pixel_z units assumed cm).

    dE/dx from a simple stopping-muon Bethe-Bloch: dE/dx_MIP ≈ 2 MeV/cm plus
    a Bragg tail ~ K / R^0.6 for R < 20 cm. Uses the crude parametrisation
    dEdx(R) = a + b / (R+eps)^p       (a=1.5, b=6.5, p=0.6, eps=0.5 cm)
    which reproduces the MIP plateau (~2 MeV/cm) and the ×5-10 rise in the
    last 2 cm. Order-of-magnitude, not calibration-grade.
    """
    ev_unique, ev_inv = np.unique(event, return_inverse=True)
    n_tracks = len(ev_unique)
    z_end = np.zeros(n_tracks)
    for k in range(n_tracks):
        m = ev_inv == k
        if m.sum() == 0:
            continue
        # Endpoint = z of pixel with max Q (Bragg-peak proxy).
        z_end[k] = pixel_z[m][np.argmax(Q[m])]
    R = np.abs(pixel_z - z_end[ev_inv])   # residual range in cm
    a, b, p, eps = 1.5, 6.5, 0.6, 0.5
    dEdx = a + b / (R + eps) ** p
    return dEdx, R


def make_plot(hits, tau_nominal_us, ticks_per_us, out_png,
              input_h5=None):
    Q, tick, event, pixel_z = hits["Q"], hits["tick"], hits["event"], hits["pixel_z"]
    st = _per_track_stats(Q, tick, event)

    fig, axs = plt.subplots(4, 2, figsize=(14, 22))
    ax_A, ax_B = axs[0]
    ax_C, ax_D = axs[1]
    ax_E, ax_F = axs[2]
    ax_G, ax_H = axs[3]

    # Reference exponential slope for eyeballing: −1/(τ · ticks_per_us) per tick.
    ref_slope_per_tick = -1.0 / (tau_nominal_us * ticks_per_us)

    # ---------- Panel A: all hits, colored by raw residual range ----------
    # Residual range R = |z_hit − z_endpoint_of_track| in cm, where the
    # endpoint is the max-Q pixel per track (Bragg-peak proxy). Small R
    # = near the Bragg endpoint; large R = deep in the MIP region.
    _, R_for_A = _residual_range_dedx(Q, tick, event, pixel_z, ticks_per_us)
    logQ = np.log(np.maximum(Q, 0.05))
    scat = ax_A.scatter(tick, logQ, c=np.clip(R_for_A, 0, 8), s=2, alpha=0.35,
                        cmap="plasma", vmin=0, vmax=8)
    fig.colorbar(scat, ax=ax_A, label="residual range R [cm]\n"
                                       "(0 = Bragg endpoint, ≥5 = MIP-only)")
    slope, intercept = _weighted_linreg_slope(tick, logQ, np.ones_like(Q))
    xs = np.linspace(tick.min(), tick.max(), 100)
    ax_A.plot(xs, intercept + slope * xs, "r-", lw=1.5,
              label=f"per_hit fit (all): slope = {slope:.2e} / tick\n"
                    f"τ_fit ≈ {-1 / (slope * ticks_per_us):.0f} µs")

    # ---------- Lower-envelope fit (per-tick-bin minimum) ----------
    # Bin hits into `N_ENV_BINS` tick bins and pick the low-Q edge of each
    # bin. This traces the "clean line at the bottom" of the scatter that
    # your eye sees: MIP-only hits at unknown drift-z that all lie on
    # log Q = log Q_0_MIP - t/τ, with Q_0_MIP roughly constant across
    # tracks.  Two variants shown:
    #   min           — sensitive to statistical fluctuations; use only for
    #                   visualization
    #   5th quantile  — robust, less noisy; use this for the fitted slope
    # We fit both and plot the envelope curve so the eye can compare.
    # Sweep several percentiles (5, 15, 25, 40) so we can see which one the
    # eye's "line at the bottom" corresponds to. Each percentile gives a
    # different subset — the visible band is typically somewhere in the
    # low-to-mid range, not the strict floor.
    N_ENV_BINS = 40
    ENV_PERCENTILES = [5, 15, 25, 40]
    ENV_COLORS = ["#00c853", "#ff6f00", "#3f51b5", "#8e24aa"]
    tick_edges = np.linspace(tick.min(), tick.max(), N_ENV_BINS + 1)
    tick_centers = 0.5 * (tick_edges[1:] + tick_edges[:-1])
    n_per_bin = np.zeros(N_ENV_BINS, dtype=int)
    per_bin_masks = []
    for k in range(N_ENV_BINS):
        m = (tick >= tick_edges[k]) & (tick < tick_edges[k + 1])
        n_per_bin[k] = m.sum()
        per_bin_masks.append(m)
    xs = np.linspace(tick.min(), tick.max(), 100)

    env_slopes = {}
    for pct, color in zip(ENV_PERCENTILES, ENV_COLORS):
        env_vals = np.full(N_ENV_BINS, np.nan)
        for k in range(N_ENV_BINS):
            if n_per_bin[k] < 5:
                continue
            env_vals[k] = np.log(np.maximum(
                np.percentile(Q[per_bin_masks[k]], pct), 0.05))
        ok = np.isfinite(env_vals)
        if ok.sum() < 3:
            env_slopes[pct] = np.nan
            continue
        w_env = n_per_bin[ok].astype(np.float32)
        s_env, i_env = _weighted_linreg_slope(
            tick_centers[ok], env_vals[ok], w_env,
        )
        env_slopes[pct] = s_env
        ax_A.plot(tick_centers[ok], env_vals[ok], "s", ms=5, color=color,
                  markerfacecolor="none", markeredgewidth=1.2, alpha=0.9,
                  label=f"{pct}th pct per bin  (fit: τ ≈ {-1/(s_env*ticks_per_us):.0f} µs)")
        ax_A.plot(xs, i_env + s_env * xs, "-", color=color, lw=1.4, alpha=0.9)
    s_env = env_slopes.get(15, np.nan)   # keep one for the stdout summary

    # ---------- Combined MIP-and-tick cut overlay on Panel A ----------
    # The physics-motivated way to isolate the clean attenuation line at the
    # bottom of Panel A: keep only hits that satisfy BOTH
    #   (a) tick ≥ TICK_MIN         — past the integrator ramp-up region
    #   (b) residual range ≥ R_MIN  — MIP-only (drop Bragg-endpoint hits)
    # These are the same cuts that would be applied in the loss via
    # SLOPE_PENALTY_TICK_MIN (a) and a future residual-range mask (b).
    # We plot the fit on the surviving subset in a distinct color so its
    # slope can be compared to the naive per-hit slope directly.
    TICK_MIN_A = 200.0
    R_MIN_A = 5.0
    keep_A = (tick >= TICK_MIN_A) & (R_for_A >= R_MIN_A)
    if keep_A.sum() > 50:
        s_clean, i_clean = _weighted_linreg_slope(
            tick[keep_A], logQ[keep_A], np.ones(int(keep_A.sum())),
        )
        ax_A.plot(xs, i_clean + s_clean * xs, "-", color="#00c853", lw=1.7,
                  label=f"clean (R≥{R_MIN_A}cm & tick≥{TICK_MIN_A:.0f}): "
                        f"slope = {s_clean:.2e} / tick\n"
                        f"τ_fit ≈ {-1 / (s_clean * ticks_per_us):.0f} µs   "
                        f"({keep_A.sum()} hits)")
    ax_A.axvline(TICK_MIN_A, color="0.4", linestyle=":", linewidth=1,
                 label=f"tick_min = {TICK_MIN_A:.0f}")

    ax_A.plot(xs, intercept + ref_slope_per_tick * xs, "k--", lw=1,
              label=f"−1/τ_nom line (τ = {tau_nominal_us} µs)")
    ax_A.set(xlabel="tick", ylabel="log Q [log ke]",
             title=f"A — All {len(Q)} hits, colored by residual range R [cm]")
    ax_A.legend(loc="best", fontsize=7)
    ax_A.grid(alpha=0.3)

    # ---------- Panel B: between_track summary points ----------
    good = st["n_hits"] >= 3
    tb = st["t_bar"][good]
    lb = st["l_bar"][good]
    w  = st["sumQ"][good]
    bf = st["bragg_frac"][good]
    scat = ax_B.scatter(tb, lb, c=bf, s=30, alpha=0.7, cmap="magma",
                        vmin=0.15, vmax=0.6)
    fig.colorbar(scat, ax=ax_B, label="track Bragg fraction (top-20% Q)")
    slope, intercept = _weighted_linreg_slope(tb, lb, w)
    xs = np.linspace(tb.min(), tb.max(), 100)
    ax_B.plot(xs, intercept + slope * xs, "r-", lw=1.5,
              label=f"between_track fit: slope = {slope:.2e} / tick\n"
                    f"τ_fit ≈ {-1 / (slope * ticks_per_us):.0f} µs")
    ax_B.plot(xs, intercept + ref_slope_per_tick * xs, "k--", lw=1,
              label=f"−1/τ_nom line (τ = {tau_nominal_us} µs)")
    ax_B.set(xlabel="⟨tick⟩ per track", ylabel="⟨log Q⟩ per track",
             title=f"B — Between-track ({good.sum()} tracks)")
    ax_B.legend(loc="best", fontsize=8)
    ax_B.grid(alpha=0.3)

    # ---------- Panel C: MIP-only (drop top-20% Q hits per track) ----------
    mip_mask = _mask_top_Q_per_track(Q, event, top_frac=0.2)
    st_mip = _per_track_stats(Q[mip_mask], tick[mip_mask], event[mip_mask])
    good_mip = st_mip["n_hits"] >= 3
    tb_m = st_mip["t_bar"][good_mip]
    lb_m = st_mip["l_bar"][good_mip]
    w_m  = st_mip["sumQ"][good_mip]
    bf_m = st_mip["bragg_frac"][good_mip]
    scat = ax_C.scatter(tb_m, lb_m, c=bf_m, s=30, alpha=0.7, cmap="magma",
                        vmin=0.15, vmax=0.6)
    fig.colorbar(scat, ax=ax_C, label="track Bragg fraction (MIP-region)")
    slope_mip, intercept_mip = _weighted_linreg_slope(tb_m, lb_m, w_m)
    xs = np.linspace(tb_m.min(), tb_m.max(), 100)
    ax_C.plot(xs, intercept_mip + slope_mip * xs, "r-", lw=1.5,
              label=f"between_track MIP-only fit: slope = {slope_mip:.2e} / tick\n"
                    f"τ_fit ≈ {-1 / (slope_mip * ticks_per_us):.0f} µs")
    ax_C.plot(xs, intercept_mip + ref_slope_per_tick * xs, "k--", lw=1,
              label=f"−1/τ_nom line (τ = {tau_nominal_us} µs)")
    ax_C.set(xlabel="⟨tick⟩ per track (MIP-only)",
             ylabel="⟨log Q⟩ per track (MIP-only)",
             title=f"C — Between-track after dropping top-20% Q hits/track")
    ax_C.legend(loc="best", fontsize=8)
    ax_C.grid(alpha=0.3)

    # ---------- Panel D: slope vs Bragg-mask fraction sweep ----------
    fracs = np.linspace(0, 0.5, 11)
    bt_slopes = []
    ph_slopes = []
    for f in fracs:
        mask = _mask_top_Q_per_track(Q, event, top_frac=f)
        Qm, tickm, evm = Q[mask], tick[mask], event[mask]
        # between_track slope on masked hits
        st2 = _per_track_stats(Qm, tickm, evm)
        g2 = st2["n_hits"] >= 3
        s_bt, _ = _weighted_linreg_slope(st2["t_bar"][g2], st2["l_bar"][g2],
                                         st2["sumQ"][g2])
        # per_hit slope on masked hits
        logQm = np.log(np.maximum(Qm, 0.05))
        s_ph, _ = _weighted_linreg_slope(tickm, logQm, np.ones_like(Qm))
        bt_slopes.append(s_bt)
        ph_slopes.append(s_ph)
    ax_D.plot(fracs, bt_slopes, "o-", label="between_track", color="C0")
    ax_D.plot(fracs, ph_slopes, "s-", label="per_hit", color="C1")
    ax_D.axhline(ref_slope_per_tick, color="k", ls="--",
                 label=f"−1/τ_nom = {ref_slope_per_tick:.2e}")
    ax_D.set(xlabel="fraction of top-Q hits masked per track",
             ylabel="fitted slope [1/tick]",
             title="D — Slope vs Bragg-mask fraction")
    ax_D.legend(loc="best", fontsize=9)
    ax_D.grid(alpha=0.3)

    # ---------- Panel E: per-track de-meaned residuals ----------
    # Subtract per-track ⟨log Q⟩ and ⟨tick⟩; every track collapses to a
    # cloud around (0, 0) with only within-track shape info remaining.
    l_bar_hit, t_bar_hit = _per_hit_track_means(Q, tick, event)
    dlogQ = np.log(np.maximum(Q, 0.05)) - l_bar_hit
    dtick = tick - t_bar_hit
    scat = ax_E.scatter(dtick, dlogQ, c=st["hit_rank_norm"], s=2, alpha=0.35,
                        cmap="viridis")
    fig.colorbar(scat, ax=ax_E, label="within-track Q rank")
    slope_E, intercept_E = _weighted_linreg_slope(dtick, dlogQ, np.ones_like(dtick))
    xs = np.linspace(dtick.min(), dtick.max(), 100)
    ax_E.plot(xs, intercept_E + slope_E * xs, "r-", lw=1.5,
              label=f"fit: slope = {slope_E:.2e} / tick\n"
                    f"τ_fit ≈ {-1 / (slope_E * ticks_per_us):.0f} µs")
    ax_E.plot(xs, ref_slope_per_tick * xs, "k--", lw=1,
              label=f"−1/τ_nom line (τ = {tau_nominal_us} µs)")
    ax_E.axhline(0, color="0.6", lw=0.5)
    ax_E.axvline(0, color="0.6", lw=0.5)
    ax_E.set(xlabel="tick − ⟨tick⟩_track",
             ylabel="log Q − ⟨log Q⟩_track",
             title="E — Per-track de-meaned: amplitude nuisance removed")
    ax_E.legend(loc="best", fontsize=8)
    ax_E.grid(alpha=0.3)

    # ---------- Panel F: log Q vs residual range, colored by tick ----------
    # This is the "orthogonal view" to Panel A: instead of tick on x, put
    # residual range from the endpoint. If dE/dx dominates Q, we see the
    # Bragg curve here (Q sharply peaked at R≈0). If drift attenuation
    # dominates, we see mostly horizontal bands with tick-color gradient.
    # Fit lines drawn for two range regions:
    #   R < R_cut cm — near-endpoint (Bragg-dominated)
    #   R ≥ R_cut cm — MIP-only region: log(Q) should be roughly constant
    #                  with R at fixed tick, and shift with tick via −1/τ.
    R_cut_cm = 5.0
    _, R = _residual_range_dedx(Q, tick, event, pixel_z, ticks_per_us)
    logQ_all = np.log(np.maximum(Q, 0.05))
    scat = ax_F.scatter(R, logQ_all, c=tick, s=2, alpha=0.4, cmap="viridis",
                        vmin=0, vmax=tick.max())
    fig.colorbar(scat, ax=ax_F, label="tick (drift time)")
    ax_F.axvline(R_cut_cm, color="red", linestyle="--", linewidth=1,
                 label=f"R = {R_cut_cm} cm cut (right side = MIP-only proposal)")
    # Sanity: what does log Q look like vs tick if we only keep MIP hits?
    mip_R = R >= R_cut_cm
    if mip_R.sum() > 50:
        s_mip_R, _ = _weighted_linreg_slope(
            tick[mip_R], logQ_all[mip_R], np.ones(int(mip_R.sum())),
        )
        ax_F.text(
            0.02, 0.02,
            f"R ≥ {R_cut_cm} cm subset: slope of log Q vs tick = {s_mip_R:.2e}/tick\n"
            f"→ τ_fit = {-1/(s_mip_R*ticks_per_us):.0f} µs   ({mip_R.sum()} hits)",
            transform=ax_F.transAxes, fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
        )
    else:
        s_mip_R = np.nan
    ax_F.set(xlabel="residual range R from endpoint [cm]",
             ylabel="log Q [log ke]",
             title="F — log Q vs residual range (color = tick); "
                   f"vertical red = R>{R_cut_cm} cm MIP cut for penalty")
    ax_F.legend(loc="upper right", fontsize=8)
    ax_F.grid(alpha=0.3)

    for pct, s in env_slopes.items():
        if np.isfinite(s):
            print(f"envelope {pct:>2}th pct slope           = {s:.3e}  → τ = {-1/(s*ticks_per_us):.0f} µs")
    print(f"per-track demeaned slope             = {slope_E:.3e}  → τ = {-1/(slope_E*ticks_per_us):.0f} µs")
    if np.isfinite(s_mip_R):
        print(f"MIP-only (R ≥ {R_cut_cm}cm) slope of log Q vs tick "
              f"= {s_mip_R:.3e}  → τ = {-1/(s_mip_R*ticks_per_us):.0f} µs")

    # ---------- Panel G: segment-dE-normalized observable ----------
    # For each hit, look up dE/dx of the nearest segment (same event, min
    # 3-D distance) from the input h5. Plot log(Q_hit / dE/dx_seg) vs tick.
    # If the sim's per-hit Q really is dE/dx × exp(−t/τ) × const, this
    # should trace a straight line with slope = −1/(τ · ticks_per_μs) and
    # no residual amplitude scatter from Bragg-vs-MIP dE/dx variation.
    if input_h5 is not None:
        print(f"Loading segments from {input_h5} …")
        # `event` in the npz was globally offset per batch. Get the unique
        # h5 event_ids across all batches.
        ev_needed = np.unique(event % 100_000)
        segs = _load_h5_segments(input_h5, ev_needed)
        dEdx_h, dE_h, rng_h = _match_hits_to_segments(hits, segs)
        ok = np.isfinite(dEdx_h) & (dEdx_h > 0)
        if ok.sum() > 100:
            log_Q_norm = np.log(np.maximum(Q[ok], 0.05)) - np.log(dEdx_h[ok])
            tick_G = tick[ok]
            R_G = rng_h[ok]
            scat = ax_G.scatter(tick_G, log_Q_norm, c=np.clip(R_G, 0, 20),
                                s=2, alpha=0.35, cmap="plasma",
                                vmin=0, vmax=20)
            fig.colorbar(scat, ax=ax_G, label="segment residual range R [cm]")
            s_G, i_G = _weighted_linreg_slope(
                tick_G, log_Q_norm, np.ones_like(tick_G),
            )
            xs = np.linspace(tick_G.min(), tick_G.max(), 100)
            ax_G.plot(xs, i_G + s_G * xs, "r-", lw=1.6,
                      label=f"fit: slope = {s_G:.2e} / tick\n"
                            f"τ_fit ≈ {-1/(s_G*ticks_per_us):.0f} µs   "
                            f"({int(ok.sum())} hits matched)")
            ax_G.plot(xs, i_G + ref_slope_per_tick * xs, "k--", lw=1,
                      label=f"−1/τ_nom line (τ = {tau_nominal_us} µs)")
            print(f"segment-dE-normalized slope          = {s_G:.3e}  → τ = {-1/(s_G*ticks_per_us):.0f} µs")
        else:
            s_G = np.nan
            ax_G.text(0.5, 0.5, f"only {int(ok.sum())} hits matched to segments;\n"
                                 "check h5 path / event IDs",
                      transform=ax_G.transAxes, ha="center", va="center")
    else:
        s_G = np.nan
        ax_G.text(0.5, 0.5, "pass --input-h5 to enable\nsegment-dE normalization",
                  transform=ax_G.transAxes, ha="center", va="center", color="0.5")
    ax_G.set(xlabel="tick", ylabel="log(Q / dE/dx_seg)",
             title="G — Segment-dE-normalized (color = segment R)")
    ax_G.legend(loc="best", fontsize=8)
    ax_G.grid(alpha=0.3)

    # ---------- Panel H: first-hit-per-pixel (Panel A but decluttered) ----------
    # Keep the earliest-tick hit at each (event, pixel_x, pixel_y). Each
    # pixel contributes at most one point → removes multi-hit pileup and
    # the resulting scatter is cleaner. This is the "one-point-per-source"
    # version of Panel A.
    hits_first = _first_hit_per_pixel(hits)
    Q_h, tick_h = hits_first["Q"], hits_first["tick"]
    # Recompute residual range on the filtered subset for coloring.
    _, R_h = _residual_range_dedx(Q_h, tick_h, hits_first["event"],
                                  hits_first["pixel_z"], ticks_per_us)
    logQ_h = np.log(np.maximum(Q_h, 0.05))
    scat = ax_H.scatter(tick_h, logQ_h, c=np.clip(R_h, 0, 20), s=3,
                        alpha=0.45, cmap="plasma", vmin=0, vmax=8)
    fig.colorbar(scat, ax=ax_H, label="residual range R [cm]")
    s_H, i_H = _weighted_linreg_slope(tick_h, logQ_h, np.ones_like(Q_h))
    xs = np.linspace(tick_h.min(), tick_h.max(), 100)
    ax_H.plot(xs, i_H + s_H * xs, "r-", lw=1.5,
              label=f"first-hit fit: slope = {s_H:.2e} / tick\n"
                    f"τ_fit ≈ {-1/(s_H*ticks_per_us):.0f} µs   "
                    f"({len(Q_h)} unique pixels)")
    ax_H.plot(xs, i_H + ref_slope_per_tick * xs, "k--", lw=1,
              label=f"−1/τ_nom line (τ = {tau_nominal_us} µs)")
    ax_H.set(xlabel="tick", ylabel="log Q [log ke]",
             title=f"H — First hit per pixel ({len(Q_h)} of {len(Q)} total hits)")
    ax_H.legend(loc="best", fontsize=8)
    ax_H.grid(alpha=0.3)
    print(f"first-hit-per-pixel slope             = {s_H:.3e}  → τ = {-1/(s_H*ticks_per_us):.0f} µs")

    fig.suptitle(f"log Q vs drift-tick — target hits from "
                 f"{hits['n_batches_read']} batches. Panels G/H add "
                 f"segment-dE normalization and first-hit-per-pixel filter.",
                 y=1.005)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Numerical summary to stdout.
    print(f"per_hit slope             (full)     = {slope:.3e}  → τ = {-1/(slope*ticks_per_us):.0f} µs")
    print(f"between_track slope       (full)     = {_weighted_linreg_slope(st['t_bar'][st['n_hits']>=3], st['l_bar'][st['n_hits']>=3], st['sumQ'][st['n_hits']>=3])[0]:.3e}")
    print(f"between_track slope       (MIP-only) = {slope_mip:.3e}  → τ = {-1/(slope_mip*ticks_per_us):.0f} µs")
    print(f"reference (−1/τ_nom)      = {ref_slope_per_tick:.3e}    (τ_nom = {tau_nominal_us} µs)")
    print(f"\nSaved → {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target-dir", default=None,
                    help="target_<label>/ folder. Defaults to newest "
                         "target_A2-B5-C2-D10_* under project root.")
    ap.add_argument("--n-batches", type=int, default=3,
                    help="How many batch*_target.npz files to concatenate.")
    ap.add_argument("--tau-nominal", type=float, default=2200.0,
                    help="Reference lifetime in µs for the −1/τ line.")
    ap.add_argument("--ticks-per-us", type=float, default=10.0,
                    help="Sample rate — number of ticks per microsecond. "
                         "Default 10 matches the sim (dt = 0.1 µs).")
    ap.add_argument("--adc-scale", type=float, default=0.25,
                    help="Scalar mapping adcs → Q in ke for the y-axis. "
                         "Only affects intercept; slope is invariant.")
    ap.add_argument("--input-h5", default=DEFAULT_INPUT_H5,
                    help="Segments h5 for the dE lookup in Panel G. Pass "
                         "an empty string to skip Panel G.")
    ap.add_argument("--out", default=str(BASE / "plots" / "logQ_spread_diagnostic.png"))
    args = ap.parse_args()

    if args.target_dir is None:
        candidates = sorted(glob.glob(str(BASE / "target_A2-B5-C2-D10_*")),
                            key=os.path.getmtime)
        if not candidates:
            raise SystemExit("No target_A2-B5-C2-D10_* dir found; pass --target-dir.")
        args.target_dir = candidates[-1]
        print(f"Using newest target dir: {args.target_dir}")

    hits = _load_target_hits(args.target_dir, args.n_batches,
                             adc_to_ke=args.adc_scale)
    print(f"Loaded {len(hits['Q'])} hits from {hits['n_batches_read']} batches, "
          f"{len(np.unique(hits['event']))} tracks total.")

    out_png = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    make_plot(hits, args.tau_nominal, args.ticks_per_us, out_png,
              input_h5=(args.input_h5 if args.input_h5 else None))


if __name__ == "__main__":
    main()
