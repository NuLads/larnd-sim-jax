"""
Figure 3 (fig:probabilistic_method) for the NuLads paper, electronics section
(Sec. "Probabilistic and Differentiable Electronics Simulation").

Replaces figures/placeholder_probabilistic_method.png. Everything plotted is computed
from the production model code (src/larndsim/fee_jax.py, sim_jax.py) on ONE real pixel
waveform from a real track batch (through-going muon segments, paper_common.load /
params / TRUTH calibration point) -- nothing is hand-drawn.

Single-column figure (3.4 in wide), three panels STACKED on one shared time axis,
cropped to ONE strike -- the one genuine close double-hit (n=0,1, ~34 ticks apart) this
pixel shows. (The same pixel is also struck a second, unrelated time ~128 us later by a
different track segment sharing the batch; that second strike is simply cropped out,
not shown -- see "Data" below.) No insets, no twin axis.

Panels:
  (a) Charge: Q(t), the quantity the n=0 search compares to threshold, and, after the
      first hit, the charge since that reset, Q(t) - c_1 (c_1 = the charge the n=0
      search's surviving path had collected when hit 0 was found, from a replay of the
      beam search of Sec. "Path-Based Beam Search"). Both against the discrimination
      threshold V_th and the reset-noise band V_th +/- sigma_r; the hold window delta
      after the expected first hit is shaded (Sec. "The self-triggering front end and
      its noise model"), with a marker for the expected digitised charge <Q>_0 of
      Eq. (charge_expectation) at its end.
  (b) The first-hit (n=0) analytical crossing probabilities of Eq. (first_crossing):
      P(cross <= t) = Phi((Qbar(t)-V_th)/sigma_r) [Qbar = running max of Q] and its
      complement P(no cross < t), left axis 0-1.
  (c) Hit-time distributions P(hit n at t) for n=0,1 from the production beam search
      (get_adc_values_average_noise_vmap), on their own natural scale -- the n=0 curve
      IS the first-crossing probability g(t) of Eq. (first_crossing), refined by the
      hold-window collection and uncorrelated-noise acceptance of Sec. "Analytical
      Triggering Probabilities" -- overlaid with step histograms from N independent
      stochastic throws (get_adc_values) of the SAME waveform. Legend gives
      lambda_{jn}, E[t]_{jn} of Eq. (lambda).

Data: paper_common.load(max_batch_len=200., max_nbatch=1, data_seed=1) /
      paper_common.params() (TRUTH calibration point, Module-0 detector properties,
      same recipe as the rest of the review/paper pipeline). The pixel is selected, not
      hand-picked: among all active pixels of this track batch, the one that maximizes
      the model's own confidence of a 4th hit (lambda_3), subject to hits 0,1,2 being
      essentially certain (lambda_n > 0.99) AND the 4 hits forming one compact burst
      (span tight in units of the hold window) rather than two unrelated episodes of
      the same pixel ID far apart in the readout window -- in this batch only 2 pixels
      satisfy lambda_0,1,2 > 0.99 at all, and both turn out to show two such episodes;
      the figure itself only plots the first one (hits n=0,1), cropped tight.

Runtime: the K=100-path beam search over ~1000 pixels x 10 hit slots takes a few
minutes on CPU; results are cached to fig3_probabilistic_method_cache.npz next to this
script (a few hundred kB) so replotting is instant. Delete the cache (or pass --force)
to recompute from scratch.

Usage:  JAX_PLATFORMS=cpu python fig3_probabilistic_method.py [--force] [--n_throws 10000]
Writes: /sdf/group/neutrino/pgranger/NuLads_draft/figures/probabilistic_method.{pdf,png}
LaTeX: single-column `figure` environment, width=\\linewidth (NOT figure*/0.9\\linewidth
-- this version is sized for one column, not the full page width).
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # must precede any jax import (CPU-only, per task)

import sys
import argparse
import pathlib
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
CACHE_PATH = SCRIPT_DIR / 'fig3_probabilistic_method_cache.npz'
CENSUS_CACHE_PATH = SCRIPT_DIR / 'fig3_census_cache.npz'
PAPERJOBS = '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS'
OUT_DIR = '/sdf/group/neutrino/pgranger/NuLads_draft/figures'
OUT_BASENAME = 'probabilistic_method'

# Okabe-Ito colorblind-safe categorical palette (shared paper style)
BLUE, VERM, GREEN, ORANGE, SKY, PURPLE, BLACK = (
    '#0072B2', '#D55E00', '#009E73', '#E69F00', '#56B4E9', '#CC79A7', '#000000')
HIT_COLORS = [BLUE, VERM, GREEN, ORANGE]

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
    'font.family': 'sans-serif',
})

N_HIT_SHOW = 4  # hit indices 0..3 shown in panel (c)


# ----------------------------------------------------------------------------------
# Stage 1 (expensive, ~3 min on CPU): simulate the whole track batch and run the
# production K=100-path beam search on every active pixel, then reduce to small
# per-pixel summaries (lambda_n, E[t]_n) via get_average_hit_values (Eq. lambda / E[t]).
# This full-grid pass need only run once; wfs + the summaries are cached so pixel
# SELECTION can be redone instantly without repeating the beam search.
# ----------------------------------------------------------------------------------
def compute_census(max_batch_len=200., data_seed=1, force=False):
    if CENSUS_CACHE_PATH.exists() and not force:
        print(f'[fig3] using census cache {CENSUS_CACHE_PATH}')
        return dict(np.load(CENSUS_CACHE_PATH))

    sys.path.insert(0, PAPERJOBS)
    import jax
    import jax.numpy as jnp
    import paper_common as pc
    from larndsim.sim_jax import simulate_wfs
    from larndsim.fee_jax import get_adc_values_average_noise_chunked, digitize, get_average_hit_values

    assert jax.default_backend() == 'cpu', f'expected CPU backend, got {jax.default_backend()}'

    t0 = time.time()
    ds_sim, ds_tgt, fields = pc.load(max_batch_len=max_batch_len, max_nbatch=1, data_seed=data_seed)
    truth, response = pc.params()
    tracks = jnp.asarray(np.asarray(ds_tgt[0]).reshape(-1, len(fields)))
    print(f'[fig3] loaded {tracks.shape[0]} track segments in {time.time()-t0:.1f}s')

    t0 = time.time()
    wfs, unique_pixels = simulate_wfs(truth, response, tracks, fields)
    wfs = jax.block_until_ready(wfs)
    valid = np.asarray(unique_pixels) >= 0
    print(f'[fig3] simulate_wfs: {wfs.shape[0]} pixel slots ({int(valid.sum())} active), '
          f'{wfs.shape[1]} ticks, {time.time()-t0:.1f}s')

    # Full production probabilistic pass (K=100 beam search, pixel-chunked) on every
    # active pixel, exactly the function used in the loss. This is the expensive step;
    # only the small per-pixel reductions are kept (not the (Npix,10,Nticks-1) grid).
    t0 = time.time()
    log_prob_all, charge_all = get_adc_values_average_noise_chunked(truth, wfs, chunk_size=128)
    adcs_distrib_all = digitize(truth, charge_all)
    E_t_all, E_adc_all, lam_all = get_average_hit_values(log_prob_all, adcs_distrib_all)
    print(f'[fig3] production beam search (K={int(truth.fee_paths_scaling)}) over '
          f'{wfs.shape[0]} pixels x {truth.MAX_ADC_VALUES} hit slots: {time.time()-t0:.1f}s')

    interval = round((3 * float(truth.CLOCK_CYCLE) + float(truth.ADC_HOLD_DELAY) * float(truth.CLOCK_CYCLE))
                      / float(truth.t_sampling))

    out = dict(
        wfs=np.asarray(wfs), unique_pixels=np.asarray(unique_pixels), valid=valid,
        lam_all=np.asarray(lam_all), Et_all=np.asarray(E_t_all),
        Vth=float(truth.DISCRIMINATION_THRESHOLD), sigma_r=float(truth.RESET_NOISE_CHARGE),
        sigma_u=float(truth.UNCORRELATED_NOISE_CHARGE), t_sampling=float(truth.t_sampling),
        interval=interval, K_beam=int(truth.fee_paths_scaling), max_adc_values=int(truth.MAX_ADC_VALUES),
        n_active_pixels=int(valid.sum()), n_pixel_slots=int(wfs.shape[0]),
        n_track_segments=int(tracks.shape[0]), data_seed=data_seed, max_batch_len=max_batch_len,
    )
    np.savez_compressed(CENSUS_CACHE_PATH, **out)
    print(f'[fig3] wrote census cache {CENSUS_CACHE_PATH} ({CENSUS_CACHE_PATH.stat().st_size/1e6:.1f} MB)')
    return out


def select_pixel(census, verbose=True):
    """Pick the pixel for the figure: hits 0,1,2 essentially certain, maximizing the
    model's own confidence of a compact 4th hit, among pixels where the 4 hits form
    ONE burst (not two unrelated episodes of the same pixel ID far apart in the
    readout window -- this track batch stacks many independent track segments into
    one nominal 'event', so lambda alone can pick a pixel re-struck ~100us later by
    an unrelated segment)."""
    lam_np, Et_np, valid = census['lam_all'], census['Et_all'], census['valid']
    interval = int(census['interval'])
    span_all = Et_np[:, N_HIT_SHOW - 1] - Et_np[:, 0]
    confident3 = valid & (lam_np[:, 0] > 0.99) & (lam_np[:, 1] > 0.99) & (lam_np[:, 2] > 0.99)

    if verbose:
        cand = np.where(confident3)[0]
        order = cand[np.argsort(-lam_np[cand, 3])]
        print(f'[fig3] {cand.size} pixels with lambda_0,1,2 > 0.99; by lambda_3 (desc):')
        for i in order[:12]:
            print(f'    idx {i:5d}  id {int(census["unique_pixels"][i]):7d}  '
                  f'lambda={np.round(lam_np[i, :4], 4)}  span={span_all[i]:8.1f} ticks')

    pix_idx = None
    for mult in (8, 12, 20, 35, 60, 100, 1e9):
        cand = np.where(confident3 & (span_all < mult * interval))[0]
        if cand.size:
            pix_idx = int(cand[np.argmax(lam_np[cand, 3])])
            break
    if pix_idx is None:
        confident2 = valid & (lam_np[:, 0] > 0.99) & (lam_np[:, 1] > 0.9)
        cand = np.where(confident2)[0]
        pix_idx = int(cand[np.argmax(lam_np[cand, 2])])
    print(f'[fig3] selected pixel array-index {pix_idx}, unique_pixel_id '
          f'{int(census["unique_pixels"][pix_idx])}, lambda_0..3 = {lam_np[pix_idx, :4]}, '
          f'burst span (hit3-hit0) = {span_all[pix_idx]:.1f} ticks '
          f'({span_all[pix_idx]*float(census["t_sampling"]):.1f} us)')
    return pix_idx


# ----------------------------------------------------------------------------------
# Stage 2 (cheap, a few seconds): given one pixel's waveform (sliced from the census
# cache -- no need to redo simulate_wfs or the all-pixel beam search), rerun the
# SAME production single-pixel functions plus the stochastic validation throws.
# ----------------------------------------------------------------------------------
def compute_detail(census, pix_idx, n_throws=2000, throw_seed=12345):
    sys.path.insert(0, PAPERJOBS)
    import jax
    import jax.numpy as jnp
    import paper_common as pc
    from larndsim.fee_jax import (get_adc_values_average_noise_vmap, digitize,
                                   get_adc_values, _find_one_hit_step, get_average_hit_values)
    from larndsim.detsim_jax import id2pixel, get_pixel_coordinates
    from jax.scipy.special import ndtr

    truth, _response = pc.params()  # cheap (~2.5s): just need the params struct + constants
    wfs_pixel = np.asarray(census['wfs'][pix_idx])
    unique_pixel_id = int(census['unique_pixels'][pix_idx])

    px, py, plane, ev = id2pixel(truth, jnp.asarray([unique_pixel_id]))
    coords = np.asarray(get_pixel_coordinates(truth, px, py, plane))[0]
    pixel_meta = dict(unique_pixel_id=unique_pixel_id, x_cm=float(coords[0]), y_cm=float(coords[1]),
                       event=int(np.asarray(ev)[0]))

    # Re-run the actual production single-pixel beam search (identical function the
    # chunked multi-pixel pass calls; no chunking needed for Npix=1).
    log_prob_pixel_j, charge_pixel_j = get_adc_values_average_noise_vmap(truth, wfs_pixel[None, :])
    adcs_distrib_j = digitize(truth, charge_pixel_j)
    Et_pixel_j, _E_adc_j, lam_pixel_j = get_average_hit_values(log_prob_pixel_j, adcs_distrib_j)
    log_prob_pixel = np.asarray(log_prob_pixel_j[0])   # (MAX_ADC_VALUES, Nticks-1)
    charge_pixel = np.asarray(charge_pixel_j[0])
    lam_pixel = np.asarray(lam_pixel_j[0])
    Et_pixel = np.asarray(Et_pixel_j[0])

    # Analytic crossing CDF from the real Q(t): same primitive (ndtr) the model's
    # log_diff_ndtr uses internally, evaluated on the running maximum Qbar(t).
    Nticks = wfs_pixel.shape[0]
    q = wfs_pixel.astype(np.float64) * float(truth.t_sampling)
    Q = np.cumsum(q)
    Qbar = np.maximum.accumulate(Q)
    sigma_r, sigma_u = float(truth.RESET_NOISE_CHARGE), float(truth.UNCORRELATED_NOISE_CHARGE)
    Vth = float(truth.DISCRIMINATION_THRESHOLD)
    cdf = np.asarray(ndtr((jnp.asarray(Qbar) - Vth) / sigma_r))
    g_pure = np.diff(cdf)  # Eq. first_crossing, no hold-window / acceptance refinement

    min_log_prob = -18.42  # log(p_min), p_min = 1e-8 (Table fee_params), production default
    interval = int(census['interval'])

    # "Universe tracking": replay the single-pixel beam search explicitly (same
    # _find_one_hit_step the production vmap/scan calls, same K) and record the charge
    # offset c_k of the SURVIVING (highest-probability) path before each of the first 4
    # hits. Plotting Q(t) - c_k for k=0..3 directly shows the paper's "remaining
    # waveform Q_k(t) = Q(t) - c_k" (Sec. "Path-Based Beam Search", item 1).
    K = int(truth.fee_paths_scaling)
    q_sum_jnp = jnp.asarray(Q, dtype=jnp.float32)
    charges = jnp.zeros((K,), dtype=jnp.float32)
    logp = jnp.full((K,), min_log_prob, dtype=jnp.float32).at[0].set(0.0)
    c_top = [0.0]
    for _n in range(N_HIT_SHOW):
        (charges, logp), _ = _find_one_hit_step(
            q_sum_jnp, charges, logp, sigma_r, sigma_u, Vth, interval, K, min_log_prob,
            float(truth.excess_offset), float(truth.smooth_sigma_scale))
        top = int(jnp.argmax(logp))
        c_top.append(float(charges[top]))
    c_top = np.array(c_top[:N_HIT_SHOW], dtype=np.float64)
    print(f'[fig3] top-path charge offsets c_0..c_3 (e-): {c_top}')

    # Stochastic validation: replicate this one waveform n_throws times and run the
    # real self-trigger stochastic path (independent reset+uncorrelated noise per row,
    # single RNG key -> n_throws i.i.d. throws).
    t0 = time.time()
    wfs_rep = jnp.tile(jnp.asarray(wfs_pixel)[None, :], (n_throws, 1))
    adc_s, ticks_s = get_adc_values(truth, wfs_rep, jax.random.key(throw_seed))
    adc_s, ticks_s = np.asarray(adc_s), np.asarray(ticks_s)
    print(f'[fig3] {n_throws} stochastic throws of the chosen pixel: {time.time()-t0:.1f}s')

    out = dict(
        wfs_pixel=wfs_pixel, Q=Q, Qbar=Qbar, cdf=cdf, g_pure=g_pure, c_top=c_top,
        log_prob_pixel=log_prob_pixel, charge_pixel=charge_pixel,
        Et_pixel=Et_pixel, lam_pixel=lam_pixel,
        adc_s=adc_s, ticks_s=ticks_s,
        Vth=Vth, sigma_r=sigma_r, sigma_u=sigma_u, t_sampling=float(truth.t_sampling),
        interval=interval, K_beam=K, max_adc_values=int(truth.MAX_ADC_VALUES),
        n_throws=n_throws, n_active_pixels=int(census['n_active_pixels']),
        n_pixel_slots=int(census['n_pixel_slots']), n_track_segments=int(census['n_track_segments']),
        data_seed=int(census['data_seed']), max_batch_len=float(census['max_batch_len']),
        pix_idx=pix_idx,
        **{f'meta_{k}': v for k, v in pixel_meta.items()},
    )
    np.savez_compressed(CACHE_PATH, **out)
    print(f'[fig3] wrote cache {CACHE_PATH} ({CACHE_PATH.stat().st_size/1e3:.0f} kB)')
    return out


def load_or_compute(force=False, reselect=False, max_batch_len=200., data_seed=1,
                     n_throws=2000, throw_seed=12345):
    if CACHE_PATH.exists() and not force and not reselect:
        print(f'[fig3] using cache {CACHE_PATH}')
        return dict(np.load(CACHE_PATH))
    census = compute_census(max_batch_len=max_batch_len, data_seed=data_seed, force=force)
    pix_idx = select_pixel(census)
    return compute_detail(census, pix_idx, n_throws=n_throws, throw_seed=throw_seed)


# ----------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------
def make_figure(d, out_dir=OUT_DIR, out_basename=OUT_BASENAME):
    """Single-column, 3-row figure cropped to ONE strike: the genuine close double-hit
    (n=0,1) this pixel shows (see module docstring -- this same pixel is also struck a
    second, unrelated time much later in the readout window by another track segment;
    that second strike is simply cropped out here, not shown). No insets, no twin axis.
    """
    Vth, sigma_r, sigma_u = float(d['Vth']), float(d['sigma_r']), float(d['sigma_u'])
    interval, K, t_samp = int(d['interval']), int(d['K_beam']), float(d['t_sampling'])
    Q, c_top = d['Q'], d['c_top']
    cdf = d['cdf']
    log_prob_pixel, Et_pixel = d['log_prob_pixel'], d['Et_pixel']
    lam_pixel = np.clip(d['lam_pixel'], 0.0, 1.0)
    ticks_s, n_throws = d['ticks_s'], int(d['n_throws'])
    Nticks = d['wfs_pixel'].shape[0]
    Nbins = Nticks - 1

    model_prob = np.exp(np.clip(log_prob_pixel, -40.0, 0.0))  # (MAX_ADC_VALUES, Nbins)

    N_SHOW = 2  # n = 0, 1: the one close double-hit shown in this cropped figure
    SHOW_COLORS = [BLUE, VERM]

    # --- empirical per-tick probabilities from the stochastic throws (n=0,1 only) ---
    bin_edges = np.arange(Nbins + 1) - 0.5
    emp_prob = np.zeros((N_SHOW, Nbins))
    emp_frac = np.zeros(N_SHOW)
    emp_mean_t = np.full(N_SHOW, np.nan)
    for n in range(N_SHOW):
        tn = ticks_s[:, n]
        ok = tn < (Nticks - 3)
        counts, _ = np.histogram(tn[ok], bins=bin_edges)
        emp_prob[n] = counts / n_throws
        emp_frac[n] = ok.mean()
        if ok.any():
            emp_mean_t[n] = tn[ok].mean()

    tv_dist = 0.5 * np.abs(model_prob[:N_SHOW] - emp_prob).sum(axis=1)  # total-variation distance, per hit

    print('[fig3] --- agreement: probabilistic model vs stochastic throws (N={}) ---'.format(n_throws))
    for n in range(N_SHOW):
        print(f'  n={n}: lambda_model={lam_pixel[n]:.4f}  frac_empirical={emp_frac[n]:.4f}  '
              f'E[t]_model={Et_pixel[n]:.2f}  mean_t_empirical={emp_mean_t[n]:.2f}  '
              f'TV_dist={tv_dist[n]:.4f}')

    # --- ONE shared window across all 3 panels, cropped to the n=0,1 double-hit ---
    margin_lead = 18.0
    margin_trail = interval + 4.0
    t_lo = max(0, int(np.floor(Et_pixel[0] - margin_lead)))
    t_hi = min(Nbins - 1, int(np.ceil(Et_pixel[1] + margin_trail)))
    tb = np.arange(Nbins)
    sl = slice(t_lo, t_hi)
    print(f'[fig3] shared window: ticks {t_lo}-{t_hi} ({t_hi - t_lo} ticks, '
          f'{(t_hi - t_lo) * t_samp:.1f} us)')

    fig, (axa, axb, axcc) = plt.subplots(
        3, 1, figsize=(3.4, 5.6), sharex=True,
        gridspec_kw=dict(hspace=0.15))

    # ================= (a) charge: Q(t) [n=0 search] and Q(t)-c1 [n=1 search] ======
    hold_lo, hold_hi = Et_pixel[0], Et_pixel[0] + interval
    qa = Q[sl] / 1e3
    qb = (Q[sl] - c_top[1]) / 1e3
    ylo = min(qa.min(), qb.min())
    yhi = max(qa.max(), qb.max(), (Vth + sigma_r) / 1e3)
    pad = 0.10 * (yhi - ylo)

    axa.axvspan(hold_lo, hold_hi, color='0.88', lw=0, zorder=0)
    axa.axhspan((Vth - sigma_r) / 1e3, (Vth + sigma_r) / 1e3, color='0.55', alpha=0.25, lw=0, zorder=1)
    axa.axhline(Vth / 1e3, color=BLACK, ls='--', lw=0.9, zorder=2)
    axa.plot(tb[sl], qa, color=BLUE, lw=1.3, zorder=3, label=r'$Q(t)$  ($n=0$ search)')
    axa.plot(tb[sl], qb, color=VERM, lw=1.3, zorder=3, label=r'$Q(t)-c_1$  ($n=1$ search)')

    idx_hold = int(np.clip(round(hold_hi), 0, Nbins - 1))
    Qexp0 = d['charge_pixel'][0, idx_hold] / 1e3
    axa.plot([hold_hi], [Qexp0], marker='o', ms=3.2, color=BLACK, zorder=5)
    axa.annotate(r'$\langle Q\rangle_0$', xy=(hold_hi, Qexp0),
                 xytext=(hold_hi + 0.03 * (t_hi - t_lo), Qexp0 - 0.35 * (yhi - ylo)),
                 fontsize=6.5, va='top', ha='left')
    axa.text((hold_lo + hold_hi) / 2, yhi + 0.62 * pad, r'hold $\delta$',
              fontsize=6.5, ha='center', va='bottom', color='0.3')
    axa.text(0.985, (Vth + sigma_r) / 1e3, r'$V_{th}\pm\sigma_r$', transform=axa.get_yaxis_transform(),
              fontsize=6.5, ha='right', va='bottom', color='0.3')

    axa.set_ylim(ylo - pad, yhi + 1.9 * pad)
    axa.set_ylabel(r'charge ($10^3\,e^-$)')
    axa.legend(frameon=False, loc='lower right', fontsize=6.5, handlelength=1.3,
               labelspacing=0.3, borderaxespad=0.3)
    axa.text(0.0, 1.02, '(a)', transform=axa.transAxes, fontsize=7, va='bottom', ha='left')

    # ================= (b) first-hit (n=0) crossing CDFs ===========================
    p_cross = cdf[t_lo + 1:t_hi + 1]
    p_nocross = 1.0 - cdf[t_lo:t_hi]
    axb.plot(tb[sl], p_cross, color=BLUE, lw=1.3, label=r'$P(\mathrm{cross}\leq t_i)$')
    axb.plot(tb[sl], p_nocross, color='0.25', lw=1.3, ls='--', label=r'$P(\mathrm{no\ cross}<t_i)$')
    axb.axhline(0, color='0.6', lw=0.5, zorder=0)
    axb.axhline(1, color='0.6', lw=0.5, zorder=0)
    axb.set_ylim(-0.06, 1.12)
    axb.set_yticks([0, 0.5, 1])
    axb.set_ylabel('probability')
    axb.legend(frameon=False, loc='center right', fontsize=6.5, handlelength=1.3,
               labelspacing=0.3, borderaxespad=0.3)
    axb.text(0.0, 1.02, '(b)', transform=axb.transAxes, fontsize=7, va='bottom', ha='left')

    # ================= (c) hit-time distributions, n=0,1, vs stochastic throws =====
    ymax_c = max(model_prob[:N_SHOW, sl].max(), emp_prob[:, sl].max()) * 1.18
    for n in range(N_SHOW):
        lbl = fr'$n={n}$: $\lambda_{n}={lam_pixel[n]:.2f}$, $E[t]_{n}={Et_pixel[n]:.1f}$'
        axcc.plot(tb[sl], model_prob[n, sl], color=SHOW_COLORS[n], lw=1.3, zorder=3, label=lbl)
        axcc.step(tb[sl], emp_prob[n, sl], where='mid', color=SHOW_COLORS[n], lw=0.8,
                  alpha=0.75, zorder=2)
    dummy = Line2D([0], [0], color='0.35', lw=0.8, label=f'stochastic throws ($N={n_throws}$)')
    axcc.axhline(0, color='0.6', lw=0.5, zorder=0)
    axcc.set_ylim(-0.04 * ymax_c, ymax_c)
    axcc.set_ylabel(r'$P(\mathrm{hit}\ n\ \mathrm{at}\ t_i)$')
    axcc.set_xlabel(r'tick $t_i$  ($\Delta t = 0.1\,\mu$s)')
    handles, labels = axcc.get_legend_handles_labels()
    axcc.legend(handles + [dummy], labels + [dummy.get_label()], frameon=False, loc='upper left',
                fontsize=6.3, handlelength=1.3, labelspacing=0.35, borderaxespad=0.3)
    axcc.text(0.0, 1.02, '(c)', transform=axcc.transAxes, fontsize=7, va='bottom', ha='left')
    axcc.set_xlim(t_lo, t_hi)

    for ax_ in (axa, axb, axcc):
        ax_.grid(alpha=0.2, linewidth=0.4)
    fig.align_ylabels([axa, axb, axcc])

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, out_basename + '.pdf')
    png_path = os.path.join(out_dir, out_basename + '.png')
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f'[fig3] wrote {pdf_path}')
    print(f'[fig3] wrote {png_path}')
    plt.close(fig)

    return dict(tv_dist=tv_dist, emp_frac=emp_frac, emp_mean_t=emp_mean_t, t_lo=t_lo, t_hi=t_hi)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true',
                    help='recompute everything, including the expensive all-pixel census')
    p.add_argument('--reselect', action='store_true',
                    help='rerun pixel selection + the single-pixel detail stage, reusing an '
                         'existing census cache (seconds, not minutes)')
    p.add_argument('--n_throws', type=int, default=10000)
    args = p.parse_args()
    d = load_or_compute(force=args.force, reselect=args.reselect, n_throws=args.n_throws)
    print('[fig3] selected pixel: unique_pixel_id={}  (x={:.2f} cm, y={:.2f} cm)  event={}'.format(
        int(d['meta_unique_pixel_id']), float(d['meta_x_cm']), float(d['meta_y_cm']), int(d['meta_event'])))
    print('[fig3] active pixels in batch: {}/{}  track segments: {}'.format(
        int(d['n_active_pixels']), int(d['n_pixel_slots']), int(d['n_track_segments'])))
    make_figure(d)
