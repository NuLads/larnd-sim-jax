"""
Module that simulates the front-end electronics (triggering, ADC)
"""

import jax.numpy as jnp
from jax.profiler import annotate_function
from jax import jit, vmap, lax, random, checkpoint
from jax.scipy.special import erf
from functools import partial
import jax.scipy.special as jss
from jax.scipy.special import log_ndtr
import jax
import os

# GN curvature perf-investigation knob (env-gated; default OFF => identical behaviour).
# The pixel-chunk scan in get_adc_values_average_noise_chunked already computes its primal
# chunk-by-chunk, but reverse-mode AD through lax.scan still needs, for every chunk, the
# vjp residuals w.r.t. the closed-over `params` pytree that the Hessian/gradient differentiate
# through -- i.e. peak AD memory scales with num_chunks even though the primal never
# materializes more than one chunk at a time. Wrapping the per-chunk body in jax.checkpoint
# makes the outer scan save only (params, wfs_chunk) per iteration and recompute the
# per-chunk forward+backward on demand.
_CHUNK_SCAN_REMAT = os.environ.get('LARND_CHUNK_SCAN_REMAT', '0') == '1'

def _soft_max(x, lo, sharpness=100.0):
    """Smooth approximation to jnp.maximum(x, lo).

    Uses softplus: max(x, lo) ≈ lo + softplus((x - lo) * s) / s.
    Unlike jnp.maximum, this is C∞ and supports higher-order differentiation.
    """
    return lo + jax.nn.softplus((x - lo) * sharpness) / sharpness


def _soft_where(condition_val, true_val, false_val, sharpness=100.0):
    """Smooth approximation to jnp.where(condition_val > 0, true_val, false_val).

    Uses sigmoid blending instead of hard switching, enabling higher-order
    differentiation without NaN.
    """
    w = jax.nn.sigmoid(condition_val * sharpness)
    return w * true_val + (1 - w) * false_val


def log_diff_ndtr(a, b, min_log_prob=-18.42):
    """
    Compute log(Φ(a) - Φ(b)) where Φ is standard normal CDF.

    Uses numerically stable formulation with smooth clamping (softplus/sigmoid)
    to support higher-order automatic differentiation without NaN.
    """
    la = jss.log_ndtr(a)
    lb = jss.log_ndtr(b)

    safe_diff = jnp.where(a > b, lb - la, -1.0)

    neg_expm1 = -jnp.expm1(safe_diff)
    eps = 1e-30
    neg_expm1_safe = _soft_max(neg_expm1, eps, sharpness=1e10)

    log_term = jnp.log(neg_expm1_safe)
    log_term_safe = _soft_max(log_term, min_log_prob, sharpness=10.0)

    log_prob = la + log_term_safe

    return jnp.where(a > b, log_prob, min_log_prob)

@annotate_function
@jit
def digitize(params, integral_list):
    """
    The function takes as input the integrated charge and returns the digitized
    ADC counts.

    Args:
        integral_list (:obj:`numpy.ndarray`): list of charge collected by each pixel

    Returns:
        numpy.ndarray: list of ADC values for each pixel
    """
    adcs = jnp.minimum((jnp.maximum((integral_list*params.GAIN+params.V_PEDESTAL - params.V_CM), 0) \
                        * params.ADC_COUNTS/(params.V_REF-params.V_CM)), params.ADC_COUNTS)

    return adcs

@jit
def get_adc_values_average_noise(params, pixels_signals):
    q = pixels_signals*params.t_sampling
    q_sum = q.cumsum(axis=-1)  # Cumulative sum over time ticks

    Nvalues_scaling = params.fee_paths_scaling
    Npix = pixels_signals.shape[0]
    Nvalues = Nvalues_scaling*Npix

    def find_hit(carry, _):
        q_sum_loc, previous_prob, pixid = carry # q_sum_loc[Nvalues, Nticks] ; previous_prob[Nvalues] ; pixid[Nvalues]
        sigma = params.RESET_NOISE_CHARGE #Found out that only considering the reset noise was sufficient
        # sigma = jnp.sqrt(params.RESET_NOISE_CHARGE**2 + params.UNCORRELATED_NOISE_CHARGE**2)
        eps = 1e-10

        _, Nticks = q_sum_loc.shape

        erf_term = erf((q_sum_loc - params.DISCRIMINATION_THRESHOLD)/(jnp.sqrt(2)*sigma)) # erf_term[Nvalues, Nticks]
        # erf_term_signal = jnp.frompyfunc(jnp.maximum, 2, 1).accumulate(erf_term, axis=-1) #erf is increasing so should be faster to make the re-ordering afterwards
        erf_term_signal = lax.cummax(erf_term, axis=1)  # erf_term_signal[Nvalues, Nticks]
        # max_future_signal = jnp.frompyfunc(jnp.maximum, 2, 1).accumulate(q_sum_loc[:, ::-1], axis=-1)[:, ::-1]
        max_future_signal = lax.cummax(q_sum_loc, axis=1, reverse=True)  # max_future_signal[Nvalues, Nticks]
        
        guess = 0.5*(erf_term_signal[..., 1:] - erf_term_signal[..., :-1]) # guess[Nvalues, Nticks - 1]
        # guess = jnp.clip(guess, 0, 1)  # Ensure guess is between 0 and 1, should not be needed but erf got odd behavior for some reason

        interval = round((3 * params.CLOCK_CYCLE + params.ADC_HOLD_DELAY * params.CLOCK_CYCLE) / params.t_sampling)
        shifted_ticks = jnp.arange(Nticks - 1) + interval + 1 # shifted_ticks[Nticks - 1]
        shifted_ticks = jnp.clip(shifted_ticks, 0, Nticks - 1)

        prob_event = jnp.clip(0.5*(erf_term[..., shifted_ticks] - erf_term[..., :-1]), 0, guess)

        esperance_value = q_sum_loc[..., shifted_ticks] + params.DISCRIMINATION_THRESHOLD - 0.5*(q_sum_loc[..., 1:] + q_sum_loc[..., :-1]) # esperance_value[Nvalues, Nticks - 1]

        norm = jnp.clip(jnp.sum(prob_event, axis=-1) + eps, 0, 1)  # Adding a small epsilon to avoid division by zero ; norm[Nvalues]

        previous_prob_norm = jnp.full(Npix, eps, dtype=jnp.float32)  # Avoid division by zero
        previous_prob_norm = previous_prob_norm.at[pixid].add(previous_prob)

        no_hit_prob_across = jnp.ones(Npix, dtype=jnp.float32)  # Initialize no-hit probability across universes
        no_hit_prob_across = no_hit_prob_across.at[pixid].subtract(norm*previous_prob)  # No-hit probability across universes

        prob_distrib = prob_event*previous_prob[:, None] # prob_distrib[Nvalues, Nticks - 1]

        prob_distrib_across = jnp.zeros((Npix, Nticks - 1), dtype=jnp.float32)  # Initialize probability distribution across universes
        prob_distrib_across = prob_distrib_across.at[pixid].add(prob_distrib)  # prob_distrib_across[Npix, Nticks - 1]

        norm_across = jnp.sum(prob_distrib_across, axis=-1) + eps  # Normalize across universes ; norm_across[Npix]

        tick_avg = jnp.sum(prob_distrib_across/(norm_across[:, None])*jnp.arange(Nticks - 1), axis=-1)  # + 0.5

        charge_avg = jnp.sum(prob_event*esperance_value, axis=-1) # charge_avg[Nvalues] ; average charge for each path

        charge_avg_across = jnp.zeros((Npix,), dtype=jnp.float32)  # Initialize average charge across universes
        charge_avg_across = charge_avg_across.at[pixid].add(charge_avg*previous_prob)/norm_across  # charge_avg_across[Npix]

        future_hit_earliest_end = jnp.clip(shifted_ticks + interval + 1, 0, Nticks - 1) #We shift again
        future_hit_prob = 0.5*(1 + erf((max_future_signal[:, future_hit_earliest_end] - q_sum_loc[:, jnp.clip(shifted_ticks + 1, 0, Nticks - 1)] - params.DISCRIMINATION_THRESHOLD)/(jnp.sqrt(2)*sigma)))
        best_paths = jnp.argsort(prob_distrib*future_hit_prob, axis=None, descending=True)[:Nvalues] # best_paths[Nvalues]
        best_path_universe = best_paths // (Nticks - 1)
        best_path_ticks = best_paths % (Nticks - 1)
        best_path_pixid = pixid[best_path_universe]  # best_path_pixid[Nvalues]

        best_path_next_ticks = jnp.clip(shifted_ticks + 1, 0, Nticks - 1) # best_path_next_ticks[Nticks - 1]
        best_path_esperance = q_sum_loc[best_path_universe, best_path_next_ticks[best_path_ticks]]  # best_path_esperance[Nvalues]
        best_path_wfs = q_sum_loc[best_path_universe, :]
        q_sum_new = best_path_wfs - best_path_esperance[..., None]
        new_prob = prob_distrib[best_path_universe, best_path_ticks] # new_prob[Nvalues]
        return (q_sum_new, new_prob, best_path_pixid), (charge_avg_across, tick_avg, no_hit_prob_across)
    
    q_sum_multi = jnp.zeros((Nvalues, q_sum.shape[1]), dtype=jnp.float32)  # Initialize q_sum for multiple universes
    q_sum_multi = q_sum_multi.at[:q_sum.shape[0], :].set(q_sum)
    previous_prob = jnp.zeros(Nvalues, dtype=jnp.float32)  # Initialize previous probability
    previous_prob = previous_prob.at[:q_sum.shape[0]].set(1.0)  # Set the first universe probability to 1.0
    pixid = jnp.full((Nvalues,), -1, dtype=jnp.int32)  # Initialize pixel IDs for multiple universes
    pixid = pixid.at[:q_sum.shape[0]].set(jnp.arange(q_sum.shape[0]))  # Set pixel IDs for the first universe

    init_loop = (q_sum_multi, previous_prob, pixid)  # Initialize previous_prob
    _, (charge_avg, tick_avg, no_hit_prob) = lax.scan(find_hit, init_loop, jnp.arange(0, params.MAX_ADC_VALUES))
    # new_loop, (charge_avg, tick_avg, no_hit_prob) = find_hit(init_loop, 0)
    # _, (charge_avg, tick_avg, no_hit_prob) = find_hit(new_loop, 1)

    return (charge_avg.T, tick_avg.T, no_hit_prob.T)

def select_roi(params, wfs):
    roi_threshold = params.roi_threshold
    Npix, Nticks = wfs.shape
    roi_start = jnp.argmax(wfs > roi_threshold, axis=1)
    roi_end = Nticks - jnp.argmax(wfs[:, ::-1] > roi_threshold, axis=1) - 1
    with_roi = roi_start > 0
    largest_roi = jnp.max((roi_end-roi_start)[with_roi])
    wfs_with_roi = wfs.at[jnp.arange(Npix)[:, None], (jnp.arange(largest_roi) + roi_start[:, None])].get(unique_indices=True)
    return wfs_with_roi, roi_start


@annotate_function
@jit
def get_adc_values(params, pixels_signals, noise_rng_key):
    """
    Implementation of self-trigger logic

    Args:
        pixels_signals (:obj:`numpy.ndarray`): list of induced currents for
            each pixel
        time_ticks (:obj:`numpy.ndarray`): list of time ticks for each pixel
        adc_list (:obj:`numpy.ndarray`): list of integrated charges for each
            pixel
        adc_ticks_list (:obj:`numpy.ndarray`): list of the time ticks that
            correspond to each integrated charge.
    """

    #Baseline level of noise on integrated charge

    q_sum_base = random.normal(noise_rng_key, (pixels_signals.shape[0],)) * params.RESET_NOISE_CHARGE

    # Charge
    q = pixels_signals*params.t_sampling

    # Collect cumulative charge over all time ticks + add baseline noise
    # q_cumsum = q #Assuming the input is cumulated signal
    q_cumsum = q.cumsum(axis=-1)  # Cumulative sum over time ticks
    q_sum = q_sum_base[:, jnp.newaxis] + q_cumsum


    def find_hit(carry, it):
        key, q_sum, q_cumsum = carry
        # Index of first threshold passing. For nice time axis differentiability: first find index window around threshold.
        selec_func = lambda x: jnp.where((x[1:] >= params.DISCRIMINATION_THRESHOLD) & 
                    (x[:-1] <= params.DISCRIMINATION_THRESHOLD), size=1, fill_value=q_sum.shape[1]-2)
        idx_t, = vmap(selec_func, 0, 0)(q_sum)
        idx_t = idx_t.ravel()
        idx_pix = jnp.arange(0, q_sum.shape[0])
        # Then linearly interpolate for the intersection point.

        dq = jnp.where(q_sum[idx_pix, idx_t + 1] == q_sum[idx_pix, idx_t], q_sum[idx_pix, idx_t + 1] - params.DISCRIMINATION_THRESHOLD, q_sum[idx_pix, idx_t + 1] - q_sum[idx_pix, idx_t])

        idx_val = idx_t + 1 - (q_sum[idx_pix, idx_t+1] - params.DISCRIMINATION_THRESHOLD)/dq
        # debug.print("idx_val: {idx_val}", idx_val=idx_val)

        ic = jnp.zeros((q_sum.shape[0],))
        ic = ic.at[idx_pix].set(idx_t)

        # End point of integration
        interval = round((3 * params.CLOCK_CYCLE + params.ADC_HOLD_DELAY * params.CLOCK_CYCLE) / params.t_sampling)
        # integrate_end = ic+interval

        #Protect against ic+integrate_end past last index
        #TODO: This is really weird. How is this thing even differentiable?
        #TODO: Hardfixing to something reasonable, clearly not diff
        integrate_end = idx_t + 1 + interval
        integrate_end = jnp.where(integrate_end >= q_sum.shape[1], q_sum.shape[1]-1, integrate_end)
        end2d_idx = tuple(jnp.stack([jnp.arange(0, ic.shape[0]).astype(int), integrate_end]))
        # integrate_end = jnp.where(integrate_end >= q_sum.shape[1], q_sum.shape[1]-1, integrate_end)
        # end2d_idx = tuple(jnp.stack([jnp.arange(0, ic.shape[0]).astype(int), integrate_end.astype(int)]))

        end2d_idx_next = tuple(jnp.stack([jnp.arange(0, ic.shape[0]).astype(int), jnp.clip(integrate_end + 1, 0, q_sum.shape[1]-1)]))
        q_vals = q_sum[end2d_idx] #+ (idx_val - idx_t) *(q_sum[end2d_idx_next] - q_sum[end2d_idx])
        q_vals_no_noise = q_cumsum[end2d_idx] #+ (idx_val - idx_t) *(q_cumsum[end2d_idx_next] - q_cumsum[end2d_idx])

        # Cumulative => value at end is desired value
        # q_vals = q_sum[end2d_idx] 
        # q_vals_no_noise = q_cumsum[end2d_idx]

        # Uncorrelated noise
        key, = random.split(key, 1)
        extra_noise = random.normal(key, (pixels_signals.shape[0],))  * params.UNCORRELATED_NOISE_CHARGE

        # Only include noise if nonzero
        adc = jnp.where(q_vals_no_noise != 0, q_vals + extra_noise, q_vals_no_noise)

        cond_adc = adc < params.DISCRIMINATION_THRESHOLD
        #TODO: Check if the following selection makes sense
        cond_adc = jnp.logical_or(cond_adc, idx_t == q_sum.shape[1]-2) #Force set to zero if no initial threshold crossing

        # Only include if passes threshold     
        adc = jnp.where(cond_adc, 0, adc)
        ic = jnp.where(cond_adc, q_sum.shape[1]-2, ic)

        # Setup for next loop: baseline noise set to based on adc passing disc. threshold
        key, = random.split(key, 1)
        q_adc_pass = random.normal(key, (pixels_signals.shape[0],)) * params.RESET_NOISE_CHARGE
        key, = random.split(key, 1)
        q_adc_fail = random.normal(key, (pixels_signals.shape[0],)) * params.UNCORRELATED_NOISE_CHARGE
        q_sum_base = jnp.where(cond_adc, q_adc_fail, q_adc_pass)

        # Remove charge already counted
        integrate_end = idx_t + 1 + interval + 1 #Additional +1 present in the original code 
        integrate_end = jnp.where(integrate_end >= q_sum.shape[1], q_sum.shape[1]-1, integrate_end)
        end2d_idx = tuple(jnp.stack([jnp.arange(0, ic.shape[0]).astype(int), integrate_end]))
        q_vals_no_noise = q_cumsum[end2d_idx]
        q_cumsum = q_cumsum - q_vals_no_noise[..., jnp.newaxis]
        q_cumsum = jnp.where(q_cumsum < 0, 0, q_cumsum)
        q_sum = q_sum_base[:, jnp.newaxis] + q_cumsum
        

        return (key, q_sum, q_cumsum), (adc, ic)


    init_loop = (random.split(noise_rng_key, 1)[0], q_sum, q_cumsum)
    
    _, (full_adc, full_ticks) = lax.scan(find_hit, init_loop, jnp.arange(0, params.MAX_ADC_VALUES))
    # Single iteration to detect NaNs
    # _, (full_adc, full_ticks) = find_hit(init_loop, 0)
    # full_adc = jnp.repeat(full_adc[:, jnp.newaxis], params.MAX_ADC_VALUES, axis=1).T
    # full_ticks = jnp.repeat(full_ticks[:, jnp.newaxis], params.MAX_ADC_VALUES, axis=1).T

    return full_adc.T, full_ticks.T

# def _find_one_hit_step(q_sum, prev_charges, previous_prob, sigma, threshold, Nvalues):
#     """
#     Calculates a single hit-finding step for one pixel. This function is designed
#     to be vmapped across all pixels.
#     """
#     # This function contains the logic from the previous `_active_branch`.
#     Nticks = q_sum.shape[0]
#     inv_sqrt2_sigma = 1.0 / (jnp.sqrt(2) * sigma)
#     eps = 1e-9 # Using a small internal epsilon
#     interval = 18
#     shifted_ticks = jnp.arange(Nticks - 1) + interval + 1
#     shifted_ticks = jnp.clip(shifted_ticks, 0, Nticks - 1)
#     tick_indices = jnp.arange(Nticks - 1)

#     q_sum_loc = q_sum - prev_charges[..., None]

#     # Steps 1-2: Calculate Event Probabilities
#     erf_term = erf((q_sum_loc - threshold) * inv_sqrt2_sigma)
#     erf_term_signal = lax.cummax(erf_term, axis=1)
#     max_future_signal = lax.cummax(q_sum_loc, axis=1, reverse=True)
#     guess = 0.5 * jnp.diff(erf_term_signal, axis=-1)
#     prob_event = jnp.clip(0.5 * (erf_term[..., shifted_ticks] - erf_term[..., :-1]), 0, guess)
#     esperance_value = q_sum_loc[..., shifted_ticks] + threshold - 0.5 * (q_sum_loc[..., 1:] + q_sum_loc[..., :-1])

#     # Step 3: Aggregate Results
#     prob_distrib = prob_event * previous_prob[:, None]
#     total_hit_prob_per_path = jnp.sum(prob_event, axis=-1) * previous_prob
#     no_hit_prob = 1.0 - jnp.sum(total_hit_prob_per_path)
#     total_distrib_prob_per_tick = jnp.sum(prob_distrib, axis=0)
#     norm_across = jnp.sum(total_hit_prob_per_path)
#     safe_norm_across = jnp.maximum(norm_across, eps)
#     tick_avg = jnp.sum(total_distrib_prob_per_tick * tick_indices) / safe_norm_across
#     charge_avg_paths = jnp.sum(prob_event * esperance_value, axis=-1)
#     charge_avg = jnp.sum(charge_avg_paths * previous_prob) / safe_norm_across
    
#     # Step 4: Optimized Merging & Selection
#     future_hit_earliest_end = jnp.clip(shifted_ticks + interval + 1, 0, Nticks - 1)
#     next_q_sum = q_sum_loc[:, jnp.clip(shifted_ticks + 1, 0, Nticks - 1)]
#     future_hit_prob = 0.5 * (1 + erf((max_future_signal[:, future_hit_earliest_end] - next_q_sum - threshold) * inv_sqrt2_sigma))
#     path_selection_prob = prob_distrib * future_hit_prob
#     total_prob_per_tick = jnp.sum(path_selection_prob, axis=0)
#     _, top_k_ticks = lax.top_k(total_prob_per_tick, k=Nvalues)
    
#     # Step 5: Construct the State for the Next Iteration
#     new_prob = total_distrib_prob_per_tick[top_k_ticks]
#     best_path_next_ticks_indices = jnp.clip(shifted_ticks[top_k_ticks] + 1, 0, Nticks - 1)
#     charges_new = q_sum[best_path_next_ticks_indices]
    
#     # Return the new state and the outputs for this step for this pixel
#     return (charges_new, new_prob), (charge_avg, tick_avg, no_hit_prob, total_distrib_prob_per_tick)
#     # return (charges_new, new_prob), (charge_avg_across, tick_avg, no_hit_prob_across, prob_distrib_across)


# ---------------------------------------------------------------------------------------------------------------------
# LARND_FEE_FAST (gpu_sim opt, 22 Oct; adopted 22 Oct: default ON, =0 restores the original code path). Same values, less work in the beam search:
#  (1) q_max_future = cummax(q_sum - c[:, None]) is computed as cummax(q_sum)[None, :] - c[:, None] (one prefix max per
#      pixel instead of one per beam path: x -> x - c is monotone in IEEE arithmetic, so the two are bitwise equal);
#  (2) log_ndtr is evaluated ONCE on the full (V, T) argument array and sliced, instead of once per overlapping slice
#      (log_diff_ndtr(A[1:], A[:-1]) evaluated log_ndtr twice on the same 1998 of 1999 values; the shifted window too);
#  (3) log_ndtr itself (_log_ndtr_fast) evaluates _ndtr (erf + erfc) once on the selected argument instead of twice.
# All are custom_jvp functions whose tangent rules replay exactly the ops JAX's autodiff applies to the original
# expressions (same slices/gathers of the same tangent, same derivative exp(norm_logpdf(x) - log_ndtr(x)), same products in
# the same order), so the transposed (reverse-mode) computation is the same sequence of floating-point operations.
# ---------------------------------------------------------------------------------------------------------------------
_FEE_FAST = os.environ.get('LARND_FEE_FAST', '1') == '1'
# LARND_FEE_PEEL0 (fee2, 23 Oct; adopted 24 Oct: default ON, =0 restores; KN10R2 consistent): beam-search step 0 outside the scan, its per-path work done once (all
# paths start at charge 0). Forward values identical; backward = same terms summed in another order (round-off).
_FEE_PEEL0 = os.environ.get('LARND_FEE_PEEL0', '1') == '1'
# LARND_FEE_PIXEL_STOP (fee2, 23 Oct; adopted 24 Oct: default 3, =0 restores; NOT round-off: a (tiny) behaviour change;
# full-length fit KN10R2 consistent with the same-data replicas, review_2026_09/REVIEW.md): inside an ACTIVE chunk,
# the beam-search step is no longer computed for "dead" pixels, i.e. pixels whose total path probability is below
# stop_threshold * exp(-LARND_FEE_STOP_MARGIN) (their probability can grow by at most (1 + 2.2e-5) per step through the
# 1e-8 floors, so with the margin they can never cross stop_threshold again and the chunk activity -- the lax.cond --
# is exactly the original one). Their outputs: LARND_FEE_STOP_DEAD=low (default, closest to the original):
# log-prob -1e4 (the original computes values <= log(stop_threshold) + 1e-8, i.e. < 1e-9 total per step) and the
# charge output = esperance (path- and state-independent, i.e. exactly the original's); =floor: the inactive-chunk
# outputs (min_log_prob, 0) instead. =1: emulation (everything computed, then selected; no speed gain).
_FEE_PIXEL_STOP = os.environ.get('LARND_FEE_PIXEL_STOP', '3')
_FEE_PIXEL_STOP = _FEE_PIXEL_STOP if _FEE_PIXEL_STOP != '0' else ''
# =2: the real implementation: per chunk-step a lax.switch on the number n of live pixels: n = 0 -> inactive outputs,
# n <= K (ladder LARND_FEE_STOP_LADDER, default "16,64") -> the live rows are gathered into K rows, stepped, scattered
# back; otherwise all rows stepped + selected. Same semantics as =1 (round-off apart); cost ~ K / 128 per cell.
_FEE_STOP_LADDER = tuple(sorted(int(k) for k in os.environ.get('LARND_FEE_STOP_LADDER', '16,64').split(',') if k.strip()))
_FEE_STOP_MARGIN = float(os.environ.get('LARND_FEE_STOP_MARGIN', '0.01'))
_FEE_STOP_DEAD = os.environ.get('LARND_FEE_STOP_DEAD', 'low')
assert _FEE_STOP_DEAD in ('low', 'floor'), _FEE_STOP_DEAD
_FEE_DEAD_LOGP = -1e4
_FEE_STOP_BLOCK = int(os.environ.get('LARND_FEE_STOP_BLOCK', '0') or 0)   # =3: rows per compute block (0 = chunk_size)
from jax.custom_derivatives import SymbolicZero as _SymZero
from jax._src.scipy.special import _norm_logpdf as _jss_norm_logpdf


def _is_zero(t):
    return isinstance(t, _SymZero)


from jax._src.scipy.special import _ndtr as _jss_ndtr, _log_ndtr_lower as _jss_log_ndtr_lower
from jax._src.scipy.special import _log_ndtr_asymptotic_series as _jss_log_ndtr_series
import numpy as np
_FEE_FAST_LOGMERGE = os.environ.get('LARND_FEE_FAST_LOGMERGE', '1') == '1'    # part of LARND_FEE_FAST (=0: old 2-log form)
from jax._src.scipy.special import _LOGNDTR_FLOAT32_LOWER, _LOGNDTR_FLOAT32_UPPER, _LOGNDTR_FLOAT64_LOWER, _LOGNDTR_FLOAT64_UPPER


def _log_ndtr_value(x):
    """Values of jax.scipy.special.log_ndtr(x) (series_order 3) with ONE _ndtr evaluation instead of two: the original
    evaluates _ndtr(-x) for the x > upper branch and _ndtr(max(x, lower)) for the middle branch and selects; here the
    argument is selected first. Likewise the middle-branch log(ndtr) and the lower-branch log(asymptotic series) are
    one log of the selected argument. Same elementwise functions on the same values => same result."""
    x_arr = jnp.asarray(x)
    dtype = lax.dtype(x_arr)
    lower, upper = ((_LOGNDTR_FLOAT64_LOWER, _LOGNDTR_FLOAT64_UPPER) if dtype == jnp.float64
                    else (_LOGNDTR_FLOAT32_LOWER, _LOGNDTR_FLOAT32_UPPER))
    hi = lax.gt(x_arr, upper)
    n = _jss_ndtr(jnp.where(hi, -x_arr, lax.max(x_arr, lower)))
    if not _FEE_FAST_LOGMERGE:
        return jnp.where(hi, -n, jnp.where(lax.gt(x_arr, lower), lax.log(n), _jss_log_ndtr_lower(lax.min(x_arr, lower), 3)))
    # middle branch log(n) and lower branch log(series) share ONE log of the selected argument (same values)
    xl = lax.min(x_arr, lower)
    dt = lax.dtype(xl).type
    x_2 = lax.square(xl)
    log_scale = -dt(0.5) * x_2 - lax.log(-xl) - dt(0.5 * np.log(2. * np.pi))     # == _log_ndtr_lower's
    mid = lax.gt(x_arr, lower)
    L = lax.log(jnp.where(mid, n, _jss_log_ndtr_series(xl, 3)))
    return jnp.where(hi, -n, jnp.where(mid, L, log_scale + L))


@jax.custom_jvp
def _log_ndtr_fast(x):
    """== jax.scipy.special.log_ndtr (values and JVP rule t * exp(norm_logpdf(x) - log_ndtr(x)))."""
    return _log_ndtr_value(x)


@_log_ndtr_fast.defjvp
def _log_ndtr_fast_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    ans = _log_ndtr_fast(x)
    return ans, lax.mul(t, lax.exp(lax.sub(_jss_norm_logpdf(x), ans)))


@partial(jax.custom_jvp, nondiff_argnums=(3,))
def _cummax_loc(q_sum_loc, q_sum, c, reverse):
    """== lax.cummax(q_sum_loc, axis=1, reverse=reverse) with q_sum_loc = q_sum - c[:, None] (bitwise)."""
    return lax.cummax(q_sum, axis=0, reverse=reverse)[None, :] - c[:, None]


def _cummax_loc_jvp(reverse, primals, tangents):
    q_sum_loc, q_sum, c = primals
    t = tangents[0]
    if _is_zero(t):
        return _cummax_loc(q_sum_loc, q_sum, c, reverse), jnp.zeros_like(q_sum_loc)
    # the tangent of the original expression, computed by JAX's own cummax JVP on the original argument
    out, t_out = jax.jvp(partial(lax.cummax, axis=1, reverse=reverse), (q_sum_loc,), (t,))
    return out, t_out


_cummax_loc.defjvp(_cummax_loc_jvp, symbolic_zeros=True)


def _scaled_tangent(t_sl, t_thr, X_sl, thr, z, t_z):
    """Tangent of (X_sl - thr) * z exactly as JAX's sub/mul JVPs build it (symbolic zeros skipped)."""
    t = None
    if t_sl is not None:
        t = t_sl if _is_zero(t_thr) else t_sl - t_thr
    elif not _is_zero(t_thr):
        t = -t_thr * jnp.ones_like(X_sl)
    out = None if t is None else t * z
    if not _is_zero(t_z):
        o2 = (X_sl - thr) * t_z
        out = o2 if out is None else out + o2
    return jnp.zeros_like(X_sl) if out is None else out


def _pair_slices(X, idx):
    """(X[..., idx] or X[..., 1:], X[..., :-1])"""
    first = X[..., 1:] if idx is None else X[..., idx]
    return first, X[..., :-1]


def _lndtr_pair_impl(X, thr, z, idx):
    A = (X - thr) * z
    LA = _log_ndtr_fast(A)
    a, b = _pair_slices(A, idx)
    la, lb = _pair_slices(LA, idx)
    return a, b, la, lb


@jax.custom_jvp
def _lndtr_pair_guess(X, thr, z):
    """== (a, b, log_ndtr(a), log_ndtr(b)) with a = (X[..., 1:] - thr) * z, b = (X[..., :-1] - thr) * z."""
    return _lndtr_pair_impl(X, thr, z, None)


@jax.custom_jvp
def _lndtr_pair_event(X, thr, z, idx):
    """== (a, b, log_ndtr(a), log_ndtr(b)) with a = (X[..., idx] - thr) * z, b = (X[..., :-1] - thr) * z."""
    return _lndtr_pair_impl(X, thr, z, idx)


def _lndtr_pair_jvp_common(X, thr, z, idx, tX, tthr, tz):
    A = (X - thr) * z
    LA = _log_ndtr_fast(A)
    D = lax.exp(lax.sub(_jss_norm_logpdf(A), LA))          # log_ndtr's own JVP factor, elementwise
    a, b = _pair_slices(A, idx)
    la, lb = _pair_slices(LA, idx)
    Da, Db = _pair_slices(D, idx)
    Xa, Xb = _pair_slices(X, idx)
    # same order as the original: tangent slice/gather for a first, then for b
    tXa = None if _is_zero(tX) else _pair_slices(tX, idx)[0]
    tXb = None if _is_zero(tX) else tX[..., :-1]
    ta = _scaled_tangent(tXa, tthr, Xa, thr, z, tz)
    tb = _scaled_tangent(tXb, tthr, Xb, thr, z, tz)
    tla = lax.mul(ta, Da)
    tlb = lax.mul(tb, Db)
    return (a, b, la, lb), (ta, tb, tla, tlb)


@_lndtr_pair_guess.defjvp
def _lndtr_pair_guess_jvp(primals, tangents):
    X, thr, z = primals
    tX, tthr, tz = tangents
    return _lndtr_pair_jvp_common(X, thr, z, None, tX, tthr, tz)


@_lndtr_pair_event.defjvp
def _lndtr_pair_event_jvp(primals, tangents):
    X, thr, z, idx = primals
    tX, tthr, tz, _ = tangents
    return _lndtr_pair_jvp_common(X, thr, z, idx, tX, tthr, tz)


_lndtr_pair_guess.defjvp(_lndtr_pair_guess_jvp, symbolic_zeros=True)
_lndtr_pair_event.defjvp(_lndtr_pair_event_jvp, symbolic_zeros=True)


def _log_diff_ndtr_pre(a, b, la, lb, min_log_prob=-18.42):
    """log_diff_ndtr(a, b) given la = log_ndtr(a), lb = log_ndtr(b) (same ops as log_diff_ndtr after those two)."""
    safe_diff = jnp.where(a > b, lb - la, -1.0)
    neg_expm1 = -jnp.expm1(safe_diff)
    eps = 1e-30
    neg_expm1_safe = _soft_max(neg_expm1, eps, sharpness=1e10)
    log_term = jnp.log(neg_expm1_safe)
    log_term_safe = _soft_max(log_term, min_log_prob, sharpness=10.0)
    log_prob = la + log_term_safe
    return jnp.where(a > b, log_prob, min_log_prob)


def _keys_interp(arr, pos):
    """Keys/Catmull-Rom cubic interpolation of arr (..., Nticks) at fractional positions pos (M,).
    Exact at integer positions; weights sum to 1 (so it commutes with per-path constant offsets)."""
    Nticks = arr.shape[-1]
    fl = jnp.floor(pos)
    f = pos - fl
    taps = jnp.clip(fl.astype(int)[:, None] + jnp.array([-1, 0, 1, 2])[None, :], 0, Nticks - 1)
    w = jnp.stack([-0.5 * f ** 3 + f ** 2 - 0.5 * f,
                   1.5 * f ** 3 - 2.5 * f ** 2 + 1.0,
                   -1.5 * f ** 3 + 2.0 * f ** 2 + 0.5 * f,
                   0.5 * f ** 3 - 0.5 * f ** 2], axis=-1)
    return jnp.sum(arr[..., taps] * w, axis=-1)


_BS3_K = 14                                                   # prefilter half-length: |sqrt(3)-2|^15 ~ 3e-9
_BS3_H = (3.0 ** 0.5) * (3.0 ** 0.5 - 2.0) ** jnp.abs(jnp.arange(-_BS3_K, _BS3_K + 1)).astype(jnp.float32)


def _bspline3_coeffs(arr):
    """Cubic B-spline coefficients of arr (..., N) for C2 interpolation: c = arr * h with h[k] = sqrt(3)(sqrt(3)-2)^|k|
    (the exact inverse of the [1, 4, 1]/6 sampling filter, truncated at |k| <= 14), edge-replicated boundaries.
    Returns c on the extended index range -2 .. N+1 (shape (..., N+4))."""
    N = arr.shape[-1]; P = _BS3_K + 2
    pad = jnp.pad(arr, [(0, 0)] * (arr.ndim - 1) + [(P, P)], mode='edge')
    c = jnp.zeros(arr.shape[:-1] + (N + 4,), arr.dtype)
    for j, k in enumerate(range(-_BS3_K, _BS3_K + 1)):
        c = c + _BS3_H[j].astype(arr.dtype) * pad[..., P + k - 2: P + k - 2 + N + 4]
    return c


def _bspline3_eval(c_ext, pos):
    """Evaluate the cubic B-spline with coefficients c_ext (from _bspline3_coeffs) at fractional positions pos (M,).
    C2 in pos everywhere (Keys is only C1, with second-derivative jumps at every integer); reproduces the samples at
    integer positions to the prefilter truncation (~1e-7 relative in float32)."""
    Ne = c_ext.shape[-1]
    fl = jnp.floor(pos)
    f = pos - fl
    taps = jnp.clip(fl.astype(int)[:, None] + jnp.array([1, 2, 3, 4])[None, :], 0, Ne - 1)   # samples i-1 .. i+2
    w = jnp.stack([(1.0 - f) ** 3 / 6.0,
                   (3.0 * f ** 3 - 6.0 * f ** 2 + 4.0) / 6.0,
                   (-3.0 * f ** 3 + 3.0 * f ** 2 + 3.0 * f + 1.0) / 6.0,
                   f ** 3 / 6.0], axis=-1)
    return jnp.sum(c_ext[..., taps] * w, axis=-1)


def _drift_mode_fee_precompute(params, q_sum_all, interval):
    """Drift-coordinate mode: the electronics clocks are fixed in REAL time, so on the u-grid the ADC/dead-time
    window spans interval * r u-ticks (fractional), r = v(E)/v_ref. All fractional-position interpolations are
    hoisted out of the beam-search scan (see _find_one_hit_step_frac). At r = 1 every position is an integer and
    the interpolations reproduce the integer-window step's q_sum[shifted_ticks] etc. (exactly for Keys, to ~1e-7 for
    the default C2 B-spline).

    Each step is real time, so all of it scales with r, measured from the mean crossing position k + 1/2 of tick k (the
    expected ADC value subtracts 0.5 * (q[k] + q[k+1])): ADC read after interval + 1/2 ticks, reset 1 tick later, earliest
    next hit end after interval + 1 more. Scaling only the interval closes the ADC window (r - 1)/2 u-tick early and leaves
    the reset step at 1 u-tick, i.e. 1/r real ticks: dead time 1 - 1/r too short (run_efield_gap.py: first-hit charge low by
    0.09-0.11 points and all-hit charge high by 0.11-0.16 points at +10 % eField). Legacy mode keeps that behaviour."""
    from larndsim.consts_jax import get_vdrift as _gv
    Nticks = q_sum_all.shape[-1]
    r = _gv(params) / params.u_grid_vdrift
    k = jnp.arange(Nticks - 1).astype(jnp.float32)
    if params.drift_mode_stretch:
        wpos = k + interval * r + 1.0
        wpos1, fhe = wpos + 1.0, wpos + interval * r + 1.0
    else:
        wpos = k + 0.5 + r * (interval + 0.5)
        wpos1, fhe = k + 0.5 + r * (interval + 1.5), k + 0.5 + r * (2 * interval + 1.5)
    wpos, wpos1, fhe = (jnp.clip(x, 0.0, Nticks - 1.0) for x in (wpos, wpos1, fhe))
    qmax_all = lax.cummax(q_sum_all, axis=1)
    q_rmax = lax.cummax(q_sum_all, axis=1, reverse=True)
    if params.drift_mode_c2:
        # C2 window edges: every window-end knot sits at r = m/interval -- including r = 1, the usual expansion point --
        # so with Keys (C1) the AD Hessian in eField is one-sided there.
        cq = _bspline3_coeffs(q_sum_all)
        qs_w_all, qs_w1_all = _bspline3_eval(cq, wpos), _bspline3_eval(cq, wpos1)
        mf_fhe_all = _bspline3_eval(_bspline3_coeffs(q_rmax), fhe)
    else:
        qs_w_all, qs_w1_all = _keys_interp(q_sum_all, wpos), _keys_interp(q_sum_all, wpos1)
        mf_fhe_all = _keys_interp(q_rmax, fhe)
    return qs_w_all, qs_w1_all, qmax_all, mf_fhe_all


def _find_one_hit_step_frac(q_sum, qs_w, qs_w1, qmax, mf_fhe, prev_charges, previous_log_prob,
                            sigma, sigma_uncorr, threshold, Nvalues, min_log_prob,
                            excess_offset=0.0, smooth_sigma_scale=1.0):
    """Drift-coordinate-mode step with FRACTIONAL window edges (real-time-fixed electronics clocks).

    Same physics as _find_one_hit_step (min_log_prob floors, jnp.where in log_diff_ndtr, unified Gaussian
    acceptance over the uncorrelated noise) with the integer-window gathers replaced by the hoisted fractional
    ones: because interpolation weights sum to 1 and cummax commutes with constant offsets,
    interp(q_sum - c, pos) = interp(q_sum, pos) - c, so qs_w = interp(q_sum, wpos), qs_w1 = interp(q_sum, wpos+1),
    qmax = cummax(q_sum), mf_fhe = interp(cummax_rev(q_sum), fhe) suffice and the step body is gather-free.
    Mirrors the LARND_FEE_FAST=0 expressions of _find_one_hit_step term by term (no custom_jvp fast path).
    """
    z_scale = 1.0 / sigma
    c = prev_charges[..., None]

    # 1. Log-probability of a hit starting
    q_max_future = qmax - c                          # cummax(q_sum - c) = cummax(q_sum) - c
    log_guess = log_diff_ndtr(
        (q_max_future[..., 1:] - threshold) * z_scale,
        (q_max_future[..., :-1] - threshold) * z_scale,
        min_log_prob=min_log_prob
    )

    # 2. Log-probability of a hit being collected in the (fractional) shifted window
    log_prob_event = log_diff_ndtr(
        (qs_w - c - threshold) * z_scale,            # interp(q_sum - c, wpos) = qs_w - c
        (q_sum[..., :-1] - c - threshold) * z_scale,
        min_log_prob=min_log_prob
    )
    log_prob_event = jnp.minimum(log_prob_event, log_guess)
    log_prob_event = _soft_max(log_prob_event, min_log_prob, sharpness=1.0)

    esperance_value = qs_w + threshold - 0.5 * (q_sum[..., 1:] + q_sum[..., :-1])

    # Unified Gaussian acceptance correction (same as _find_one_hit_step)
    safe_sigma_uncorr = jnp.maximum(sigma_uncorr * smooth_sigma_scale, 1.0)
    log_accept = jss.log_ndtr((esperance_value - threshold + excess_offset) / safe_sigma_uncorr)
    log_prob_event = log_prob_event + log_accept

    # 3. Aggregate Results
    log_prob_distrib = log_prob_event + previous_log_prob[:, None]
    log_total_hit_dist_tick = jax.nn.logsumexp(log_prob_distrib, axis=0)
    log_total_dist_tick = jax.nn.logsumexp(log_guess + previous_log_prob[:, None], axis=0)

    # 4. Path Selection
    next_q_sum = qs_w1 - c
    log_future_hit_prob = jss.log_ndtr(
        ((mf_fhe - c) - next_q_sum - threshold) * z_scale
    )
    log_selection_prob = log_guess + log_future_hit_prob + previous_log_prob[:, None]
    log_total_sel_prob_tick = jax.nn.logsumexp(log_selection_prob, axis=0)

    # 5. Construct State for Next Iteration (stop_gradient on discrete indices)
    _, top_k_ticks = lax.top_k(log_total_sel_prob_tick, k=Nvalues)
    top_k_ticks = lax.stop_gradient(top_k_ticks)
    new_log_prob = log_total_dist_tick[top_k_ticks]
    charges_new = qs_w1[top_k_ticks]                 # q_sum at (fractional) window-end + 1

    return (charges_new, new_log_prob), (log_total_hit_dist_tick, esperance_value)


def _find_one_hit_step(q_sum, prev_charges, previous_log_prob, sigma, sigma_uncorr,
                       threshold, interval, Nvalues, min_log_prob,
                       excess_offset=0.0, smooth_sigma_scale=1.0, n_paths=None):
    """n_paths (LARND_FEE_PEEL0, fee2 23 Oct): prev_charges holds ONE charge shared by all n_paths beam paths (step 0:
    every path starts at charge 0). The per-path arrays are then computed once, (1, T), and broadcast to (n_paths, T)
    where they meet previous_log_prob: the same values as the n_paths identical rows of the original (forward
    bitwise); the backward sums the path cotangents before the shared computation instead of after it (round-off)."""
    Nticks = q_sum.shape[0]
    z_scale = 1.0 / sigma
    bc = (lambda x: x) if n_paths is None else (lambda x: jnp.broadcast_to(x, (n_paths,) + x.shape[1:]))

    shifted_ticks = jnp.arange(Nticks - 1) + interval + 1
    shifted_ticks = jnp.clip(shifted_ticks, 0, Nticks - 1)
    q_sum_loc = q_sum - prev_charges[..., None]

    if _FEE_FAST:
        q_max_future = _cummax_loc(q_sum_loc, q_sum, prev_charges, False)
        log_guess = _log_diff_ndtr_pre(*_lndtr_pair_guess(q_max_future, threshold, z_scale), min_log_prob=min_log_prob)
        log_prob_event = _log_diff_ndtr_pre(*_lndtr_pair_event(q_sum_loc, threshold, z_scale, shifted_ticks),
                                            min_log_prob=min_log_prob)
    else:
        # 1. Log-probability of a hit starting
        q_max_future = lax.cummax(q_sum_loc, axis=1)
        log_guess = log_diff_ndtr(
            (q_max_future[..., 1:] - threshold) * z_scale,
            (q_max_future[..., :-1] - threshold) * z_scale,
            min_log_prob=min_log_prob
        )

        # 2. Log-probability of a hit being collected in the shifted window
        log_prob_event = log_diff_ndtr(
            (q_sum_loc[..., shifted_ticks] - threshold) * z_scale,
            (q_sum_loc[..., :-1] - threshold) * z_scale,
            min_log_prob=min_log_prob
        )
    log_prob_event = jnp.minimum(log_prob_event, log_guess)
    log_prob_event = _soft_max(log_prob_event, min_log_prob, sharpness=1.0)

    # esperance_value: expected ADC given a crossing at tick t.
    # esperance_value is kept because it is returned as the expected-charge output.
    esperance_value = q_sum[..., shifted_ticks] + threshold - 0.5 * (q_sum[..., 1:] + q_sum[..., :-1])

    # Unified Gaussian acceptance correction (Fix 2, corrected)
    # Replace both log_adc_correction and _soft_where with a single correction
    # that integrates over the uncorrelated noise U, since reset noise N cancels out.
    safe_sigma_uncorr = jnp.maximum(sigma_uncorr * smooth_sigma_scale, 1.0)
    log_accept = (_log_ndtr_fast if _FEE_FAST else jss.log_ndtr)((esperance_value - threshold + excess_offset) / safe_sigma_uncorr)
    log_prob_event = log_prob_event + log_accept

    # 3. Aggregate Results
    log_prob_distrib = bc(log_prob_event) + previous_log_prob[:, None]
    log_total_hit_dist_tick = jax.nn.logsumexp(log_prob_distrib, axis=0)
    log_total_dist_tick = jax.nn.logsumexp(bc(log_guess) + previous_log_prob[:, None], axis=0)

    # 4. Path Selection
    next_q_sum = q_sum_loc[:, jnp.clip(shifted_ticks + 1, 0, Nticks - 1)]
    max_future_signal = _cummax_loc(q_sum_loc, q_sum, prev_charges, True) if _FEE_FAST \
        else lax.cummax(q_sum_loc, axis=1, reverse=True)
    future_hit_earliest_end = jnp.clip(shifted_ticks + interval + 1, 0, Nticks - 1)
    log_future_hit_prob = (_log_ndtr_fast if _FEE_FAST else jss.log_ndtr)(
        (max_future_signal[:, future_hit_earliest_end] - next_q_sum - threshold) * z_scale
    )
    log_selection_prob = bc(log_guess + log_future_hit_prob) + previous_log_prob[:, None]
    log_total_sel_prob_tick = jax.nn.logsumexp(log_selection_prob, axis=0)

    # 5. Construct State for Next Iteration (stop_gradient on discrete indices)
    _, top_k_ticks = lax.top_k(log_total_sel_prob_tick, k=Nvalues)
    top_k_ticks = lax.stop_gradient(top_k_ticks)
    new_log_prob = log_total_dist_tick[top_k_ticks]
    best_path_next_ticks = jnp.clip(shifted_ticks[top_k_ticks] + 1, 0, Nticks - 1)
    charges_new = q_sum[best_path_next_ticks]

    return (charges_new, new_log_prob), (log_total_hit_dist_tick, esperance_value)

@partial(jit, static_argnums=(2, 3))
def get_adc_values_average_noise_vmap(params, wfs, stop_threshold=1e-9, min_log_prob=-18.42):
    """
    A globally-stopped, vmap-based implementation of the beam search. This is a highly optimized pattern for parallel hardware.
    """

    Npix, Nticks = wfs.shape
    Nvalues = params.fee_paths_scaling
    interval = round((3 * params.CLOCK_CYCLE + params.ADC_HOLD_DELAY * params.CLOCK_CYCLE) / params.t_sampling)
    
    # --- Vectorize the single-step function ---
    vmapped_step_fun = vmap(
        _find_one_hit_step,
        in_axes=(0, 0, 0, None, None, None, None, None, None, None, None)  # Map over q_sum, charges, probs
    )

    # --- Pre-calculate q_sum for all pixels ---
    q = wfs * params.t_sampling
    q_sum_all = q.cumsum(axis=-1)

    if params.drift_coordinate_mode:
        # Drift-coordinate mode: fractional (real-time-fixed) FEE windows, see _drift_mode_fee_precompute.
        # The LARND_FEE_PEEL0 / LARND_FEE_PIXEL_STOP=2 paths below hard-code integer windows and are bypassed.
        qs_w_all, qs_w1_all, qmax_all, mf_fhe_all = _drift_mode_fee_precompute(params, q_sum_all, interval)
        vmapped_step_fun_frac = vmap(
            _find_one_hit_step_frac,
            in_axes=(0, 0, 0, 0, 0, 0, 0) + (None,) * 7
        )
    

    def global_scan_fun(carry, _):
        """The main scan function with a global conditional stop."""
        prev_charges_all, prev_probs_all, is_active_global = carry

        # Apply jax.checkpoint to the expensive branch of the computation.
        # This tells JAX not to store intermediate values from this function,
        # saving memory at the cost of some re-computation.
        @checkpoint
        def _active_branch(operand):
            """The expensive vmapped computation, only run when globally active."""
            charges, probs, _ = operand
            # Run one hit-finding step for all pixels in parallel
            if params.drift_coordinate_mode:
                (new_charges, new_probs), (prob_dist, charge_dist) = vmapped_step_fun_frac(
                    q_sum_all, qs_w_all, qs_w1_all, qmax_all, mf_fhe_all, charges, probs,
                    params.RESET_NOISE_CHARGE, params.UNCORRELATED_NOISE_CHARGE,
                    params.DISCRIMINATION_THRESHOLD, Nvalues, min_log_prob,
                    params.excess_offset, params.smooth_sigma_scale
                )
            else:
                (new_charges, new_probs), (prob_dist, charge_dist) = vmapped_step_fun(
                    q_sum_all, charges, probs,
                    params.RESET_NOISE_CHARGE, params.UNCORRELATED_NOISE_CHARGE,
                    params.DISCRIMINATION_THRESHOLD, interval,
                    Nvalues, min_log_prob,
                    params.excess_offset, params.smooth_sigma_scale
                )
            if _FEE_PIXEL_STOP:
                # per-pixel stop, emulation (=1: everything computed, then selected; see LARND_FEE_PIXEL_STOP)
                alive = jax.nn.logsumexp(probs, axis=1) > jnp.log(stop_threshold) - _FEE_STOP_MARGIN
                new_charges = jnp.where(alive[:, None], new_charges, charges)
                new_probs = jnp.where(alive[:, None], new_probs, probs)
                low = _FEE_STOP_DEAD == 'low'
                prob_dist = jnp.where(alive[:, None], prob_dist, jnp.float32(_FEE_DEAD_LOGP if low else min_log_prob))
                if not low:
                    charge_dist = jnp.where(alive[:, None], charge_dist, jnp.float32(0.0))
            # LogSumExp gives the log of the total probability
            total_prob_per_pixel = jax.nn.logsumexp(new_probs, axis=1)
            new_active = jnp.any(total_prob_per_pixel > jnp.log(stop_threshold))
            return (new_charges, new_probs, new_active), (prob_dist, charge_dist)

        def _inactive_branch(operand):
            """A cheap pass-through, executed when the whole batch is inactive."""
            return operand, (
                jnp.full((Npix, Nticks - 1), min_log_prob, dtype=jnp.float32),
                jnp.zeros((Npix, Nticks - 1), dtype=jnp.float32),
                )

        # --- Global Conditional Execution ---
        final_carry, final_outputs = lax.cond(
            is_active_global,
            _active_branch,
            _inactive_branch,
            carry
        )
        return final_carry, final_outputs

    # per-pixel expected charge of a crossing (== esperance_value of _find_one_hit_step: path/state independent)
    esp_all = None
    if _FEE_PIXEL_STOP == '2' and _FEE_STOP_DEAD == 'low':
        _sh = jnp.clip(jnp.arange(Nticks - 1) + interval + 1, 0, Nticks - 1)
        esp_all = q_sum_all[:, _sh] + params.DISCRIMINATION_THRESHOLD - 0.5 * (q_sum_all[:, 1:] + q_sum_all[:, :-1])

    def _rows_step(q_rows, charges, probs):
        return vmapped_step_fun(q_rows, charges, probs,
                                params.RESET_NOISE_CHARGE, params.UNCORRELATED_NOISE_CHARGE,
                                params.DISCRIMINATION_THRESHOLD, interval, Nvalues, min_log_prob,
                                params.excess_offset, params.smooth_sigma_scale)

    def pixel_scan_fun(carry, _):
        """LARND_FEE_PIXEL_STOP=2: per-pixel stop with the live rows compacted.
        carry = (charges, probs, live[Npix] (rows to compute: total > stop * exp(-margin)), chunk_active)."""
        log_stop = jnp.log(stop_threshold)
        low = _FEE_STOP_DEAD == 'low'
        dead_lp = jnp.float32(_FEE_DEAD_LOGP if low else min_log_prob)

        def _new_flags(npr):
            tot = jax.nn.logsumexp(npr, axis=1)
            return tot > log_stop - _FEE_STOP_MARGIN, jnp.any(tot > log_stop)

        def _inactive(operand):
            return operand, (jnp.full((Npix, Nticks - 1), min_log_prob, dtype=jnp.float32),
                             jnp.zeros((Npix, Nticks - 1), dtype=jnp.float32))

        @checkpoint
        def _full(operand):
            c, p, al, _ = operand
            (nc, npr), (pd, cd) = _rows_step(q_sum_all, c, p)
            nc = jnp.where(al[:, None], nc, c)
            npr = jnp.where(al[:, None], npr, p)
            pd = jnp.where(al[:, None], pd, dead_lp)
            if not low:
                cd = jnp.where(al[:, None], cd, jnp.float32(0.0))
            return (nc, npr) + _new_flags(npr), (pd, cd)

        def _compact(K):
            @checkpoint
            def br(operand):
                c, p, al, _ = operand
                idx = jnp.nonzero(al, size=K, fill_value=Npix)[0]          # live rows first, then out-of-range
                q_r = jnp.take(q_sum_all, idx, axis=0, mode='clip')
                c_r = jnp.take(c, idx, axis=0, mode='clip')
                p_r = jnp.take(p, idx, axis=0, mode='clip')
                (nc, npr), (pd, cd) = _rows_step(q_r, c_r, p_r)
                nc = c.at[idx].set(nc, mode='drop')
                npr = p.at[idx].set(npr, mode='drop')
                pd = jnp.full((Npix, Nticks - 1), dead_lp, dtype=jnp.float32).at[idx].set(pd, mode='drop')
                cd = (esp_all if low else jnp.zeros((Npix, Nticks - 1), dtype=jnp.float32)).at[idx].set(cd, mode='drop')
                return (nc, npr) + _new_flags(npr), (pd, cd)
            return br

        ks = [k for k in _FEE_STOP_LADDER if k < Npix]
        n = jnp.sum(carry[2].astype(jnp.int32))
        index = jnp.where(carry[3], 1 + (jnp.sum((n > jnp.array(ks, dtype=jnp.int32)).astype(jnp.int32)) if ks else 0), 0)
        return lax.switch(index, [_inactive] + [_compact(k) for k in ks] + [_full], carry)

    # --- Setup and Execute the Global Scan ---
    initial_charges = jnp.zeros((Npix, Nvalues), dtype=jnp.float32)
    initial_log_probs = jnp.full((Npix, Nvalues), min_log_prob, dtype=jnp.float32).at[:, 0].set(0.0)
    initial_active = jnp.array(True)
    
    init_loop = (initial_charges, initial_log_probs, initial_active)

    if _FEE_PEEL0 and params.MAX_ADC_VALUES > 1 and not params.drift_coordinate_mode:
        # Step 0 peeled out of the scan (LARND_FEE_PEEL0): it is always active (initial_active = True), and its 100 beam
        # paths all start at charge 0, so the per-path work is done once per pixel (n_paths, see _find_one_hit_step).
        vstep0 = vmap(partial(_find_one_hit_step, n_paths=Nvalues),
                      in_axes=(0, 0, 0, None, None, None, None, None, None, None, None))

        @checkpoint
        def _step0(q_sum, log_probs0):
            (new_charges, new_probs), (prob_dist, charge_dist) = vstep0(
                q_sum, jnp.zeros((Npix, 1), dtype=jnp.float32), log_probs0,
                params.RESET_NOISE_CHARGE, params.UNCORRELATED_NOISE_CHARGE,
                params.DISCRIMINATION_THRESHOLD, interval,
                Nvalues, min_log_prob,
                params.excess_offset, params.smooth_sigma_scale
            )
            total_prob_per_pixel = jax.nn.logsumexp(new_probs, axis=1)
            new_active = jnp.any(total_prob_per_pixel > jnp.log(stop_threshold))
            return (new_charges, new_probs, new_active), (prob_dist, charge_dist)

        carry1, (prob0, charge0) = _step0(q_sum_all, initial_log_probs)
        if _FEE_PIXEL_STOP == '2':
            _tot = jax.nn.logsumexp(carry1[1], axis=1)
            carry1 = (carry1[0], carry1[1], _tot > jnp.log(stop_threshold) - _FEE_STOP_MARGIN, carry1[2])
        _, (lp_rest, ch_rest) = lax.scan(pixel_scan_fun if _FEE_PIXEL_STOP == '2' else global_scan_fun, carry1,
                                         jnp.arange(1, params.MAX_ADC_VALUES))
        log_prob_distrib = jnp.concatenate([prob0[None], lp_rest], axis=0)
        charge_distrib = jnp.concatenate([charge0[None], ch_rest], axis=0)
    elif _FEE_PIXEL_STOP == '2' and not params.drift_coordinate_mode:
        _, (log_prob_distrib, charge_distrib) = lax.scan(
            pixel_scan_fun, (initial_charges, initial_log_probs, jnp.ones((Npix,), dtype=bool), initial_active),
            jnp.arange(0, params.MAX_ADC_VALUES))
    else:
        _, (log_prob_distrib, charge_distrib) = lax.scan(global_scan_fun, init_loop, jnp.arange(0, params.MAX_ADC_VALUES))
    # log_prob_distrib = jnp.full((Npix, params.MAX_ADC_VALUES, Nticks - 1), -jnp.inf, dtype=jnp.float32)
    # charge_distrib = jnp.zeros((Npix, params.MAX_ADC_VALUES, Nticks - 1), dtype=jnp.float32)
    # carry = init_loop
    # for _ in range(params.MAX_ADC_VALUES):
    #     carry, (log_prob_step, charge_step) = global_scan_fun(carry, None)
    #     log_prob_distrib = log_prob_distrib.at[:, _, :].set(log_prob_step)
    #     charge_distrib = charge_distrib.at[:, _, :].set(charge_step)


    return jnp.moveaxis(log_prob_distrib, 0, 1), jnp.moveaxis(charge_distrib, 0, 1)

@jit
def get_adc_values_markov(params, wfs, min_log_prob=-18.42):
    """
    Computes the Markov transition matrices for hit-finding.
    
    Returns:
        log_P1: (Npix, Nticks) log-prob of the first hit.
        log_T: (Npix, Nticks, Nticks) transition matrix T[a, b] = P(hit_i+1 = b | hit_i = a).
        expected_Q: (Npix, Nticks, Nticks) expected charge for a hit at b given previous at a.
        log_P_none: (Npix, Nticks) log-prob of no further hits given current at a.
    """
    Npix, Nticks = wfs.shape
    interval = round((3 * params.CLOCK_CYCLE + params.ADC_HOLD_DELAY * params.CLOCK_CYCLE) / params.t_sampling)
    sigma = params.RESET_NOISE_CHARGE
    threshold = params.DISCRIMINATION_THRESHOLD
    z_scale = 1.0 / sigma

    q = wfs * params.t_sampling
    q_sum = q.cumsum(axis=-1)

    # We need to compute log_P_first(t | start_at r) for all r
    # This can be done by vectorizing over r.
    # To save memory/compute, we only consider r that are valid reset points.
    
    def compute_first_hit_dist(r_idx, q_sum_pix):
        # q_sum_pix: (Nticks,)
        # r_idx: scalar start tick
        
        # Charge relative to the reset point
        # If r=0, prev_charge = 0. Else prev_charge = q_sum_pix[r-1]
        prev_charge = jnp.where(r_idx > 0, q_sum_pix[jnp.clip(r_idx-1, 0, Nticks-1)], 0.0)
        q_sum_loc = q_sum_pix - prev_charge
        
        # Mask out ticks before r
        mask_past = jnp.arange(Nticks) < r_idx
        q_sum_loc = jnp.where(mask_past, -1e10, q_sum_loc)
        
        q_max_future = lax.cummax(q_sum_loc, axis=0)
        
        # Log-probability of a hit starting at tick t
        # (Using the same logic as _find_one_hit_step)
        log_guess = log_diff_ndtr(
            (q_max_future[1:] - threshold) * z_scale,
            (q_max_future[:-1] - threshold) * z_scale,
            min_log_prob=min_log_prob
        )
        # Pad back to Nticks
        log_guess = jnp.concatenate([jnp.full((1,), min_log_prob), log_guess])
        
        # Expected charge if hit is found at tick t
        # In the Markov case, we don't have a window, it's exactly at t.
        shifted_ticks = jnp.clip(jnp.arange(Nticks) + interval, 0, Nticks - 1)
        expected_q = q_sum_pix[shifted_ticks] + threshold - 0.5 * (q_sum_pix[jnp.clip(jnp.arange(Nticks), 0, Nticks-1)] + q_sum_pix[jnp.clip(jnp.arange(Nticks)-1, 0, Nticks-1)])
        # Expected charge Q represents absolute charge difference in the window + threshold
        
        # Probability of NO hit being found at all starting from r
        log_p_none = jss.log_ndtr((threshold - jnp.max(q_sum_loc)) * z_scale)
        
        return log_guess, expected_q, log_p_none

    @checkpoint
    def compute_batch_for_r(r_idx, q_sum_batch):
        # We vmap over the pixel dimension (axis 0 of q_sum_batch)
        # for a single previous tick r_idx.
        return vmap(compute_first_hit_dist, in_axes=(None, 0))(r_idx, q_sum_batch)

    def scan_over_r(carry, r_idx):
        return carry, compute_batch_for_r(r_idx, q_sum)

    # Use lax.scan to compute the transition matrix rows sequentially.
    # This prevents JAX from materializing (Npix, Nticks, Nticks) intermediates at once.
    all_r = jnp.arange(Nticks)
    _, (log_T_raw, expected_Q_raw, log_P_none_raw) = lax.scan(scan_over_r, None, all_r)
    
    # Moveaxis from (Nticks_r, Npix, Nticks_t) -> (Npix, Nticks_r, Nticks_t)
    log_T_raw = jnp.moveaxis(log_T_raw, 0, 1)
    expected_Q_raw = jnp.moveaxis(expected_Q_raw, 0, 1)
    log_P_none_raw = jnp.moveaxis(log_P_none_raw, 0, 1)
    
    # log_P1 is the hit distribution starting at r=0
    log_P1 = log_T_raw[:, 0, :]
    Q1 = expected_Q_raw[:, 0, :]
    
    # Transition matrix T[a, b]: prob of next hit at b given previous at a
    # If previous hit was at a, the BEAM SEARCH reset logic sets next search start at r = a + interval + 2
    # (See _find_one_hit_step: best_path_next_ticks = shifted_ticks + 1, and shifted_ticks = t + interval + 1)
    next_r = jnp.clip(jnp.arange(Nticks) + interval + 2, 0, Nticks - 1)
    
    log_T = log_T_raw[:, next_r, :]
    expected_Q = expected_Q_raw[:, next_r, :]
    log_P_none = log_P_none_raw[:, next_r]
    log_p_none_at_zero = log_P_none_raw[:, 0]
    
    # Apply causality mask: T[a, b] = 0 if b <= a + interval
    causality_mask = jnp.arange(Nticks)[None, :] > (jnp.arange(Nticks)[:, None] + interval)
    log_T = jnp.where(causality_mask[None, :, :], log_T, min_log_prob)
    
    return log_P1, log_T, expected_Q, log_P_none, Q1, log_p_none_at_zero

@jit
def get_average_hit_values(log_ticks_prob, adcs_distrib, min_log_prob=-18.42):
    """
    Computes expected hit values (tick, ADC, probability) from the probabilistic grid.

    Args:
        log_ticks_prob: Log-probabilities for each bin (Npix, Nhits, Nticks).
        adcs_distrib: ADC distribution for each bin.
        min_log_prob: Minimum log-probability floor (log(1e-8) ~ -18.42).
    """
    Npix, Nhits, Nticks = log_ticks_prob.shape

    # Exponentiate to linear probabilities
    # We clip to min_log_prob to avoid extremely negative values affecting the sum
    ticks_prob = jnp.exp(jnp.maximum(log_ticks_prob, min_log_prob))

    # For each (pixel, hit_index), compute λ = Σ_t P(tick | pixel, hit_index)
    # This represents the expected probability of this particular hit existing
    lambda_per_hit = jnp.sum(ticks_prob, axis=2)  # (Npix, Nhits)

    # For each (pixel, hit_index), compute expected tick
    # E[tick | pixel, hit_index] = Σ_t t * P(tick | pixel, hit_index) / λ
    tick_range = jnp.arange(Nticks)  # (Nticks,)
    expected_ticks_per_hit = jnp.sum(
            tick_range[None, None, :] * ticks_prob, axis=2
        ) / jnp.maximum(lambda_per_hit, 1e-10)  # (Npix, Nhits)

    # For each (pixel, hit_index), compute expected ADC
    # E[ADC | pixel, hit_index] = Σ_t ADC(hit_index, tick) * P(tick | pixel, hit_index) / λ
    expected_adcs_per_hit = jnp.sum(
        adcs_distrib * ticks_prob, axis=2
    ) / jnp.maximum(lambda_per_hit, 1e-10)  # (Npix, Nhits)

    return expected_ticks_per_hit, expected_adcs_per_hit, lambda_per_hit

def _fee_global_pixel_stop(params, wfs, chunk_size, stop_threshold=1e-9, min_log_prob=-18.42):
    """LARND_FEE_PIXEL_STOP=3: the per-pixel-stop semantics of =1/=2 with the loops swapped: an outer scan over the hit
    steps and, inside, a scan over blocks of chunk_size rows of the LIVE pixels of the whole batch (gathered with
    jnp.nonzero; one lax.cond per block, false for the blocks past ceil(n_live / chunk_size)). The original per-chunk
    activity (the cond of get_adc_values_average_noise_vmap) is tracked exactly, per chunk. One copy of the step body
    is compiled (like the original), and the work is ~ the number of live pixels instead of 128 x active chunks."""
    P, Nticks = wfs.shape
    C = P // chunk_size
    B = _FEE_STOP_BLOCK or chunk_size                   # rows per compute block (the activity stays per chunk)
    NB = -(-P // B)
    Nvalues = params.fee_paths_scaling
    interval = round((3 * params.CLOCK_CYCLE + params.ADC_HOLD_DELAY * params.CLOCK_CYCLE) / params.t_sampling)
    log_stop = jnp.log(stop_threshold)
    low = _FEE_STOP_DEAD == 'low'
    pargs = (params.RESET_NOISE_CHARGE, params.UNCORRELATED_NOISE_CHARGE, params.DISCRIMINATION_THRESHOLD, interval,
             Nvalues, min_log_prob, params.excess_offset, params.smooth_sigma_scale)
    vstep = vmap(_find_one_hit_step, in_axes=(0, 0, 0, None, None, None, None, None, None, None, None))
    q_sum_all = (wfs * params.t_sampling).cumsum(axis=-1)
    sh = jnp.clip(jnp.arange(Nticks - 1) + interval + 1, 0, Nticks - 1)
    esp_all = q_sum_all[:, sh] + params.DISCRIMINATION_THRESHOLD - 0.5 * (q_sum_all[:, 1:] + q_sum_all[:, :-1])
    chunk_of_row = jnp.arange(P) // chunk_size
    f32 = jnp.float32

    def _flags(chunk_act, p_new):
        tot = jax.nn.logsumexp(p_new, axis=1)
        return chunk_act & jnp.any((tot > log_stop).reshape(C, chunk_size), axis=1), tot > log_stop - _FEE_STOP_MARGIN

    def step_fun(carry, _):
        c_all, p_all, chunk_act, live = carry
        act_rows = chunk_act[chunk_of_row]
        gate = live & act_rows
        n = jnp.sum(gate.astype(jnp.int32))
        nb = (n + B - 1) // B
        order = jnp.nonzero(gate, size=NB * B, fill_value=P)[0]

        def block_fun(_, xs):
            b, rows = xs

            @checkpoint
            def _act(rows_):
                q_r = jnp.take(q_sum_all, rows_, axis=0, mode='clip')
                c_r = jnp.take(c_all, rows_, axis=0, mode='clip')
                p_r = jnp.take(p_all, rows_, axis=0, mode='clip')
                (nc, npr), (pd, cd) = vstep(q_r, c_r, p_r, *pargs)
                return nc, npr, pd, cd

            def _skip(rows_):
                return (jnp.zeros((B, Nvalues), f32), jnp.zeros((B, Nvalues), f32),
                        jnp.zeros((B, Nticks - 1), f32), jnp.zeros((B, Nticks - 1), f32))
            return None, lax.cond(b < nb, _act, _skip, rows)

        _, (nc_s, np_s, pd_s, cd_s) = lax.scan(block_fun, None, (jnp.arange(NB), order.reshape(NB, B)))
        c_new = c_all.at[order].set(nc_s.reshape(NB * B, Nvalues), mode='drop')
        p_new = p_all.at[order].set(np_s.reshape(NB * B, Nvalues), mode='drop')
        lp0 = jnp.where(act_rows[:, None], f32(_FEE_DEAD_LOGP if low else min_log_prob), f32(min_log_prob))
        lp_out = jnp.broadcast_to(lp0, (P, Nticks - 1)).at[order].set(pd_s.reshape(NB * B, Nticks - 1), mode='drop')
        cd0 = jnp.where(act_rows[:, None], esp_all, f32(0.0)) if low else jnp.zeros((P, Nticks - 1), f32)
        cd_out = cd0.at[order].set(cd_s.reshape(NB * B, Nticks - 1), mode='drop')
        return (c_new, p_new) + _flags(chunk_act, p_new), (lp_out, cd_out)

    init_c = jnp.zeros((P, Nvalues), f32)
    init_p = jnp.full((P, Nvalues), min_log_prob, dtype=f32).at[:, 0].set(0.0)
    init_act = jnp.ones((C,), dtype=bool)
    if _FEE_PEEL0 and params.MAX_ADC_VALUES > 1:
        # step 0 (always active, all rows): per chunk, the 1-path peeled step (see LARND_FEE_PEEL0)
        vstep0 = vmap(partial(_find_one_hit_step, n_paths=Nvalues), in_axes=(0, 0, 0) + (None,) * 8)

        def chunk0(_, xs):
            @checkpoint
            def _s0(q, lp):
                (nc, npr), (pd, cd) = vstep0(q, jnp.zeros((chunk_size, 1), f32), lp, *pargs)
                return nc, npr, pd, cd
            return None, _s0(*xs)
        _, (nc0, np0, pd0, cd0) = lax.scan(chunk0, None, (q_sum_all.reshape(C, chunk_size, Nticks),
                                                          init_p.reshape(C, chunk_size, Nvalues)))
        np0 = np0.reshape(P, Nvalues)
        carry1 = (nc0.reshape(P, Nvalues), np0) + _flags(init_act, np0)
        _, (lp_rest, cd_rest) = lax.scan(step_fun, carry1, jnp.arange(1, params.MAX_ADC_VALUES))
        lp_all = jnp.concatenate([pd0.reshape(1, P, Nticks - 1), lp_rest], axis=0)
        cd_all = jnp.concatenate([cd0.reshape(1, P, Nticks - 1), cd_rest], axis=0)
    else:
        _, (lp_all, cd_all) = lax.scan(step_fun, (init_c, init_p, init_act, jnp.ones((P,), dtype=bool)),
                                       jnp.arange(0, params.MAX_ADC_VALUES))
    return jnp.moveaxis(lp_all, 0, 1), jnp.moveaxis(cd_all, 0, 1)


def get_adc_values_average_noise_chunked(params, wfs, chunk_size=128):
    """
    Chunked version of get_adc_values_average_noise_vmap using jax.lax.scan.
    This reduces peak memory footprint by processing the waveform array in sequential chunks.
    """
    Npix, Nticks = wfs.shape
    # Ensure Npix is a multiple of chunk_size
    pad_len = ((Npix + chunk_size - 1) // chunk_size) * chunk_size
    wfs_padded = jnp.pad(wfs, ((0, pad_len - Npix), (0, 0)), mode='constant', constant_values=0.0)
    
    num_chunks = pad_len // chunk_size
    # drift-coordinate mode: _fee_global_pixel_stop uses the integer-window step; the per-chunk path below runs the
    # fractional-window step (with the =1 per-pixel-stop emulation, same semantics).
    if _FEE_PIXEL_STOP == '3' and not params.drift_coordinate_mode:
        log_prob_dist, charge_dist = _fee_global_pixel_stop(params, wfs_padded, chunk_size)
        return log_prob_dist[:Npix], charge_dist[:Npix]
    wfs_chunks = wfs_padded.reshape(num_chunks, chunk_size, Nticks)
    
    _chunk_body = checkpoint(get_adc_values_average_noise_vmap) if _CHUNK_SCAN_REMAT \
        else get_adc_values_average_noise_vmap

    def scan_fn(carry, wfs_chunk):
        log_prob_step, charge_step = _chunk_body(params, wfs_chunk)
        return carry, (log_prob_step, charge_step)

    _, (log_prob_chunks, charge_chunks) = lax.scan(scan_fn, None, wfs_chunks)
    
    Nvalues = params.MAX_ADC_VALUES
    log_prob_dist = log_prob_chunks.reshape(pad_len, Nvalues, Nticks - 1)
    charge_dist = charge_chunks.reshape(pad_len, Nvalues, Nticks - 1)
    
    return log_prob_dist[:Npix], charge_dist[:Npix]