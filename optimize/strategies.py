import jax
import jax.numpy as jnp
from larndsim.sim_jax import simulate_wfs, simulate_stochastic, simulate_parametrized, simulate_probabilistic, pad_size
from larndsim.losses_jax import adc2charge, mmd
from larndsim.detsim_jax import id2pixel, get_hit_z
from larndsim.fee_jax import get_average_hit_values
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@jax.jit
def compute_occurrence_indices(ids):
    """
    Compute occurrence index (0, 1, 2, ...) for each ID in the array.

    For sorted IDs, this counts how many times each ID has appeared so far.
    Example: [100, 100, 100, 200, 200, 300] -> [0, 1, 2, 0, 1, 0]

    Args:
        ids: Array of IDs (should be sorted for meaningful results)

    Returns:
        occurrence_indices: Array where each element is its occurrence count within its ID group
    """
    id_changes = jnp.concatenate([
        jnp.array([True]), # First element is always a new group
        ids[1:] != ids[:-1]  # Compare consecutive elements
    ])

    cumsum = jnp.cumsum(jnp.ones_like(ids, dtype=jnp.int32))
    reset_values = jnp.where(id_changes, cumsum, 0)

    # JAX equivalent of np.maximum.accumulate
    reset_at_boundary = jax.lax.associative_scan(jnp.maximum, reset_values)

    occurrence_indices = cumsum - reset_at_boundary
    return occurrence_indices

def pad_to_closest_multiple(x, dims_to_pad=None, multiple=128, pad_value=0, pad_front=False):
    """
    Efficiently pads array x to the closest multiple of a given value using update-in-place syntax.
    Works with arrays of any number of dimensions.
    
    Args:
        x: Input array to pad
        dims_to_pad: List of dimension indices to pad (default: all dimensions)
        multiple: The multiple to pad to (default: 128)
        pad_value: Value to use for padding (default: 0)
    
    Returns:
        Padded array with shape target_shape
    """

    # Compute target shape by padding each dimension to the closest multiple
    if dims_to_pad is None:
        dims_to_pad = range(x.ndim)
    target_shape = list(x.shape)
    for dim in dims_to_pad:
        target_shape[dim] = ((x.shape[dim] + multiple - 1) // multiple) * multiple
    target_shape = tuple(target_shape)

    logger.info(f"Padding from shape {x.shape} to target shape {target_shape} with pad value {pad_value}")


    # 1. Create a buffer of the target static shape (allocates memory)
    buffer = jnp.full(target_shape, pad_value, dtype=x.dtype)
    
    # 2. Copy 'x' into the start of the buffer
    # Create slice tuple for all dimensions: [:x.shape[0], :x.shape[1], ...]
    if pad_front:
        slices = tuple(slice(target_shape[idim] - dim_size, None) for idim, dim_size in enumerate(x.shape))
    else:
        slices = tuple(slice(0, dim_size) for dim_size in x.shape)
    padded_x = buffer.at[slices].set(x)
    
    return padded_x

class SimulationStrategy:
    def predict(self, params, tracks, fields, rngkey):
        """
        Runs the simulation and returns a dictionary of outputs.
        """
        raise NotImplementedError

class LUTSimulation(SimulationStrategy):
    def __init__(self, response):
        self.response = response

    def predict(self, params, tracks, fields, rngkey):
        wfs, unique_pixels = simulate_wfs(params, self.response, tracks, fields)
        adcs, x, y, z, ticks, hit_prob, event, hit_pixels = simulate_stochastic(params, wfs, unique_pixels, rngseed=rngkey)
        return {
            'adcs': adcs,
            'pixel_x': x,
            'pixel_y': y,
            'pixel_z': z,
            'ticks': ticks,
            'hit_prob': hit_prob,
            'event': event,
            'hit_pixels': hit_pixels,
            'unique_pixels': unique_pixels,
            'wfs': wfs
        }

class LUTProbabilisticSimulation(SimulationStrategy):
    def __init__(self, response):
        self.response = response

    def predict(self, params, tracks, fields, rngkey):
        
        wfs, unique_pixels = simulate_wfs(params, self.response, tracks, fields)

        unique_pixels = pad_to_closest_multiple(unique_pixels, multiple=128, pad_value=-1, pad_front=True)
        wfs = pad_to_closest_multiple(wfs, dims_to_pad=(0,), multiple=128, pad_value=0.0, pad_front=True)

        adcs_distrib, pixel_x, pixel_y, ticks_prob, event = simulate_probabilistic(params, wfs, unique_pixels)
        
        # Extract pixel plane for z-coordinate calculation
        _, _, pixel_plane, _ = id2pixel(params, unique_pixels)
        
        # We return the raw distributions for the ProbabilisticLossStrategy
        return {
            'adcs_distrib': adcs_distrib, # (Npix, Nvalues, Nticks)
            'hit_prob': ticks_prob,     # (Npix, Nvalues, Nticks)
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
            'pixel_plane': pixel_plane,   # Needed for z-coordinate calculation
            'event': event,
            'unique_pixels': unique_pixels, 
            'hit_pixels': unique_pixels,
            'wfs': wfs
        }

class ParametrizedSimulation(SimulationStrategy):
    def predict(self, params, tracks, fields, rngkey):
        adcs, x, y, z, ticks, hit_prob, event, unique_pixels = simulate_parametrized(params, tracks, fields, rngseed=rngkey)
        return {
            'adcs': adcs,
            'pixel_x': x,
            'pixel_y': y,
            'pixel_z': z,
            'ticks': ticks,
            'hit_prob': hit_prob,
            'event': event,
            'unique_pixels': unique_pixels

        }

class LossStrategy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def compute(self, params, prediction, target):
        """
        Computes the loss between prediction (dict) and target (dict).
        Target is expected to have keys like 'adcs', 'pixel_x', etc.
        """
        raise NotImplementedError

class GenericLossStrategy(LossStrategy):
    def __init__(self, loss_fn, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def compute(self, params, prediction, target):
        # We need to adapt the dict output to the function signature of loss_fn
        # Most loss functions in losses_jax.py expect:
        # params, Q, x, y, z, ticks, hit_prob, event, ref_Q, ref_x, ref_y, ref_z, ref_ticks, ref_hit_prob, ref_event
        
        Q = adc2charge(prediction['adcs'], params)
        ref_Q = adc2charge(target['adcs'], params)

        return self.loss_fn(
            params, 
            Q, prediction['pixel_x'], prediction['pixel_y'], prediction['pixel_z'], prediction['ticks'], prediction['hit_prob'], prediction['event'],
            ref_Q, target['pixel_x'], target['pixel_y'], target['pixel_z'], target['ticks'], target['hit_prob'], target['event'],
            **self.kwargs
        )

class CollapsedProbabilisticLossStrategy(LossStrategy):
    def __init__(self, loss_fn, hit_threshold=1e-8, collapsed=True, prob_target=False, **kwargs):
        """
        Collapses probabilistic distributions into expected values and applies a deterministic loss.
        
        For each predicted pixel:
        - Computes λ = Σ_t P(tick|pixel) = expected number of hits
        - If λ > threshold: generates a "pseudo-hit" with expected tick and charge
        - Applies the provided loss_fn as if these were sampled hits
        
        This allows using existing loss functions (MSE, Chamfer, etc.) with probabilistic predictions.
        
        Args:
            loss_fn: A loss function with signature (params, Q, x, y, z, ticks, ..., ref_Q, ref_x, ...)
            hit_threshold: Minimum λ to generate a pseudo-hit (default 1e-8)
        """
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.hit_threshold = hit_threshold
        self.collapsed = collapsed
        self.prob_target = prob_target
    # def _generate_pseudo_hits(self, ticks_prob, adcs_distrib):
    #     Npix, Nhits, Nticks = ticks_prob.shape
    #     expected_ticks_per_hit, expected_adcs_per_hit, lambda_per_hit = get_average_hit_values(ticks_prob, adcs_distrib)
    #     # Filter out hits with negligible probability
    #     has_hit_mask = lambda_per_hit > self.hit_threshold  # (Npix, Nhits)
        
    #     # Flatten to create list of pseudo-hits
    #     # We need to replicate pixel coordinates for each hit
    #     pred_ticks = expected_ticks_per_hit[has_hit_mask]  # (N_total_hits,)
    #     pred_adcs = expected_adcs_per_hit[has_hit_mask]  # (N_total_hits,)
    #     pred_lambda = lambda_per_hit[has_hit_mask]  # (N_total_hits,)
        
    #     # For pixel coordinates, we need to replicate them for each hit
    #     # Create indices for which pixel each hit belongs to
    #     pixel_indices = jnp.arange(Npix)[:, None] * jnp.ones((Npix, Nhits), dtype=jnp.int32)  # (Npix, Nhits)
    #     pred_pixel_idx = pixel_indices[has_hit_mask]  # (N_total_hits,)
        
        
    #     return pred_ticks, pred_adcs, pred_lambda, pred_pixel_idx

    def _generate_distribution_hits(self, params, output):
        # This function can be used to prepare the probabilistic output for loss computation
        # For example, it can compute expected values or filter out low-probability hits
        ticks_prob = output['hit_prob']
        adcs_distrib = output['adcs_distrib']
        pixel_x = output['pixel_x']
        pixel_y = output['pixel_y']

        Npix, Nhits, Nticks = ticks_prob.shape

        mask = ticks_prob > self.hit_threshold

        
        hit_prob = ticks_prob[mask]
        hit_adc = adcs_distrib[mask]
        Q = adc2charge(hit_adc, params)

        all_ticks = jnp.arange(Nticks)[None, None, :]*jnp.ones((Npix, Nhits, Nticks))
        selected_ticks = all_ticks[mask]

        # Get z-coordinates and event IDs from prediction (if available)
        # If not available, compute from drift time or use same default as target
        if 'pixel_z' in output:
            # If prediction has pixel_z (from stochastic simulation), replicate for each hit
            pred_z = output['pixel_z']
        else:
            # For probabilistic predictions without z, compute from drift time
            # z = v_drift * t_drift (same approach as in simulate_stochastic)
            # Get pixel plane for z calculation
            pixel_plane = output.get('pixel_plane')
            selected_planes = (pixel_plane[:, None, None] * jnp.ones((Npix, Nhits, Nticks), dtype=jnp.int32))[mask]
            pred_z = get_hit_z(params, selected_ticks, selected_planes)

        # Event is per-pixel, replicate for each hit
        pred_event_per_pixel = output['event'][:, None, None] * jnp.ones((Npix, Nhits, Nticks), dtype=jnp.int32)  # (Npix, Nhits, Nticks)
        pixel_x_per_event = pixel_x[:, None, None] * jnp.ones((Npix, Nhits, Nticks), dtype=jnp.int32)
        pixel_y_per_event = pixel_y[:, None, None] * jnp.ones((Npix, Nhits, Nticks), dtype=jnp.int32)

        return Q, pixel_x_per_event[mask], pixel_y_per_event[mask], pred_z, selected_ticks, hit_prob, pred_event_per_pixel[mask]

    def _prepare_probabilistic_output(self, params, output):
        # This function can be used to prepare the probabilistic output for loss computation
        # For example, it can compute expected values or filter out low-probability hits
        ticks_prob = output['hit_prob']
        adcs_distrib = output['adcs_distrib']
        pixel_x = output['pixel_x']
        pixel_y = output['pixel_y']
        
        expected_ticks_per_hit, expected_adcs_per_hit, hit_prob = get_average_hit_values(ticks_prob, adcs_distrib)
        Npix, Nhits, Nticks = ticks_prob.shape

        Q = adc2charge(expected_adcs_per_hit, params)

        # Get z-coordinates and event IDs from prediction (if available)
        # If not available, compute from drift time or use same default as target
        if 'pixel_z' in output:
            # If prediction has pixel_z (from stochastic simulation), replicate for each hit
            pred_z = output['pixel_z']
        else:
            # For probabilistic predictions without z, compute from drift time
            # z = v_drift * t_drift (same approach as in simulate_stochastic)
            # Get pixel plane for z calculation
            pixel_plane = output.get('pixel_plane')
            pred_z = get_hit_z(params, expected_ticks_per_hit, pixel_plane[:, None] * jnp.ones((Npix, Nhits), dtype=jnp.int32))

        # Event is per-pixel, replicate for each hit
        pred_event_per_pixel = output['event'][:, None] * jnp.ones((Npix, Nhits), dtype=jnp.int32)  # (Npix, Nhits)
        pixel_x_per_event = pixel_x[:, None] * jnp.ones((Npix, Nhits), dtype=jnp.int32)
        pixel_y_per_event = pixel_y[:, None] * jnp.ones((Npix, Nhits), dtype=jnp.int32)

        return Q, pixel_x_per_event, pixel_y_per_event, pred_z, expected_ticks_per_hit, hit_prob, pred_event_per_pixel
        
        

    def compute(self, params, prediction, target):
        """
        Convert probabilistic predictions to pseudo-hits and apply deterministic loss.
        
        Important: ticks_prob and adcs_distrib have shape (Npix, Nhits, Nticks), where:
        - Npix: number of pixels
        - Nhits: maximum number of triggered hits per pixel (different hits, not charge values)
        - Nticks: time ticks
        
        Each (pixel, hit_index) combination should be treated independently.
        """

        if self.prob_target:
            ref_Q, target_x_per_event, target_y_per_event, ref_z, target_ticks, ref_hit_prob, ref_event = self._prepare_probabilistic_output(params, target)
        else:
            ref_Q = adc2charge(target['adcs'], params)
            ref_z = target['pixel_z']
            ref_event = target.get('event', jnp.zeros_like(target['ticks'], dtype=jnp.int32))
            ref_hit_prob = target.get('hit_prob', jnp.ones_like(target['ticks']))
            target_x_per_event = target['pixel_x']
            target_y_per_event = target['pixel_y']
            target_ticks = target['ticks']

        if self.collapsed:
            pred_Q, pixel_x_per_event, pixel_y_per_event, pred_z, pred_ticks, pred_hit_prob, pred_event_per_pixel = self._prepare_probabilistic_output(params, prediction)
        else:
            pred_Q, pixel_x_per_event, pixel_y_per_event, pred_z, pred_ticks, pred_hit_prob, pred_event_per_pixel = self._generate_distribution_hits(params, prediction)

        
        # Apply the deterministic loss function
        loss_val, aux = self.loss_fn(
            params,
            pred_Q.flatten(), pixel_x_per_event.flatten(), pixel_y_per_event.flatten(), pred_z.flatten(), pred_ticks.flatten(), pred_hit_prob.flatten(), pred_event_per_pixel.flatten(),
            ref_Q.flatten(), target_x_per_event.flatten(), target_y_per_event.flatten(), ref_z.flatten(), target_ticks.flatten(), ref_hit_prob.flatten(), ref_event.flatten(),
            **self.kwargs
        )
        
        # Return loss with auxiliary info
        return loss_val, aux


class ProbabilisticLossStrategy(LossStrategy):
    def __init__(self, sigma_charge=500.0, eps=1e-10, **kwargs):
        """
        Computes negative log-likelihood of observed hits given predicted probability distributions.
        
        Implements a complete probabilistic loss that accounts for:
        1. Observed hits: -log P(tick|pixel) - log P(charge|tick,pixel)
        2. False positives: penalty for predicting hits where none observed (Σλ for unobserved pixels)
        
        This ensures the model learns to concentrate probability only at pixels with actual hits.
        
        NUMERICAL STABILITY NOTE:
        The gradient of log(x) is 1/x, which explodes when x → 0. When probabilities are
        very small (e.g., 1e-20), this causes gradient instability. We handle this by:
        - Clipping probabilities to [eps, 1] before taking log
        - This limits max gradient to 1/eps (e.g., 1e10 for eps=1e-10)
        
        ALTERNATIVE (for future): Work entirely in log-space by having the model output
        log-probabilities directly using log_softmax, then loss = -log_prob (no additional log).
        
        Args:
            sigma_charge: Standard deviation for Gaussian charge likelihood (in electrons)
            eps: Small constant to avoid log(0). Also sets minimum probability floor.
        """
        super().__init__(**kwargs)
        self.sigma_charge = sigma_charge
        self.eps = eps

    def compute(self, params, prediction, target):
        """
        Compute negative log-likelihood loss with false positive penalty.
        
        Loss = -Σ[log P(tick|pixel) + log P(charge|tick,pixel)]  [observed hits]
               + Σλ(pixel)                                         [false positive penalty]
        
        where λ(pixel) = Σ_t P(tick|pixel) = expected number of hits per pixel
        
        Prediction contains:
            - adcs_distrib: (Npix, Nvalues, Nticks) - predicted ADC distributions
            - ticks_prob: (Npix, Nvalues, Nticks) - joint probability P(value, tick | pixel has hit)
            - unique_pixels: (Npix,) - pixel IDs in sorted order
            - pixel_x, pixel_y: pixel coordinates
            
        Target contains:
            - pixel_id: (Nhits,) - pixel ID for each observed hit
            - ticks: (Nhits,) - observed tick for each hit
            - adcs: (Nhits,) - observed ADC for each hit
        """
        # Step 1: Match target hits to predicted pixel distributions
        target_pixel_ids = target['pixel_id']
        sim_unique_pixels = prediction['unique_pixels']
        
        # Find indices of target pixels in simulation output (unique_pixels is sorted)
        pixel_indices = jnp.searchsorted(sim_unique_pixels, target_pixel_ids)
        # jax.debug.print("pixel_indices={pixel_indices}", pixel_indices=pixel_indices)
        # Validate matches (check if pixel was actually simulated)
        pixel_indices_safe = jnp.clip(pixel_indices, 0, sim_unique_pixels.shape[0] - 1)
        pixel_match_valid = (sim_unique_pixels[pixel_indices_safe] == target_pixel_ids) & (target_pixel_ids >= 0)

        # jax.debug.print("target_pixel_ids={target_pixel_ids}", target_pixel_ids=target_pixel_ids[:5])
        # jax.debug.print("sim_unique_pixels={sim_unique_pixels}", sim_unique_pixels=sim_unique_pixels[:5])
        # jax.debug.print("Matched {sim} to {target} pixels", sim=sim_unique_pixels[pixel_indices_safe][:5], target=target_pixel_ids[:5])
        # jax.debug.print("pixel_match_valid={pixel_match_valid}", pixel_match_valid=pixel_match_valid[:5])

        
        # Step 2: Extract probability distributions for matched pixels
        # ticks_prob shape: (Npix, Nvalues, Nticks)
        # We need P(tick, charge | pixel_id) for the observed (tick, charge) pairs
        
        ticks_prob = prediction['hit_prob']  # (Npix, Nvalues, Nticks)
        adcs_distrib = prediction['adcs_distrib']  # (Npix, Nvalues, Nticks)
        
        # Compute marginal probability P(tick | pixel) = sum_values P(tick, value | pixel)
        # marginal_tick_prob = jnp.sum(ticks_prob, axis=1)  # (Npix, Nticks)
        
        # Step 3: For each target hit, compute likelihood
        target_ticks = target['ticks'].astype(int)
        target_adcs = target['adcs']
        target_charge = adc2charge(target_adcs, params)
        
        # Gather probabilities for the matched pixels at observed ticks
        # For each hit i: marginal_tick_prob[pixel_indices[i], target_ticks[i]]

        trigger_nb = compute_occurrence_indices(target_pixel_ids)
        # jax.debug.print("trigger_nb={trigger_nb}", trigger_nb=trigger_nb)

        hit_tick_probs = ticks_prob[pixel_indices_safe, trigger_nb, target_ticks]
        # jax.debug.print("hit_tick_probs={pixel_indices_safe}", pixel_indices_safe=pixel_indices_safe[:5])
        # jax.debug.print("trigger_nb={trigger_nb}", trigger_nb=trigger_nb[:5])
        # jax.debug.print("target_ticks={target_ticks}", target_ticks=target_ticks[:5])
        # jax.debug.print("hit_tick_probs={hit_tick_probs}", hit_tick_probs=hit_tick_probs[:5])
        # hit_tick_probs = jnp.sum(ticks_prob[pixel_indices_safe, :, target_ticks], axis=1)  # Sum over values to get P(tick|pixel)
        
        # Step 4: Compute expected charge at observed tick for each pixel
        # E[charge | pixel, tick] = sum_values charge(value, tick) * P(value | tick, pixel)
        # where P(value | tick, pixel) = P(value, tick | pixel) / P(tick | pixel)
        
        # Get conditional probability distributions: P(value | tick, pixel)
        # safe_marginal = jnp.where(marginal_tick_prob > self.eps, marginal_tick_prob, 1.0)
        # conditional_value_prob = ticks_prob / safe_marginal[:, None, :]  # (Npix, Nvalues, Nticks)
        
        # # Expected charge at each (pixel, tick)
        # expected_charge_adc = jnp.sum(adcs_distrib * conditional_value_prob, axis=1)  # (Npix, Nticks)
        # expected_charge = adc2charge(expected_charge_adc, params)
        
        # Gather expected charges for observed hits
        hit_expected_charges = adc2charge(adcs_distrib[pixel_indices_safe, trigger_nb, target_ticks], params)  # (Nhits,)
        hit_expected_charges = jnp.where(pixel_match_valid, hit_expected_charges, 0.0)  # Set to 0 for invalid matches
        # Step 5: Compute log-likelihood components
        
        # (a) Tick likelihood: log P(tick | pixel)
        # IMPORTANT: For numerical stability in gradient computation, we need to handle
        # very small probabilities carefully. The issue is that d/dx log(x) = 1/x
        # becomes huge when x → 0, causing gradient instability.
        # 
        # Solution: Clip probabilities to a reasonable range BEFORE taking log.
        # This prevents gradients from exploding when probabilities are tiny.
        # The clipping acts as a "soft floor" - probabilities below eps are treated
        # as if they were eps, limiting the maximum gradient magnitude to 1/eps.
        # prob_floor = 1e-5  # Ensure eps is not too small
        # clipped_tick_probs = jnp.clip(hit_tick_probs, prob_floor, 1.0)
        eps = 1e-10
        # p_safe = hit_tick_probs * (1 - 2 * eps) + eps
        # log_likelihood_tick = jnp.log(p_safe)
        # log_likelihood_tick = jnp.sqrt(jnp.square(hit_tick_probs) + jnp.square(eps)) - eps
        log_likelihood_tick = jnp.maximum(hit_tick_probs, jnp.log(eps))
        
        # (b) Charge likelihood: log P(charge | tick, pixel) assuming Gaussian
        #     P(charge_obs | charge_expected, sigma) ~ N(charge_expected, sigma^2)
        charge_diff = target_charge - hit_expected_charges
        # jax.debug.print("target_charge={target_charge}", target_charge=target_charge)
        # jax.debug.print("hit_expected_charges={hit_expected_charges}", hit_expected_charges=hit_expected_charges)
        log_likelihood_charge = (
            -0.5 * (charge_diff / (self.sigma_charge/1000)) ** 2 
            - 0.5 * jnp.log(2 * jnp.pi * (self.sigma_charge/1000)**2)
        )

        # log_likelihood_charge = jnp.where(
        #     pixel_match_valid, 
        #     log_likelihood_charge, 
        #     0.0
        # )

        log_likelihood_tick = jnp.where(
            pixel_match_valid, 
            log_likelihood_tick,
            0.0
        )


        # jax.debug.print("log_likelihood_tick={log_likelihood_tick}", log_likelihood_tick=log_likelihood_tick[:5])
        
        # (c) Cap likelihood instead of masking for very small probabilities
        # When P(tick) is extremely small, cap the penalty instead of setting to 0
        # This ensures bad predictions are still penalized, preventing loss from artificially decreasing
        # tick_prob_threshold = 1e-8
        # max_negative_ll_tick = -jnp.log(tick_prob_threshold + self.eps)  # ≈ 18.4 for 1e-8
        
        # Cap the tick likelihood: use actual value if reasonable, otherwise use max penalty

        # capped_log_likelihood_tick = jnp.where(
        #     hit_tick_probs > tick_prob_threshold,
        #     log_likelihood_tick,
        #     -max_negative_ll_tick  # Large negative value (strong penalty)
        # )
        # capped_log_likelihood_tick = log_likelihood_tick
        
        # For charge: only compute when tick probability is significant
        # When P(tick) is tiny, the charge term is meaningless, so set to 0
        # tick_mask = hit_tick_probs > tick_prob_threshold
        # masked_log_likelihood_charge = jnp.where(tick_mask, log_likelihood_charge, 0.0)
        
        # (d) Combined log-likelihood per hit
        # log_likelihood_per_hit = log_likelihood_tick #+ log_likelihood_charge
        # log_likelihood_per_hit = log_likelihood_charge
        log_likelihood_per_hit = log_likelihood_tick*100 + log_likelihood_charge
        
        # # Step 6: Handle invalid matches (pixels not in simulation)
        # # For invalid matches, assign a very negative log-likelihood (low probability)
        
        # # Step 7: Sum log-likelihood over observed hits
        # total_log_likelihood_hits = jnp.sum(log_likelihood_per_hit)

        no_match_penalty = jnp.log(eps)*jnp.sum(sim_unique_pixels[pixel_indices_safe] != target_pixel_ids)
        total_log_likelihood_time = jnp.sum(log_likelihood_tick) + no_match_penalty

        total_log_likelihood_hits = -jnp.sqrt(total_log_likelihood_time*jnp.sum(log_likelihood_charge))
        # jax.debug.print("total_log_likelihood_hits={total_log_likelihood_hits}", total_log_likelihood_hits=total_log_likelihood_hits)
        
        # # Step 8: Add penalty for false positives (predicted hits where none observed)
        # # For each predicted pixel, compute λ = Σ_t P(tick|pixel) = expected number of hits
        lambda_per_pixel = jnp.sum(ticks_prob, axis=(1, 2))  # (Npix,)
        
        # # Check which predicted pixels have at least one observed hit
        # # For each predicted pixel, check if it appears in target_pixel_ids
        # pred_pixels = prediction['unique_pixels']
        
        # def pixel_has_hit(pred_pixel):
        #     return jnp.sum(target_pixel_ids == pred_pixel) > 0
        
        # pred_pixel_has_hit = jax.vmap(pixel_has_hit)(pred_pixels)  # (Npix,) boolean
        
        # # # For pixels with no observed hits: penalty = λ (from Poisson P(n=0|λ) = exp(-λ))
        # # # For pixels with hits: already accounted for in Step 7
        # penalty_per_pixel = jnp.where(pred_pixel_has_hit, 0.0, lambda_per_pixel)
        # total_false_positive_penalty = jnp.sum(penalty_per_pixel)
        
        # # Step 9: Combined loss (negative log-likelihood with false positive penalty)
        # nll = -total_log_likelihood_hits + total_false_positive_penalty
        
        # Auxiliary info for debugging
        # aux = {
        #     'n_hits': target_pixel_ids.shape[0],
        #     'n_valid_matches': jnp.sum(pixel_match_valid),
        #     'mean_tick_prob': jnp.mean(hit_tick_probs),
        #     'mean_charge_diff': jnp.mean(jnp.abs(charge_diff)),
        #     'n_capped_hits': jnp.sum(~tick_mask),  # Renamed: now counts capped hits, not masked
        #     'false_positive_penalty': total_false_positive_penalty,
        #     'n_pred_pixels': pred_pixels.shape[0],
        #     'n_pixels_with_hits': jnp.sum(pred_pixel_has_hit),
        # }

        aux = {
            "log_likelihood_charge": -jnp.sum(log_likelihood_charge),
            "log_likelihood_tick": -total_log_likelihood_time,
            "no_match_penalty": -no_match_penalty
            # "total_false_positive_penalty": total_false_positive_penalty
        }

        return -total_log_likelihood_hits, aux


class DQDtRadialLossStrategy(LossStrategy):
    """Reco-observable loss with per-(track, tick_bin) 2D Gaussian NLL on (dQ/dt, <r^2>).

    Observables per slice (track, tick_bin):
      Q = sum of charge over all pixels in the slice          (constrains Ab, kb, eField, tau_e, DL)
      V = charge-weighted <r^2> around the slice's own charge (constrains DT)
          centroid, i.e. V = <(x-xbar)^2 + (y-ybar)^2>_Q
    The 2-vector (Q, V) is scored jointly by a 2D Gaussian NLL:
      nll = 0.5 resid^T Sigma^-1 resid + 0.5 log det(Sigma)
    with Sigma the 2x2 prediction-derived covariance from cell-independence propagation.

    Fix 1 (numerical): center pixel coordinates on `stop_gradient(centroid)` before the
    moment sums. This eliminates the (10 cm)^2 vs (1 cm)^2 cancellation in the raw-coord
    delta method that used to inflate Var[V] by ~10^4. In centered coords the first
    moment beta_c is identically zero at the forward pass and contributes zero to the
    gradient, so V collapses to gamma_c/alpha (no beta_c^2/alpha^2 term needed) and the
    delta-method Jacobian reduces to three terms.

    Fix 2 (physical): soft sigmoid mask on E[Q_pt] excludes prediction cells whose
    expected charge is below q_pt_threshold_ke. The target only has hits above the
    stochastic sim's discriminator; without this mask the prediction integrates noise-
    only cells across the whole detector that contribute negligibly to Q but heavily
    to V via the r^2 weighting.

    Per-slice own-centroid computation makes the loss insensitive to bulk transverse
    shifts (V is centroid-invariant by construction).

    Track-vs-event note: sim's `event` field is per-eventID from the packed pixel ID.
    On the current dataset (proton stopping / stopping muon with track_len_sel) events
    hold one track, so per-slice == per-track. Multi-track events would lump tracks
    together; fix upstream by plumbing a compound (eventID, trackID) id.
    """

    def __init__(self,
                 t_bin_width=1,
                 sigma_floor_ke=0.5,
                 sigma_floor_v_cm2=0.01,
                 rho_max=0.99,
                 q_pt_threshold_ke=0.1,
                 q_pt_softness_ke=0.02,
                 max_tracks_stochastic=64,
                 max_t_bins_stochastic=5000,
                 mask_empty=True,
                 distance_metric="gaussian_nll",
                 mmd_sigma_Q_ke=5.0,
                 mmd_sigma_drift_tick=500.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.t_bin_width = int(t_bin_width)
        self.sigma_floor_ke = float(sigma_floor_ke)
        self.sigma_floor_v_cm2 = float(sigma_floor_v_cm2)
        self.rho_max = float(rho_max)
        self.q_pt_threshold_ke = float(q_pt_threshold_ke)
        self.q_pt_softness_ke = float(q_pt_softness_ke)
        self.max_tracks_stochastic = int(max_tracks_stochastic)
        self.max_t_bins_stochastic = int(max_t_bins_stochastic)
        self.mask_empty = bool(mask_empty)
        # Distributional loss (skips slice pairing). "gaussian_nll" = current
        # per-slice L2; "mmd" = compare 2D point clouds (Q_hit, tick_hit) via
        # the RBF-kernel MMD from losses_jax.mmd.
        self.distance_metric = str(distance_metric)
        self.mmd_sigma_Q_ke = float(mmd_sigma_Q_ke)
        self.mmd_sigma_drift_tick = float(mmd_sigma_drift_tick)

    def _probabilistic_prediction(self, params, prediction):
        """Prediction moments from a probabilistic sim (adcs_distrib, hit_prob).

        Collapses (Npix, Nvalues, Nticks) to (Npix, Nticks) per-cell (E[Q], Var[Q]),
        applies fix 2 (soft threshold on E[Q_pt]) and fix 1 (stop_gradient centering),
        then aggregates per (event, tick_bin) slice into (Q_pred, V_pred) and their
        prediction-derived variances (Var_Q, Var_V, Cov_QV) via the 3-term delta method.
        """
        t_bin_width = self.t_bin_width

        # `hit_prob` in the sim output is a log-probability (see fee_jax.py:474 and
        # the exp() in get_average_hit_values at fee_jax.py:490); convert to linear.
        Q_v = adc2charge(prediction['adcs_distrib'], params)          # (Npix, Nvalues, Nticks)
        p_v = jnp.exp(jnp.maximum(prediction['hit_prob'], -18.42))    # linear probability
        valid_pixel = (prediction['unique_pixels'] >= 0).astype(p_v.dtype)  # (Npix,)
        p_v = p_v * valid_pixel[:, None, None]

        E_Q_pt = jnp.sum(Q_v * p_v, axis=1)                           # (Npix, Nticks)
        E_Q2_pt = jnp.sum(Q_v * Q_v * p_v, axis=1)
        Var_Q_pt = jnp.maximum(E_Q2_pt - E_Q_pt * E_Q_pt, 0.0)

        # Fix 2: soft-threshold on E[Q_pt]. Mean gets w_pt, variance gets w_pt^2
        # (Var[w*X] = w^2 * Var[X] under linear scaling).
        softness = jnp.maximum(self.q_pt_softness_ke, 1e-12)
        w_pt = jax.nn.sigmoid((E_Q_pt - self.q_pt_threshold_ke) / softness)
        E_Q_pt_prethresh_sum = jnp.sum(E_Q_pt)
        E_Q_pt = E_Q_pt * w_pt
        Var_Q_pt = Var_Q_pt * w_pt * w_pt

        Npix, Nticks = E_Q_pt.shape
        n_t_bins = int((Nticks + t_bin_width - 1) // t_bin_width)
        tick_grid = jnp.arange(Nticks)
        t_bin_row = (tick_grid // t_bin_width).astype(jnp.int32)

        event_p = jnp.clip(prediction['event'].astype(jnp.int32), 0, Npix - 1)
        n_tracks = Npix
        N_et = n_tracks * n_t_bins

        et_bin_pt = event_p[:, None] * n_t_bins + t_bin_row[None, :]
        flat_et = et_bin_pt.ravel()
        flat_EQ = E_Q_pt.ravel()
        flat_VQ = Var_Q_pt.ravel()
        flat_x = jnp.broadcast_to(prediction['pixel_x'][:, None], (Npix, Nticks)).ravel()
        flat_y = jnp.broadcast_to(prediction['pixel_y'][:, None], (Npix, Nticks)).ravel()

        # Fix 1: stop_gradient centering, then V = gamma_c / alpha (no beta_c^2/alpha^2 term).
        alpha_pred = jax.ops.segment_sum(flat_EQ, flat_et, num_segments=N_et)
        beta_x_raw = jax.ops.segment_sum(flat_EQ * flat_x, flat_et, num_segments=N_et)
        beta_y_raw = jax.ops.segment_sum(flat_EQ * flat_y, flat_et, num_segments=N_et)
        safe_alpha_pred = jnp.where(alpha_pred > 0, alpha_pred, 1.0)
        x_ref_pred = jax.lax.stop_gradient(beta_x_raw / safe_alpha_pred)
        y_ref_pred = jax.lax.stop_gradient(beta_y_raw / safe_alpha_pred)

        dx_c_pt = flat_x - x_ref_pred[flat_et]
        dy_c_pt = flat_y - y_ref_pred[flat_et]
        r2_c_pt = dx_c_pt * dx_c_pt + dy_c_pt * dy_c_pt

        gamma_c_pred = jax.ops.segment_sum(flat_EQ * r2_c_pt, flat_et, num_segments=N_et)
        V_pred = jnp.maximum(gamma_c_pred / safe_alpha_pred, 0.0)
        Q_pred = alpha_pred

        # 3-term delta method for Var[V], Cov[Q,V].
        Vaa = jax.ops.segment_sum(flat_VQ, flat_et, num_segments=N_et)
        Vag = jax.ops.segment_sum(flat_VQ * r2_c_pt, flat_et, num_segments=N_et)
        Vgg = jax.ops.segment_sum(flat_VQ * r2_c_pt * r2_c_pt, flat_et, num_segments=N_et)

        Var_Q = Vaa
        alpha2 = safe_alpha_pred * safe_alpha_pred
        Var_V = jnp.maximum(
            (V_pred * V_pred * Vaa - 2.0 * V_pred * Vag + Vgg) / alpha2,
            0.0,
        )
        Cov_QV = (Vag - V_pred * Vaa) / safe_alpha_pred

        return (Q_pred, V_pred, Var_Q, Var_V, Cov_QV,
                n_tracks, n_t_bins, N_et,
                E_Q_pt_prethresh_sum, jnp.mean(w_pt))

    def _stochastic_prediction(self, params, prediction):
        """Prediction moments from a stochastic sim (per-hit adcs, pixel_x/y, ticks, event).

        Same segment_sum aggregation as the target side, using stop_gradient centering.
        No prediction-derived variance is available (single realisation), so Var_Q/Var_V
        are zeros and the sigma_floor_ke / sigma_floor_v_cm2 config knobs set the scale
        of the NLL denominator. Cov_QV is zero -> the loss reduces to two diagonal
        Gaussian NLLs (Option-1 collapse).
        """
        t_bin_width = self.t_bin_width
        n_tracks = self.max_tracks_stochastic
        n_t_bins = int((self.max_t_bins_stochastic + t_bin_width - 1) // t_bin_width)
        N_et = n_tracks * n_t_bins

        Q_h = adc2charge(prediction['adcs'], params)
        event_h = jnp.clip(prediction['event'].astype(jnp.int32), 0, n_tracks - 1)
        t_bin_h = jnp.clip(prediction['ticks'].astype(jnp.int32) // t_bin_width, 0, n_t_bins - 1)
        et_bin_h = event_h * n_t_bins + t_bin_h
        x_h = prediction['pixel_x']
        y_h = prediction['pixel_y']

        alpha_pred  = jax.ops.segment_sum(Q_h,        et_bin_h, num_segments=N_et)
        beta_x_pred = jax.ops.segment_sum(Q_h * x_h,  et_bin_h, num_segments=N_et)
        beta_y_pred = jax.ops.segment_sum(Q_h * y_h,  et_bin_h, num_segments=N_et)
        safe_alpha_pred = jnp.where(alpha_pred > 0, alpha_pred, 1.0)
        x_ref_pred = jax.lax.stop_gradient(beta_x_pred / safe_alpha_pred)
        y_ref_pred = jax.lax.stop_gradient(beta_y_pred / safe_alpha_pred)

        dx_c_h = x_h - x_ref_pred[et_bin_h]
        dy_c_h = y_h - y_ref_pred[et_bin_h]
        r2_c_h = dx_c_h * dx_c_h + dy_c_h * dy_c_h

        gamma_c_pred = jax.ops.segment_sum(Q_h * r2_c_h, et_bin_h, num_segments=N_et)
        V_pred = jnp.maximum(gamma_c_pred / safe_alpha_pred, 0.0)
        Q_pred = alpha_pred

        # No sim-derived variance for a single realisation.
        zeros = jnp.zeros_like(Q_pred)
        Var_Q = zeros
        Var_V = zeros
        Cov_QV = zeros

        # Diagnostic keys not applicable in stochastic mode: report as sentinels.
        sum_Q_pred_prethresh = jnp.sum(Q_h)
        mean_pass_frac = jnp.asarray(1.0)

        return (Q_pred, V_pred, Var_Q, Var_V, Cov_QV,
                n_tracks, n_t_bins, N_et,
                sum_Q_pred_prethresh, mean_pass_frac)

    def _compute_loss_mmd(self, params, prediction, target, is_probabilistic):
        """Distributional MMD loss on (Q_hit, tick_hit) 2D point clouds.

        No per-slice pairing: treats each hit as a point in a 2D plane where the
        first axis is charge (scaled by mmd_sigma_Q_ke) and the second is drift
        tick (scaled by mmd_sigma_drift_tick). Compares the target point cloud
        to the prediction point cloud with the RBF-kernel MMD from
        larndsim.losses_jax.mmd (unit bandwidth after scaling).

        Currently implemented for the stochastic sim only (hit-list output). For
        the probabilistic sim we would need to fold the (Npix, Nvalues, Nticks)
        distribution into a weighted point cloud, deferred for now.
        """
        assert not is_probabilistic, (
            "distance_metric='mmd' currently supports the stochastic sim only. "
            "Set --probabilistic_sim off or extend _compute_loss_mmd for probabilistic."
        )

        # Target-side point cloud: one point per hit (Q, tick), scaled per-axis.
        Q_tgt_h = adc2charge(target['adcs'], params)
        tick_tgt_h = target['ticks'].astype(Q_tgt_h.dtype)
        # Valid-hit weight: event >= 0 marks a real hit, otherwise padded slot.
        event_tgt_h = target['event']
        w_tgt = (event_tgt_h >= 0).astype(Q_tgt_h.dtype)

        # Prediction-side point cloud from the stochastic sim's hit list.
        Q_pred_h = adc2charge(prediction['adcs'], params)
        tick_pred_h = prediction['ticks'].astype(Q_pred_h.dtype)
        event_pred_h = prediction['event']
        w_pred = (event_pred_h >= 0).astype(Q_pred_h.dtype)

        # Scale each axis so kernel bandwidth sigma=1 is natural post-scaling.
        sQ = self.mmd_sigma_Q_ke
        st = self.mmd_sigma_drift_tick
        tgt_pts = jnp.stack([Q_tgt_h / sQ, tick_tgt_h / st], axis=1)
        pred_pts = jnp.stack([Q_pred_h / sQ, tick_pred_h / st], axis=1)

        loss = mmd(pred_pts, tgt_pts, w_pred, w_tgt, sigma=1.0)

        n_pred = jnp.sum(w_pred)
        n_tgt = jnp.sum(w_tgt)
        aux = {
            'mmd_loss': loss,
            'n_pred_hits': n_pred,
            'n_tgt_hits': n_tgt,
            'sum_Q_pred': jnp.sum(Q_pred_h * w_pred),
            'sum_Q_tgt': jnp.sum(Q_tgt_h * w_tgt),
            # Keys expected by the notebook cells for compat.
            'n_bins_populated': n_pred + n_tgt,
            'mean_mu_pred': jnp.sum(Q_pred_h * w_pred) / (n_pred + 1e-30),
            'mean_Q_tgt': jnp.sum(Q_tgt_h * w_tgt) / (n_tgt + 1e-30),
            # Placeholders so cell 7's autodetect for the "option-2 covariance"
            # schema still works, though these are not part of the MMD loss.
            'chi2_total': loss,
            'logdet_total': jnp.zeros_like(loss),
            'nll_total': loss,
            'sum_V_pred_wQ': jnp.zeros_like(loss),
            'sum_V_tgt_wQ': jnp.zeros_like(loss),
            'mean_sigma2_Q': jnp.asarray(sQ * sQ),
            'mean_sigma2_V': jnp.asarray(st * st),
            'mean_rho': jnp.zeros_like(loss),
            'mean_abs_rho': jnp.zeros_like(loss),
            'sum_Q_pred_prethresh': jnp.sum(Q_pred_h * w_pred),
            'mean_pass_frac': jnp.asarray(1.0),
        }
        return loss, aux

    def compute(self, params, prediction, target):
        # Dispatch on sim output type.
        # Probabilistic sim -> (Npix, Nvalues, Nticks) distribution; moment collapse + delta method.
        # Stochastic sim    -> per-hit list on the prediction side (same schema as target).
        is_probabilistic = 'adcs_distrib' in prediction and 'hit_prob' in prediction
        t_bin_width = self.t_bin_width

        # Distributional loss path: skip the per-slice pairing entirely.
        if self.distance_metric == "mmd":
            return self._compute_loss_mmd(params, prediction, target, is_probabilistic)

        if is_probabilistic:
            (Q_pred, V_pred, Var_Q, Var_V, Cov_QV,
             n_tracks, n_t_bins, N_et,
             sum_Q_pred_prethresh, mean_pass_frac) = self._probabilistic_prediction(params, prediction)
        else:
            (Q_pred, V_pred, Var_Q, Var_V, Cov_QV,
             n_tracks, n_t_bins, N_et,
             sum_Q_pred_prethresh, mean_pass_frac) = self._stochastic_prediction(params, prediction)

        # -----------------------------------------------------------------
        # Target-side (Q_tgt, V_tgt) from hit list, target's own centroid.
        # Unified across both sim types.
        # -----------------------------------------------------------------
        Q_h = adc2charge(target['adcs'], params)
        event_h = jnp.clip(target['event'].astype(jnp.int32), 0, n_tracks - 1)
        t_bin_h = jnp.clip(target['ticks'].astype(jnp.int32) // t_bin_width, 0, n_t_bins - 1)
        et_bin_h = event_h * n_t_bins + t_bin_h
        x_h = target['pixel_x']
        y_h = target['pixel_y']

        alpha_tgt  = jax.ops.segment_sum(Q_h,        et_bin_h, num_segments=N_et)
        beta_x_tgt = jax.ops.segment_sum(Q_h * x_h,  et_bin_h, num_segments=N_et)
        beta_y_tgt = jax.ops.segment_sum(Q_h * y_h,  et_bin_h, num_segments=N_et)
        safe_alpha_tgt = jnp.where(alpha_tgt > 0, alpha_tgt, 1.0)
        x_ref_tgt = jax.lax.stop_gradient(beta_x_tgt / safe_alpha_tgt)
        y_ref_tgt = jax.lax.stop_gradient(beta_y_tgt / safe_alpha_tgt)

        dx_c_h = x_h - x_ref_tgt[et_bin_h]
        dy_c_h = y_h - y_ref_tgt[et_bin_h]
        r2_c_h = dx_c_h * dx_c_h + dy_c_h * dy_c_h

        gamma_c_tgt = jax.ops.segment_sum(Q_h * r2_c_h, et_bin_h, num_segments=N_et)
        V_tgt = jnp.maximum(gamma_c_tgt / safe_alpha_tgt, 0.0)
        Q_tgt = alpha_tgt

        # -----------------------------------------------------------------
        # Step 6: 2D Gaussian NLL per slice.
        # -----------------------------------------------------------------
        # === TEMP CHARGE-ONLY LOSS (BEGIN) ==============================
        # Isolates the Q term to test whether the V/Mahalanobis machinery is
        # the source of the wrong-sign gradients. To recover the original 2D
        # loss: replace this whole block (down to the TEMP END marker) with
        # the 2D Mahalanobis code kept in the git history / commented below.
        sigma_QQ = Var_Q + self.sigma_floor_ke * self.sigma_floor_ke
        # sigma_VV / sigma_QV / det / rho are kept for aux compatibility only;
        # they do not enter the loss under this temporary configuration.
        sigma_VV = Var_V + self.sigma_floor_v_cm2 * self.sigma_floor_v_cm2
        rho = jnp.zeros_like(sigma_QQ)

        dQ = Q_tgt - Q_pred
        dV = V_tgt - V_pred                                              # aux only
        chi2_per_bin = 0.5 * dQ * dQ / sigma_QQ                          # Q-only chi^2
        logdet_per_bin = 0.5 * jnp.log(sigma_QQ)                         # Q-only log-det
        nll_per_bin = chi2_per_bin + logdet_per_bin
        # -- Original 2D NLL (uncomment to restore) ----------------------
        # sigma_QV = Cov_QV
        # denom = jnp.sqrt(jnp.maximum(sigma_QQ * sigma_VV, 1e-30))
        # rho_raw = sigma_QV / denom
        # rho = jnp.clip(rho_raw, -self.rho_max, self.rho_max)
        # sigma_QV_clipped = rho * denom
        # det = sigma_QQ * sigma_VV - sigma_QV_clipped * sigma_QV_clipped
        # safe_det = jnp.maximum(det, 1e-30)
        # mahal = (sigma_VV * dQ * dQ - 2.0 * sigma_QV_clipped * dQ * dV + sigma_QQ * dV * dV) / safe_det
        # chi2_per_bin = 0.5 * mahal
        # logdet_per_bin = 0.5 * jnp.log(safe_det)
        # nll_per_bin = chi2_per_bin + logdet_per_bin
        # === TEMP CHARGE-ONLY LOSS (END) ================================

        # Populated slice mask: at least one side has charge.
        if self.mask_empty:
            populated = ((Q_pred > 0) | (Q_tgt > 0)).astype(Q_pred.dtype)
        else:
            populated = jnp.ones_like(Q_pred)

        wsum = jnp.sum(populated) + 1e-30
        loss = jnp.sum(populated * nll_per_bin) / wsum

        n_pop = jnp.sum(populated) + 1e-30
        aux = {
            # Total per-slice NLL components (batch sums; average per slice = sum / n_bins_populated).
            'chi2_total':      jnp.sum(populated * chi2_per_bin),
            'logdet_total':    jnp.sum(populated * logdet_per_bin),
            'nll_total':       jnp.sum(populated * nll_per_bin),
            # Charge-closure diagnostic (per-slice sum of Q).
            'sum_Q_pred':      jnp.sum(populated * Q_pred),
            'sum_Q_tgt':       jnp.sum(populated * Q_tgt),
            # Spread-closure diagnostic (charge-weighted average <r^2>).
            'sum_V_pred_wQ':   jnp.sum(populated * Q_pred * V_pred),
            'sum_V_tgt_wQ':    jnp.sum(populated * Q_tgt * V_tgt),
            # Averaged sigmas the loss actually saw.
            'mean_sigma2_Q':   jnp.sum(populated * sigma_QQ) / n_pop,
            'mean_sigma2_V':   jnp.sum(populated * sigma_VV) / n_pop,
            # Correlation coefficient rho = Cov[Q,V] / (sigma_Q sigma_V).
            'mean_rho':        jnp.sum(populated * rho) / n_pop,
            'mean_abs_rho':    jnp.sum(populated * jnp.abs(rho)) / n_pop,
            # Populated-slice count.
            'n_bins_populated': jnp.sum(populated),
            # Fix-2 threshold diagnostics (probabilistic only; NaN under stochastic).
            'sum_Q_pred_prethresh': sum_Q_pred_prethresh,
            'mean_pass_frac':       mean_pass_frac,
            # Kept for backward compatibility with older notebook cells.
            'mean_mu_pred':    jnp.sum(populated * Q_pred) / n_pop,
            'mean_Q_tgt':      jnp.sum(populated * Q_tgt) / n_pop,
        }

        return loss, aux

