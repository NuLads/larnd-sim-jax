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
                 per_track_scale_mode="off",
                 per_track_scale_eps_ke=1.0,
                 per_track_scale_clip=(0.1, 10.0),
                 per_track_scale_prior_log_sigma=None,
                 mmd_extra_features="none",
                 mmd_sigma_std_Q_ke=2.0,
                 mmd_sigma_std_tick=100.0,
                 mmd_include_skew_tick=False,
                 mmd_include_kurt_tick=False,
                 mmd_sigma_skew_tick=1.0,
                 mmd_sigma_kurt_tick=2.0,
                 mmd_var_tick_floor=1.0,
                 moment_match_weight=0.0,
                 moment_match_use_var=True,
                 marginal_llhd_sigma_Q_ke=0.5,
                 marginal_llhd_eps=1e-10,
                 marginal_llhd_include_Q=True,
                 marginal_llhd_min_log_prob=-18.42,
                 slope_penalty_weight=0.0,
                 slope_penalty_sigma=1.0,
                 slope_penalty_min_hits_per_track=3,
                 slope_penalty_Q_floor_ke=0.05,
                 slope_penalty_diag=False,
                 slope_penalty_tick_min=0.0,
                 slope_penalty_kind="within_track",
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
        # Per-track ΣQ nuisance. Rescales sim Q per hit by
        #     α = (ΣQ_tgt + eps) / (ΣQ_sim + eps)
        # closed-form each forward call. "per_track" applies one α per event;
        # "per_batch" applies a single α to the whole batch (keeps track-to-track
        # ΣQ *variation* as physics signal, only kills the bulk offset). Optional
        # Gaussian prior on log α (weighted by track charge) keeps some absolute
        # scale info: prior_log_sigma≈0.2 restricts α to ~[0.8, 1.25] softly.
        mode = str(per_track_scale_mode).lower()
        if mode not in ("off", "per_track", "per_batch"):
            raise ValueError(
                f"per_track_scale_mode must be 'off' | 'per_track' | 'per_batch', got {per_track_scale_mode!r}"
            )
        self.per_track_scale_mode = mode
        self.per_track_scale_eps_ke = float(per_track_scale_eps_ke)
        lo, hi = per_track_scale_clip
        self.per_track_scale_clip = (float(lo), float(hi))
        self.per_track_scale_prior_log_sigma = (
            None if per_track_scale_prior_log_sigma is None
            else float(per_track_scale_prior_log_sigma)
        )
        # ------------------------------------------------------------------
        # MMD-feature extensions.
        # `mmd_extra_features`:
        #   "none" — 2-D point cloud (E[Q], E[tick]) as before.
        #   "std"  — 4-D by concatenating (√Var[Q], √Var[tick]) per pseudo-hit.
        #            Sim populates these from per-slot 2nd moments; stochastic-sim
        #            and target hits have Var=0 (single realisation → no per-hit
        #            uncertainty). See the docstring for the DL-bias caveat.
        #   "var"  — 4-D with (Var[Q], Var[tick]) directly (no sqrt).
        # `mmd_sigma_std_Q_ke`, `mmd_sigma_std_tick` set the bandwidths on the
        # extra axes (units chosen so σ ≈ typical per-slot √Var).
        feat = str(mmd_extra_features).lower()
        if feat not in ("none", "std", "var"):
            raise ValueError(
                f"mmd_extra_features must be 'none' | 'std' | 'var', got {mmd_extra_features!r}"
            )
        self.mmd_extra_features = feat
        self.mmd_sigma_std_Q_ke = float(mmd_sigma_std_Q_ke)
        self.mmd_sigma_std_tick = float(mmd_sigma_std_tick)
        # Optional per-hit (per-slot) higher moments of the tick distribution,
        # added as MMD axes on top of the Var axes. Skewness picks up
        # asymmetry (Bragg-peak / recombination signatures); excess kurtosis
        # picks up tail behaviour (heavy-tail diffusion / threshold shapes).
        # Both are dimensionless and computed with a Var floor to avoid
        # division blow-up on near-empty slots. Target hits are single-sample
        # → skew=kurt=0 with the same target-bias caveat as the Var axes.
        self.mmd_include_skew_tick = bool(mmd_include_skew_tick)
        self.mmd_include_kurt_tick = bool(mmd_include_kurt_tick)
        self.mmd_sigma_skew_tick = float(mmd_sigma_skew_tick)
        self.mmd_sigma_kurt_tick = float(mmd_sigma_kurt_tick)
        # Var-floor (in tick^2) applied when computing skew = m3 / Var^{3/2}
        # and kurt = m4 / Var^2. Slots with true Var < floor get compressed
        # skew/kurt values instead of NaN/inf.
        self.mmd_var_tick_floor = float(mmd_var_tick_floor)
        # Global moment-matching term (option 2). Additive to the MMD/prior loss.
        # Comparison: normalized (mean_Q, mean_tick, [var_Q, var_tick]) between
        # weighted sim and target distributions. Weight 0 disables the term.
        self.moment_match_weight = float(moment_match_weight)
        self.moment_match_use_var = bool(moment_match_use_var)
        # Pixel-marginalized LLHD (distance_metric="marginal_llhd"). Scores each
        # target hit under the sim's marginal-over-pixel density at (t_i, Q_i).
        # Position-insensitive because the pixel identity is summed out.
        # marginal_llhd_sigma_Q_ke: Gaussian kernel width (ke) over sim's Q grid,
        #   softens the discrete adcs_distrib grid into a differentiable density.
        # marginal_llhd_eps: density floor to prevent -log(0) blow-up.
        # marginal_llhd_include_Q: if False, drop the Q axis (marginal-tick LLHD).
        # marginal_llhd_min_log_prob: floor applied to log_ticks_prob to stay
        #   numerically finite; matches the convention in get_average_hit_values.
        self.marginal_llhd_sigma_Q_ke = float(marginal_llhd_sigma_Q_ke)
        self.marginal_llhd_eps = float(marginal_llhd_eps)
        self.marginal_llhd_include_Q = bool(marginal_llhd_include_Q)
        self.marginal_llhd_min_log_prob = float(marginal_llhd_min_log_prob)
        # ------------------------------------------------------------------
        # Shape penalty (breaks the (Ab, τ) ΣQ-valley).
        # Adds a squared-residual term on the per-track slope of log(Q) vs
        # drift-tick. Physics: exponential attenuation gives
        #     log(Q(t)) = const − t / (τ · v_drift · ticks_per_us)
        # so the slope IS the (negated inverse) lifetime — direction Ab
        # and τ cannot both shift while preserving. It is exactly α-invariant:
        #     log(α · Q) = log(α) + log(Q)   → additive constant → slope unchanged.
        # So the term is orthogonal to the per-track ΣQ nuisance.
        #
        # Weight = 0 disables. Pick `slope_penalty_sigma` from the target-side
        # slope std across tracks (batch-derived reference σ); the penalty is
        # then O(1) at the true minimum and grows quadratically as ⟨slope⟩
        # drifts.
        # `slope_penalty_min_hits_per_track`: tracks with fewer valid hits
        # get zeroed out — the slope is ill-conditioned when the (tick, Q)
        # cloud has no lever arm.
        # `slope_penalty_Q_floor_ke`: floor applied inside log() so near-zero
        # sim slots don't blow up the residual; target hits are naturally
        # above the ADC threshold so the floor rarely bites the target side.
        # ------------------------------------------------------------------
        self.slope_penalty_weight = float(slope_penalty_weight)
        self.slope_penalty_sigma = float(slope_penalty_sigma)
        self.slope_penalty_min_hits_per_track = int(slope_penalty_min_hits_per_track)
        self.slope_penalty_Q_floor_ke = float(slope_penalty_Q_floor_ke)
        # Dry-run switch. When True, _slope_penalty is computed and its aux
        # exposed (mean_slope_sim/tgt/resid, std_slope_resid, frac_tracks_good)
        # even if slope_penalty_weight == 0. Purpose: calibrate σ and pick a
        # weight by *observing* the slope distribution on your actual data
        # before letting the penalty influence gradients. When weight > 0 the
        # diagnostics are always computed regardless of this flag.
        self.slope_penalty_diag = bool(slope_penalty_diag)
        # Minimum tick to include in the slope regression. Hits with
        # `tick < slope_penalty_tick_min` are given weight 0 in both the sim
        # and target arms of the observable. Purpose: exclude the pixel
        # electronics' signal-integrator ramp-up region at very short drift,
        # which has a non-exponential relationship between drift-time and
        # measured Q and biases any linear slope fit. Empirical from the
        # diagnostic (plot_input_logQ_spread) the ramp-up sits inside the
        # first ~100 ticks; a `tick_min` of 200-300 is safely past it.
        # Zero (default) = keep all hits (legacy behavior).
        self.slope_penalty_tick_min = float(slope_penalty_tick_min)
        # Which slope observable to use inside the penalty. All three share
        # weight/σ and produce the same aux-field names so downstream tooling
        # (reports, notebooks) needs no changes; the semantics differ per kind.
        #   "within_track"  — original: fit log(Q) vs tick INSIDE each track,
        #                     then average residuals across tracks. On stopping
        #                     muons this is Bragg-shape-dominated and τ-blind
        #                     (see 2-D Ab-τ scan diagnostic).
        #   "between_track" — one point per track (⟨t⟩, ⟨log Q⟩), then fit
        #                     ONE line across tracks. Uses inter-track drift-
        #                     depth variation, which is 10× the lever arm of
        #                     the within-track drift span → τ-sensitive.
        #   "per_hit"       — pool all hits regardless of track and fit ONE
        #                     line of log(Q_hit) vs tick_hit. Same signal as
        #                     between_track but each hit is a separate point;
        #                     more statistics but per-hit noise (Bragg included)
        #                     inflates variance vs the per-track summary.
        kind = str(slope_penalty_kind).lower()
        if kind not in ("within_track", "between_track", "per_hit"):
            raise ValueError(
                f"slope_penalty_kind must be 'within_track' | 'between_track' | 'per_hit', "
                f"got {slope_penalty_kind!r}"
            )
        self.slope_penalty_kind = kind

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

    def _per_track_alpha(self, Q_pred_h, w_pred, event_pred_h,
                         Q_tgt_h, w_tgt, event_tgt_h):
        """Closed-form per-track (or per-batch) ΣQ nuisance α.

        α_track = (ΣQ_tgt_track + eps) / (ΣQ_sim_track + eps), clipped.
        Gradient flows through α, so the physics-parameter effect on ΣQ_sim
        gets absorbed by α and the loss becomes ΣQ-invariant per track
        (profile-likelihood view of the nuisance).

        Returns:
          alpha_per_pred_hit: (Nhit_pred,) — α indexed by each prediction hit's
              event, broadcast to hit level.
          alpha_track: (n_tracks,) — the α per event, for aux/diagnostics.
          w_track: (n_tracks,) — per-track charge weight for the log-prior
              (kills empty/padded tracks).
          frac_clipped: scalar — fraction of tracks whose α hit the clip.
        """
        n_tracks = self.max_tracks_stochastic
        ep = jnp.clip(event_pred_h.astype(jnp.int32), 0, n_tracks - 1)
        et = jnp.clip(event_tgt_h.astype(jnp.int32), 0, n_tracks - 1)
        sumQ_pred = jax.ops.segment_sum(Q_pred_h * w_pred, ep, num_segments=n_tracks)
        sumQ_tgt = jax.ops.segment_sum(Q_tgt_h * w_tgt, et, num_segments=n_tracks)

        eps = self.per_track_scale_eps_ke
        alpha_raw = (sumQ_tgt + eps) / (sumQ_pred + eps)
        lo, hi = self.per_track_scale_clip
        alpha_track = jnp.clip(alpha_raw, lo, hi)
        frac_clipped = jnp.mean(
            (jnp.abs(alpha_track - alpha_raw) > 1e-12).astype(Q_pred_h.dtype)
        )

        if self.per_track_scale_mode == "per_batch":
            # One α for the whole batch — kills only the bulk offset, keeps
            # track-to-track ΣQ spread as physics signal.
            alpha_batch_raw = (jnp.sum(sumQ_tgt) + eps) / (jnp.sum(sumQ_pred) + eps)
            alpha_batch = jnp.clip(alpha_batch_raw, lo, hi)
            alpha_per_pred_hit = jnp.broadcast_to(alpha_batch, Q_pred_h.shape)
            # For diagnostics: repeat the batch scalar into the alpha_track slot.
            alpha_track = jnp.full_like(alpha_track, alpha_batch)
        else:
            alpha_per_pred_hit = alpha_track[ep]

        # Weight tracks by their charge in the prior, so empty/padded tracks
        # (α ≈ 1 by construction) don't dominate the "log α ≈ 0" statistic.
        w_track = sumQ_tgt + sumQ_pred
        return alpha_per_pred_hit, alpha_track, w_track, frac_clipped

    def _per_track_logQ_slope(self, Q, tick, event, weight):
        """Weighted per-track slope of log(Q) vs drift-tick.

        Returns (slope, W_track, n_hits_track):
          slope       (n_tracks,)  — least-squares slope, computed via the
                                     centered form
                                       Σ w (t − t̄_w) (log Q − log_Q_bar_w)
                                       ────────────────────────────────────
                                             Σ w (t − t̄_w)²
                                     which is exactly α-invariant
                                     (log(αQ) = log α + log Q makes the
                                     centered log-Q unchanged).
          W_track     (n_tracks,)  — Σ w per track (total weight; use as a
                                     track-level prior weight so empty/pad
                                     tracks don't contribute to the mean
                                     residual).
          n_hits_track (n_tracks,) — number of hits with w > 0 per track;
                                     used to threshold at min_hits_per_track.
        """
        n_tracks = self.max_tracks_stochastic
        ev = jnp.clip(event.astype(jnp.int32), 0, n_tracks - 1)
        # Floor Q inside log() so pad/near-zero slots don't blow up.
        Q_safe = jnp.maximum(Q, self.slope_penalty_Q_floor_ke)
        logQ = jnp.log(Q_safe)
        # Weighted means per track.
        W = jax.ops.segment_sum(weight, ev, num_segments=n_tracks)
        W_safe = W + 1e-30
        t_bar = jax.ops.segment_sum(weight * tick, ev, num_segments=n_tracks) / W_safe
        l_bar = jax.ops.segment_sum(weight * logQ, ev, num_segments=n_tracks) / W_safe
        # Centered hit-level residuals (broadcast the per-track mean back to hits).
        t_c = tick - t_bar[ev]
        l_c = logQ - l_bar[ev]
        num = jax.ops.segment_sum(weight * t_c * l_c, ev, num_segments=n_tracks)
        den = jax.ops.segment_sum(weight * t_c * t_c, ev, num_segments=n_tracks)
        slope = num / (den + 1e-30)
        # Count valid hits per track (weight-thresholded; a slot with w = 0
        # contributes nothing to the slope numerator/denominator anyway,
        # but we want to gate on hit *count*, not weight, so pad tracks
        # with all-w=0 slots don't accidentally survive the min-hits filter.
        valid_hit = (weight > 0).astype(jnp.int32)
        n_hits = jax.ops.segment_sum(valid_hit, ev, num_segments=n_tracks)
        return slope, W, n_hits

    def _weighted_linreg_slope(self, x, y, w):
        """Weighted 1-D linear regression slope of `y` vs `x`, weighted by
        `w`. Uses the centered form so the intercept is absorbed:
            slope = Σ w (x − x̄_w) (y − ȳ_w) / Σ w (x − x̄_w)²
        Returns a scalar. When Σw ≈ 0 (empty batch), returns 0.
        """
        W = jnp.sum(w) + 1e-30
        x_bar = jnp.sum(w * x) / W
        y_bar = jnp.sum(w * y) / W
        num = jnp.sum(w * (x - x_bar) * (y - y_bar))
        den = jnp.sum(w * (x - x_bar) ** 2) + 1e-30
        return num / den

    def _per_hit_slope(self, Q, tick, weight):
        """Pool ALL hits regardless of track and fit one line of
        log(Q_hit) vs tick_hit. Simplest observable; highest statistics but
        also includes within-track Bragg noise as scatter around the fit.
        """
        Q_safe = jnp.maximum(Q, self.slope_penalty_Q_floor_ke)
        return self._weighted_linreg_slope(tick, jnp.log(Q_safe), weight)

    def _between_track_slope(self, Q, tick, event, weight):
        """Between-track drift regression: one point per track, then fit ONE
        line across tracks. Physics:
            ⟨Q⟩_track ≈ ⟨dE/dx⟩_MIP · recomb · exp(−⟨t_drift⟩_track / τ)
        so slope of log ⟨Q⟩_track vs ⟨t_drift⟩_track across the population
        of stopping-muon tracks is −1/τ.

        Averaging inside each track suppresses:
          - within-track Bragg-peak dE/dx variation (dominant noise on the
            within-track slope, cancels here);
          - correlated hit-level noise (√N_hits_per_track shrinkage).
        Each track is one data point weighted by its total Σw (heavier
        tracks contribute more; matches how α is weighted).

        α-caveat: this observable is NOT α-invariant. Applying per-track α
        before averaging shifts each track's y-coordinate by log(α_track);
        because α is derived from ΣQ_tgt/ΣQ_sim (which includes drift-
        attenuation), it can absorb the τ signal. Callers pass RAW sim Q
        (no α) — that's what our sim path already does.
        """
        n_tracks = self.max_tracks_stochastic
        ev = jnp.clip(event.astype(jnp.int32), 0, n_tracks - 1)
        Q_safe = jnp.maximum(Q, self.slope_penalty_Q_floor_ke)
        logQ = jnp.log(Q_safe)
        W = jax.ops.segment_sum(weight, ev, num_segments=n_tracks)
        W_safe = W + 1e-30
        t_bar = jax.ops.segment_sum(weight * tick, ev, num_segments=n_tracks) / W_safe
        l_bar = jax.ops.segment_sum(weight * logQ, ev, num_segments=n_tracks) / W_safe
        # Track-level weight for the outer regression = each track's total
        # `w`. Empty/pad tracks (W≈0) contribute nothing.
        return self._weighted_linreg_slope(t_bar, l_bar, W)

    def _slope_penalty(self, params, prediction, target):
        """Drift-attenuation shape penalty. Three kinds selected by
        `self.slope_penalty_kind`; all share the (weight, σ) machinery and
        produce the same aux field names so downstream tooling doesn't care.

        Target-side inputs: raw hit list (Q, tick, event, valid-mask).
        Sim-side inputs: per-slot probabilistic expectations
        (E[Q_slot], E[tick_slot], λ_slot, event_pix) with NO α rescaling —
        applying α destroys the between-track τ signal (see helper docstring).

        Returns (penalty_scalar, aux_dict).
        """
        # --- target side ---
        Q_tgt = adc2charge(target['adcs'], params)
        tick_tgt = target['ticks'].astype(Q_tgt.dtype)
        event_tgt = target['event']
        w_tgt = (event_tgt >= 0).astype(Q_tgt.dtype)

        # --- sim side (probabilistic-only path) ---
        E_ticks, E_adcs, lam = get_average_hit_values(
            prediction['hit_prob'], prediction['adcs_distrib'],
        )
        E_Q_slot = adc2charge(E_adcs, params)
        valid_pixel = (prediction['unique_pixels'] >= 0).astype(E_Q_slot.dtype)
        w_pred_slot = (lam * valid_pixel[:, None]).ravel()
        Q_pred_flat = E_Q_slot.ravel()
        tick_pred_flat = E_ticks.ravel().astype(Q_pred_flat.dtype)
        event_pred_flat = jnp.broadcast_to(
            prediction['event'][:, None], E_Q_slot.shape,
        ).ravel()

        # Optional tick cut: drop hits inside the pixel-electronics ramp-up
        # region at short drift. Applied identically to sim and target.
        # Multiplicative on the weights (not an index cut) so JAX static
        # shape is preserved.
        if self.slope_penalty_tick_min > 0:
            tick_ok_tgt  = (tick_tgt      >= self.slope_penalty_tick_min).astype(Q_tgt.dtype)
            tick_ok_pred = (tick_pred_flat >= self.slope_penalty_tick_min).astype(Q_pred_flat.dtype)
            w_tgt        = w_tgt        * tick_ok_tgt
            w_pred_slot  = w_pred_slot  * tick_ok_pred

        kind = self.slope_penalty_kind
        sigma = self.slope_penalty_sigma

        if kind == "within_track":
            slope_sim, W_sim_track, n_hits_sim = self._per_track_logQ_slope(
                Q_pred_flat, tick_pred_flat, event_pred_flat, w_pred_slot,
            )
            slope_tgt, W_tgt_track, n_hits_tgt = self._per_track_logQ_slope(
                Q_tgt, tick_tgt, event_tgt, w_tgt,
            )
            min_hits = self.slope_penalty_min_hits_per_track
            good = ((n_hits_tgt >= min_hits) & (n_hits_sim >= min_hits)).astype(Q_tgt.dtype)
            w_track = good * W_tgt_track
            resid2 = (slope_sim - slope_tgt) ** 2 / (sigma * sigma)
            W = jnp.sum(w_track) + 1e-30
            penalty = jnp.sum(w_track * resid2) / W
            aux = {
                'slope_penalty':    penalty,
                'mean_slope_sim':   jnp.sum(w_track * slope_sim) / W,
                'mean_slope_tgt':   jnp.sum(w_track * slope_tgt) / W,
                'mean_slope_resid': jnp.sum(w_track * (slope_sim - slope_tgt)) / W,
                'std_slope_resid':  jnp.sqrt(
                    jnp.sum(w_track * (slope_sim - slope_tgt) ** 2) / W
                ),
                'frac_tracks_good': jnp.mean(good),
            }
            return penalty, aux

        # "between_track" and "per_hit" both produce a SINGLE scalar slope on
        # each side, so the penalty collapses to a plain squared residual.
        if kind == "between_track":
            slope_sim = self._between_track_slope(
                Q_pred_flat, tick_pred_flat, event_pred_flat, w_pred_slot,
            )
            slope_tgt = self._between_track_slope(
                Q_tgt, tick_tgt, event_tgt, w_tgt,
            )
        else:  # "per_hit"
            slope_sim = self._per_hit_slope(Q_pred_flat, tick_pred_flat, w_pred_slot)
            slope_tgt = self._per_hit_slope(Q_tgt, tick_tgt, w_tgt)

        resid = slope_sim - slope_tgt
        penalty = (resid * resid) / (sigma * sigma)
        # Same aux names as within_track so the report generator's plots and
        # tables work without changes. `mean_slope_*` = the single slope
        # scalar for this batch; `std_slope_resid` = 0 by design (single
        # scalar, no spread); `frac_tracks_good` = 1 (no per-track gating).
        aux = {
            'slope_penalty':    penalty,
            'mean_slope_sim':   slope_sim,
            'mean_slope_tgt':   slope_tgt,
            'mean_slope_resid': resid,
            'std_slope_resid':  jnp.zeros_like(penalty),
            'frac_tracks_good': jnp.ones_like(penalty),
        }
        return penalty, aux

    def _compute_loss_mmd(self, params, prediction, target, is_probabilistic):
        """Distributional MMD loss on (Q_hit, tick_hit) 2D point clouds.

        Two prediction sources are supported:
        - stochastic sim: raw per-hit list (adcs, ticks, event); one point per hit,
          weight 1 for real hits and 0 for padded slots.
        - probabilistic sim: (Npix, Nhits, Nticks) distribution collapsed per hit
          via get_average_hit_values. Each (pixel, hit_slot) becomes one point at
          (E[Q], E[tick]) with weight lambda_per_hit = Σ_t P(tick, value | pixel,
          hit_slot) = P(this pseudo-hit exists). Padded pixels (unique_pixels < 0)
          get weight zero.

        Target-side is always the stochastic hit list (loaded from the input h5),
        one point per (Q, tick) hit.

        Optional per-track ΣQ nuisance (per_track_scale_mode != "off"): sim Q per
        hit is rescaled by α = (ΣQ_tgt + eps) / (ΣQ_sim + eps) per track (or per
        batch). This makes the MMD ΣQ-invariant per track, so reco-vs-truth input
        bias in total dE doesn't contaminate the physics-param minimum. A soft
        Gaussian prior on log α (per_track_scale_prior_log_sigma) optionally
        preserves some absolute-scale info.

        No per-slice pairing: MMD compares the two 2-D point clouds directly.
        Axes are scaled by (mmd_sigma_Q_ke, mmd_sigma_drift_tick) so the RBF
        kernel bandwidth sigma=1 is natural post-scaling.
        """
        # Target-side point cloud: one point per hit (Q, tick), scaled per-axis.
        Q_tgt_h = adc2charge(target['adcs'], params)
        tick_tgt_h = target['ticks'].astype(Q_tgt_h.dtype)
        # Valid-hit weight: event >= 0 marks a real hit, otherwise padded slot.
        event_tgt_h = target['event']
        w_tgt = (event_tgt_h >= 0).astype(Q_tgt_h.dtype)

        # Prediction-side point cloud.
        if is_probabilistic:
            # Collapse (Npix, Nhits, Nticks) -> (Npix, Nhits) per-slot expectations,
            # weighted by lambda_per_hit = P(this hit exists). get_average_hit_values
            # exponentiates the log-prob 'hit_prob' internally (fee_jax.py:490).
            log_ticks_prob = prediction['hit_prob']      # (Npix, Nhits, Nticks)
            adcs_distrib = prediction['adcs_distrib']    # (Npix, Nhits, Nticks)
            E_ticks, E_adcs, lam = get_average_hit_values(log_ticks_prob, adcs_distrib)
            E_Q = adc2charge(E_adcs, params)             # (Npix, Nhits)

            valid_pixel = (prediction['unique_pixels'] >= 0).astype(E_Q.dtype)  # (Npix,)
            w_pred = (lam * valid_pixel[:, None]).ravel()
            Q_pred_h = E_Q.ravel()
            tick_pred_h = E_ticks.ravel()
            # event is per-pixel in probabilistic output; broadcast to per hit_slot.
            event_pred_h = jnp.broadcast_to(
                prediction['event'][:, None], E_Q.shape
            ).ravel()

            # Per-slot second (and optional 3rd/4th) moments for the extended
            # MMD point cloud and the moment-matching term. Computed via the
            # *central-moment* form (Σ (x - μ)^k P / λ) rather than raw-moment
            # subtraction, to avoid big-minus-big cancellation error for the
            # tick axis where μ ~ few×10³ and dt² > E[t²] individually.
            need_moments = (
                self.mmd_extra_features != "none"
                or self.moment_match_weight > 0
                or self.mmd_include_skew_tick
                or self.mmd_include_kurt_tick
            )
            if need_moments:
                ticks_prob_lin = jnp.exp(jnp.maximum(log_ticks_prob, -18.42))
                Q_v = adc2charge(adcs_distrib, params)               # (Npix, Nhits, Nticks)
                Nticks = ticks_prob_lin.shape[-1]
                tick_grid = jnp.arange(Nticks, dtype=Q_v.dtype)      # (Nticks,)
                lam_safe = jnp.maximum(lam, 1e-10)
                # (Npix, Nhits, Nticks) deviations from per-slot mean.
                dt = tick_grid[None, None, :] - E_ticks[:, :, None]
                dQ = Q_v - E_Q[:, :, None]
                Var_Q_slot = jnp.maximum(
                    jnp.sum(dQ * dQ * ticks_prob_lin, axis=2) / lam_safe, 0.0
                )                                                    # (Npix, Nhits)
                Var_tick_slot = jnp.maximum(
                    jnp.sum(dt * dt * ticks_prob_lin, axis=2) / lam_safe, 0.0
                )                                                    # (Npix, Nhits)
                Var_Q_pred_h = Var_Q_slot.ravel()
                Var_tick_pred_h = Var_tick_slot.ravel()

                if self.mmd_include_skew_tick or self.mmd_include_kurt_tick:
                    # Normalize by Var^{3/2} / Var^2. Floor Var to keep near-
                    # empty slots from producing extreme skew/kurt values.
                    var_safe_hm = jnp.maximum(Var_tick_slot, self.mmd_var_tick_floor)
                    m3_tick = jnp.sum(dt * dt * dt * ticks_prob_lin, axis=2) / lam_safe
                    m4_tick = jnp.sum(dt * dt * dt * dt * ticks_prob_lin, axis=2) / lam_safe
                    skew_tick_pred_h = (m3_tick / (var_safe_hm ** 1.5)).ravel()
                    # Excess kurtosis: 0 for a Gaussian, so a Gaussian target-side
                    # slot (theoretical) would match the sim exactly.
                    kurt_tick_pred_h = (m4_tick / (var_safe_hm ** 2) - 3.0).ravel()
                else:
                    skew_tick_pred_h = jnp.zeros_like(Var_tick_pred_h)
                    kurt_tick_pred_h = jnp.zeros_like(Var_tick_pred_h)
            else:
                Var_Q_pred_h = jnp.zeros_like(Q_pred_h)
                Var_tick_pred_h = jnp.zeros_like(tick_pred_h)
                skew_tick_pred_h = jnp.zeros_like(Q_pred_h)
                kurt_tick_pred_h = jnp.zeros_like(Q_pred_h)
        else:
            # Stochastic sim's hit list.
            Q_pred_h = adc2charge(prediction['adcs'], params)
            tick_pred_h = prediction['ticks'].astype(Q_pred_h.dtype)
            event_pred_h = prediction['event']
            w_pred = (event_pred_h >= 0).astype(Q_pred_h.dtype)
            # No per-slot moments available from a single sample.
            Var_Q_pred_h = jnp.zeros_like(Q_pred_h)
            Var_tick_pred_h = jnp.zeros_like(tick_pred_h)
            skew_tick_pred_h = jnp.zeros_like(Q_pred_h)
            kurt_tick_pred_h = jnp.zeros_like(Q_pred_h)

        # Target is always a stochastic hit list → per-hit higher moments are 0.
        Var_Q_tgt_h = jnp.zeros_like(Q_tgt_h)
        Var_tick_tgt_h = jnp.zeros_like(tick_tgt_h)
        skew_tick_tgt_h = jnp.zeros_like(Q_tgt_h)
        kurt_tick_tgt_h = jnp.zeros_like(Q_tgt_h)

        # Per-track (or per-batch) ΣQ nuisance rescaling of sim Q.
        if self.per_track_scale_mode != "off":
            alpha_per_pred_hit, alpha_track, w_track, frac_alpha_clipped = self._per_track_alpha(
                Q_pred_h, w_pred, event_pred_h, Q_tgt_h, w_tgt, event_tgt_h,
            )
            Q_pred_h = Q_pred_h * alpha_per_pred_hit
        else:
            alpha_track = jnp.ones((self.max_tracks_stochastic,), dtype=Q_pred_h.dtype)
            w_track = jnp.zeros_like(alpha_track)
            frac_alpha_clipped = jnp.zeros((), dtype=Q_pred_h.dtype)

        # Scale each axis so kernel bandwidth sigma=1 is natural post-scaling.
        sQ = self.mmd_sigma_Q_ke
        st = self.mmd_sigma_drift_tick
        tgt_axes = [Q_tgt_h / sQ, tick_tgt_h / st]
        pred_axes = [Q_pred_h / sQ, tick_pred_h / st]

        # Optional 4-D features: append per-slot Var (or √Var) axes. Sim gets
        # non-zero values on the probabilistic branch; target and stochastic-sim
        # get 0. Note this creates a systematic offset on the extra axes that
        # tends to bias DL / diffusion parameters toward *smaller* Var — turn
        # off by setting mmd_extra_features="none" (default) if that dominates.
        if self.mmd_extra_features == "std":
            ssQ = self.mmd_sigma_std_Q_ke
            sst = self.mmd_sigma_std_tick
            tgt_axes.append(jnp.sqrt(Var_Q_tgt_h) / ssQ)
            tgt_axes.append(jnp.sqrt(Var_tick_tgt_h) / sst)
            pred_axes.append(jnp.sqrt(Var_Q_pred_h) / ssQ)
            pred_axes.append(jnp.sqrt(Var_tick_pred_h) / sst)
        elif self.mmd_extra_features == "var":
            # Bandwidths on variance axes: use the *squared* std sigmas so the
            # scale is consistent (Var lives on the (ke)^2 / tick^2 axis).
            ssQ2 = self.mmd_sigma_std_Q_ke * self.mmd_sigma_std_Q_ke
            sst2 = self.mmd_sigma_std_tick * self.mmd_sigma_std_tick
            tgt_axes.append(Var_Q_tgt_h / ssQ2)
            tgt_axes.append(Var_tick_tgt_h / sst2)
            pred_axes.append(Var_Q_pred_h / ssQ2)
            pred_axes.append(Var_tick_pred_h / sst2)

        # Optional per-hit skew / kurt of tick — describe the *shape* of the
        # sim's per-slot tick distribution beyond variance. Non-Gaussian shape
        # (Bragg-peak / recombination tail, LUT time response) carries physics
        # info that std alone cannot capture. Target values are 0 (single-hit
        # → no per-hit shape); same target-bias caveat as the Var axes applies.
        if self.mmd_include_skew_tick:
            ss_skew = self.mmd_sigma_skew_tick
            tgt_axes.append(skew_tick_tgt_h / ss_skew)
            pred_axes.append(skew_tick_pred_h / ss_skew)
        if self.mmd_include_kurt_tick:
            ss_kurt = self.mmd_sigma_kurt_tick
            tgt_axes.append(kurt_tick_tgt_h / ss_kurt)
            pred_axes.append(kurt_tick_pred_h / ss_kurt)

        tgt_pts = jnp.stack(tgt_axes, axis=1)
        pred_pts = jnp.stack(pred_axes, axis=1)

        loss = mmd(pred_pts, tgt_pts, w_pred, w_tgt, sigma=1.0)
        mmd_only = loss  # pure MMD before prior/moment terms, saved for aux

        # Optional soft Gaussian prior on log α, weighted by track charge so
        # empty tracks (α=1 by construction) don't dominate the statistic.
        prior_sigma = self.per_track_scale_prior_log_sigma
        if self.per_track_scale_mode != "off" and prior_sigma is not None and prior_sigma > 0:
            log_alpha = jnp.log(jnp.clip(alpha_track, 1e-12, None))
            prior_num = jnp.sum(w_track * log_alpha * log_alpha)
            prior_den = jnp.sum(w_track) + 1e-30
            prior_term = 0.5 * prior_num / (prior_den * prior_sigma * prior_sigma)
            loss = loss + prior_term
        else:
            prior_term = jnp.zeros_like(loss)

        # ---------------------------------------------------------------
        # Option 2: global moment-matching term (weighted L2 on normalized
        # global (mean, var) of (Q, tick) between sim and target). Cheap,
        # symmetric (both sides computed from their own weighted sample),
        # and physics-meaningful: mean_Q constrains ΣQ per hit, var_tick
        # is a *direct* diffusion signature (unlike option 1 where target
        # Var is 0). Disable by setting moment_match_weight=0 (default).
        # ---------------------------------------------------------------
        if self.moment_match_weight > 0:
            eps_w = 1e-30
            # E[X²] uses per-slot second moment on the sim side (adds the
            # within-slot spread that a per-slot mean would otherwise miss)
            # and X² on the target side (no per-hit spread available).
            W_pred = jnp.sum(w_pred) + eps_w
            W_tgt = jnp.sum(w_tgt) + eps_w
            mean_Q_pred = jnp.sum(w_pred * Q_pred_h) / W_pred
            mean_Q_tgt = jnp.sum(w_tgt * Q_tgt_h) / W_tgt
            mean_t_pred = jnp.sum(w_pred * tick_pred_h) / W_pred
            mean_t_tgt = jnp.sum(w_tgt * tick_tgt_h) / W_tgt

            # E_agg[X²] under prob sim: E[X² | slot] = E[X|slot]² + Var[X|slot],
            # so the outer weighted average captures both between-slot spread
            # (from Q_pred_h, tick_pred_h) and within-slot spread (from Var).
            E_Q2_pred = jnp.sum(w_pred * (Q_pred_h * Q_pred_h + Var_Q_pred_h)) / W_pred
            E_Q2_tgt = jnp.sum(w_tgt * (Q_tgt_h * Q_tgt_h + Var_Q_tgt_h)) / W_tgt
            E_t2_pred = jnp.sum(
                w_pred * (tick_pred_h * tick_pred_h + Var_tick_pred_h)
            ) / W_pred
            E_t2_tgt = jnp.sum(
                w_tgt * (tick_tgt_h * tick_tgt_h + Var_tick_tgt_h)
            ) / W_tgt
            var_Q_pred = jnp.maximum(E_Q2_pred - mean_Q_pred * mean_Q_pred, 0.0)
            var_Q_tgt = jnp.maximum(E_Q2_tgt - mean_Q_tgt * mean_Q_tgt, 0.0)
            var_t_pred = jnp.maximum(E_t2_pred - mean_t_pred * mean_t_pred, 0.0)
            var_t_tgt = jnp.maximum(E_t2_tgt - mean_t_tgt * mean_t_tgt, 0.0)

            # Normalize by the same axis sigmas as MMD so scales are consistent.
            dmean_Q = (mean_Q_pred - mean_Q_tgt) / sQ
            dmean_t = (mean_t_pred - mean_t_tgt) / st
            moment_term = dmean_Q * dmean_Q + dmean_t * dmean_t
            if self.moment_match_use_var:
                # Var lives on (ke)^2 / tick^2 — normalize by σ².
                dvar_Q = (var_Q_pred - var_Q_tgt) / (sQ * sQ)
                dvar_t = (var_t_pred - var_t_tgt) / (st * st)
                moment_term = moment_term + dvar_Q * dvar_Q + dvar_t * dvar_t
            moment_term = self.moment_match_weight * moment_term
            loss = loss + moment_term
        else:
            moment_term = jnp.zeros_like(loss)

        # Charge-weighted mean / std of log α for diagnostics (matches prior
        # weighting so the aux number corresponds to what the prior sees).
        w_track_safe = jnp.sum(w_track) + 1e-30
        log_alpha_diag = jnp.log(jnp.clip(alpha_track, 1e-12, None))
        mean_log_alpha = jnp.sum(w_track * log_alpha_diag) / w_track_safe
        var_log_alpha = jnp.sum(w_track * (log_alpha_diag - mean_log_alpha) ** 2) / w_track_safe

        n_pred = jnp.sum(w_pred)
        n_tgt = jnp.sum(w_tgt)
        aux = {
            'mmd_loss': loss,
            'mmd_only': mmd_only,
            'moment_match_term': moment_term,
            'n_pred_hits': n_pred,
            'n_tgt_hits': n_tgt,
            'sum_Q_pred': jnp.sum(Q_pred_h * w_pred),
            'sum_Q_tgt': jnp.sum(Q_tgt_h * w_tgt),
            # Per-track scale diagnostics.
            'alpha_prior_term': prior_term,
            'mean_log_alpha_track': mean_log_alpha,
            'std_log_alpha_track': jnp.sqrt(jnp.maximum(var_log_alpha, 0.0)),
            'frac_alpha_clipped': frac_alpha_clipped,
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

    def _compute_loss_marginal_llhd(self, params, prediction, target):
        """Pixel-marginalized log-likelihood on (t, Q) — LLHD without pixel matching.

        For each valid target hit i at (t_i, Q_i):
            p_sim(t_i, Q_i) = (1/N_total) · Σ_{pixel, slot}
                              exp(log_ticks_prob[pixel, slot, t_i])
                            · N(Q_i; adcs_grid[pixel, slot, t_i], σ_Q)     [if include_Q]
        Loss = -mean_i w_tgt_i · log max(p_sim(t_i, Q_i), eps).

        Position-insensitive because the pixel identity is summed out; a bulk
        x/y/z shift changes *which* pixels contribute mass at (t_i, Q_i) but
        typically not whether *some* pixel does. Contrast full LLHD, which
        requires the sim to match target's exact pixel and blows up under any
        shift.

        Q handling: sim's (pixel, slot, tick) grid is irregular in ADC. A
        Gaussian kernel of width σ_Q over the slot dimension softens the grid
        into a differentiable density at target's continuous Q_i. Set
        marginal_llhd_include_Q=False to drop the Q axis entirely (marginal-
        tick LLHD — simpler, ignores Q-tick correlation, retains full drift-
        time shape info).

        Memory: builds a (Npix, Nhits, N_target) intermediate ~600 MB for
        typical batch/pad sizes. Well within A100. If OOM, reduce batch size
        or chunk target hits.

        Per-track α (per_track_scale_mode != "off"): closed-form profile-α is
        applied to sim's Q_grid *before* the Gaussian-Q kernel evaluation, so
        target's Q_i aligns with the α-rescaled sim grid. Removes per-track
        systematic Q offset (reco vs truth positions, dE-density flow), which
        the batch-level rate normalization (below) cannot handle. The kernel
        keeps sensitivity to *within-slot* Q spread, so σ_Q can stay tight.
        Optional log-α² Gaussian prior with width per_track_scale_prior_log_sigma
        limits how much scale info α can absorb.
        """
        assert 'adcs_distrib' in prediction and 'hit_prob' in prediction, (
            "distance_metric='marginal_llhd' requires --probabilistic_sim."
        )

        log_ticks_prob = prediction['hit_prob']       # (Npix, Nhits, Nticks)
        adcs_distrib = prediction['adcs_distrib']     # (Npix, Nhits, Nticks)
        Nticks = log_ticks_prob.shape[-1]

        # Target-side hit list. Q in ke.
        Q_tgt_h = adc2charge(target['adcs'], params)
        tick_tgt_h = target['ticks'].astype(jnp.int32)
        tick_tgt_clip = jnp.clip(tick_tgt_h, 0, Nticks - 1)
        w_tgt = (target['event'] >= 0).astype(Q_tgt_h.dtype)
        event_tgt_h = target['event']

        # Sim-side log P and Q grid, floored / converted.
        log_p_all = jnp.maximum(log_ticks_prob, self.marginal_llhd_min_log_prob)
        Q_grid = adc2charge(adcs_distrib, params)     # (Npix, Nhits, Nticks)

        # Valid-pixel mask: pad slots (unique_pixels < 0) contribute zero mass.
        valid_pixel = (prediction['unique_pixels'] >= 0).astype(Q_tgt_h.dtype)
        log_valid = jnp.log(valid_pixel + 1e-30)                       # (Npix,)

        # Optional per-track α — rescale sim Q_grid to align per-track ΣQ.
        # Sim's per-slot expected charge Q_slot ≈ E[Q|slot], weight λ_slot.
        # We reuse _per_track_alpha by supplying (Q_slot.ravel(), λ.ravel()) as
        # the sim-side per-hit list; segment_sum by event gives per-track ΣQ,
        # α_track = (ΣQ_tgt + eps) / (ΣQ_sim + eps).
        if self.per_track_scale_mode != "off":
            E_ticks, E_adcs, lam = get_average_hit_values(log_ticks_prob, adcs_distrib)
            E_Q_slot = adc2charge(E_adcs, params)                              # (Npix, Nhits)
            w_pred_slot = (lam * valid_pixel[:, None]).ravel()
            Q_pred_slot = E_Q_slot.ravel()
            event_pred_slot = jnp.broadcast_to(
                prediction['event'][:, None], E_Q_slot.shape
            ).ravel()
            _, alpha_track, w_track, frac_alpha_clipped = self._per_track_alpha(
                Q_pred_slot, w_pred_slot, event_pred_slot,
                Q_tgt_h, w_tgt, event_tgt_h,
            )
            n_tracks = self.max_tracks_stochastic
            event_pix = jnp.clip(prediction['event'].astype(jnp.int32), 0, n_tracks - 1)
            alpha_per_pixel = alpha_track[event_pix]                           # (Npix,)
            # Rescale sim's Q_grid before Q kernel evaluation.
            Q_grid = Q_grid * alpha_per_pixel[:, None, None]
        else:
            alpha_track = jnp.ones((self.max_tracks_stochastic,), dtype=Q_tgt_h.dtype)
            w_track = jnp.zeros_like(alpha_track)
            frac_alpha_clipped = jnp.zeros((), dtype=Q_tgt_h.dtype)

        # -----------------------------------------------------------
        # Hits-per-pixel diagnostic (informational; not part of loss).
        # For the full-LLHD D2 path, each target hit occupies a slot
        # 0..(Nhits-1) selected by its within-pixel occurrence count.
        # If any pixel has more than MAX_ADC_VALUES hits, D2's slot
        # index is silently clamped by XLA → wrong. Here we report:
        #   max_hits_per_pixel: worst pixel's hit count in this batch
        #   frac_hits_over_slot_cap: fraction of target hits that would
        #     be assigned to slot idx >= MAX_ADC_VALUES (i.e. clamped)
        # These are informational for D10 (which doesn't use slots)
        # but useful for diagnosing whether the current dataset would
        # trigger D2's slot-collision bug.
        # -----------------------------------------------------------
        pix_id_tgt = (target['pixel_x'].astype(jnp.int32) * 100000
                      + target['pixel_y'].astype(jnp.int32))
        # Push padded/invalid target hits (event < 0) to a distinct
        # sentinel bucket so they don't inflate the count on real pixels.
        pix_id_tgt = jnp.where(w_tgt > 0, pix_id_tgt, jnp.int32(-1))
        sort_idx = jnp.argsort(pix_id_tgt)
        sorted_pix_id = pix_id_tgt[sort_idx]
        sorted_w = w_tgt[sort_idx]
        # Occurrence count within sorted groups: 0, 1, 2, ... per pixel_id.
        occ_tgt = compute_occurrence_indices(sorted_pix_id)
        max_hits_per_pixel = jnp.max(occ_tgt * sorted_w.astype(jnp.int32)) + 1
        nhits_cap = params.MAX_ADC_VALUES
        frac_hits_over_slot_cap = (
            jnp.sum((occ_tgt >= nhits_cap).astype(sorted_w.dtype) * sorted_w)
            / (jnp.sum(sorted_w) + 1e-30)
        )

        # Normalization: E[N_hits_sim] = Σ_{pixel, slot, tick} P · valid_pixel.
        # Use logsumexp for numerical stability across the (Npix·Nhits·Nticks) grid.
        log_p_masked = log_p_all + log_valid[:, None, None]
        log_N_total = jax.scipy.special.logsumexp(log_p_masked.reshape(-1))

        # Gather log P and Q at target ticks. Shape: (Npix, Nhits, N_tgt).
        log_p_at_ticks = jnp.take(log_p_all, tick_tgt_clip, axis=2)
        Q_at_ticks = jnp.take(Q_grid, tick_tgt_clip, axis=2)

        # Broadcast mask over the target-hit axis.
        log_mask = log_valid[:, None, None]  # (Npix, 1, 1)

        # Density at (t_i, Q_i): log P + log Gaussian(Q_i; Q_grid, σ_Q).
        if self.marginal_llhd_include_Q:
            sigma_Q = self.marginal_llhd_sigma_Q_ke
            log_norm_Q = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_Q)
            Q_i = Q_tgt_h[None, None, :]                                # (1, 1, N_tgt)
            log_gauss = -0.5 * ((Q_i - Q_at_ticks) / sigma_Q) ** 2 + log_norm_Q
            log_p_joint = log_p_at_ticks + log_gauss + log_mask         # (Npix, Nhits, N_tgt)
        else:
            # Marginal-tick LLHD: sum P over slots to get P(t_i | pixel),
            # then sum over pixels to get p_sim(t_i). No Q axis.
            log_p_joint = log_p_at_ticks + log_mask

        # logsumexp over (pixel, slot) → log-density per target hit.
        log_sum_per_hit = jax.scipy.special.logsumexp(
            log_p_joint.reshape(-1, log_p_joint.shape[-1]), axis=0
        )                                                                # (N_tgt,)
        log_p_per_hit = log_sum_per_hit - log_N_total

        # Floor to avoid -inf on target hits sim assigns near-zero density.
        log_eps = jnp.log(self.marginal_llhd_eps)
        log_p_per_hit_floored = jnp.maximum(log_p_per_hit, log_eps)
        frac_underflow = jnp.mean(
            (log_p_per_hit < log_eps).astype(Q_tgt_h.dtype) * w_tgt
        ) / (jnp.mean(w_tgt) + 1e-30)

        # Weighted per-hit average of -log p.
        W_tgt = jnp.sum(w_tgt) + 1e-30
        loss = -jnp.sum(w_tgt * log_p_per_hit_floored) / W_tgt
        mllhd_only = loss  # pure marginal-LLHD, before α prior

        # Optional soft Gaussian prior on log α (same form as the MMD path).
        prior_sigma = self.per_track_scale_prior_log_sigma
        if self.per_track_scale_mode != "off" and prior_sigma is not None and prior_sigma > 0:
            log_alpha = jnp.log(jnp.clip(alpha_track, 1e-12, None))
            prior_num = jnp.sum(w_track * log_alpha * log_alpha)
            prior_den = jnp.sum(w_track) + 1e-30
            prior_term = 0.5 * prior_num / (prior_den * prior_sigma * prior_sigma)
            loss = loss + prior_term
        else:
            prior_term = jnp.zeros_like(loss)

        # Shape penalty: per-track log(Q) vs drift-tick slope residual. Zero
        # weight is a no-op; strictly additive to the mllhd + α-prior loss.
        # Fully α-orthogonal by construction (see _per_track_logQ_slope
        # docstring) so it stacks cleanly on top of the ΣQ profile.
        if self.slope_penalty_weight > 0 or self.slope_penalty_diag:
            slope_term, slope_aux = self._slope_penalty(params, prediction, target)
            # Only fold into the loss when the weight is nonzero; a diag-only
            # dry-run computes the stats but leaves gradients untouched.
            if self.slope_penalty_weight > 0:
                loss = loss + self.slope_penalty_weight * slope_term
        else:
            slope_term = jnp.zeros_like(loss)
            slope_aux = {
                'slope_penalty':    slope_term,
                'mean_slope_sim':   jnp.zeros_like(loss),
                'mean_slope_tgt':   jnp.zeros_like(loss),
                'mean_slope_resid': jnp.zeros_like(loss),
                'std_slope_resid':  jnp.zeros_like(loss),
                'frac_tracks_good': jnp.zeros_like(loss),
            }

        # Charge-weighted log-α diagnostics.
        w_track_safe = jnp.sum(w_track) + 1e-30
        log_alpha_diag = jnp.log(jnp.clip(alpha_track, 1e-12, None))
        mean_log_alpha = jnp.sum(w_track * log_alpha_diag) / w_track_safe
        var_log_alpha = jnp.sum(
            w_track * (log_alpha_diag - mean_log_alpha) ** 2
        ) / w_track_safe

        aux = {
            'marginal_llhd_loss': loss,
            'mmd_loss': loss,                # alias for scripts expecting mmd_loss
            'mmd_only': mllhd_only,
            'moment_match_term': jnp.zeros_like(loss),
            'alpha_prior_term': prior_term,
            **slope_aux,
            'mean_log_p': jnp.sum(w_tgt * log_p_per_hit) / W_tgt,
            'frac_underflow': frac_underflow,
            'log_N_total': log_N_total,
            'n_pred_hits': jnp.exp(log_N_total),
            'n_tgt_hits': W_tgt,
            'sum_Q_pred': jnp.exp(log_N_total),  # rough sanity handle
            'sum_Q_tgt': jnp.sum(Q_tgt_h * w_tgt),
            # Target-side hit-count diagnostic (would be D2's slot cap):
            'max_hits_per_pixel': max_hits_per_pixel.astype(loss.dtype),
            'frac_hits_over_slot_cap': frac_hits_over_slot_cap,
            'slot_cap': jnp.asarray(nhits_cap, dtype=loss.dtype),
            # Real α diagnostics when per_track_scale_mode != "off"; zeros otherwise.
            'mean_log_alpha_track': mean_log_alpha,
            'std_log_alpha_track': jnp.sqrt(jnp.maximum(var_log_alpha, 0.0)),
            'frac_alpha_clipped': frac_alpha_clipped,
            # Placeholders for compat with the option-2 covariance aux schema.
            'n_bins_populated': W_tgt,
            'mean_mu_pred': jnp.zeros_like(loss),
            'mean_Q_tgt': jnp.sum(Q_tgt_h * w_tgt) / W_tgt,
            'chi2_total': loss,
            'logdet_total': jnp.zeros_like(loss),
            'nll_total': loss,
            'sum_V_pred_wQ': jnp.zeros_like(loss),
            'sum_V_tgt_wQ': jnp.zeros_like(loss),
            'mean_sigma2_Q': jnp.asarray(self.marginal_llhd_sigma_Q_ke ** 2),
            'mean_sigma2_V': jnp.zeros_like(loss),
            'mean_rho': jnp.zeros_like(loss),
            'mean_abs_rho': jnp.zeros_like(loss),
            'sum_Q_pred_prethresh': jnp.exp(log_N_total),
            'mean_pass_frac': jnp.asarray(1.0),
        }
        return loss, aux

    def _compute_loss_marginal_llhd_prod(self, params, prediction, target):
        """Pixel-marginalized LLHD, factorized product-of-marginals variant.

        Instead of scoring the joint density p_sim(t, Q) at each target hit
        (as _compute_loss_marginal_llhd does), this factorizes:

            p_sim_tick(t) = (1/N_total) · Σ_{p, s} P(t | p, s) · v_p
            p_sim_Q(Q)    = (1/Λ_total) · Σ_{p, s} λ_{p,s} ·
                                          𝒩(Q − α·E[Q | p, s]; σ_Q) · v_p
            L_prod = -mean_i log p_sim_tick(t_i) - mean_i log p_sim_Q(Q_i)

        Interpretation: treats tick and Q as *independent* marginals (product
        in linear space = sum in log space at loss level), as opposed to the
        joint marginal in the standard variant. Same α rescale as the joint
        version. Loses tick-Q correlation info from each (p, s) cell but is
        less prone to joint-density underflow when target's (t_i, Q_i) is far
        from every sim (p, s) grid point simultaneously.

        Uses the same kwargs (marginal_llhd_sigma_Q_ke, marginal_llhd_eps,
        per_track_scale_*). marginal_llhd_include_Q=False here drops the Q
        term entirely and gives pure tick-marginal LLHD.
        """
        assert 'adcs_distrib' in prediction and 'hit_prob' in prediction, (
            "distance_metric='marginal_llhd_prod' requires --probabilistic_sim."
        )

        log_ticks_prob = prediction['hit_prob']
        adcs_distrib = prediction['adcs_distrib']
        Nticks = log_ticks_prob.shape[-1]

        Q_tgt_h = adc2charge(target['adcs'], params)
        tick_tgt_h = target['ticks'].astype(jnp.int32)
        tick_tgt_clip = jnp.clip(tick_tgt_h, 0, Nticks - 1)
        w_tgt = (target['event'] >= 0).astype(Q_tgt_h.dtype)
        event_tgt_h = target['event']

        log_p_all = jnp.maximum(log_ticks_prob, self.marginal_llhd_min_log_prob)
        valid_pixel = (prediction['unique_pixels'] >= 0).astype(Q_tgt_h.dtype)
        log_valid = jnp.log(valid_pixel + 1e-30)

        # Per-slot expected Q and λ (used for both α and the Q marginal).
        E_ticks, E_adcs, lam = get_average_hit_values(log_ticks_prob, adcs_distrib)
        E_Q_slot = adc2charge(E_adcs, params)

        # Per-track α (same profile as the joint variant).
        if self.per_track_scale_mode != "off":
            w_pred_slot = (lam * valid_pixel[:, None]).ravel()
            Q_pred_slot = E_Q_slot.ravel()
            event_pred_slot = jnp.broadcast_to(
                prediction['event'][:, None], E_Q_slot.shape
            ).ravel()
            _, alpha_track, w_track, frac_alpha_clipped = self._per_track_alpha(
                Q_pred_slot, w_pred_slot, event_pred_slot,
                Q_tgt_h, w_tgt, event_tgt_h,
            )
            n_tracks = self.max_tracks_stochastic
            event_pix = jnp.clip(prediction['event'].astype(jnp.int32), 0, n_tracks - 1)
            alpha_per_pixel = alpha_track[event_pix]
            E_Q_slot_scaled = E_Q_slot * alpha_per_pixel[:, None]
        else:
            alpha_track = jnp.ones((self.max_tracks_stochastic,), dtype=Q_tgt_h.dtype)
            w_track = jnp.zeros_like(alpha_track)
            frac_alpha_clipped = jnp.zeros((), dtype=Q_tgt_h.dtype)
            E_Q_slot_scaled = E_Q_slot

        # ---- Tick marginal density ----
        # log_N_total_tick = logsumexp over full (p, s, t) grid.
        log_p_masked = log_p_all + log_valid[:, None, None]
        log_N_total_tick = jax.scipy.special.logsumexp(log_p_masked.reshape(-1))

        # Density at each target tick, summed over (p, s).
        log_p_at_ticks = jnp.take(log_p_masked, tick_tgt_clip, axis=2)   # (Npix, Nhits, N_tgt)
        log_p_tick_sum = jax.scipy.special.logsumexp(
            log_p_at_ticks.reshape(-1, log_p_at_ticks.shape[-1]), axis=0
        )
        log_p_tick_per_hit = log_p_tick_sum - log_N_total_tick

        # ---- Q marginal density (only if include_Q; else set to zero) ----
        if self.marginal_llhd_include_Q:
            sigma_Q = self.marginal_llhd_sigma_Q_ke
            log_norm_Q = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_Q)
            # Λ_slot = λ · v_p is the "weight" of each pixel/slot in the Q
            # marginal. Λ_total = Σ_{p, s} Λ_slot = expected total hit count.
            log_lam_slot = jnp.log(jnp.maximum(lam, 1e-30)) + log_valid[:, None]  # (Npix, Nhits)
            log_Lambda_total = jax.scipy.special.logsumexp(log_lam_slot.reshape(-1))

            # Kernel at each target Q_i against sim's per-slot rescaled E[Q].
            # Shape: (Npix, Nhits, N_tgt).
            Q_i = Q_tgt_h[None, None, :]                      # (1, 1, N_tgt)
            EQ = E_Q_slot_scaled[:, :, None]                  # (Npix, Nhits, 1)
            log_gauss_Q = -0.5 * ((Q_i - EQ) / sigma_Q) ** 2 + log_norm_Q
            log_Q_summand = log_lam_slot[:, :, None] + log_gauss_Q  # (Npix, Nhits, N_tgt)
            log_p_Q_sum = jax.scipy.special.logsumexp(
                log_Q_summand.reshape(-1, log_Q_summand.shape[-1]), axis=0
            )
            log_p_Q_per_hit = log_p_Q_sum - log_Lambda_total
        else:
            log_p_Q_per_hit = jnp.zeros_like(log_p_tick_per_hit)

        # ---- Combine as product in linear space (sum in log space) ----
        log_p_per_hit = log_p_tick_per_hit + log_p_Q_per_hit

        # Floor to prevent -∞.
        log_eps = jnp.log(self.marginal_llhd_eps)
        log_p_per_hit_floored = jnp.maximum(log_p_per_hit, log_eps)
        frac_underflow = jnp.mean(
            (log_p_per_hit < log_eps).astype(Q_tgt_h.dtype) * w_tgt
        ) / (jnp.mean(w_tgt) + 1e-30)

        W_tgt = jnp.sum(w_tgt) + 1e-30
        loss = -jnp.sum(w_tgt * log_p_per_hit_floored) / W_tgt
        mllhd_only = loss

        # α prior (same as joint variant).
        prior_sigma = self.per_track_scale_prior_log_sigma
        if self.per_track_scale_mode != "off" and prior_sigma is not None and prior_sigma > 0:
            log_alpha = jnp.log(jnp.clip(alpha_track, 1e-12, None))
            prior_num = jnp.sum(w_track * log_alpha * log_alpha)
            prior_den = jnp.sum(w_track) + 1e-30
            prior_term = 0.5 * prior_num / (prior_den * prior_sigma * prior_sigma)
            loss = loss + prior_term
        else:
            prior_term = jnp.zeros_like(loss)

        # Shape penalty (same term as the joint variant; see the docstring on
        # _slope_penalty). α-orthogonal so it stacks cleanly on top of the
        # factorized marginals.
        if self.slope_penalty_weight > 0 or self.slope_penalty_diag:
            slope_term, slope_aux = self._slope_penalty(params, prediction, target)
            # Only fold into the loss when the weight is nonzero; a diag-only
            # dry-run computes the stats but leaves gradients untouched.
            if self.slope_penalty_weight > 0:
                loss = loss + self.slope_penalty_weight * slope_term
        else:
            slope_term = jnp.zeros_like(loss)
            slope_aux = {
                'slope_penalty':    slope_term,
                'mean_slope_sim':   jnp.zeros_like(loss),
                'mean_slope_tgt':   jnp.zeros_like(loss),
                'mean_slope_resid': jnp.zeros_like(loss),
                'std_slope_resid':  jnp.zeros_like(loss),
                'frac_tracks_good': jnp.zeros_like(loss),
            }

        # Diagnostics.
        w_track_safe = jnp.sum(w_track) + 1e-30
        log_alpha_diag = jnp.log(jnp.clip(alpha_track, 1e-12, None))
        mean_log_alpha = jnp.sum(w_track * log_alpha_diag) / w_track_safe
        var_log_alpha = jnp.sum(
            w_track * (log_alpha_diag - mean_log_alpha) ** 2
        ) / w_track_safe

        # Hits-per-pixel (informational; unchanged from joint variant).
        pix_id_tgt = (target['pixel_x'].astype(jnp.int32) * 100000
                      + target['pixel_y'].astype(jnp.int32))
        pix_id_tgt = jnp.where(w_tgt > 0, pix_id_tgt, jnp.int32(-1))
        sort_idx = jnp.argsort(pix_id_tgt)
        sorted_pix_id = pix_id_tgt[sort_idx]
        sorted_w = w_tgt[sort_idx]
        occ_tgt = compute_occurrence_indices(sorted_pix_id)
        max_hits_per_pixel = jnp.max(occ_tgt * sorted_w.astype(jnp.int32)) + 1
        nhits_cap = params.MAX_ADC_VALUES
        frac_hits_over_slot_cap = (
            jnp.sum((occ_tgt >= nhits_cap).astype(sorted_w.dtype) * sorted_w)
            / (jnp.sum(sorted_w) + 1e-30)
        )

        aux = {
            'marginal_llhd_loss': loss,
            'mmd_loss': loss,
            'mmd_only': mllhd_only,
            'moment_match_term': jnp.zeros_like(loss),
            'alpha_prior_term': prior_term,
            **slope_aux,
            # Per-marginal breakdown — mean unfloored log-density for each factor.
            'mean_log_p_tick': jnp.sum(w_tgt * log_p_tick_per_hit) / W_tgt,
            'mean_log_p_Q': jnp.sum(w_tgt * log_p_Q_per_hit) / W_tgt,
            'mean_log_p': jnp.sum(w_tgt * log_p_per_hit) / W_tgt,
            'frac_underflow': frac_underflow,
            'log_N_total': log_N_total_tick,
            'n_pred_hits': jnp.exp(log_N_total_tick),
            'n_tgt_hits': W_tgt,
            'sum_Q_pred': jnp.exp(log_N_total_tick),
            'sum_Q_tgt': jnp.sum(Q_tgt_h * w_tgt),
            'max_hits_per_pixel': max_hits_per_pixel.astype(loss.dtype),
            'frac_hits_over_slot_cap': frac_hits_over_slot_cap,
            'slot_cap': jnp.asarray(nhits_cap, dtype=loss.dtype),
            'mean_log_alpha_track': mean_log_alpha,
            'std_log_alpha_track': jnp.sqrt(jnp.maximum(var_log_alpha, 0.0)),
            'frac_alpha_clipped': frac_alpha_clipped,
            # Notebook-compat placeholders.
            'n_bins_populated': W_tgt,
            'mean_mu_pred': jnp.zeros_like(loss),
            'mean_Q_tgt': jnp.sum(Q_tgt_h * w_tgt) / W_tgt,
            'chi2_total': loss,
            'logdet_total': jnp.zeros_like(loss),
            'nll_total': loss,
            'sum_V_pred_wQ': jnp.zeros_like(loss),
            'sum_V_tgt_wQ': jnp.zeros_like(loss),
            'mean_sigma2_Q': jnp.asarray(self.marginal_llhd_sigma_Q_ke ** 2),
            'mean_sigma2_V': jnp.zeros_like(loss),
            'mean_rho': jnp.zeros_like(loss),
            'mean_abs_rho': jnp.zeros_like(loss),
            'sum_Q_pred_prethresh': jnp.exp(log_N_total_tick),
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
        if self.distance_metric == "marginal_llhd":
            return self._compute_loss_marginal_llhd(params, prediction, target)
        if self.distance_metric == "marginal_llhd_prod":
            return self._compute_loss_marginal_llhd_prod(params, prediction, target)

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

