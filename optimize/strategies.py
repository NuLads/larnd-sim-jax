import jax
import jax.numpy as jnp
import math
from larndsim.sim_jax import simulate_wfs, simulate_stochastic, simulate_parametrized, simulate_probabilistic
from larndsim.losses_jax import adc2charge
from larndsim.detsim_jax import id2pixel, get_hit_z
from larndsim.fee_jax import get_average_hit_values
from larndsim.consts_jax import compute_smear_gaussian_weights, _build_tpc_z_metadata

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


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
            'ticks_prob': ticks_prob,     # (Npix, Nvalues, Nticks)
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
    
class Q1dLossStrategy(LossStrategy):
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
            Q, ref_Q, prediction['event'],
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

    def _generate_distribution_hits(self, params, output):
        # This function can be used to prepare the probabilistic output for loss computation
        # For example, it can compute expected values or filter out low-probability hits
        ticks_prob = output['ticks_prob']
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
        ticks_prob = output['ticks_prob']
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
    #eps=1e-10
    def __init__(self, eps=1e-6, 
                 target_gaussian_3d_radius_cm=0.3,
                 target_gaussian_3d_sigma_cm=0.1,
                 sobolev_pool_nbin_medium=70,
                 sobolev_pool_nbin_global=10,
                 sobolev_pool_layer_balance='weights',
                 w_sobolev_3d_grad_local=0.15,
                 w_sobolev_3d_grad_medium=0.02,
                 w_sobolev_3d_grad_global=0.015,
                 w_sobolev_pool_local=1.0,
                 w_sobolev_pool_medium=30.0,
                 w_sobolev_pool_global=600.0,
                 emit_sobolev_pool_report=True,
                 sobolev_norm_target_source='smeared',
                 smear_gaussian_weights=None,
                 max_events_per_batch=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.w_sobolev_3d_grad_local = float(w_sobolev_3d_grad_local)
        self.w_sobolev_3d_grad_medium = float(w_sobolev_3d_grad_medium)
        self.w_sobolev_3d_grad_global = float(w_sobolev_3d_grad_global)

        # Notebook-style parameters in physical units (cm).
        self.target_gaussian_3d_radius_cm = (
            None if target_gaussian_3d_radius_cm is None else max(float(target_gaussian_3d_radius_cm), 0.0)
        )
        self.target_gaussian_3d_sigma_cm = (
            None if target_gaussian_3d_sigma_cm is None else max(float(target_gaussian_3d_sigma_cm), 1e-6)
        )

        if sobolev_pool_nbin_medium %2 != 0 or sobolev_pool_nbin_global %2 != 0:
            raise ValueError("sobolev_pool_nbin_medium and sobolev_pool_nbin_global must be even integers to make a break in z at the TPC cathode.")
        
        self.sobolev_pool_nbin_x_medium = max(int(sobolev_pool_nbin_medium), 1)
        self.sobolev_pool_nbin_y_medium = 2 * self.sobolev_pool_nbin_x_medium
        self.sobolev_pool_nbin_z_medium = max(int(sobolev_pool_nbin_medium), 1)

        self.sobolev_pool_medium_bins_xyz = (
            self.sobolev_pool_nbin_x_medium,
            self.sobolev_pool_nbin_y_medium,
            self.sobolev_pool_nbin_z_medium,
        )

        self.sobolev_pool_nbin_x_global = max(int(sobolev_pool_nbin_global), 1)
        self.sobolev_pool_nbin_y_global = 2 * self.sobolev_pool_nbin_x_global
        self.sobolev_pool_nbin_z_global = max(int(sobolev_pool_nbin_global), 1)

        self.sobolev_pool_global_bins_xyz = (
            self.sobolev_pool_nbin_x_global,
            self.sobolev_pool_nbin_y_global,
            self.sobolev_pool_nbin_z_global,
        )

        self.sobolev_pool_layer_balance = str(sobolev_pool_layer_balance).lower()
        if self.sobolev_pool_layer_balance not in ('weights', 'none'):
            raise ValueError(
                "sobolev_pool_layer_balance must be one of 'weights', or 'none'"
            )
        self.sobolev_pool_manual_weights = {
            'local': max(float(w_sobolev_pool_local), 0.0),
            'medium': max(float(w_sobolev_pool_medium), 0.0),
            'global': max(float(w_sobolev_pool_global), 0.0),
        }

        self.emit_sobolev_pool_report = bool(emit_sobolev_pool_report)
        self.sobolev_norm_target_source = str(sobolev_norm_target_source).lower()
        if self.sobolev_norm_target_source == 'unsmeared':
            self.sobolev_norm_target_source = 'non_smeared'
        if self.sobolev_norm_target_source not in ('smeared', 'non_smeared'):
            raise ValueError(
                "sobolev_norm_target_source must be either 'smeared' or 'non_smeared'"
            )

        # Full z extent is derived from geometry (tpc_borders) in compute().
        # Fixed event-axis budget for per-event vectorized loss in one batch.
        self.max_events_per_batch = max(int(max_events_per_batch), 1)

        # Optional precomputed Gaussian smear weights from detector constants.
        self.smear_gaussian_weights = smear_gaussian_weights

    def _compute_smear_gaussian_weights(self, radius_cm, sigma_cm, pixel_pitch, z_tick_size):
        return compute_smear_gaussian_weights(radius_cm, sigma_cm, pixel_pitch, z_tick_size)

    def _gaussian_smear_sparse(self, target_xyz_unsmeared, smear_gaussian_weights,
                               max_hits_per_batch=256):
        """Sparse Gaussian smearing: apply kernel only around active hit locations.
        
        Args:
            target_xyz_unsmeared: (nx, ny, nz) dense unsmeared grid
            smear_gaussian_weights: iterable of (oy, ox, oz, weight)
            max_hits_per_batch: maximum number of hits to pad to (for JAX static shape)
        
        Returns:
            smeared: (nx, ny, nz) smeared grid
        """
        nx, ny, nz = target_xyz_unsmeared.shape

        # Start with the original (unsmeared) grid
        smeared = target_xyz_unsmeared.astype(jnp.float32)

        # Build a fixed-capacity list of active hits so array shapes stay static under JIT.
        active_mask = target_xyz_unsmeared > self.eps
        hit_x, hit_y, hit_z = jnp.nonzero(
            active_mask,
            size=max_hits_per_batch,
            fill_value=0,
        )
        n_hits = jnp.minimum(jnp.sum(active_mask.astype(jnp.int32)), max_hits_per_batch)
        hit_mask = jnp.arange(max_hits_per_batch, dtype=jnp.int32) < n_hits
        hit_vals = target_xyz_unsmeared[hit_x, hit_y, hit_z].astype(jnp.float32)
        
        # Apply each offset in the kernel
        def apply_offset(smeared_acc, smear_weight):
            oy, ox, oz, w = smear_weight

            oy = oy.astype(jnp.int32)
            ox = ox.astype(jnp.int32)
            oz = oz.astype(jnp.int32)

            # Compute new positions for all (padded) hits.
            # Kernel offsets are in (oy, ox, oz), while field layout is (x, y, z).
            x_new = hit_x + ox
            y_new = hit_y + oy
            z_new = hit_z + oz

            # Bounds check
            in_bounds = (x_new >= 0) & (x_new < nx) & \
                        (y_new >= 0) & (y_new < ny) & \
                        (z_new >= 0) & (z_new < nz)

            # Mask: only apply if in bounds AND hit is not padding
            valid_mask = in_bounds & hit_mask

            # Clip indices for gather/scatter (set invalid ones to 0)
            x_safe = jnp.where(valid_mask, x_new, 0)
            y_safe = jnp.where(valid_mask, y_new, 0)
            z_safe = jnp.where(valid_mask, z_new, 0)

            # Contribution: hit value × kernel weight, masked out for invalid entries
            contrib = jnp.where(valid_mask, hit_vals * w, 0.0)

            # Scatter-add all contributions in one vectorized operation
            smeared_new = smeared_acc.at[x_safe, y_safe, z_safe].add(contrib)

            return smeared_new, None

        # Scan over offsets (static loop, unrolled at compile time)
        smear_gaussian_weights = jnp.asarray(smear_gaussian_weights, dtype=jnp.float32)
        smeared_final, _ = jax.lax.scan(apply_offset, smeared, smear_gaussian_weights)

        return smeared_final

    def _gaussian_smear_target(self, target_xyz, smear_gaussian_weights):
        """Apply 3D Gaussian smearing on (nx, ny, nz) target field in physical units.
        
        Uses sparse smearing: only processes voxels within the smearing radius of active hits.
        """
        return self._gaussian_smear_sparse(
            target_xyz, smear_gaussian_weights
        )

    def _pool_xyz_bins(self, field_xyz, bins_xyz):
        """Pool a dense [nx, ny, nz] field into coarse xyz bins."""
        field_xyz = jnp.asarray(field_xyz, dtype=jnp.float32)
        bin_x = max(int(bins_xyz[0]), 1)
        bin_y = max(int(bins_xyz[1]), 1)
        bin_z = max(int(bins_xyz[2]), 1)
        nx, ny, nz = field_xyz.shape

        x_bin = jnp.minimum((jnp.arange(nx, dtype=jnp.int32) * bin_x) // max(nx, 1), bin_x - 1)
        y_bin = jnp.minimum((jnp.arange(ny, dtype=jnp.int32) * bin_y) // max(ny, 1), bin_y - 1)
        z_bin = jnp.minimum((jnp.arange(nz, dtype=jnp.int32) * bin_z) // max(nz, 1), bin_z - 1)

        flat_x = jnp.repeat(x_bin, ny * nz)
        flat_y = jnp.tile(jnp.repeat(y_bin, nz), nx)
        flat_z = jnp.tile(z_bin, nx * ny)
        flat_idx = ((flat_x * bin_y) + flat_y) * bin_z + flat_z
        flat_vals = field_xyz.reshape(-1)

        pooled_sum = jnp.zeros(bin_x * bin_y * bin_z, dtype=jnp.float32).at[flat_idx].add(flat_vals)
        pooled_count = jnp.zeros(bin_x * bin_y * bin_z, dtype=jnp.float32).at[flat_idx].add(
            jnp.ones_like(flat_vals, dtype=jnp.float32)
        )
        pooled = pooled_sum / jnp.maximum(pooled_count, 1.0)
        return pooled.reshape(bin_x, bin_y, bin_z)

    def _sobolev_layer_metrics(self, pooled_pred, pooled_target, pooled_norm_source,
                               pixel_pitch_cm, z_bin_cm, w_sobolev_3d_grad=None,
                               axis_pool_factors_xyz=(1.0, 1.0, 1.0)):
        eps = self.eps
        residual = pooled_pred - pooled_target
        active_mask = (jnp.abs(pooled_pred) > eps) | (jnp.abs(pooled_target) > eps)
        norm_mask = jnp.abs(pooled_norm_source) > eps
        norm_voxels = jnp.sum(norm_mask.astype(jnp.float32))
        pooled_norm = jnp.maximum(norm_voxels, 1.0)
        pool_fx = max(float(axis_pool_factors_xyz[0]), 1.0)
        pool_fy = max(float(axis_pool_factors_xyz[1]), 1.0)
        pool_fz = max(float(axis_pool_factors_xyz[2]), 1.0)

        value = jnp.sum((residual ** 2) * active_mask.astype(jnp.float32)) / pooled_norm

        grad_x_e = jnp.array(0.0, dtype=jnp.float32)
        if residual.shape[0] > 1:
            dx = residual[1:, :, :] - residual[:-1, :, :]
            mx = active_mask[1:, :, :] & active_mask[:-1, :, :]
            grad_x_e = jnp.sum((dx ** 2) * mx.astype(jnp.float32)) / (
                pool_fx * max(float(pixel_pitch_cm) ** 2, 1e-12)
            )

        grad_y_e = jnp.array(0.0, dtype=jnp.float32)
        if residual.shape[1] > 1:
            dy = residual[:, 1:, :] - residual[:, :-1, :]
            my = active_mask[:, 1:, :] & active_mask[:, :-1, :]
            grad_y_e = jnp.sum((dy ** 2) * my.astype(jnp.float32)) / (
                pool_fy * max(float(pixel_pitch_cm) ** 2, 1e-12)
            )

        grad_z_e = jnp.array(0.0, dtype=jnp.float32)
        if residual.shape[2] > 1:
            dz = residual[:, :, 1:] - residual[:, :, :-1]
            mz = active_mask[:, :, 1:] & active_mask[:, :, :-1]
            # Adjust z-gradient contribution by voxel anisotropy (xy pitch over z bin size).
            z_grad_scale = max(float(z_bin_cm) / max(float(pixel_pitch_cm), 1e-12), 1e-12)
            # z_grad_scale = 1.0
            grad_z_e = jnp.sum((dz ** 2) * mz.astype(jnp.float32)) / (
                pool_fz * max(float(z_bin_cm) ** 2, 1e-12)
            ) / z_grad_scale

        sobolev_3d_grad = (grad_x_e + grad_y_e + grad_z_e) / 3.0
        _w_grad = float(self.w_sobolev_3d_grad) if w_sobolev_3d_grad is None else float(w_sobolev_3d_grad)
        total = value + _w_grad * sobolev_3d_grad
        active_voxels = jnp.sum(active_mask.astype(jnp.float32))

        return {
            'norm_voxels': norm_voxels,
            'active_voxels': active_voxels,
            'value': value,
            'grad_x_e': grad_x_e,
            'grad_y_e': grad_y_e,
            'grad_z_e': grad_z_e,
            'sobolev_3d_grad': sobolev_3d_grad,
            'total': total,
        }

    def _sobolev_pool_layer_weights(self):
        if self.sobolev_pool_layer_balance == 'weights':
            raw_weights = jnp.asarray([
                float(self.sobolev_pool_manual_weights['local']),
                float(self.sobolev_pool_manual_weights['medium']),
                float(self.sobolev_pool_manual_weights['global']),
            ], dtype=jnp.float32)
        else:
            raw_weights = jnp.ones(3, dtype=jnp.float32)

        raw_mean = jnp.maximum(jnp.mean(raw_weights), 1e-12)
        return raw_weights / raw_mean

    def _compute_three_layer_sobolev_pooling(
        self,
        pred_xyz,
        target_xyz,
        target_norm_xyz,
        pixel_pitch_cm,
        z_bin_cm,
    ):
        """Compute 3-layer Sobolev pooling: local, medium, global."""
        results = {}

        layer_specs = [
            ('local', 'local'),
            ('medium', 'medium'),
            ('global', 'global'),
        ]

        for layer_name, mode in layer_specs:
            if mode == 'local':
                pooled_pred = pred_xyz
                pooled_target = target_xyz
                pooled_norm_source = target_norm_xyz
                axis_pool_factors_xyz = (1.0, 1.0, 1.0)
            elif mode == 'medium':
                pooled_pred = self._pool_xyz_bins(pred_xyz, self.sobolev_pool_medium_bins_xyz)
                pooled_target = self._pool_xyz_bins(target_xyz, self.sobolev_pool_medium_bins_xyz)
                pooled_norm_source = self._pool_xyz_bins(target_norm_xyz, self.sobolev_pool_medium_bins_xyz)
                axis_pool_factors_xyz = (
                    max(float(pred_xyz.shape[0]) / max(float(pooled_pred.shape[0]), 1.0), 1.0),
                    max(float(pred_xyz.shape[1]) / max(float(pooled_pred.shape[1]), 1.0), 1.0),
                    max(float(pred_xyz.shape[2]) / max(float(pooled_pred.shape[2]), 1.0), 1.0),
                )
            elif mode == 'global':
                pooled_pred = self._pool_xyz_bins(pred_xyz, self.sobolev_pool_global_bins_xyz)
                pooled_target = self._pool_xyz_bins(target_xyz, self.sobolev_pool_global_bins_xyz)
                pooled_norm_source = self._pool_xyz_bins(target_norm_xyz, self.sobolev_pool_global_bins_xyz)
                axis_pool_factors_xyz = (
                    max(float(pred_xyz.shape[0]) / max(float(pooled_pred.shape[0]), 1.0), 1.0),
                    max(float(pred_xyz.shape[1]) / max(float(pooled_pred.shape[1]), 1.0), 1.0),
                    max(float(pred_xyz.shape[2]) / max(float(pooled_pred.shape[2]), 1.0), 1.0),
                )

            layer_w_grad = {
                'local': self.w_sobolev_3d_grad_local,
                'medium': self.w_sobolev_3d_grad_medium,
                'global': self.w_sobolev_3d_grad_global,
            }[layer_name]
            results[layer_name] = self._sobolev_layer_metrics(
                pooled_pred,
                pooled_target,
                pooled_norm_source,
                pixel_pitch_cm,
                z_bin_cm,
                w_sobolev_3d_grad=layer_w_grad,
                axis_pool_factors_xyz=axis_pool_factors_xyz,
            )

        return results

    def _format_three_layer_sobolev_pooling(self):
        return (
            "3-layer Sobolev pooling report:\n"
            f"  normalization source: {self.sobolev_norm_target_source}\n"
            f"  layer balance: {self.sobolev_pool_layer_balance}\n"
            "  layer 1: local, no pooling\n"
            f"  layer 2: medium xyz binning with bins_xyz={self.sobolev_pool_medium_bins_xyz}\n"
            f"  layer 3: global xyz binning with bins_xyz={self.sobolev_pool_global_bins_xyz}"
        )

    def print_three_layer_sobolev_pooling(self):
        print(self._format_three_layer_sobolev_pooling())

    def sobolev_pooling_bins(self):
        return {
            'medium_bins_xyz': self.sobolev_pool_medium_bins_xyz,
            'global_bins_xyz': self.sobolev_pool_global_bins_xyz,
        }

    def compute(self, params, prediction, target):
        """Compute loss on fixed-shape detector voxels (nx, ny, nz).

        nx = number of detector pixels along x.
        ny = number of detector pixels along y.
        nz = full detector z bins by concatenating TPC drift bins in increasing-z order.
        """
        target_adcs = target['adcs']
        target_x = jnp.asarray(target['pixel_x'], dtype=jnp.int32) if 'pixel_x' in target else None
        target_y = jnp.asarray(target['pixel_y'], dtype=jnp.int32) if 'pixel_y' in target else None
        target_z = jnp.asarray(target['pixel_z'], dtype=jnp.float32) if 'pixel_z' in target else None
        target_event = jnp.asarray(target['event'], dtype=jnp.int32) if 'event' in target else None

        pred_x = jnp.asarray(prediction['pixel_x'], dtype=jnp.int32) if 'pixel_x' in prediction else None
        pred_y = jnp.asarray(prediction['pixel_y'], dtype=jnp.int32) if 'pixel_y' in prediction else None
        pred_plane = jnp.asarray(prediction['pixel_plane'], dtype=jnp.int32) if 'pixel_plane' in prediction else None
        pred_event = jnp.asarray(prediction['event'], dtype=jnp.int32) if 'event' in prediction else None

        ticks_prob = prediction['ticks_prob']
        adcs_distrib = prediction['adcs_distrib']

        n_ticks = ticks_prob.shape[2]
        nx = int(getattr(params, 'n_pixels_x', 0))
        ny = int(getattr(params, 'n_pixels_y', 0))
        nz = int(getattr(params, 'nz', 0))
        ntpc = int(getattr(params, 'ntpc', getattr(params, 'tpc_borders').shape[0]))
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError(
                "Invalid detector geometry in ProbabilisticLossStrategy: "
                f"n_pixels_x={nx}, n_pixels_y={ny}, nz={nz}."
            )
        if n_ticks <= 0:
            raise ValueError(f"Invalid probabilistic prediction shape: n_ticks={n_ticks}.")

        z_tick_size = float(getattr(params, 't_sampling', 1.0)) * float(getattr(params, 'vdrift_static', 1.0))
        pixel_pitch = float(getattr(params, 'pixel_pitch', 1.0))
        z_bin_size_sparse = float(z_tick_size)

        tpc_borders = jnp.asarray(getattr(params, 'tpc_borders'))
        z_lo_per_tpc = jnp.minimum(tpc_borders[:, 2, 0], tpc_borders[:, 2, 1])
        z_hi_per_tpc = jnp.maximum(tpc_borders[:, 2, 0], tpc_borders[:, 2, 1])

        z_min_per_tpc = getattr(params, 'z_min_per_tpc', None)
        nz_per_tpc = getattr(params, 'nz_per_tpc', None)
        offset_by_tpc = getattr(params, 'offset_by_tpc', None)

        z_min_per_tpc = jnp.asarray(z_min_per_tpc, dtype=jnp.float32)
        nz_per_tpc = jnp.asarray(nz_per_tpc, dtype=jnp.int32)
        offset_by_tpc = jnp.asarray(offset_by_tpc, dtype=jnp.int32)

        target_x = jnp.asarray(target_x, dtype=jnp.int32)
        target_y = jnp.asarray(target_y, dtype=jnp.int32)
        target_z = jnp.asarray(target_z, dtype=jnp.float32)
        target_event = jnp.asarray(target_event, dtype=jnp.int32) if target_event is not None else jnp.zeros_like(target_x, dtype=jnp.int32)
        target_charge = adc2charge(target_adcs, params)
        total_target_charge = jnp.maximum(jnp.sum(jnp.abs(target_charge)), 1.0)

        pred_x = jnp.asarray(pred_x, dtype=jnp.int32)
        pred_y = jnp.asarray(pred_y, dtype=jnp.int32)
        pred_plane = jnp.asarray(pred_plane, dtype=jnp.int32)
        pred_event = jnp.asarray(pred_event, dtype=jnp.int32) if pred_event is not None else jnp.zeros_like(pred_x, dtype=jnp.int32)
        pred_charge_raw = adc2charge(adcs_distrib, params)

        valid_pred_pix = (
            (pred_x >= 0) & (pred_x < nx)
            & (pred_y >= 0) & (pred_y < ny)
            & (pred_plane >= 0) & (pred_plane < ntpc)
            & (pred_event >= 0)
        )
        pred_px_safe = jnp.where(valid_pred_pix, pred_x, 0)
        pred_py_safe = jnp.where(valid_pred_pix, pred_y, 0)
        pred_plane_safe = jnp.where(valid_pred_pix, pred_plane, 0)

        target_xy_valid = (
            (target_x >= 0) & (target_x < nx)
            & (target_y >= 0) & (target_y < ny)
            & (target_event >= 0)
        )
        overflow_target_hits = jnp.sum(
            ((target_event >= self.max_events_per_batch) & target_xy_valid).astype(jnp.float32)
        )
        overflow_pred_pixels = jnp.sum(
            ((pred_event >= self.max_events_per_batch) & valid_pred_pix).astype(jnp.float32)
        )

        layer_weights = self._sobolev_pool_layer_weights()
        event_ids = jnp.arange(self.max_events_per_batch, dtype=jnp.int32)

        target_evt_valid = target_xy_valid & (target_event < self.max_events_per_batch)
        pred_evt_valid = valid_pred_pix & (pred_event < self.max_events_per_batch)

        target_chunk_size = max(1, (int(target_x.shape[0]) + self.max_events_per_batch - 1) // self.max_events_per_batch)
        pred_chunk_size = max(1, (int(pred_x.shape[0]) + self.max_events_per_batch - 1) // self.max_events_per_batch)

        def _build_event_chunks(event_arr, valid_arr, chunk_size):
            def _one_event(evt_id):
                evt_mask = valid_arr & (event_arr == evt_id)
                idx = jnp.nonzero(evt_mask, size=chunk_size, fill_value=0)[0]
                raw_count = jnp.sum(evt_mask.astype(jnp.int32))
                count = jnp.minimum(raw_count, chunk_size)
                return idx, count, raw_count

            return jax.vmap(_one_event)(event_ids)

        target_idx_chunks, target_chunk_counts, target_chunk_raw_counts = _build_event_chunks(
            target_event,
            target_evt_valid,
            target_chunk_size,
        )
        pred_idx_chunks, pred_chunk_counts, pred_chunk_raw_counts = _build_event_chunks(
            pred_event,
            pred_evt_valid,
            pred_chunk_size,
        )

        def _validate_event_chunk_counts(t_counts, p_counts):
            bad_events = [
                idx for idx, (tc, pc) in enumerate(zip(t_counts.tolist(), p_counts.tolist()))
                if (tc == 0) or (pc == 0)
            ]
            if bad_events:
                raise ValueError(
                    "ProbabilisticLossStrategy requires both target and prediction entries per event; "
                    f"found zero-sized chunks for events {bad_events}."
                )

        jax.debug.callback(_validate_event_chunk_counts, target_chunk_counts, pred_chunk_counts)

        overflow_target_hits = overflow_target_hits + jnp.sum(
            jnp.maximum(target_chunk_raw_counts - target_chunk_counts, 0).astype(jnp.float32)
        )
        overflow_pred_pixels = overflow_pred_pixels + jnp.sum(
            jnp.maximum(pred_chunk_raw_counts - pred_chunk_counts, 0).astype(jnp.float32)
        )

        zero_metrics = {
            'sobolev_3d_value': jnp.array(0.0, dtype=jnp.float32),
            'sobolev_3d_grad': jnp.array(0.0, dtype=jnp.float32),
            'sobolev_3d': jnp.array(0.0, dtype=jnp.float32),
            'local_value': jnp.array(0.0, dtype=jnp.float32),
            'local_grad_x_e': jnp.array(0.0, dtype=jnp.float32),
            'local_grad_y_e': jnp.array(0.0, dtype=jnp.float32),
            'local_grad_z_e': jnp.array(0.0, dtype=jnp.float32),
            'local_sobolev_3d_grad': jnp.array(0.0, dtype=jnp.float32),
            'local_norm_voxels': jnp.array(0.0, dtype=jnp.float32),
            'local_active_voxels': jnp.array(0.0, dtype=jnp.float32),
            'local_total': jnp.array(0.0, dtype=jnp.float32),
            'medium_value': jnp.array(0.0, dtype=jnp.float32),
            'medium_grad_x_e': jnp.array(0.0, dtype=jnp.float32),
            'medium_grad_y_e': jnp.array(0.0, dtype=jnp.float32),
            'medium_grad_z_e': jnp.array(0.0, dtype=jnp.float32),
            'medium_sobolev_3d_grad': jnp.array(0.0, dtype=jnp.float32),
            'medium_norm_voxels': jnp.array(0.0, dtype=jnp.float32),
            'medium_active_voxels': jnp.array(0.0, dtype=jnp.float32),
            'medium_total': jnp.array(0.0, dtype=jnp.float32),
            'global_value': jnp.array(0.0, dtype=jnp.float32),
            'global_grad_x_e': jnp.array(0.0, dtype=jnp.float32),
            'global_grad_y_e': jnp.array(0.0, dtype=jnp.float32),
            'global_grad_z_e': jnp.array(0.0, dtype=jnp.float32),
            'global_sobolev_3d_grad': jnp.array(0.0, dtype=jnp.float32),
            'global_norm_voxels': jnp.array(0.0, dtype=jnp.float32),
            'global_active_voxels': jnp.array(0.0, dtype=jnp.float32),
            'global_total': jnp.array(0.0, dtype=jnp.float32),
            'mean_pred_occupancy': jnp.array(0.0, dtype=jnp.float32),
            'mean_target_occupancy': jnp.array(0.0, dtype=jnp.float32),
            'residual_mean_abs': jnp.array(0.0, dtype=jnp.float32),
            'pred_field_mean': jnp.array(0.0, dtype=jnp.float32),
            'target_field_mean': jnp.array(0.0, dtype=jnp.float32),
            'z_win_start_tick': jnp.array(0.0, dtype=jnp.float32),
            'event_weight': jnp.array(0.0, dtype=jnp.float32),
            'event_target_charge': jnp.array(0.0, dtype=jnp.float32),
            'active_event': jnp.array(0.0, dtype=jnp.float32),
            'sobolev_pool_nbin_medium': jnp.array(0.0, dtype=jnp.float32),
            'sobolev_pool_nbin_global': jnp.array(0.0, dtype=jnp.float32),
        }

        def _event_metrics(evt_id):
            evt_tgt_idx = target_idx_chunks[evt_id]
            evt_tgt_count = target_chunk_counts[evt_id]
            evt_pred_idx = pred_idx_chunks[evt_id]
            evt_pred_count = pred_chunk_counts[evt_id]

            evt_tgt_valid = jnp.arange(target_chunk_size, dtype=jnp.int32) < evt_tgt_count
            evt_pred_valid = jnp.arange(pred_chunk_size, dtype=jnp.int32) < evt_pred_count
            has_evt = (evt_tgt_count > 0) | (evt_pred_count > 0)

            def _compute_for_event(_):
                # Put target smeared hits on a dense grid
                target_x_evt = target_x[evt_tgt_idx]
                target_y_evt = target_y[evt_tgt_idx]
                target_z_evt = target_z[evt_tgt_idx]
                target_charge_evt = target_charge[evt_tgt_idx]

                target_in_tpc_evt = (
                    (target_z_evt[:, None] >= z_lo_per_tpc[None, :])
                    & (target_z_evt[:, None] <= z_hi_per_tpc[None, :])
                )
                target_has_tpc_evt = jnp.any(target_in_tpc_evt, axis=1)
                target_plane_evt = jnp.argmax(target_in_tpc_evt.astype(jnp.int32), axis=1)
                target_plane_safe_evt = jnp.clip(target_plane_evt, 0, ntpc - 1)

                target_z_local_bin_evt = jnp.round(
                    (target_z_evt - z_min_per_tpc[target_plane_safe_evt]) / max(z_tick_size, 1e-12)
                ).astype(jnp.int32)
                target_z_valid_evt = (
                    target_z_local_bin_evt >= 0
                ) & (target_z_local_bin_evt < nz_per_tpc[target_plane_safe_evt])
                target_z_global_evt = offset_by_tpc[target_plane_safe_evt] + jnp.clip(
                    target_z_local_bin_evt,
                    0,
                    nz_per_tpc[target_plane_safe_evt] - 1,
                )

                evt_tgt = evt_tgt_valid & target_has_tpc_evt & target_z_valid_evt
                evt_abs_charge = jnp.where(evt_tgt, jnp.abs(target_charge_evt), 0.0)
                evt_target_charge = jnp.sum(evt_abs_charge)
                evt_weight = jnp.maximum(evt_target_charge, 1.0)

                target_x_idx_evt = jnp.where(evt_tgt, target_x_evt, 0)
                target_y_idx_evt = jnp.where(evt_tgt, target_y_evt, 0)
                target_z_idx_evt = jnp.where(evt_tgt, target_z_global_evt, 0)

                target_occ_xyz_evt = jnp.clip(
                    jnp.zeros((nx, ny, nz), dtype=jnp.float32).at[
                        target_x_idx_evt,
                        target_y_idx_evt,
                        target_z_idx_evt,
                    ].add(jnp.where(evt_tgt, 1.0, 0.0)),
                    0.0,
                    1.0,
                )
                target_expected_xyz_unsmeared_evt = jnp.zeros((nx, ny, nz), dtype=jnp.float32).at[
                    target_x_idx_evt,
                    target_y_idx_evt,
                    target_z_idx_evt,
                ].add(jnp.where(evt_tgt, target_charge_evt, 0.0))

                if has_target_smearing:
                    target_expected_xyz_evt = self._gaussian_smear_target(
                        target_expected_xyz_unsmeared_evt,
                        smear_gaussian_weights,
                    )
                else:
                    target_expected_xyz_evt = target_expected_xyz_unsmeared_evt

                target_norm_xyz_evt = (
                    target_expected_xyz_evt if self.sobolev_norm_target_source == 'smeared'
                    else target_expected_xyz_unsmeared_evt
                )

                # Put predicted pixels on the same dense grid, weighted by expected charge and occupancy.
                pred_px_evt = pred_px_safe[evt_pred_idx]
                pred_py_evt = pred_py_safe[evt_pred_idx]
                pred_plane_evt = pred_plane_safe[evt_pred_idx]
                pred_hit_prob_evt = ticks_prob[evt_pred_idx]
                pred_charge_raw_evt = pred_charge_raw[evt_pred_idx]

                pred_expected_evt = jnp.sum(pred_hit_prob_evt * pred_charge_raw_evt, axis=1)
                pred_occ_evt = jnp.clip(1.0 - jnp.prod(1.0 - pred_hit_prob_evt, axis=1), 0.0, 1.0)

                tick_axis_evt = jnp.broadcast_to(
                    jnp.arange(n_ticks, dtype=jnp.float32)[None, :],
                    (pred_chunk_size, n_ticks),
                )
                pred_z_cm_evt = get_hit_z(
                    params,
                    tick_axis_evt,
                    jnp.broadcast_to(pred_plane_evt[:, None], (pred_chunk_size, n_ticks)),
                    fixed_v=False,
                )
                pred_z_local_bin_evt = jnp.round(
                    (pred_z_cm_evt - z_min_per_tpc[pred_plane_evt][:, None]) / max(z_tick_size, 1e-12)
                ).astype(jnp.int32)
                pred_z_valid = (
                    pred_z_local_bin_evt >= 0
                ) & (pred_z_local_bin_evt < nz_per_tpc[pred_plane_evt][:, None])
                pred_z_global = offset_by_tpc[pred_plane_evt][:, None] + jnp.clip(
                    pred_z_local_bin_evt,
                    0,
                    nz_per_tpc[pred_plane_evt][:, None] - 1,
                )

                pred_in_evt = evt_pred_valid[:, None] & pred_z_valid
                pred_x_idx_evt = jnp.where(pred_in_evt, pred_px_evt[:, None], 0)
                pred_y_idx_evt = jnp.where(pred_in_evt, pred_py_evt[:, None], 0)
                pred_z_idx_evt = jnp.where(pred_in_evt, pred_z_global, 0)

                pred_expected_xyz_evt = jnp.zeros((nx, ny, nz), dtype=jnp.float32).at[
                    pred_x_idx_evt,
                    pred_y_idx_evt,
                    pred_z_idx_evt,
                ].add(jnp.where(pred_in_evt, pred_expected_evt, 0.0))
                pred_occ_xyz_evt = jnp.clip(
                    jnp.zeros((nx, ny, nz), dtype=jnp.float32).at[
                        pred_x_idx_evt,
                        pred_y_idx_evt,
                        pred_z_idx_evt,
                    ].add(jnp.where(pred_in_evt, pred_occ_evt, 0.0)),
                    0.0,
                    1.0,
                )

                # Sobolev loss
                sobolev_pool_layers_evt = self._compute_three_layer_sobolev_pooling(
                    pred_expected_xyz_evt,
                    target_expected_xyz_evt,
                    target_norm_xyz_evt,
                    pixel_pitch,
                    z_bin_size_sparse,
                )
                sobolev_pool_local_evt = sobolev_pool_layers_evt['local']
                sobolev_pool_medium_evt = sobolev_pool_layers_evt['medium']
                sobolev_pool_global_evt = sobolev_pool_layers_evt['global']

                sobolev_3d_value_evt = (
                    layer_weights[0] * sobolev_pool_local_evt['value']
                    + layer_weights[1] * sobolev_pool_medium_evt['value']
                    + layer_weights[2] * sobolev_pool_global_evt['value']
                ) / 3.0
                sobolev_3d_grad_evt = (
                    layer_weights[0] * sobolev_pool_local_evt['sobolev_3d_grad']
                    + layer_weights[1] * sobolev_pool_medium_evt['sobolev_3d_grad']
                    + layer_weights[2] * sobolev_pool_global_evt['sobolev_3d_grad']
                ) / 3.0
                sobolev_3d_evt = (
                    layer_weights[0] * sobolev_pool_local_evt['total']
                    + layer_weights[1] * sobolev_pool_medium_evt['total']
                    + layer_weights[2] * sobolev_pool_global_evt['total']
                ) / 3.0

                residual_xyz_evt = pred_expected_xyz_evt - target_expected_xyz_evt
                mean_pred_occupancy_evt = jnp.sum(pred_occ_xyz_evt) / float(ny * nx * nz)
                mean_target_occupancy_evt = jnp.sum(target_occ_xyz_evt) / float(ny * nx * nz)

                return {
                    'sobolev_3d_value': sobolev_3d_value_evt,
                    'sobolev_3d_grad': sobolev_3d_grad_evt,
                    'sobolev_3d': sobolev_3d_evt,
                    'local_value': sobolev_pool_local_evt['value'],
                    'local_grad_x_e': sobolev_pool_local_evt['grad_x_e'],
                    'local_grad_y_e': sobolev_pool_local_evt['grad_y_e'],
                    'local_grad_z_e': sobolev_pool_local_evt['grad_z_e'],
                    'local_sobolev_3d_grad': sobolev_pool_local_evt['sobolev_3d_grad'],
                    'local_norm_voxels': sobolev_pool_local_evt['norm_voxels'],
                    'local_active_voxels': sobolev_pool_local_evt['active_voxels'],
                    'local_total': sobolev_pool_local_evt['total'],
                    'medium_value': sobolev_pool_medium_evt['value'],
                    'medium_grad_x_e': sobolev_pool_medium_evt['grad_x_e'],
                    'medium_grad_y_e': sobolev_pool_medium_evt['grad_y_e'],
                    'medium_grad_z_e': sobolev_pool_medium_evt['grad_z_e'],
                    'medium_sobolev_3d_grad': sobolev_pool_medium_evt['sobolev_3d_grad'],
                    'medium_norm_voxels': sobolev_pool_medium_evt['norm_voxels'],
                    'medium_active_voxels': sobolev_pool_medium_evt['active_voxels'],
                    'medium_total': sobolev_pool_medium_evt['total'],
                    'global_value': sobolev_pool_global_evt['value'],
                    'global_grad_x_e': sobolev_pool_global_evt['grad_x_e'],
                    'global_grad_y_e': sobolev_pool_global_evt['grad_y_e'],
                    'global_grad_z_e': sobolev_pool_global_evt['grad_z_e'],
                    'global_sobolev_3d_grad': sobolev_pool_global_evt['sobolev_3d_grad'],
                    'global_norm_voxels': sobolev_pool_global_evt['norm_voxels'],
                    'global_active_voxels': sobolev_pool_global_evt['active_voxels'],
                    'global_total': sobolev_pool_global_evt['total'],
                    'mean_pred_occupancy': mean_pred_occupancy_evt,
                    'mean_target_occupancy': mean_target_occupancy_evt,
                    'residual_mean_abs': jnp.mean(jnp.abs(residual_xyz_evt)),
                    'pred_field_mean': jnp.mean(pred_expected_xyz_evt),
                    'target_field_mean': jnp.mean(target_expected_xyz_evt),
                    'z_win_start_tick': jnp.array(0.0, dtype=jnp.float32),
                    'event_weight': evt_weight,
                    'event_target_charge': evt_target_charge,
                    'active_event': jnp.array(1.0, dtype=jnp.float32),
                    'sobolev_pool_nbin_medium': jnp.array(self.sobolev_pool_medium_bins_xyz, dtype=jnp.float32),
                    'sobolev_pool_nbin_global': jnp.array(self.sobolev_pool_global_bins_xyz, dtype=jnp.float32),
                }

            return jax.lax.cond(has_evt, _compute_for_event, lambda _: zero_metrics, operand=None)

        def _scan_body(carry, evt_id):
            evt_metrics = _event_metrics(evt_id)
            w = evt_metrics['event_weight']
            for key in zero_metrics.keys():
                carry[key] = carry[key] + w * evt_metrics[key]
            carry['sum_weights'] = carry['sum_weights'] + w
            carry['sum_active_events'] = carry['sum_active_events'] + evt_metrics['active_event']
            carry['sum_event_target_charge'] = carry['sum_event_target_charge'] + evt_metrics['event_target_charge']
            return carry, 0

        weighted_sums = {k: jnp.array(0.0, dtype=jnp.float32) for k in zero_metrics.keys()}
        weighted_sums['sum_weights'] = jnp.array(0.0, dtype=jnp.float32)
        weighted_sums['sum_active_events'] = jnp.array(0.0, dtype=jnp.float32)
        weighted_sums['sum_event_target_charge'] = jnp.array(0.0, dtype=jnp.float32)
        weighted_sums, _ = jax.lax.scan(_scan_body, weighted_sums, event_ids)

        denom_weights = jnp.maximum(weighted_sums['sum_weights'], 1e-12)
        sobolev_pool_reports = {
            'local_norm_voxels': weighted_sums['local_norm_voxels'] / denom_weights,
            'local_active_voxels': weighted_sums['local_active_voxels'] / denom_weights,
            'local_value': weighted_sums['local_value'] / denom_weights,
            'local_grad_x_e': weighted_sums['local_grad_x_e'] / denom_weights,
            'local_grad_y_e': weighted_sums['local_grad_y_e'] / denom_weights,
            'local_grad_z_e': weighted_sums['local_grad_z_e'] / denom_weights,
            'local_sobolev_3d_grad': weighted_sums['local_sobolev_3d_grad'] / denom_weights,
            'local_total': weighted_sums['local_total'] / denom_weights,
            'medium_norm_voxels': weighted_sums['medium_norm_voxels'] / denom_weights,
            'medium_active_voxels': weighted_sums['medium_active_voxels'] / denom_weights,
            'medium_value': weighted_sums['medium_value'] / denom_weights,
            'medium_grad_x_e': weighted_sums['medium_grad_x_e'] / denom_weights,
            'medium_grad_y_e': weighted_sums['medium_grad_y_e'] / denom_weights,
            'medium_grad_z_e': weighted_sums['medium_grad_z_e'] / denom_weights,
            'medium_sobolev_3d_grad': weighted_sums['medium_sobolev_3d_grad'] / denom_weights,
            'medium_total': weighted_sums['medium_total'] / denom_weights,
            'global_norm_voxels': weighted_sums['global_norm_voxels'] / denom_weights,
            'global_active_voxels': weighted_sums['global_active_voxels'] / denom_weights,
            'global_value': weighted_sums['global_value'] / denom_weights,
            'global_grad_x_e': weighted_sums['global_grad_x_e'] / denom_weights,
            'global_grad_y_e': weighted_sums['global_grad_y_e'] / denom_weights,
            'global_grad_z_e': weighted_sums['global_grad_z_e'] / denom_weights,
            'global_sobolev_3d_grad': weighted_sums['global_sobolev_3d_grad'] / denom_weights,
            'global_total': weighted_sums['global_total'] / denom_weights,
            'sobolev_pool_nbin_medium': jnp.array(self.sobolev_pool_medium_bins_xyz, dtype=jnp.float32),
            'sobolev_pool_nbin_global': jnp.array(self.sobolev_pool_global_bins_xyz, dtype=jnp.float32),
        }

        target_gaussian_radius_cm_eff = radius_cm_eff
        target_gaussian_sigma_cm_eff = sigma_cm_eff

        sobolev_pool_reports['layer_weight_local'] = layer_weights[0]
        sobolev_pool_reports['layer_weight_medium'] = layer_weights[1]
        sobolev_pool_reports['layer_weight_global'] = layer_weights[2]

        if self.emit_sobolev_pool_report:
            print(self._format_three_layer_sobolev_pooling())
            jax.debug.print(
                (
                    "3-layer Sobolev metrics\n"
                    "  local : norm_vox={ln:.1f}, active_vox={la:.1f}, value={lv:.6e}, grad_x={lx:.6e}, grad_y={ly:.6e}, grad_z={lz:.6e}, grad_3d={l3:.6e}, total={lt:.6e}\n"
                    "  medium: norm_vox={mn:.1f}, active_vox={ma:.1f}, value={mv:.6e}, grad_x={mx:.6e}, grad_y={my:.6e}, grad_z={mz:.6e}, grad_3d={m3:.6e}, total={mt:.6e}\n"
                    "  global: norm_vox={gn:.1f}, active_vox={ga:.1f}, value={gv:.6e}, grad_x={gx:.6e}, grad_y={gy:.6e}, grad_z={gz:.6e}, grad_3d={g3:.6e}, total={gt:.6e}"
                ),
                ln=sobolev_pool_reports['local_norm_voxels'],
                la=sobolev_pool_reports['local_active_voxels'],
                lv=sobolev_pool_reports['local_value'],
                lx=sobolev_pool_reports['local_grad_x_e'],
                ly=sobolev_pool_reports['local_grad_y_e'],
                lz=sobolev_pool_reports['local_grad_z_e'],
                l3=sobolev_pool_reports['local_sobolev_3d_grad'],
                lt=sobolev_pool_reports['local_total'],
                mn=sobolev_pool_reports['medium_norm_voxels'],
                ma=sobolev_pool_reports['medium_active_voxels'],
                mv=sobolev_pool_reports['medium_value'],
                mx=sobolev_pool_reports['medium_grad_x_e'],
                my=sobolev_pool_reports['medium_grad_y_e'],
                mz=sobolev_pool_reports['medium_grad_z_e'],
                m3=sobolev_pool_reports['medium_sobolev_3d_grad'],
                mt=sobolev_pool_reports['medium_total'],
                gn=sobolev_pool_reports['global_norm_voxels'],
                ga=sobolev_pool_reports['global_active_voxels'],
                gv=sobolev_pool_reports['global_value'],
                gx=sobolev_pool_reports['global_grad_x_e'],
                gy=sobolev_pool_reports['global_grad_y_e'],
                gz=sobolev_pool_reports['global_grad_z_e'],
                g3=sobolev_pool_reports['global_sobolev_3d_grad'],
                gt=sobolev_pool_reports['global_total'],
            )

        sobolev_3d_value = weighted_sums['sobolev_3d_value'] / denom_weights
        sobolev_3d_grad = weighted_sums['sobolev_3d_grad'] / denom_weights
        sobolev_3d = weighted_sums['sobolev_3d'] / denom_weights

        total_loss = sobolev_3d
        mean_pred_occupancy = weighted_sums['mean_pred_occupancy'] / denom_weights
        mean_target_occupancy = weighted_sums['mean_target_occupancy'] / denom_weights

        aux = {
            'total_target_charge': total_target_charge,
            'sobolev_integrated_field_mean_pred': weighted_sums['pred_field_mean'] / denom_weights,
            'sobolev_integrated_field_mean_target': weighted_sums['target_field_mean'] / denom_weights,
            'sobolev_3d_value': sobolev_3d_value,
            'sobolev_3d_grad': sobolev_3d_grad,
            'sobolev_3d': sobolev_3d,
            'sobolev_pool_local_value': sobolev_pool_reports['local_value'],
            'sobolev_pool_local_grad_x_e': sobolev_pool_reports['local_grad_x_e'],
            'sobolev_pool_local_grad_y_e': sobolev_pool_reports['local_grad_y_e'],
            'sobolev_pool_local_grad_z_e': sobolev_pool_reports['local_grad_z_e'],
            'sobolev_pool_local_sobolev_3d_grad': sobolev_pool_reports['local_sobolev_3d_grad'],
            'sobolev_pool_local_norm_voxels': sobolev_pool_reports['local_norm_voxels'],
            'sobolev_pool_local_active_voxels': sobolev_pool_reports['local_active_voxels'],
            'sobolev_pool_local_total': sobolev_pool_reports['local_total'],
            'sobolev_pool_medium_value': sobolev_pool_reports['medium_value'],
            'sobolev_pool_medium_grad_x_e': sobolev_pool_reports['medium_grad_x_e'],
            'sobolev_pool_medium_grad_y_e': sobolev_pool_reports['medium_grad_y_e'],
            'sobolev_pool_medium_grad_z_e': sobolev_pool_reports['medium_grad_z_e'],
            'sobolev_pool_medium_sobolev_3d_grad': sobolev_pool_reports['medium_sobolev_3d_grad'],
            'sobolev_pool_medium_norm_voxels': sobolev_pool_reports['medium_norm_voxels'],
            'sobolev_pool_medium_active_voxels': sobolev_pool_reports['medium_active_voxels'],
            'sobolev_pool_medium_total': sobolev_pool_reports['medium_total'],
            'sobolev_pool_global_value': sobolev_pool_reports['global_value'],
            'sobolev_pool_global_grad_x_e': sobolev_pool_reports['global_grad_x_e'],
            'sobolev_pool_global_grad_y_e': sobolev_pool_reports['global_grad_y_e'],
            'sobolev_pool_global_grad_z_e': sobolev_pool_reports['global_grad_z_e'],
            'sobolev_pool_global_sobolev_3d_grad': sobolev_pool_reports['global_sobolev_3d_grad'],
            'sobolev_pool_global_norm_voxels': sobolev_pool_reports['global_norm_voxels'],
            'sobolev_pool_global_active_voxels': sobolev_pool_reports['global_active_voxels'],
            'sobolev_pool_global_total': sobolev_pool_reports['global_total'],
            'sobolev_pool_layer_balance_mode': self.sobolev_pool_layer_balance,
            'sobolev_pool_layer_weight_local': sobolev_pool_reports['layer_weight_local'],
            'sobolev_pool_layer_weight_medium': sobolev_pool_reports['layer_weight_medium'],
            'sobolev_pool_layer_weight_global': sobolev_pool_reports['layer_weight_global'],
            'target_gaussian_3d_radius_cm': jnp.array(target_gaussian_radius_cm_eff, dtype=jnp.float32),
            'target_gaussian_3d_sigma_cm': jnp.array(target_gaussian_sigma_cm_eff, dtype=jnp.float32),
            'z_tick_size': jnp.array(z_tick_size, dtype=jnp.float32),
            'mean_pred_occupancy': mean_pred_occupancy,
            'mean_target_occupancy': mean_target_occupancy,
            'z_win_start_tick': weighted_sums['z_win_start_tick'] / denom_weights,
            'nz': jnp.array(nz, dtype=jnp.float32),
            'z_bin_size_sparse': jnp.array(z_bin_size_sparse, dtype=jnp.float32),
            'residual_mean_abs': weighted_sums['residual_mean_abs'] / denom_weights,
            'event_weighted_mean_denom': denom_weights,
            'event_active_count': weighted_sums['sum_active_events'],
            'event_target_charge_sum': weighted_sums['sum_event_target_charge'],
            'max_events_per_batch': jnp.array(self.max_events_per_batch, dtype=jnp.float32),
            'event_overflow_target_hits': overflow_target_hits,
            'event_overflow_pred_pixels': overflow_pred_pixels,
            'sobolev_pool_nbin_medium': jnp.array(self.sobolev_pool_medium_bins_xyz, dtype=jnp.float32),
            'sobolev_pool_nbin_global': jnp.array(self.sobolev_pool_global_bins_xyz, dtype=jnp.float32),
        }

        return total_loss, aux
