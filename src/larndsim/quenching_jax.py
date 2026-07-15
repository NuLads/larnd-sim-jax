"""
Module to implement the quenching of the ionized electrons
through the detector
"""

import jax
import jax.numpy as jnp
from jax import jit
import jax.scipy as jsp
import numpy as np
from functools import partial
from larndsim.consts_jax import RecombinationMode, get_dedx_density_data, get_dedx_flow_model
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.info("QUENCHING MODULE PARAMETERS")

# Per-particle dE/dx histogram config.
# Fields:
#   is_1d          – histogram is 1D (no R axis); broadcast marginal_pdf to all tracks
#   r_clip         – hard-clip R to this max before bin lookup (None = no clip)
#   r_tg_threshold – optional internal fallback to throughgoing_muon profile
_PARTICLE_HIST_CONFIG = {
    "throughgoing_muon": dict(is_1d=True,  r_clip=None, r_tg_threshold=None),
    "stopping_muon":     dict(is_1d=False, r_clip=50.0, r_tg_threshold=50.0),
    "stopping_proton":   dict(is_1d=False, r_clip=20.0, r_tg_threshold=None),
}

# Per-particle flow quadrature hyperparameters.
# Empirically determined via flow_quadrature_hyperparam_recommendation.ipynb
# to achieve <0.5% relative error in n_electrons-related integrands.
# Fields:
#   y_clip: Truncation in normalized log-dEdx space [-y_clip, y_clip]
#   n_nodes: Number of Gauss-Legendre quadrature nodes
_PARTICLE_FLOW_QUADRATURE_CONFIG = {
    "throughgoing_muon": dict(y_clip=4.97, n_nodes=24),
    "stopping_muon":     dict(y_clip=5.24, n_nodes=8),
    "stopping_proton":   dict(y_clip=5.25, n_nodes=12),
}

# Precompute deterministic quadrature rules used by flow integration mode.
_FLOW_QUADRATURE_RULES = {}
for _n in (8, 12, 16, 24, 32):
    _nodes, _weights = np.polynomial.legendre.leggauss(_n)
    _FLOW_QUADRATURE_RULES[_n] = (
        jnp.asarray(_nodes, dtype=jnp.float32),
        jnp.asarray(_weights, dtype=jnp.float32),
    )


def _get_flow_quadrature_rule(n_nodes):
    if n_nodes not in _FLOW_QUADRATURE_RULES:
        raise ValueError(
            f"Unsupported flow_quadrature_nodes={n_nodes}. "
            f"Supported: {sorted(_FLOW_QUADRATURE_RULES)}"
        )
    return _FLOW_QUADRATURE_RULES[n_nodes]

def _get_particle_quadrature_params(particle_type, global_n_nodes=None, global_y_clip=None):
    """Get flow quadrature parameters for a particle type.
    
    Prioritizes per-particle optimized values, falls back to global parameters,
    then defaults to the empirically determined values from the recommendation notebook.
    
    Args:
        particle_type: One of 'throughgoing_muon', 'stopping_muon', 'stopping_proton'
        global_n_nodes: Global override for n_nodes (optional)
        global_y_clip: Global override for y_clip (optional)
    
    Returns:
        (n_nodes, y_clip): Recommended quadrature parameters
    """
    # Start with per-particle optimized defaults
    particle_params = _PARTICLE_FLOW_QUADRATURE_CONFIG.get(
        particle_type,
        dict(y_clip=5.25, n_nodes=24)  # Fallback to highest values
    )
    
    # Apply global overrides if provided
    n_nodes = global_n_nodes if global_n_nodes is not None else particle_params["n_nodes"]
    y_clip = global_y_clip if global_y_clip is not None else particle_params["y_clip"]
    
    return int(n_nodes), float(y_clip)

def _dedx_density_histogram(n_tracks, R_values, particle_type):
    """Unified particle-type histogram dE/dx density lookup.

    Behaviour is controlled by ``_PARTICLE_HIST_CONFIG``:
    - ``is_1d``: 1D histogram (throughgoing muon) → same marginal PDF for every track.
    - ``r_clip``: clip R to this max before histogram lookup (stopping proton: 20 cm).
    - ``r_tg_threshold``: replace density with throughgoing muon profile when R exceeds this value.
    """
    cfg = _PARTICLE_HIST_CONFIG[particle_type]
    hist = get_dedx_density_data(particle_type=particle_type)

    dedx_edges = jnp.asarray(hist["dedx_edges"])
    dedx_centers = 0.5 * (dedx_edges[:-1] + dedx_edges[1:])
    n_dedx_bins = dedx_centers.shape[0]
    dEdx_samples = jnp.broadcast_to(dedx_centers[None, :], (n_tracks, n_dedx_bins))

    if cfg["is_1d"]:
        marginal_pdf = jnp.asarray(hist["marginal_pdf"])
        dEdx_weights = jnp.broadcast_to(marginal_pdf[None, :], (n_tracks, n_dedx_bins))
        return dEdx_samples, dEdx_weights

    # --- 2-D lookup ---
    R_eff = jnp.minimum(R_values, cfg["r_clip"]) if cfg["r_clip"] is not None else R_values
    r_edges = jnp.asarray(hist["r_edges"])
    n_r_bins = r_edges.shape[0] - 1
    r_bin_idx = jnp.clip(jnp.searchsorted(r_edges, R_eff, side="right") - 1, 0, n_r_bins - 1)

    counts = jnp.asarray(hist["counts"])
    row_counts = counts[r_bin_idx, :]
    row_sums = jnp.sum(row_counts, axis=1, keepdims=True)
    marginal_pdf = jnp.asarray(hist["marginal_pdf"])[None, :]
    row_pdf = jnp.where(row_sums > 0.0, row_counts / (row_sums + 1e-12), marginal_pdf)

    # High-R fallback: substitute throughgoing-muon profile
    if cfg["r_tg_threshold"] is not None:
        tg_hist = get_dedx_density_data(particle_type="throughgoing_muon")
        tg_pdf = jnp.asarray(tg_hist["marginal_pdf"])[None, :]
        row_pdf = jnp.where(R_values[:, None] > cfg["r_tg_threshold"], tg_pdf, row_pdf)

    return dEdx_samples, row_pdf

def _dedx_density_flow(n_tracks, R_values, particle_type):
    """Build p(dE/dx | R) from trained conditional flow using fixed dE/dx support."""
    cfg = _PARTICLE_HIST_CONFIG[particle_type]
    R_eff = jnp.minimum(R_values, cfg["r_clip"]) if cfg["r_clip"] is not None else R_values

    flow_bundle = get_dedx_flow_model(particle_type=particle_type)
    flow = flow_bundle["flow"]
    norm = flow_bundle["norm_params"]
    flow_cond_shape = getattr(flow, "cond_shape", None)
    cond_dim = int(flow_bundle.get("cond_dim", flow_cond_shape[0] if flow_cond_shape else 0))
    use_condition = ("R_mean" in norm and "R_std" in norm) and cond_dim > 0

    # Reuse histogram dE/dx support (bin centers) to keep integration behavior identical.
    hist = get_dedx_density_data(particle_type=particle_type)
    dedx_edges = jnp.asarray(hist["dedx_edges"])
    dedx_centers = 0.5 * (dedx_edges[:-1] + dedx_edges[1:])
    n_dedx_bins = dedx_centers.shape[0]

    y_mean = jnp.asarray(float(norm["dEdx_log_mean"]), dtype=jnp.float32)
    y_std = jnp.asarray(float(norm["dEdx_log_std"]), dtype=jnp.float32)
    eps = jnp.asarray(float(norm.get("dEdx_eps", 1e-8)), dtype=jnp.float32)

    # Normalize dE/dx support to flow space.
    dEdx_samples = jnp.broadcast_to(dedx_centers[None, :], (n_tracks, n_dedx_bins))
    log_dedx = jnp.log(dEdx_samples + eps)
    y_norm = (log_dedx - y_mean) / (y_std + 1e-12)

    # Evaluate flow log_prob for all (track, support_bin) pairs.
    y_flat = y_norm.reshape(-1, 1)
    if use_condition:
        R_mean = jnp.asarray(float(norm["R_mean"]), dtype=jnp.float32)
        R_std = jnp.asarray(float(norm["R_std"]), dtype=jnp.float32)
        R_norm = (R_eff - R_mean) / (R_std + 1e-12)  # (n_tracks,)
        cond_flat = jnp.repeat(R_norm, n_dedx_bins)[:, None]
        log_py = flow.log_prob(y_flat, condition=cond_flat)
    else:
        cond_flat = jnp.zeros((y_flat.shape[0], cond_dim), dtype=jnp.float32)
        log_py = flow.log_prob(y_flat, condition=cond_flat)

    # Convert p(y|R) to p(dE/dx|R) up to a constant, then row-normalize.
    # y = (log(dEdx+eps) - mean) / std => |dy/ddEdx| proportional to 1/(dEdx+eps)
    density_unnorm = jnp.exp(log_py).reshape(n_tracks, n_dedx_bins) / (dEdx_samples + eps)
    density_sum = jnp.sum(density_unnorm, axis=1, keepdims=True)

    # Fallback to histogram marginal if a row underflows numerically.
    marginal_pdf = jnp.asarray(hist["marginal_pdf"])[None, :]
    row_pdf = jnp.where(
        density_sum > 0.0,
        density_unnorm / (density_sum + 1e-12),
        marginal_pdf,
    )

    # High-R muon fallback: substitute throughgoing-muon 1D histogram profile.
    if cfg["r_tg_threshold"] is not None:
        tg_hist = get_dedx_density_data(particle_type="throughgoing_muon")
        tg_pdf = jnp.asarray(tg_hist["marginal_pdf"])[None, :]
        row_pdf = jnp.where(R_values[:, None] > cfg["r_tg_threshold"], tg_pdf, row_pdf)

    return dEdx_samples, row_pdf


def _dedx_density_flow_sample(R_values, particle_type, n_samples, key):
    """Draw *n_samples* continuous dE/dx values per track from the conditional flow.

    Unlike the grid-based paths (``_dedx_density_flow``, ``_dedx_density_flow_quadrature``),
    this function does NOT evaluate on fixed bin centers or GL nodes. Each track receives
    *n_samples* independent draws from p(dEdx | R), so no discrete binning artifacts appear
    when samples are propagated to downstream observables.

    Args:
        R_values: (n_tracks,) range values in cm.
        particle_type: 'stopping_muon', 'stopping_proton', or 'throughgoing_muon'.
        n_samples: number of dE/dx draws per track.
        key: JAX PRNG key.

    Returns:
        dEdx_samples: (n_tracks, n_samples) — continuous dE/dx values.
        weights:      (n_tracks, n_samples) — uniform 1/n_samples per row.
    """
    cfg = _PARTICLE_HIST_CONFIG[particle_type]
    R_eff = jnp.minimum(R_values, cfg["r_clip"]) if cfg["r_clip"] is not None else R_values

    flow_bundle = get_dedx_flow_model(particle_type=particle_type)
    flow = flow_bundle["flow"]
    norm = flow_bundle["norm_params"]
    flow_cond_shape = getattr(flow, "cond_shape", None)
    cond_dim = int(flow_bundle.get("cond_dim", flow_cond_shape[0] if flow_cond_shape else 0))
    use_condition = bool(flow_bundle.get("use_condition", ("R_mean" in norm and "R_std" in norm and cond_dim > 0)))

    y_mean = jnp.asarray(float(norm["dEdx_log_mean"]), dtype=jnp.float32)
    y_std = jnp.asarray(float(norm["dEdx_log_std"]), dtype=jnp.float32)
    eps = jnp.asarray(float(norm.get("dEdx_eps", 1e-8)), dtype=jnp.float32)

    n_tracks = R_values.shape[0]

    if use_condition:
        R_mean = jnp.asarray(float(norm["R_mean"]), dtype=jnp.float32)
        R_std = jnp.asarray(float(norm["R_std"]), dtype=jnp.float32)
        R_norm = (R_eff - R_mean) / (R_std + 1e-12)  # (n_tracks,)
        cond = R_norm[:, None] if cond_dim == 1 else jnp.broadcast_to(R_norm[:, None], (n_tracks, cond_dim))
        try:
            # sample_shape=(n_samples,) with batched condition (n_tracks, cond_dim)
            # → output shape (n_samples, n_tracks, event_dim)
            y_samples = flow.sample(key, sample_shape=(n_samples,), condition=cond)
            y_samples = y_samples[..., 0].T
        except TypeError:
            y_samples = flow.sample(key, sample_shape=(n_tracks * n_samples,))
            y_samples = y_samples.reshape(n_tracks, n_samples)
    else:
        # Unconditional flow: draw n_tracks * n_samples scalars and reshape.
        try:
            y_samples = flow.sample(key, sample_shape=(n_tracks * n_samples,))
            y_samples = y_samples.reshape(n_tracks, n_samples)
        except TypeError:
            if "R_mean" in norm and "R_std" in norm:
                R_mean = jnp.asarray(float(norm["R_mean"]), dtype=jnp.float32)
                R_std = jnp.asarray(float(norm["R_std"]), dtype=jnp.float32)
                R_norm = (R_eff - R_mean) / (R_std + 1e-12)
                cond = R_norm[:, None] if cond_dim == 1 else jnp.broadcast_to(R_norm[:, None], (n_tracks, cond_dim))
            elif flow_cond_shape is not None:
                cond = jnp.zeros((n_tracks, *tuple(flow_cond_shape)), dtype=jnp.float32)
            else:
                raise
            y_samples = flow.sample(key, sample_shape=(n_samples,), condition=cond)
            y_samples = y_samples[..., 0].T

    # Convert from normalized log-space back to linear dE/dx.
    dEdx_samples = jnp.exp(y_samples * y_std + y_mean) - eps
    dEdx_samples = jnp.maximum(dEdx_samples, eps)

    # Uniform importance weights — all samples equally likely under this estimator.
    weights = jnp.full((n_tracks, n_samples), 1.0 / float(n_samples), dtype=dEdx_samples.dtype)
    return dEdx_samples, weights


def _dedx_density_flow_quadrature(R_values, particle_type, n_nodes=16, y_clip=5.0):
    """Deterministic quadrature over flow latent y to avoid dE/dx bin-center bias.

    Returns:
        dEdx_samples: (n_tracks, n_nodes)
        quad_weights: (n_tracks, n_nodes), normalized per track.
    """
    cfg = _PARTICLE_HIST_CONFIG[particle_type]
    R_eff = jnp.minimum(R_values, cfg["r_clip"]) if cfg["r_clip"] is not None else R_values

    flow_bundle = get_dedx_flow_model(particle_type=particle_type)
    flow = flow_bundle["flow"]
    norm = flow_bundle["norm_params"]
    flow_cond_shape = getattr(flow, "cond_shape", None)
    cond_dim = int(flow_bundle.get("cond_dim", flow_cond_shape[0] if flow_cond_shape else 0))
    use_condition = flow_cond_shape is not None and cond_dim > 0

    y_mean = jnp.asarray(float(norm["dEdx_log_mean"]), dtype=jnp.float32)
    y_std = jnp.asarray(float(norm["dEdx_log_std"]), dtype=jnp.float32)
    eps = jnp.asarray(float(norm.get("dEdx_eps", 1e-8)), dtype=jnp.float32)

    nodes, base_weights = _get_flow_quadrature_rule(n_nodes)
    y_nodes = y_clip * nodes
    quad_base = y_clip * base_weights

    n_tracks = R_values.shape[0]
    y_grid = jnp.broadcast_to(y_nodes[None, :], (n_tracks, n_nodes))

    y_flat = y_grid.reshape(-1, 1)
    if use_condition:
        R_mean = jnp.asarray(float(norm["R_mean"]), dtype=jnp.float32)
        R_std = jnp.asarray(float(norm["R_std"]), dtype=jnp.float32)
        R_norm = (R_eff - R_mean) / (R_std + 1e-12)
        cond_flat = jnp.repeat(R_norm, n_nodes)[:, None]
        log_py = flow.log_prob(y_flat, condition=cond_flat).reshape(n_tracks, n_nodes)
    else:
        cond_flat = jnp.zeros((y_flat.shape[0], 0), dtype=jnp.float32)
        log_py = flow.log_prob(y_flat, condition=cond_flat).reshape(n_tracks, n_nodes)

    log_quad = jnp.log(jnp.maximum(quad_base[None, :], 1e-30))
    log_norm = jsp.special.logsumexp(log_py + log_quad, axis=1, keepdims=True)
    quad_weights = jnp.exp(log_py + log_quad - log_norm)

    dEdx_samples = jnp.exp(y_grid * y_std + y_mean) - eps
    dEdx_samples = jnp.maximum(dEdx_samples, eps)
    return dEdx_samples, quad_weights


def _recombination_samples(params, tracks, fields, dEdx_samples):
    if params.recombination_mode == RecombinationMode.BOX:
        return box_model(dEdx_samples, params.eField, params.lArDensity, params.alpha, params.beta)
    if params.recombination_mode == RecombinationMode.BIRKS:
        return birks_model(dEdx_samples, params.eField, params.lArDensity, params.Ab, params.kb)
    if params.recombination_mode == RecombinationMode.ELLIPSOID:
        cosphi = jnp.abs(tracks[:, fields.index("z_end")] - tracks[:, fields.index("z_start")]) / (
            tracks[:, fields.index("dx")] + 1e-10
        )
        return ellipsoid_box_model(
            dEdx_samples,
            cosphi[:, None],
            params.eField,
            params.lArDensity,
            params.alpha,
            params.beta,
            params.R_param,
        )
    raise ValueError(
        f"Invalid recombination mode {params.recombination_mode}: "
        "must be RecombinationMode.BOX, RecombinationMode.BIRKS, or RecombinationMode.ELLIPSOID"
    )


def _nelectrons_samples_from_dedx(params, tracks, fields, dEdx_samples, dE_override=None):
    dx = tracks[:, fields.index("dx")][:, None]
    dE_samples = dEdx_samples * dx if dE_override is None else dE_override
    recomb_samples = _recombination_samples(params, tracks, fields, dEdx_samples)
    return get_nelectrons(dE_samples, recomb_samples, params.MeVToElectrons)


def _hist_expected_nelectrons(params, tracks, fields, R_values, particle_type):
    n_tracks = tracks.shape[0]
    dEdx_samples, dEdx_weights = _dedx_density_histogram(n_tracks, R_values, particle_type)
    n_electrons_samples = _nelectrons_samples_from_dedx(params, tracks, fields, dEdx_samples)
    return jnp.sum(n_electrons_samples * dEdx_weights, axis=1)


def _flow_expected_nelectrons_quadrature(params, tracks, fields, R_values, particle_type, n_nodes, y_clip):
    dEdx_samples, quad_weights = _dedx_density_flow_quadrature(
        R_values=R_values,
        particle_type=particle_type,
        n_nodes=n_nodes,
        y_clip=y_clip,
    )
    n_electrons_samples = _nelectrons_samples_from_dedx(params, tracks, fields, dEdx_samples)
    return jnp.sum(n_electrons_samples * quad_weights, axis=1)


def _flow_expected_nelectrons_sample(params, tracks, fields, R_values, particle_type, n_samples, key):
    """Monte Carlo estimate of E[n_electrons | R] using continuous flow samples."""
    dEdx_samples, sample_weights = _dedx_density_flow_sample(
        R_values=R_values,
        particle_type=particle_type,
        n_samples=n_samples,
        key=key,
    )
    n_electrons_samples = _nelectrons_samples_from_dedx(params, tracks, fields, dEdx_samples)
    return jnp.sum(n_electrons_samples * sample_weights, axis=1)


def _throughgoing_flow_sample_config(params):
    """Flow-sampling controls used for throughgoing-muon expectation in flow mode."""
    n_samples = int(getattr(params, "throughgoing_flow_sample_n_samples", None)) \
        if hasattr(params, "throughgoing_flow_sample_n_samples") and getattr(params, "throughgoing_flow_sample_n_samples") is not None else None
    if n_samples is None:
        n_samples = int(getattr(params, "flow_sample_n_samples", None)) \
            if hasattr(params, "flow_sample_n_samples") and getattr(params, "flow_sample_n_samples") is not None else 64

    seed = int(getattr(params, "throughgoing_flow_sample_seed", None)) \
        if hasattr(params, "throughgoing_flow_sample_seed") and getattr(params, "throughgoing_flow_sample_seed") is not None else None
    if seed is None:
        seed = int(getattr(params, "flow_sample_seed", None)) \
            if hasattr(params, "flow_sample_seed") and getattr(params, "flow_sample_seed") is not None else 0

    key = jax.random.key(seed) if hasattr(jax.random, "key") else jax.random.PRNGKey(seed)
    return n_samples, key


def box_model(dEdx, eField, lArDensity, alpha, beta):
    # Baller, 2013 JINST 8 P08005
    csi = beta * dEdx / (eField * lArDensity)
    return jnp.maximum(0, jnp.log(alpha + csi) / csi)

def birks_model(dEdx, eField, lArDensity, Ab, kb):
    # Amoruso, et al NIM A 523 (2004) 275
    return Ab / (1 + kb * dEdx / (eField * lArDensity))

def get_nelectrons(dE, recomb, MeVToElectrons):
    return recomb * dE * MeVToElectrons

def ellipsoid_box_model(dEdx, cosphi, eField, lArDensity, alpha, beta, R_param):
    # ICARUS EMB Model incorporating track angle dependence
    # b_phi represents the angular-dependent B(phi) parameter
    b_phi = beta / jnp.sqrt(1 - cosphi**2 + (1.0 / R_param**2) * cosphi**2)
    csi = b_phi * dEdx / (eField * lArDensity)
    return jnp.maximum(0, jnp.log(alpha + csi) / (csi + 1e-10))

@partial(jit, static_argnames=['fields'])
def quench(params, tracks, fields):
    """
    This function takes as input an (unstructured) array of track segments and calculates
    the number of electrons that reach the anode plane after recombination.
    It is possible to pick among two models: Box (Baller, 2013 JINST 8 P08005) or
    Birks (Amoruso, et al NIM A 523 (2004) 275).

    Args:
        tracks (:obj:`numpy.ndarray`, `JAX Tensor`): array containing the tracks segment information
        mode (RecombinationMode): recombination model.
        fields (list): an ordered string list of field/column name of the tracks structured array
    """
    n_tracks = tracks.shape[0]
    if "dx" not in fields:
        raise ValueError("quench requires a 'dx' entry in fields for segment length.")

    dx = tracks[:, fields.index("dx")][:, None]
    use_density = bool(getattr(params, "use_dedx_density", False))
    n_electrons = None
    if use_density:
        if "range" not in fields:
            raise ValueError("quench requires a 'range' entry in fields for dE/dx density lookup.")
        if "pdg_id" not in fields:
            raise ValueError("quench requires a 'pdg_id' entry in fields for particle identification.")

        R_values = tracks[:, fields.index("range")]
        pdg_ids = tracks[:, fields.index("pdg_id")]
        density_mode = getattr(params, "dedx_density_mode", "histogram")
        flow_expectation_mode = getattr(params, "flow_expectation_mode", "sample")

        if density_mode == "flow":
            if flow_expectation_mode == "quadrature":
                # Get global overrides from params (optional; if not set, use per-particle defaults)
                global_nodes = int(getattr(params, "flow_quadrature_nodes", None)) \
                    if hasattr(params, "flow_quadrature_nodes") and getattr(params, "flow_quadrature_nodes") is not None else None
                global_y_clip = float(getattr(params, "flow_quadrature_y_clip", None)) \
                    if hasattr(params, "flow_quadrature_y_clip") and getattr(params, "flow_quadrature_y_clip") is not None else None

                is_muon = jnp.abs(pdg_ids) == 13
                is_proton = jnp.abs(pdg_ids) == 2212
                use_tg_muon = is_muon & ((R_values > 50.0) | (R_values < 0.0))
                use_stop_muon = is_muon & (~use_tg_muon)

                # Compute n_electrons for each particle type using optimized hyperparameters
                stop_mu_nodes, stop_mu_y_clip = _get_particle_quadrature_params(
                    "stopping_muon", global_nodes, global_y_clip
                )
                stop_mu_ne = _flow_expected_nelectrons_quadrature(
                    params=params,
                    tracks=tracks,
                    fields=fields,
                    R_values=R_values,
                    particle_type="stopping_muon",
                    n_nodes=stop_mu_nodes,
                    y_clip=stop_mu_y_clip,
                )
                
                stop_p_nodes, stop_p_y_clip = _get_particle_quadrature_params(
                    "stopping_proton", global_nodes, global_y_clip
                )
                stop_p_ne = _flow_expected_nelectrons_quadrature(
                    params=params,
                    tracks=tracks,
                    fields=fields,
                    R_values=R_values,
                    particle_type="stopping_proton",
                    n_nodes=stop_p_nodes,
                    y_clip=stop_p_y_clip,
                )
                
                tg_nodes, tg_y_clip = _get_particle_quadrature_params(
                    "throughgoing_muon", global_nodes, global_y_clip
                )
                tg_ne = _flow_expected_nelectrons_quadrature(
                    params=params,
                    tracks=tracks,
                    fields=fields,
                    R_values=R_values,
                    particle_type="throughgoing_muon",
                    n_nodes=tg_nodes,
                    y_clip=tg_y_clip,
                )

                direct_dEdx = tracks[:, fields.index("dEdx")][:, None]
                direct_dE = tracks[:, fields.index("dE")][:, None]
                direct_ne = _nelectrons_samples_from_dedx(
                    params,
                    tracks,
                    fields,
                    direct_dEdx,
                    dE_override=direct_dE,
                )[:, 0]

                n_electrons = direct_ne
                n_electrons = jnp.where(use_stop_muon, stop_mu_ne, n_electrons)
                n_electrons = jnp.where(use_tg_muon, tg_ne, n_electrons)
                n_electrons = jnp.where(is_proton, stop_p_ne, n_electrons)
            elif flow_expectation_mode == "sample":
                sample_n = int(getattr(params, "flow_sample_n_samples", 64)) \
                    if hasattr(params, "flow_sample_n_samples") and getattr(params, "flow_sample_n_samples") is not None else 64
                sample_seed = int(getattr(params, "flow_sample_seed", 0)) \
                    if hasattr(params, "flow_sample_seed") and getattr(params, "flow_sample_seed") is not None else 0
                sample_key = jax.random.key(sample_seed) if hasattr(jax.random, "key") else jax.random.PRNGKey(sample_seed)

                is_muon = jnp.abs(pdg_ids) == 13
                is_proton = jnp.abs(pdg_ids) == 2212
                use_tg_muon = is_muon & ((R_values > 50.0) | (R_values < 0.0))
                use_stop_muon = is_muon & (~use_tg_muon)

                stop_mu_key, stop_p_key, tg_key = jax.random.split(sample_key, 3)

                stop_mu_ne = _flow_expected_nelectrons_sample(
                    params=params,
                    tracks=tracks,
                    fields=fields,
                    R_values=R_values,
                    particle_type="stopping_muon",
                    n_samples=sample_n,
                    key=stop_mu_key,
                )
                stop_p_ne = _flow_expected_nelectrons_sample(
                    params=params,
                    tracks=tracks,
                    fields=fields,
                    R_values=R_values,
                    particle_type="stopping_proton",
                    n_samples=sample_n,
                    key=stop_p_key,
                )
                tg_ne = _flow_expected_nelectrons_sample(
                    params=params,
                    tracks=tracks,
                    fields=fields,
                    R_values=R_values,
                    particle_type="throughgoing_muon",
                    n_samples=sample_n,
                    key=tg_key,
                )

                direct_dEdx = tracks[:, fields.index("dEdx")][:, None]
                direct_dE = tracks[:, fields.index("dE")][:, None]
                direct_ne = _nelectrons_samples_from_dedx(
                    params,
                    tracks,
                    fields,
                    direct_dEdx,
                    dE_override=direct_dE,
                )[:, 0]

                n_electrons = direct_ne
                n_electrons = jnp.where(use_stop_muon, stop_mu_ne, n_electrons)
                n_electrons = jnp.where(use_tg_muon, tg_ne, n_electrons)
                n_electrons = jnp.where(is_proton, stop_p_ne, n_electrons)
            elif flow_expectation_mode == "grid":
                dEdx_samples, dEdx_weights = _dedx_density_flow(
                    n_tracks=n_tracks,
                    R_values=R_values,
                    particle_type="stopping_muon",
                )
                stop_mu_samples, stop_mu_weights = dEdx_samples, dEdx_weights
                stop_p_samples, stop_p_weights = _dedx_density_flow(
                    n_tracks=n_tracks,
                    R_values=R_values,
                    particle_type="stopping_proton",
                )
                tg_n_samples, tg_key = _throughgoing_flow_sample_config(params)
                dEdx_tg_samples, dEdx_tg_weights = _dedx_density_flow_sample(
                    R_values=R_values,
                    particle_type="throughgoing_muon",
                    n_samples=tg_n_samples,
                    key=tg_key,
                )
                max_bins = max(stop_mu_samples.shape[1], stop_p_samples.shape[1], dEdx_tg_samples.shape[1])
                stop_mu_samples = jnp.pad(stop_mu_samples, ((0, 0), (0, max_bins - stop_mu_samples.shape[1])))
                stop_mu_weights = jnp.pad(stop_mu_weights, ((0, 0), (0, max_bins - stop_mu_weights.shape[1])))
                stop_p_samples = jnp.pad(stop_p_samples, ((0, 0), (0, max_bins - stop_p_samples.shape[1])))
                stop_p_weights = jnp.pad(stop_p_weights, ((0, 0), (0, max_bins - stop_p_weights.shape[1])))
                tg_samples = jnp.pad(dEdx_tg_samples, ((0, 0), (0, max_bins - dEdx_tg_samples.shape[1])))
                tg_weights = jnp.pad(dEdx_tg_weights, ((0, 0), (0, max_bins - dEdx_tg_weights.shape[1])))
            else:
                raise ValueError(
                    f"Unknown flow_expectation_mode '{flow_expectation_mode}'. "
                    "Expected 'sample', 'grid' or 'quadrature'."
                )
        elif density_mode == "histogram":
            dEdx_samples, dEdx_weights = _dedx_density_histogram(
                n_tracks,
                R_values,
                "stopping_muon",
            )
            stop_mu_samples, stop_mu_weights = dEdx_samples, dEdx_weights
            stop_p_samples, stop_p_weights = _dedx_density_histogram(n_tracks, R_values, "stopping_proton")
            tg_samples, tg_weights = _dedx_density_histogram(
                n_tracks,
                R_values,
                "throughgoing_muon",
            )  # Throughgoing-muon profile for muons with R > 50 cm or R < 0 cm.
        else:
            raise ValueError(
                f"Unknown dedx_density_mode '{density_mode}'. Expected 'histogram' or 'flow'."
            )

        if n_electrons is None:
            is_muon = jnp.abs(pdg_ids) == 13
            is_proton = jnp.abs(pdg_ids) == 2212
            supported = is_muon | is_proton
            use_tg_muon = is_muon & ((R_values > 50.0) | (R_values < 0.0))
            use_stop_muon = is_muon & (~use_tg_muon)

            dEdx_samples = jnp.where(is_proton[:, None], stop_p_samples, dEdx_samples)
            dEdx_weights = jnp.where(is_proton[:, None], stop_p_weights, dEdx_weights)

            dEdx_samples = jnp.where(use_stop_muon[:, None], stop_mu_samples, dEdx_samples)
            dEdx_weights = jnp.where(use_stop_muon[:, None], stop_mu_weights, dEdx_weights)

            dEdx_samples = jnp.where(use_tg_muon[:, None], tg_samples, dEdx_samples)
            dEdx_weights = jnp.where(use_tg_muon[:, None], tg_weights, dEdx_weights)

            dE_samples = dEdx_samples * dx

            # For unsupported particle types, fall back to per-segment dE/dx (no density propagation).
            direct_dEdx = tracks[:, fields.index("dEdx")][:, None]
            direct_dE = tracks[:, fields.index("dE")][:, None]
            dEdx_samples = jnp.where(supported[:, None], dEdx_samples, direct_dEdx)
            dE_samples = jnp.where(supported[:, None], dE_samples, direct_dE)
            dEdx_weights = jnp.where(
                supported[:, None],
                dEdx_weights,
                jnp.ones((n_tracks, 1), dtype=direct_dEdx.dtype),
            )
    else:
        # Use the per-segment dEdx value directly, no density propagation.
        dEdx_samples = tracks[:, fields.index("dEdx")][:, None]
        dE_samples = tracks[:, fields.index("dE")][:, None]
        dEdx_weights = jnp.ones((n_tracks, 1), dtype=dEdx_samples.dtype)

    if n_electrons is None:
        n_electrons_samples = _nelectrons_samples_from_dedx(
            params,
            tracks,
            fields,
            dEdx_samples,
            dE_override=dE_samples,
        )
        n_electrons = jnp.sum(n_electrons_samples * dEdx_weights, axis=1)

    #TODO: n_electrons should be int, but truncation makes gradients vanish
    updated_tracks = tracks.at[:, fields.index("n_electrons")].set(n_electrons)
    return updated_tracks