"""
Module to implement the quenching of the ionized electrons
through the detector
"""

import jax
import jax.numpy as jnp
from jax import jit
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

    # Reuse histogram dE/dx support (bin centers) to keep integration behavior identical.
    hist = get_dedx_density_data(particle_type=particle_type)
    dedx_edges = jnp.asarray(hist["dedx_edges"])
    dedx_centers = 0.5 * (dedx_edges[:-1] + dedx_edges[1:])
    n_dedx_bins = dedx_centers.shape[0]

    R_mean = jnp.asarray(float(norm["R_mean"]), dtype=jnp.float32)
    R_std = jnp.asarray(float(norm["R_std"]), dtype=jnp.float32)
    y_mean = jnp.asarray(float(norm["dEdx_log_mean"]), dtype=jnp.float32)
    y_std = jnp.asarray(float(norm["dEdx_log_std"]), dtype=jnp.float32)
    eps = jnp.asarray(float(norm.get("dEdx_eps", 1e-8)), dtype=jnp.float32)

    # Normalize condition and dE/dx support to flow space.
    R_norm = (R_eff - R_mean) / (R_std + 1e-12)  # (n_tracks,)
    dEdx_samples = jnp.broadcast_to(dedx_centers[None, :], (n_tracks, n_dedx_bins))
    log_dedx = jnp.log(dEdx_samples + eps)
    y_norm = (log_dedx - y_mean) / (y_std + 1e-12)

    # Evaluate flow log_prob for all (track, support_bin) pairs.
    cond_flat = jnp.repeat(R_norm, n_dedx_bins)[:, None]
    y_flat = y_norm.reshape(-1, 1)
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
    if use_density:
        if "range" not in fields:
            raise ValueError("quench requires a 'range' entry in fields for dE/dx density lookup.")
        if "pdg_id" not in fields:
            raise ValueError("quench requires a 'pdg_id' entry in fields for particle identification.")

        R_values = tracks[:, fields.index("range")]
        pdg_ids = tracks[:, fields.index("pdg_id")]
        density_mode = getattr(params, "dedx_density_mode", "histogram")

        if density_mode == "flow":
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
            tg_samples, tg_weights = _dedx_density_histogram(
                n_tracks,
                R_values,
                "throughgoing_muon",
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

    if params.recombination_mode == RecombinationMode.BOX:
        recomb_samples = box_model(dEdx_samples, params.eField, params.lArDensity, params.alpha, params.beta)
    elif params.recombination_mode == RecombinationMode.BIRKS:
        recomb_samples = birks_model(dEdx_samples, params.eField, params.lArDensity, params.Ab, params.kb)
    elif params.recombination_mode == RecombinationMode.ELLIPSOID:
        # Requires geometry fields (z_start, z_end, dx) to compute the track angle via cos(phi)
        cosphi = jnp.abs(tracks[:, fields.index("z_end")] - tracks[:, fields.index("z_start")]) / (tracks[:, fields.index("dx")] + 1e-10)
        recomb_samples = ellipsoid_box_model(
            dEdx_samples,
            cosphi[:, None],
            params.eField, 
            params.lArDensity, 
            params.alpha, 
            params.beta, 
            params.R_param
        )
    else:
        raise ValueError(
            f"Invalid recombination mode {params.recombination_mode}: "
            "must be RecombinationMode.BOX, RecombinationMode.BIRKS, or RecombinationMode.ELLIPSOID"
        )

    n_electrons_samples = get_nelectrons(dE_samples, recomb_samples, params.MeVToElectrons)
    n_electrons = jnp.sum(n_electrons_samples * dEdx_weights, axis=1)

    #TODO: n_electrons should be int, but truncation makes gradients vanish
    updated_tracks = tracks.at[:, fields.index("n_electrons")].set(n_electrons)
    return updated_tracks