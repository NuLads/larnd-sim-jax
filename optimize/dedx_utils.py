"""
Per-segment dEdx fitting utilities.

Provides the Student-t NLL prior used to regularise per-segment dEdx parameters
during joint fitting, plus default constants for the LAr minimum-ionising-particle
dEdx distribution.
"""

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Default Student-t prior parameters for dEdx in LAr  [MeV/cm]
# These were fitted from the minimum-ionising-particle dEdx distribution
# using scipy.stats.t.fit on simulated tracks.  Override via the
# dedx_student_{nu,loc,scale} arguments to GradientDescentFitter.
# ---------------------------------------------------------------------------
DEDX_STUDENT_NU    = jnp.array(2.715,  dtype=jnp.float32)   # degrees of freedom
DEDX_STUDENT_LOC   = jnp.array(1.864,  dtype=jnp.float32)   # location (MeV/cm)
DEDX_STUDENT_SCALE = jnp.array(0.111,  dtype=jnp.float32)   # scale     (MeV/cm)


def student_t_nll(x, df, loc, scale, weights=None):
    """Negative log-likelihood of the Student-t distribution.

    The constant log-gamma terms are omitted because they do not affect the
    gradient with respect to ``x``.

    Parameters
    ----------
    x       : jax array  — samples (e.g. per-step fitted dEdx values)
    df      : scalar     — degrees of freedom  (ν > 0)
    loc     : scalar     — location parameter
    scale   : scalar     — scale parameter     (σ > 0)
    weights : jax array  — weights for each sample (e.g. step length dx). 
                           If None, performs an unweighted mean.

    Returns
    -------
    Scalar NLL. If weights are provided, returns the weighted average.
    """
    z   = (x - loc) / scale
    nll = jnp.log(scale) + 0.5 * (df + 1.0) * jnp.log1p(z * z / df)
    
    if weights is not None:
        # Mask out invalid segments (weight <= 0)
        valid_mask = (weights > 0)
        return jnp.sum(nll * weights * valid_mask)
    else:
        return jnp.sum(nll)


# ---------------------------------------------------------------------------
# Default Split Student-t prior parameters for dEdx in LAr [MeV/cm]
# Fitted from simulated MIP tracks (see analyze_dedx.ipynb / dedx_proto.ipynb)
# ---------------------------------------------------------------------------
DEDX_SPLIT_NU_L    = jnp.array(4.785,  dtype=jnp.float32)
DEDX_SPLIT_SCALE_L = jnp.array(0.1204, dtype=jnp.float32)
DEDX_SPLIT_NU_R    = jnp.array(2.073,  dtype=jnp.float32)
DEDX_SPLIT_SCALE_R = jnp.array(0.1058, dtype=jnp.float32)
DEDX_SPLIT_LOC     = jnp.array(1.874,  dtype=jnp.float32)


def split_t_logpdf_jax(x, nu_L, nu_R, loc, scale_L, scale_R):
    """JAX-native Split (bifurcated) Student-t log-pdf.

    The two halves share the same peak height at `loc` and are
    normalized so the total integral is 1.
    """
    import jax
    import jax.scipy.stats as jstats

    def log_c(nu):
        return (jax.scipy.special.gammaln((nu + 1.0) / 2.0)
                - jax.scipy.special.gammaln(nu / 2.0)
                - 0.5 * jnp.log(nu * jnp.pi))

    log_c_L = log_c(nu_L)
    log_c_R = log_c(nu_R)

    z_L = (x - loc) / scale_L
    z_R = (x - loc) / scale_R

    log_K_L = jstats.t.logpdf(z_L, df=nu_L, loc=0.0, scale=1.0) - log_c_L
    log_K_R = jstats.t.logpdf(z_R, df=nu_R, loc=0.0, scale=1.0) - log_c_R

    Z_L = 0.5 * scale_L * jnp.exp(-log_c_L)
    Z_R = 0.5 * scale_R * jnp.exp(-log_c_R)
    log_Z = jnp.log(Z_L + Z_R)

    log_K = jnp.where(x < loc, log_K_L, log_K_R)
    return log_K - log_Z


def split_student_t_nll(x, nu_L, nu_R, loc, scale_L, scale_R, weights=None):
    """Negative log-likelihood of the Split Student-t distribution."""
    logpdf = split_t_logpdf_jax(x, nu_L, nu_R, loc, scale_L, scale_R)
    if weights is not None:
        valid_mask = (weights > 0)
        return -jnp.sum(logpdf * weights * valid_mask)
    else:
        return -jnp.sum(logpdf)
