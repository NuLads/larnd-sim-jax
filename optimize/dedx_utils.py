"""
Per-segment dEdx fitting utilities.

Provides the Student-t NLL prior used to regularise per-segment dEdx parameters
during joint fitting, plus default constants for the LAr minimum-ionising-particle
dEdx distribution.
"""

import os
import numpy as np
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


_SPLIT_T_CONSTS = {}


def _is_py_scalar(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _split_t_consts(nu_L, nu_R, loc, scale_L, scale_R, dtype):
    """x-independent part of split_t_logpdf_jax, computed ONCE per parameter set with exactly the ops the function
    (and jax.scipy.stats.t.logpdf inside it) runs on every call (opt/loss, 22 Oct)."""
    import jax
    from jax import lax
    from jax._src.lax.lax import _const as _lax_const
    from jax._src.numpy.util import promote_args_inexact
    key = (float(nu_L), float(nu_R), float(loc), float(scale_L), float(scale_R), str(dtype))
    hit = _SPLIT_T_CONSTS.get(key)
    if hit is not None:
        return hit
    with jax.ensure_compile_time_eval():
        def log_c(nu):
            return (jax.scipy.special.gammaln((nu + 1.0) / 2.0)
                    - jax.scipy.special.gammaln(nu / 2.0)
                    - 0.5 * jnp.log(nu * jnp.pi))
        log_c_L = log_c(nu_L)
        log_c_R = log_c(nu_R)
        Z_L = 0.5 * scale_L * jnp.exp(-log_c_L)
        Z_R = 0.5 * scale_R * jnp.exp(-log_c_R)
        log_Z = jnp.log(Z_L + Z_R)
        xd = jnp.zeros((1,), dtype)      # stands for the strong-typed (N,) x: same promotion of df/loc/scale

        def t_consts(df):   # jax.scipy.stats.t.logpdf(x, df, loc=0.0, scale=1.0): its x-independent lines
            x_, df_, loc_, scale_ = promote_args_inexact("t.logpdf", xd, df, 0.0, 1.0)
            two = _lax_const(x_, 2)
            df_over_two = lax.div(df_, two)
            df_plus_one_over_two = lax.add(df_over_two, _lax_const(x_, 0.5))
            normalize_term_const = lax.mul(lax.mul(scale_, scale_), _lax_const(x_, np.pi))
            normalize_term_tmp = lax.div(lax.log(lax.mul(normalize_term_const, df_)), two)
            normalize_term = lax.sub(lax.add(lax.lgamma(df_over_two), normalize_term_tmp),
                                     lax.lgamma(df_plus_one_over_two))
            return df_, df_plus_one_over_two, normalize_term
        hit = (log_c_L, log_c_R, log_Z, t_consts(nu_L), t_consts(nu_R))
    _SPLIT_T_CONSTS[key] = hit
    return hit


def _split_t_logpdf_fast(x, nu_L, nu_R, loc, scale_L, scale_R):
    """split_t_logpdf_jax with the constants cached: same lax ops on the x-dependent path. The (x - 0) / 1 of
    t.logpdf(z, df, loc=0, scale=1) is omitted: subtracting +0.0 and dividing by 1.0 are exact identities in IEEE
    arithmetic (values and cotangents), so the result is bitwise identical."""
    from jax import lax
    log_c_L, log_c_R, log_Z, (df_L, dp1_L, nt_L), (df_R, dp1_R, nt_R) = _split_t_consts(
        nu_L, nu_R, loc, scale_L, scale_R, x.dtype)
    z_L = (x - loc) / scale_L
    z_R = (x - loc) / scale_R
    # t.logpdf: neg(add(normalize_term, mul(df_plus_one_over_two, log1p(div(mul(sx, sx), df)))))
    log_K_L = lax.neg(lax.add(nt_L, lax.mul(dp1_L, lax.log1p(lax.div(lax.mul(z_L, z_L), df_L))))) - log_c_L
    log_K_R = lax.neg(lax.add(nt_R, lax.mul(dp1_R, lax.log1p(lax.div(lax.mul(z_R, z_R), df_R))))) - log_c_R
    log_K = jnp.where(x < loc, log_K_L, log_K_R)
    return log_K - log_Z


def _split_t_fast_ok(x, *params):
    if os.environ.get('LARND_SPLIT_T_FAST', '1') != '1':
        return False
    if not all(_is_py_scalar(p) for p in params):
        return False
    if not (hasattr(x, 'dtype') and x.dtype == jnp.float32 and getattr(x, 'ndim', 0) == 1):
        return False
    try:                                # JAX-internal API (0.6.x): on any surprise use the original code
        from jax._src import core as _core
        from jax._src.interpreters import partial_eval as _pe
        from jax._src.lax.lax import _const  # noqa: F401  (used by _split_t_consts)
        from jax._src.numpy.util import promote_args_inexact  # noqa: F401
        tr = _core.trace_ctx.trace      # not under jit/make_jaxpr: there XLA folds the constants itself
        for _ in range(64):
            if tr is None or tr is _core.eval_trace:
                return True
            if isinstance(tr, _pe.DynamicJaxprTrace):
                return False
            tr = getattr(tr, 'parent_trace', None)
    except Exception:
        pass
    return False


def split_t_logpdf_jax(x, nu_L, nu_R, loc, scale_L, scale_R):
    """JAX-native Split (bifurcated) Student-t log-pdf.

    The two halves share the same peak height at `loc` and are
    normalized so the total integral is 1.

    opt/loss (22 Oct): eager calls with Python-scalar parameters (the production prior) take a bitwise-identical
    fast path that caches the x-independent constants (LARND_SPLIT_T_FAST=0 disables it).
    """
    import jax
    import jax.scipy.stats as jstats
    if _split_t_fast_ok(x, nu_L, nu_R, loc, scale_L, scale_R):
        return _split_t_logpdf_fast(x, nu_L, nu_R, loc, scale_L, scale_R)

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
