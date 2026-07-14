import os, sys

larndsim_dir=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
src_dir = os.path.join(larndsim_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if larndsim_dir not in sys.path:
    sys.path.insert(0, larndsim_dir)
import shutil
import pickle
import numpy as np
from .ranges import ranges
from scipy.optimize import minimize
from larndsim.sim_jax import (
    get_size_history, simulate_drift_new, simulate_signals, simulate_probabilistic,
    pad_to_closest_multiple, get_roi_counts, simulate_wfs
)
from larndsim.losses_jax import mse_adc, mse_time, mse_time_adc, chamfer_3d, sdtw_adc, sdtw_time, sdtw_time_adc, adc2charge, nll_loss, llhd_loss #, sinkhorn_loss
from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from larndsim.detsim_jax import validate_event_ids_for_packing, validate_local_event_ids
from larndsim.softdtw_jax import SoftDTW
from jax.flatten_util import ravel_pytree
import logging
import optax
import jax
import jax.numpy as jnp
from time import time
from jax import value_and_grad, grad
import iminuit

from tqdm import tqdm

from .strategies import LUTSimulation, LUTProbabilisticSimulation, LUTProbabilisticSamplingSimulation, ParametrizedSimulation, GenericLossStrategy
from .dedx_utils import student_t_nll, DEDX_STUDENT_NU, DEDX_STUDENT_LOC, DEDX_STUDENT_SCALE, split_student_t_nll, DEDX_SPLIT_NU_L, DEDX_SPLIT_NU_R, DEDX_SPLIT_SCALE_L, DEDX_SPLIT_SCALE_R, DEDX_SPLIT_LOC

from ctypes import cdll
# libcudart = cdll.LoadLibrary('libcudart.so')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ── Chain kinematics helpers (used by fit_chain_positions mode) ───────────────

def _highland_sigma_mcs(step_cm, momentum_GeV=3.0):
    r = step_cm / 14.0  # X0_LAR_CM = 14.0 cm for LAr
    return float((0.0136 / momentum_GeV) * np.sqrt(r) * (1.0 + 0.038 * np.log(r)))

def _build_local_basis_chain(u):
    cy = jnp.linalg.cross(u, jnp.array([0., 1., 0.]))
    cx = jnp.linalg.cross(u, jnp.array([1., 0., 0.]))
    c  = jnp.where(jnp.linalg.norm(cy) > 1e-4, cy, cx)
    v  = c / jnp.linalg.norm(c)
    return v, jnp.linalg.cross(u, v)

def _deflection_chain_forward(x0, theta0, phi0, dtheta, dphi, step_len):
    """Propagate a deflection chain from entry x0 with n_chain segments.
    LEGACY: used only for backward-compat with old checkpoints.
    New code uses _absolute_chain_forward.
    """
    u0 = jnp.array([jnp.sin(theta0) * jnp.cos(phi0),
                    jnp.sin(theta0) * jnp.sin(phi0),
                    jnp.cos(theta0)])
    u0 = u0 / jnp.linalg.norm(u0)
    def _step(carry, ang):
        pos, u = carry
        v, w = _build_local_basis_chain(u)
        ct, st = jnp.cos(ang[0]), jnp.sin(ang[0])
        cp, sp = jnp.cos(ang[1]), jnp.sin(ang[1])
        u2 = ct * u + st * cp * v + st * sp * w
        u2 = u2 / jnp.linalg.norm(u2)
        return (pos + step_len * u, u2), (pos + step_len * u, u2)
    (pl, ul), (pm, um) = jax.lax.scan(_step, (x0, u0), jnp.stack([dtheta, dphi], axis=1))
    positions  = jnp.concatenate([x0[None], pm, (pl + step_len * ul)[None]], axis=0)
    directions = jnp.concatenate([u0[None], um], axis=0)
    return positions, directions


def _absolute_chain_forward(x0, thetas, phis, step_len):
    """Propagate a chain where each segment has an independent absolute direction.

    Angles layout: angles[:n_chain] = thetas, angles[n_chain:] = phis.
    Changing theta_k shifts all downstream nodes by the same fixed vector (no lever-arm growth).
    MCS prior is applied on jnp.diff(thetas) and jnp.diff(phis).

    Returns:
        positions:  (n_chain+1, 3) — node positions including start
        directions: (n_chain, 3)   — unit direction of each segment
    """
    def _step(carry, angle_pair):
        pos = carry
        theta_k, phi_k = angle_pair[0], angle_pair[1]
        u_k = jnp.array([jnp.sin(theta_k) * jnp.cos(phi_k),
                         jnp.sin(theta_k) * jnp.sin(phi_k),
                         jnp.cos(theta_k)])
        u_k = u_k / jnp.linalg.norm(u_k)
        new_pos = pos + step_len * u_k
        return new_pos, (new_pos, u_k)

    _, (nodes_rest, dirs) = jax.lax.scan(
        _step, x0, jnp.stack([thetas, phis], axis=1))
    positions = jnp.concatenate([x0[None], nodes_rest], axis=0)
    return positions, dirs


def _build_chain_meta(ctxs):
    """Precompute flat/padded metadata for the vectorized chain warp.

    Replaces the per-track Python loop with batched operations. Returns ``None``
    when there are no chain contexts. All per-segment quantities are concatenated
    into flat arrays; per-track quantities are stacked and padded to ``max_nc``.
    """
    if not ctxs:
        return None
    ncs = [int(c.n_chain) for c in ctxs]
    T = len(ctxs)
    max_nc = max(ncs)
    seg_rows, seg_track, seg_am, seg_as, seg_ae = [], [], [], [], []
    for ti, c in enumerate(ctxs):
        idxs = np.asarray(c.idxs)
        seg_rows.append(idxs)
        seg_track.append(np.full(idxs.shape[0], ti, dtype=np.int32))
        seg_am.append(np.asarray(c.alpha_mid))
        seg_as.append(np.asarray(c.alpha_start))
        seg_ae.append(np.asarray(c.alpha_end))
    return {
        'T': T, 'max_nc': max_nc,
        'ncs_np': np.asarray(ncs, np.int32),                       # static, python-side slicing
        'ncs': jnp.asarray(np.asarray(ncs, np.int32)),
        'x0': jnp.asarray(np.stack([np.asarray(c.x0_fixed) for c in ctxs]).astype(np.float32)),
        'step': jnp.asarray(np.asarray([float(c.step_len) for c in ctxs], np.float32)),
        'sigma': jnp.asarray(np.asarray([float(c.sigma_mcs) for c in ctxs], np.float32)),
        'seg_rows': jnp.asarray(np.concatenate(seg_rows).astype(np.int32)),
        'seg_track': jnp.asarray(np.concatenate(seg_track)),
        'seg_am': jnp.asarray(np.concatenate(seg_am).astype(np.float32)),
        'seg_as': jnp.asarray(np.concatenate(seg_as).astype(np.float32)),
        'seg_ae': jnp.asarray(np.concatenate(seg_ae).astype(np.float32)),
    }


def _chain_warp_vectorized(t, chain_angles_pt, meta, col_map):
    """Vectorized replacement for the per-track chain-warp loop.

    Computes all node positions with a single ``cumsum`` (absolute chain, no
    lever-arm growth) and writes every segment field with one batched scatter
    per column. Returns ``(t_warped, thetas, phis)`` with padded ``(T, max_nc)``
    angle matrices for reuse by the MCS prior.
    """
    T, max_nc = meta['T'], meta['max_nc']
    ncs, ncs_np = meta['ncs'], meta['ncs_np']
    th_rows, ph_rows = [], []
    for ti in range(T):
        a = chain_angles_pt[ti]
        nc = int(ncs_np[ti])
        th, ph = a[:nc], a[nc:]
        pad = max_nc - nc
        if pad > 0:
            th = jnp.concatenate([th, jnp.zeros(pad, th.dtype)])
            ph = jnp.concatenate([ph, jnp.zeros(pad, ph.dtype)])
        th_rows.append(th)
        ph_rows.append(ph)
    thetas = jnp.stack(th_rows)                       # (T, max_nc)
    phis = jnp.stack(ph_rows)
    st, ct = jnp.sin(thetas), jnp.cos(thetas)
    u = jnp.stack([st * jnp.cos(phis), st * jnp.sin(phis), ct], axis=-1)
    u = u / jnp.linalg.norm(u, axis=-1, keepdims=True)
    steps = meta['step'][:, None, None] * u
    nodes = jnp.concatenate(
        [meta['x0'][:, None, :], meta['x0'][:, None, :] + jnp.cumsum(steps, axis=1)],
        axis=1)                                        # (T, max_nc+1, 3)
    seg_track = meta['seg_track']
    nc_s = ncs[seg_track]

    def _interp(alpha):
        k = jnp.clip(jnp.floor(alpha * nc_s).astype(jnp.int32), 0, nc_s - 1)
        frac = alpha * nc_s - k
        p0 = nodes[seg_track, k]
        p1 = nodes[seg_track, k + 1]
        return p0 + frac[:, None] * (p1 - p0)

    nm = _interp(meta['seg_am'])
    ns = _interp(meta['seg_as'])
    ne = _interp(meta['seg_ae'])
    ndx = jnp.sqrt(jnp.sum((ne - ns) ** 2, axis=-1) + 1e-10)
    r = meta['seg_rows']
    t = t.at[r, col_map['x']].set(nm[:, 0])
    t = t.at[r, col_map['y']].set(nm[:, 1])
    t = t.at[r, col_map['z']].set(nm[:, 2])
    t = t.at[r, col_map['xs']].set(ns[:, 0])
    t = t.at[r, col_map['ys']].set(ns[:, 1])
    t = t.at[r, col_map['zs']].set(ns[:, 2])
    t = t.at[r, col_map['xe']].set(ne[:, 0])
    t = t.at[r, col_map['ye']].set(ne[:, 1])
    t = t.at[r, col_map['ze']].set(ne[:, 2])
    t = t.at[r, col_map['dx']].set(ndx)
    return t, thetas, phis


def _chain_mcs_vectorized(thetas, phis, meta, mcs_weight):
    """Vectorized Gaussian MCS prior on the padded chain angles.

    Masks out the diffs that cross the padding boundary so only within-track
    consecutive deflections contribute (matches the per-track loop exactly).
    """
    T, max_nc, ncs = meta['T'], meta['max_nc'], meta['ncs']
    dt = jnp.diff(thetas, axis=1)                      # (T, max_nc-1)
    dp = jnp.diff(phis, axis=1)
    j = jnp.arange(max_nc - 1)
    mask = (j[None, :] < (ncs[:, None] - 1)).astype(thetas.dtype)
    mcs = jnp.sum(mask * (dt ** 2 + dp ** 2) / (2.0 * (meta['sigma'][:, None] ** 2)))
    return mcs_weight * mcs / max(1, T)


# ── Spline (low-order transverse-displacement) chain basis ───────────────────
# Alternative to the tangent-angle chain: the track shape is a smooth transverse
# displacement field d(alpha) along a fixed nominal (straight linear-guess) axis,
# expanded in K sine modes  d(alpha) = sum_m beta_m sin(m*pi*alpha). This is the
# Karhunen-Loeve basis of the MCS (Brownian-bridge) process: endpoints are anchored
# (sin=0 at alpha=0,1), the number of parameters (2K) is fixed regardless of track
# length, and the data-term Jacobian d x_k / d beta = B(alpha_k) is a small, well
# conditioned design matrix (no cumsum lever-arm -> conditioning independent of length).
# dx / dE are held at nominal (the transverse warp changes WHERE charge lands, not how
# much), matching the fixed-step-length invariant the angle basis had.

def _dir_from_angles(theta, phi):
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], dtype=np.float32)


def _transverse_frame(u0):
    """Two orthonormal vectors spanning the plane perpendicular to unit vector u0."""
    ref = np.array([0., 0., 1.], np.float32)
    if abs(float(np.dot(u0, ref))) > 0.9:
        ref = np.array([1., 0., 0.], np.float32)
    e1 = np.cross(u0, ref); e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(u0, e1);  e2 /= (np.linalg.norm(e2) + 1e-12)
    return e1.astype(np.float32), e2.astype(np.float32)


def _spline_K(total_len, knot_cm):
    return int(np.clip(round(total_len / knot_cm), 3, 12))


def _build_chain_meta_spline(ctxs, knot_cm):
    """Spline analogue of _build_chain_meta: precompute nominal nodes, the sine design
    matrix at node alphas, and the transverse frame per track (all static)."""
    if not ctxs:
        return None
    T = len(ctxs)
    ncs = [int(c.n_chain) for c in ctxs]
    Ks = [_spline_K(c.total_len, knot_cm) for c in ctxs]
    max_nc, max_K = max(ncs), max(Ks)
    nominal_p = np.zeros((T, max_nc + 1, 3), np.float32)
    B_p = np.zeros((T, max_nc + 1, max_K), np.float32)
    e1s, e2s = [], []
    seg_rows, seg_track, seg_am, seg_as, seg_ae = [], [], [], [], []
    for ti, c in enumerate(ctxs):
        nc, K = ncs[ti], Ks[ti]
        u0 = _dir_from_angles(c.theta0_i, c.phi0_i)
        e1, e2 = _transverse_frame(u0)
        e1s.append(e1); e2s.append(e2)
        ks = np.arange(nc + 1)
        alpha = ks / nc
        nominal_p[ti, :nc + 1] = c.x0_fixed[None, :] + (ks[:, None] * c.step_len) * u0[None, :]
        B_p[ti, :nc + 1, :K] = np.sin(np.outer(alpha, np.pi * np.arange(1, K + 1))).astype(np.float32)
        idxs = np.asarray(c.idxs)
        seg_rows.append(idxs); seg_track.append(np.full(idxs.shape[0], ti, np.int32))
        seg_am.append(np.asarray(c.alpha_mid)); seg_as.append(np.asarray(c.alpha_start)); seg_ae.append(np.asarray(c.alpha_end))
    return {
        'basis': 'spline', 'T': T, 'max_nc': max_nc, 'max_K': max_K,
        'ncs_np': np.asarray(ncs, np.int32), 'ncs': jnp.asarray(np.asarray(ncs, np.int32)),
        'Ks_np': np.asarray(Ks, np.int32),
        'nominal': jnp.asarray(nominal_p), 'Bmat': jnp.asarray(B_p),
        'e1': jnp.asarray(np.stack(e1s)), 'e2': jnp.asarray(np.stack(e2s)),
        'step': jnp.asarray(np.asarray([float(c.step_len) for c in ctxs], np.float32)),
        'sigma': jnp.asarray(np.asarray([float(c.sigma_mcs) for c in ctxs], np.float32)),
        'seg_rows': jnp.asarray(np.concatenate(seg_rows).astype(np.int32)),
        'seg_track': jnp.asarray(np.concatenate(seg_track)),
        'seg_am': jnp.asarray(np.concatenate(seg_am).astype(np.float32)),
        'seg_as': jnp.asarray(np.concatenate(seg_as).astype(np.float32)),
        'seg_ae': jnp.asarray(np.concatenate(seg_ae).astype(np.float32)),
    }


def _chain_warp_spline(t, coeffs_pt, meta, col_map):
    """Warp via smooth transverse displacement. coeffs_pt[ti] is a flat (2K,) vector
    [beta_e1 (K), beta_e2 (K)]. dx is FROZEN (not written) -> energy decoupled."""
    T, max_nc, max_K = meta['T'], meta['max_nc'], meta['max_K']
    Ks_np = meta['Ks_np']
    b1_rows, b2_rows = [], []
    for ti in range(T):
        c = coeffs_pt[ti]
        K = int(Ks_np[ti])
        b1, b2 = c[:K], c[K:2 * K]
        pad = max_K - K
        if pad > 0:
            b1 = jnp.concatenate([b1, jnp.zeros(pad, c.dtype)])
            b2 = jnp.concatenate([b2, jnp.zeros(pad, c.dtype)])
        b1_rows.append(b1); b2_rows.append(b2)
    B1, B2 = jnp.stack(b1_rows), jnp.stack(b2_rows)          # (T, max_K)
    d1 = jnp.einsum('tnk,tk->tn', meta['Bmat'], B1)          # (T, max_nc+1) displacement along e1
    d2 = jnp.einsum('tnk,tk->tn', meta['Bmat'], B2)
    nodes = (meta['nominal']
             + d1[..., None] * meta['e1'][:, None, :]
             + d2[..., None] * meta['e2'][:, None, :])       # (T, max_nc+1, 3)
    seg_track = meta['seg_track']
    nc_s = meta['ncs'][seg_track]

    def _interp(alpha):
        k = jnp.clip(jnp.floor(alpha * nc_s).astype(jnp.int32), 0, nc_s - 1)
        frac = alpha * nc_s - k
        p0 = nodes[seg_track, k]
        p1 = nodes[seg_track, k + 1]
        return p0 + frac[:, None] * (p1 - p0)

    nm, ns, ne = _interp(meta['seg_am']), _interp(meta['seg_as']), _interp(meta['seg_ae'])
    r = meta['seg_rows']
    for col, arr in [('x', nm[:, 0]), ('y', nm[:, 1]), ('z', nm[:, 2]),
                     ('xs', ns[:, 0]), ('ys', ns[:, 1]), ('zs', ns[:, 2]),
                     ('xe', ne[:, 0]), ('ye', ne[:, 1]), ('ze', ne[:, 2])]:
        t = t.at[r, col_map[col]].set(arr)
    disp = jnp.stack([d1, d2], axis=-1)                      # (T, max_nc+1, 2) for the MCS prior
    return t, disp


def _chain_mcs_spline(disp, meta, mcs_weight):
    """MCS prior in displacement basis: penalize the discrete curvature (second difference)
    of the transverse displacement, = the per-node deflection angle ~ |Delta^2 d| / step_len."""
    T, max_nc, ncs = meta['T'], meta['max_nc'], meta['ncs']
    d2 = disp[:, 2:, :] - 2.0 * disp[:, 1:-1, :] + disp[:, :-2, :]   # (T, max_nc-1, 2)
    j = jnp.arange(max_nc - 1)
    mask = (j[None, :] < (ncs[:, None] - 1)).astype(disp.dtype)
    ang2 = jnp.sum(d2 ** 2, axis=-1) / (meta['step'][:, None] ** 2)
    mcs = jnp.sum(mask * ang2 / (2.0 * (meta['sigma'][:, None] ** 2)))
    return mcs_weight * mcs / max(1, T)


class _ChainTrackCtx:
    """Per-track deflection chain context (precomputed at batch init)."""
    __slots__ = ('track_id', 'idxs', 'n_fine', 'n_chain', 'step_len', 'total_len',
                 'x0_fixed', 'theta0_i', 'phi0_i', 'sigma_mcs',
                 'alpha_mid', 'alpha_start', 'alpha_end')

    def __init__(self, track_id, idxs, n_fine, n_chain, step_len, total_len,
                 x0_fixed, theta0_i, phi0_i, sigma_mcs, alpha_mid, alpha_start, alpha_end):
        self.track_id   = track_id
        self.idxs       = idxs
        self.n_fine     = n_fine
        self.n_chain    = n_chain
        self.step_len   = step_len
        self.total_len  = total_len
        self.x0_fixed   = x0_fixed
        self.theta0_i   = theta0_i
        self.phi0_i     = phi0_i
        self.sigma_mcs  = sigma_mcs
        self.alpha_mid  = alpha_mid
        self.alpha_start = alpha_start
        self.alpha_end  = alpha_end

# ─────────────────────────────────────────────────────────────────────────────

def value_and_grad_fwd(f, argnums=0, has_aux=False):
    """
    Computes the primal value and the gradient using forward-mode autodiff,
    with full support for auxiliary data.
    """
    def wrapper(*args, **kwargs):
        out = f(*args, **kwargs)
        if has_aux:
            y, aux = out
            # Return 'y' for jacfwd to differentiate, and (y, aux) to pass through
            return y, (y, aux)
        else:
            y = out
            # Return 'y' for jacfwd, and a copy of 'y' to pass through
            return y, y

    # We use jacfwd with has_aux=True so it captures our injected auxiliary data
    fwd_fn = jax.jacfwd(wrapper, argnums=argnums, has_aux=True)

    def val_and_grad_fn(*args, **kwargs):
        grad, aux_out = fwd_fn(*args, **kwargs)
        if has_aux:
            y, aux = aux_out
            return (y, aux), grad
        else:
            y = aux_out
            return y, grad
            
    return val_and_grad_fn

def serialize_value(v):
    if hasattr(v, 'shape'):
        # Check if it's a scalar (0-dimensional) or multi-dimensional
        if len(v.shape) == 0 or (len(v.shape) == 1 and v.shape[0] == 1):
            # Scalar JAX array -> Python float
            return float(v)
        else:
            # Multi-dimensional JAX array -> numpy array
            return np.array(v)
    elif hasattr(v, 'item'):
        # Fallback for other array-like objects with item()
        return float(v)
    else:
        # Already Python type
        return v

def serialize_param_state(params, relevant):
    """Serialize selected fields from params container into plain Python values."""
    return {par: serialize_value(getattr(params, par)) for par in relevant}

def restore_param_state(ref_params, state_dict):
    """Restore params container from serialized field dictionary."""
    return ref_params.replace(**{key: float(val) for key, val in state_dict.items()})


def normalize_param(param_val, param_name, scheme="divide", undo_norm=False):
    if scheme == "divide":
        if undo_norm:
            out_val = param_val * ranges[param_name]['nom']
        else:
            out_val = param_val / ranges[param_name]['nom']

        return out_val
    elif scheme == "standard":
        sigma = (ranges[param_name]['up'] - ranges[param_name]['down']) / 2.

        if undo_norm:
            out_val = param_val*sigma**2 + ranges[param_name]['nom']
        else:
            out_val = (param_val - ranges[param_name]['nom']) / sigma**2
            
        return out_val
    elif scheme == "none":
        return param_val
    else:
        raise ValueError(f"No normalization method called {scheme}")
    
def extract_relevant_params(params, relevant):
    return {par: getattr(params, par) for par in relevant}

def format_hessian(hess):
    flatten_hessian, _ = ravel_pytree(hess)
    return flatten_hessian.tolist()

def remove_noise_from_params(params):
    noise_params = ('RESET_NOISE_CHARGE', 'UNCORRELATED_NOISE_CHARGE')
    return params.replace(**{key: 0. for key in noise_params})


# log/exp mapping for positive-only parameters, with nominal value at exp(0) and no hard upper bound (can be extended to softplus if needed); affine mapping for real-valued parameters (shift_x/y/z), with nominal value at 0 and sigma = (up - down) / 2 from ranges.py
# def map_norm_to_phys(val, key):
#     """Map unconstrained norm value to physical space.

#     For positive-only parameters:
#         phys = nom * exp(val)
#         - val=0  → phys=nom  (starts at nominal)
#         - Jacobian = nom * exp(val) = phys  (uniform in log space, never zero)
#         - No upper hard bound, but physically motivated params rarely need it.
#           If hard upper bound is required, use softplus variant below.

#     For real-valued parameters (shift_x/y/z):
#         phys = nom + sigma * val
#         - val=0  → phys=nom
#         - Jacobian = sigma  (constant, no compression anywhere)
#         - sigma = (up - down) / 2 from ranges.py
#     """
#     ptype = param_type.get(key, 'positive')

#     if ptype == 'positive':
#         nom = ranges[key]['nom']
#         return nom * jnp.exp(val)

#     else:  # 'real'
#         nom = ranges[key]['nom']
#         sigma = (ranges[key]['up'] - ranges[key]['down']) / 2.0
#         return nom + sigma * val


# def map_phys_to_norm(val, key):
#     """Inverse of map_norm_to_phys — initialise norm from a known physical value."""
#     ptype = param_type.get(key, 'positive')

#     if ptype == 'positive':
#         nom = ranges[key]['nom']
#         # Guard against log(0) if val is at the min boundary
#         val = jnp.maximum(val, jnp.finfo(jnp.float32).tiny)
#         return jnp.log(val / nom)

#     else:  # 'real'
#         nom = ranges[key]['nom']
#         sigma = (ranges[key]['up'] - ranges[key]['down']) / 2.0
#         return (val - nom) / sigma

# Parameter normalization mappings.
def map_norm_to_phys(val, key, scheme="sigmoid", scale=1.0):
    if scheme == "sigmoid":
        low, high = ranges[key]['min'], ranges[key]['max']
        return low + (high - low) * jax.nn.sigmoid(scale * val)

    if scheme == "exp_log":
        nom = ranges[key]['nom']
        if nom <= 0:
            raise ValueError(f"exp_log normalization requires positive nominal value for '{key}', got {nom}")
        return nom * jnp.exp(scale * val)

    raise ValueError(f"Unsupported normalization scheme in map_norm_to_phys: {scheme}")


def map_phys_to_norm(val, key, scheme="sigmoid", scale=1.0):
    if scale <= 0:
        raise ValueError(f"Normalization scale must be > 0 for scheme '{scheme}', got {scale}")

    if scheme == "sigmoid":
        low, high = ranges[key]['min'], ranges[key]['max']
        eps = jnp.finfo(jnp.float32).eps
        frac = jnp.clip((val - low) / (high - low), eps, 1.0 - eps)
        return jnp.log(frac / (1.0 - frac)) / scale

    if scheme == "exp_log":
        nom = ranges[key]['nom']
        if nom <= 0:
            raise ValueError(f"exp_log normalization requires positive nominal value for '{key}', got {nom}")
        eps = jnp.finfo(jnp.float32).tiny
        return jnp.log(jnp.maximum(val, eps) / nom) / scale

    raise ValueError(f"Unsupported normalization scheme in map_phys_to_norm: {scheme}")


def get_rotation_matrix(yaw, pitch, roll):
    cy = jnp.cos(yaw)
    sy = jnp.sin(yaw)
    cp = jnp.cos(pitch)
    sp = jnp.sin(pitch)
    cr = jnp.cos(roll)
    sr = jnp.sin(roll)
    
    Rx = jnp.array([[1.0, 0.0, 0.0],
                    [0.0, cr, -sr],
                    [0.0, sr, cr]])
    
    Ry = jnp.array([[cp, 0.0, sp],
                    [0.0, 1.0, 0.0],
                    [-sp, 0.0, cp]])
    
    Rz = jnp.array([[cy, -sy, 0.0],
                    [sy, cy, 0.0],
                    [0.0, 0.0, 1.0]])
    
    return jnp.dot(Rz, jnp.dot(Ry, Rx))

def apply_transformation(tracks, fields, translation, rotation_angles):
    yaw, pitch, roll = rotation_angles
    R = get_rotation_matrix(yaw, pitch, roll)
    
    x_idx = fields.index("x")
    y_idx = fields.index("y")
    z_idx = fields.index("z")
    
    xs = tracks[:, x_idx]
    ys = tracks[:, y_idx]
    zs = tracks[:, z_idx]
    
    cog_x = jnp.mean(xs)
    cog_y = jnp.mean(ys)
    cog_z = jnp.mean(zs)
    cog = jnp.array([cog_x, cog_y, cog_z])
    
    xs_idx = fields.index("x_start")
    ys_idx = fields.index("y_start")
    zs_idx = fields.index("z_start")
    
    xe_idx = fields.index("x_end")
    ye_idx = fields.index("y_end")
    ze_idx = fields.index("z_end")
    
    pts_start = jnp.stack([tracks[:, xs_idx], tracks[:, ys_idx], tracks[:, zs_idx]], axis=-1)
    pts_end = jnp.stack([tracks[:, xe_idx], tracks[:, ye_idx], tracks[:, ze_idx]], axis=-1)
    pts_mid = jnp.stack([xs, ys, zs], axis=-1)
    
    transformed_start = jnp.dot(pts_start - cog, R.T) + cog + translation
    transformed_end = jnp.dot(pts_end - cog, R.T) + cog + translation
    transformed_mid = jnp.dot(pts_mid - cog, R.T) + cog + translation
    
    t = tracks
    t = t.at[:, xs_idx].set(transformed_start[:, 0])
    t = t.at[:, ys_idx].set(transformed_start[:, 1])
    t = t.at[:, zs_idx].set(transformed_start[:, 2])
    
    t = t.at[:, xe_idx].set(transformed_end[:, 0])
    t = t.at[:, ye_idx].set(transformed_end[:, 1])
    t = t.at[:, ze_idx].set(transformed_end[:, 2])
    
    t = t.at[:, x_idx].set(transformed_mid[:, 0])
    t = t.at[:, y_idx].set(transformed_mid[:, 1])
    t = t.at[:, z_idx].set(transformed_mid[:, 2])
    
    # Recompute dx
    dx_idx = fields.index("dx")
    dx_new = jnp.sqrt(
        (transformed_start[:, 0] - transformed_end[:, 0])**2 +
        (transformed_start[:, 1] - transformed_end[:, 1])**2 +
        (transformed_start[:, 2] - transformed_end[:, 2])**2 + 1e-10
    )
    t = t.at[:, dx_idx].set(dx_new)
    
    return t

def simulate_wfs_static(params, response_template, tracks, fields, fixed_pixels):
    main_pixels, pixels, nelectrons, t0_after_diff, long_diff, currents_idx, pIDs_neigh, currents_idx_neigh, nelectrons_neigh, t0_neigh = simulate_drift_new(params, tracks, fields)
    
    pix_renumbering_neigh = jnp.searchsorted(fixed_pixels, pIDs_neigh.ravel(), method='sort')
    mask = (pix_renumbering_neigh < fixed_pixels.size) & (fixed_pixels[pix_renumbering_neigh] == pIDs_neigh.ravel())
    pix_renumbering_neigh = jnp.where(mask, pix_renumbering_neigh, 0)
    
    wfs = simulate_signals(params, fixed_pixels, pixels, t0_after_diff, response_template, nelectrons, long_diff, currents_idx, nelectrons_neigh, pix_renumbering_neigh, t0_neigh, currents_idx_neigh)
    
    return wfs[:, 1:]

class ParamFitter:
    def __init__(self, relevant_params, set_init_params, sim_track_fields, tgt_track_fields,
                 detector_props, pixel_layouts,
                 loss_fn=None, loss_fn_kw=None, readout_noise_target=True, readout_noise_guess=False, 
                 out_label="", test_name="this_test",
                 shift_no_fit=[], set_target_vals=[], set_params={}, vary_init=False, keep_in_memory=False,
                 compute_target_hessian=False, sim_seed_strategy="different",
                 target_seed=0, target_fixed_range=None,
                 adc_norm=1, match_z=True,
                 diffusion_in_current_sim=False,
                 mc_diff = False,
                 read_target=False,
                 probabilistic_sim=False,
                 probabilistic_sampling_sim=False,
                 probabilistic_sampling_target=False,
                 normalization_scheme="sigmoid",
                 normalization_scale_sigmoid=1.0,
                 normalization_scale_exp_log=1.0,
                 sz_mini_bt=1, shuffle_bt=False, shuffle_seed=42, resume_from=None,
                 fit_dedx=False, dedx_prior_weight=0.1, dedx_lr=None, dedx_start_iter=0,
                 dedx_freeze_iter=None,
                 dedx_student_nu=None, dedx_student_loc=None, dedx_student_scale=None,
                 dedx_soft_barrier_threshold=8.5,
                 dedx_soft_barrier_weight=1.0,
                 dedx_use_split_t=True,
                 dedx_student_nu_l=None,
                 dedx_student_nu_r=None,
                 dedx_student_scale_l=None,
                 dedx_student_scale_r=None,
                 dedx_mean_constraint_weight=0.0,
                 dedx_mean_constraint_target=1.887,
                 dedx_drift_profile_weight=0.0,
                 fit_track_positions=False,
                 track_max_iter=20,
                 track_alignment_freq=1,
                 fit_chain_positions=False,
                 chain_lr=3e-5,
                 chain_start_iter=0,
                 chain_update_freq=1,
                 chain_decay_rate=1.0,
                 chain_decay_start=None,
                 pos_residual_freq=1,
                 mcs_prior_weight=1.0,
                 chain_step_len=2.0,
                 chain_momentum_GeV=3.0,
                 chain_basis='angle', chain_spline_knot_cm=40.0,
                 config = {}):

        self.read_target = read_target
        self.shift_no_fit = shift_no_fit
        self.set_params = set_params
        self.detector_props = detector_props
        self.pixel_layouts = pixel_layouts
        self.resume_from = resume_from
        self.resumed = False
        self.total_iter = 0
        self.fit_track_positions = bool(fit_track_positions)
        self.track_max_iter = int(track_max_iter)
        self.track_alignment_freq = int(track_alignment_freq)
        self.fit_chain_positions = bool(fit_chain_positions)
        self._chain_lr = float(chain_lr)
        # Phase 3: geometry-block optimiser. 'adam' (default) = current per-track Adam path (unchanged).
        # 'lbfgs' = jitted L-BFGS geometry solve (calibration fixed) alternating with the calibration step.
        self._geom_optimizer = str(getattr(config, 'geom_optimizer', 'adam') or 'adam') if config is not None else 'adam'
        self._geom_lbfgs_steps = int(getattr(config, 'geom_lbfgs_steps', 10)) if config is not None else 10
        self._geom_lbfgs_unique_size = int(getattr(config, 'geom_lbfgs_unique_size', 0)) if config is not None else 0
        self._geom_vg_cache = {}
        self._chain_start_iter = int(chain_start_iter)
        self._chain_update_freq = max(1, int(chain_update_freq))
        # Exponential decay of the chain-position learning rate. The chain optimizer
        # (optax.adam) is otherwise fixed-LR, so positions never settle and sustain a
        # position<->calibration limit cycle in the joint fit. Decaying it lets them
        # converge. chain_decay_rate=1.0 => no decay (backward compatible).
        # Position-residual is a per-step DIAGNOSTIC (not used by the fit): computing it every
        # step is costly (device<->host syncs + a per-segment Python loop). Sample it every N.
        self._pos_residual_freq = max(1, int(pos_residual_freq))
        self._chain_decay_rate = float(chain_decay_rate)
        # Decay begins at chain_decay_start (defaults to when chain fitting starts).
        self._chain_decay_start = int(chain_decay_start) if chain_decay_start is not None else int(chain_start_iter)
        self._mcs_prior_weight = float(mcs_prior_weight)
        self._chain_step_len = float(chain_step_len)
        self._chain_momentum_GeV = float(chain_momentum_GeV)
        self._chain_basis = str(chain_basis)              # 'angle' | 'spline'
        self._chain_spline_knot_cm = float(chain_spline_knot_cm)
        self._batch_chain_contexts: dict = {}
        self._batch_chain_meta: dict = {}   # cached vectorized-warp metadata per batch
        self._batch_fixed_pixels = {}
        self._batch_target_pixels = {}
        self._batch_target_expected_adc = {}
        self._batch_aligned_tracks = {}
        self._batch_padded_roi = {}
        self.readout_noise_target = readout_noise_target
        self.readout_noise_guess = readout_noise_guess
        self.vary_init = vary_init
        self.normalization_scheme = normalization_scheme
        self.normalization_scale_sigmoid = float(normalization_scale_sigmoid)
        self.normalization_scale_exp_log = float(normalization_scale_exp_log)
        self.target_seed = target_seed
        self.target_fixed_range = target_fixed_range
        self.diffusion_in_current_sim = diffusion_in_current_sim
        self.mc_diff = mc_diff

        self.out_label = out_label
        self.test_name = test_name

        self.compute_target_hessian = compute_target_hessian

        self.current_mode = config.mode
        self.electron_sampling_resolution = config.electron_sampling_resolution
        self.number_pix_neighbors = config.number_pix_neighbors
        self.signal_length = config.signal_length
        self.use_dedx_density = bool(getattr(config, "use_dedx_density", False))
        self.dedx_density_mode = getattr(config, "dedx_density_mode", "histogram")
        self.probabilistic_sim = probabilistic_sim
        self.probabilistic_sampling_sim = probabilistic_sampling_sim
        self.probabilistic_sampling_target = probabilistic_sampling_target
        if probabilistic_sim and probabilistic_sampling_sim:
            raise ValueError("probabilistic_sim and probabilistic_sampling_sim are mutually exclusive.")
        if probabilistic_sampling_target and not probabilistic_sim:
            raise ValueError("probabilistic_sampling_target requires probabilistic_sim to be set "
                             "(guess must be probabilistic so the loss can compare against sampled hits).")
        if probabilistic_sampling_target and probabilistic_sampling_sim:
            raise ValueError("probabilistic_sampling_target and probabilistic_sampling_sim are mutually exclusive.")
        self.sz_mini_bt = sz_mini_bt
        self.shuffle_bt = shuffle_bt
        self.fit_dedx = fit_dedx
        self.dedx_prior_weight = float(dedx_prior_weight)
        self.dedx_lr = dedx_lr
        self.dedx_start_iter = int(dedx_start_iter)
        self.dedx_freeze_iter = int(dedx_freeze_iter) if dedx_freeze_iter is not None else None
        self.dedx_use_split_t = bool(dedx_use_split_t)
        self.dedx_student_nu_l = jnp.array(dedx_student_nu_l, dtype=jnp.float32) if dedx_student_nu_l is not None else DEDX_SPLIT_NU_L
        self.dedx_student_nu_r = jnp.array(dedx_student_nu_r, dtype=jnp.float32) if dedx_student_nu_r is not None else DEDX_SPLIT_NU_R
        self.dedx_student_scale_l = jnp.array(dedx_student_scale_l, dtype=jnp.float32) if dedx_student_scale_l is not None else DEDX_SPLIT_SCALE_L
        self.dedx_student_scale_r = jnp.array(dedx_student_scale_r, dtype=jnp.float32) if dedx_student_scale_r is not None else DEDX_SPLIT_SCALE_R

        self.dedx_student_nu    = jnp.array(dedx_student_nu,    dtype=jnp.float32) if dedx_student_nu    is not None else DEDX_STUDENT_NU
        if dedx_student_loc is not None:
            self.dedx_student_loc = jnp.array(dedx_student_loc, dtype=jnp.float32)
        else:
            self.dedx_student_loc = DEDX_SPLIT_LOC if self.dedx_use_split_t else DEDX_STUDENT_LOC
        self.dedx_student_scale = jnp.array(dedx_student_scale, dtype=jnp.float32) if dedx_student_scale is not None else DEDX_STUDENT_SCALE
        self.dedx_soft_barrier_threshold = float(dedx_soft_barrier_threshold)
        self.dedx_soft_barrier_weight = float(dedx_soft_barrier_weight)
        self.dedx_mean_constraint_weight = float(dedx_mean_constraint_weight)
        self.dedx_mean_constraint_target = float(dedx_mean_constraint_target)
        self.dedx_drift_profile_weight = float(dedx_drift_profile_weight)
        self.shuffle_seed = shuffle_seed
        self.sim_track_fields = sim_track_fields
        self.tgt_track_fields = tgt_track_fields

        print(f"Using dE/dx density propagation: {self.use_dedx_density}, mode: {self.dedx_density_mode}")

        if self.normalization_scheme not in ("sigmoid", "exp_log", "divide"):
            raise ValueError(
                f"Unknown normalization_scheme '{self.normalization_scheme}'. "
                "Use one of: sigmoid, exp_log, divide"
            )

        if self.normalization_scale_sigmoid <= 0:
            raise ValueError(f"normalization_scale_sigmoid must be > 0, got {self.normalization_scale_sigmoid}")
        if self.normalization_scale_exp_log <= 0:
            raise ValueError(f"normalization_scale_exp_log must be > 0, got {self.normalization_scale_exp_log}")
        if 'eventID' in self.sim_track_fields:
            self.evt_id = 'eventID'
        else:
            self.evt_id = 'event_id'

        if type(relevant_params) == dict:
            self.relevant_params_list = list(relevant_params.keys())
            self.relevant_params_dict = relevant_params
        elif type(relevant_params) == list:
            self.relevant_params_list = relevant_params
            self.relevant_params_dict = None
        else:
            raise TypeError("relevant_params must be list of param names or list of dicts with learning rates")

        target_dir_name = f"target_{self.test_name}_{self.out_label}_" + "_".join(self.relevant_params_list)
        if len(target_dir_name) > 200:
            import hashlib
            hash_str = hashlib.sha256(target_dir_name.encode('utf-8')).hexdigest()[:16]
            self.target_dir = f"target_{self.test_name}_{hash_str}"
        else:
            self.target_dir = target_dir_name
        self.target_val_dict = None
        if len(set_target_vals) > 0:
            if len(set_target_vals) % 2 != 0:
                raise ValueError("Incorrect format for set_target_vals!")
            
            self.target_val_dict = {}
            for i_val in range(len(set_target_vals)//2):
                param_name = set_target_vals[2*i_val]
                param_val = set_target_vals[2*i_val+1]
                self.target_val_dict[param_name] = float(param_val)

        self.set_init_params = set_init_params

        if self.current_mode == 'lut':
            self.lut_file = config.lut_file
        else:
            self.lut_file = None

        self.setup_params()

        if not self.read_target:
            self.make_target_sim()

        loss_functions = {
            "mse_adc": (mse_adc, {}),
            "mse_time": (mse_time, {}),
            "mse_time_adc": (mse_time_adc, {'alpha': 0.5}),
            "chamfer_3d": (chamfer_3d, {'adc_norm': adc_norm, 'match_z': match_z}),
            "sdtw_adc": (sdtw_adc, {'gamma': 1.}),
            "sdtw_time": (sdtw_time, {'gamma': 1.}),
            "sdtw_time_adc": (sdtw_time_adc, {'gamma': 1., 'alpha': 0.5}),
            "nll": (nll_loss, {'adc_norm': adc_norm, 'sigma': 1.0}),
            "llhd": (llhd_loss, {}),
            "distribution": (None, {}),
            "markov_llhd": (None, {}),
            #"sinkhorn_loss": (sinkhorn_loss, {})
        }

        # Set up loss function -- can pass in directly, or choose a named one
        if loss_fn is None or loss_fn == "space_match":
            loss_fn = "mse_adc"
        elif loss_fn == "SDTW":
            loss_fn = "sdtw_adc"
    
        if isinstance(loss_fn, str):
            if loss_fn not in loss_functions:
                raise ValueError(f"Loss function {loss_fn} not supported")
            self.loss_fn, self.loss_fn_kw = loss_functions[loss_fn]
            if loss_fn_kw is not None:
                self.loss_fn_kw.update(loss_fn_kw) #Adding the user defined kwargs
            logger.info(f"Using loss function {loss_fn} with kwargs {self.loss_fn_kw}")

        else:
            self.loss_fn = loss_fn
            self.loss_fn_kw = {}
            logger.info("Using custom loss function")

        if loss_fn in ['sdtw_adc', 'sdtw_time', 'sdtw_time_adc']: #Need to setup the sdtw class for the loss function
            self.loss_fn_kw['dstw'] = SoftDTW(**self.loss_fn_kw)
        
        # Set up loss strategy based on loss function and simulation type
        if loss_fn == 'llhd':
            from .strategies import ProbabilisticLossStrategy
            self.loss_strategy = ProbabilisticLossStrategy(loss_fn=llhd_loss, **self.loss_fn_kw)
        elif loss_fn == 'distribution':
            from .strategies import DistributionLossStrategy
            dist_feature = config.dist_feature if hasattr(config, 'dist_feature') else 'charge'
            self.loss_strategy = DistributionLossStrategy(feature=dist_feature, **self.loss_fn_kw)
            logger.info(f"Using DistributionLossStrategy with feature {dist_feature}")
        elif loss_fn == 'markov_llhd':
            from .strategies import MarkovLossStrategy
            self.loss_strategy = MarkovLossStrategy(**self.loss_fn_kw)
            logger.info("Using MarkovLossStrategy")
        elif self.probabilistic_sim:
            # Use CollapsedProbabilisticLossStrategy for probabilistic simulation with deterministic losses
            from .strategies import CollapsedProbabilisticLossStrategy
            self.loss_strategy = CollapsedProbabilisticLossStrategy(loss_fn=self.loss_fn, **self.loss_fn_kw)
            logger.info(f"Using CollapsedProbabilisticLossStrategy with {loss_fn}")
        else:
            self.loss_strategy = GenericLossStrategy(self.loss_fn, **self.loss_fn_kw)

        self.training_history = {}
        for param in self.relevant_params_list:
            self.training_history[param] = []
            self.training_history[param + '_target'] = []
            self.training_history[param + '_init'] = [getattr(self.current_params, param)]

        for param in self.shift_no_fit:
            self.training_history[param + '_target'] = []

        if self.compute_target_hessian:
            self.training_history['hessian'] = []
            self.training_history['gradient'] = []
        self.training_history['size_history'] = []
        self.training_history['memory'] = []

        self.training_history['config'] = config

        self.keep_in_memory = keep_in_memory
        self.sim_seed_strategy = sim_seed_strategy
        if keep_in_memory:
            self.targets = {}

    def _norm_scale_for_scheme(self):
        if self.normalization_scheme == "sigmoid":
            return self.normalization_scale_sigmoid
        if self.normalization_scheme == "exp_log":
            return self.normalization_scale_exp_log
        return 1.0

    def setup_params(self):
        Params = build_params_class(self.relevant_params_list)
        ref_params = load_detector_properties(Params, self.detector_props, self.pixel_layouts)

        if self.set_params:
            logger.info(f"Applying global parameter overrides: {self.set_params}")
            ref_params = ref_params.replace(**{k: float(v) for k, v in self.set_params.items()})

        ref_params = ref_params.replace(
            electron_sampling_resolution=self.electron_sampling_resolution,
            number_pix_neighbors=self.number_pix_neighbors,
            signal_length=self.signal_length,
            time_window=self.signal_length)
        
        if self.lut_file is not None:
            self.response, ref_params = load_lut(self.lut_file, ref_params)
        
        params_to_apply = [
            "diffusion_in_current_sim",
            "mc_diff",
            "use_dedx_density",
            "dedx_density_mode",
        ]

        ref_params = ref_params.replace(**{key: getattr(self, key) for key in params_to_apply})

        # Initialize Simulation Strategy
        if self.current_mode == 'lut':
            if self.probabilistic_sim:
                self.sim_strategy = LUTProbabilisticSimulation(self.response)
            elif self.probabilistic_sampling_sim:
                self.sim_strategy = LUTProbabilisticSamplingSimulation(self.response)
            else:
                self.sim_strategy = LUTSimulation(self.response)
        elif self.current_mode == 'markov':
            from .strategies import LUTMarkovSimulation
            self.sim_strategy = LUTMarkovSimulation(self.response)
        elif self.current_mode == 'parametrized':
            self.sim_strategy = ParametrizedSimulation()
        else:
             raise ValueError(f"Unknown mode {self.current_mode}")

        initial_params = {}

        if self.vary_init:
            logger.info("Running with random initial guess")
            for param in self.relevant_params_list:
                if self.normalization_scheme == "divide":
                    init_val = float(np.random.uniform(low=ranges[param]['down'], high=ranges[param]['up']))
                    norm_val = 1.0
                elif self.normalization_scheme == "exp_log":
                    norm_val = np.random.uniform(low=-1.0, high=1.0)
                    init_val = float(map_norm_to_phys(norm_val, param, scheme=self.normalization_scheme, scale=self._norm_scale_for_scheme()))
                else:
                    # For sigmoid, ±3 is close to bounds but avoids hard saturation.
                    norm_val = np.random.uniform(low=-3.0, high=3.0)
                    init_val = float(map_norm_to_phys(norm_val, param, scheme=self.normalization_scheme, scale=self._norm_scale_for_scheme()))
                initial_params[param] = init_val
                logger.info(f"vary_init: {param} = {init_val:.4f} (norm={norm_val:.3f})")

        elif len(self.set_init_params) > 0:
            if len(self.set_init_params) % 2 != 0:
                raise ValueError("Incorrect format for set_init_params!")
            for i_val in range(len(self.set_init_params)//2):
                param_name = self.set_init_params[2*i_val]
                param_val = self.set_init_params[2*i_val+1]
                initial_params[param_name] = float(param_val)

        self.current_params = ref_params.replace(**initial_params)

        self.ref_params = ref_params

        self.params_normalization = ref_params.replace(**{key: getattr(self.current_params, key) if getattr(self.current_params, key) != 0. else 1. for key in self.relevant_params_list})
        if self.normalization_scheme == "divide":
            self.norm_params = ref_params.replace(**{key: 1. if getattr(self.current_params, key) != 0. else 0. for key in self.relevant_params_list})
        else:
            self.norm_params = ref_params.replace(**{
                key: float(map_phys_to_norm(getattr(self.current_params, key), key, scheme=self.normalization_scheme, scale=self._norm_scale_for_scheme()))
                for key in self.relevant_params_list
            })

        #Only do it now to not inpact current_params (ref_params?)
        #FIXME It's a problem if the noise parameters are to be fitted
        if not self.readout_noise_guess:
            logger.info("Not simulating electronics noise for guesses")
            self.current_params = remove_noise_from_params(self.current_params)
            self.params_normalization = remove_noise_from_params(self.params_normalization)
            self.norm_params = remove_noise_from_params(self.norm_params)

    def update_params(self):
        if self.normalization_scheme == "divide":
            new_physical_params = {
                key: getattr(self.norm_params, key) * getattr(self.params_normalization, key)
                for key in self.relevant_params_list
            }
        else:
            new_physical_params = {
                key: map_norm_to_phys(getattr(self.norm_params, key), key, scheme=self.normalization_scheme, scale=self._norm_scale_for_scheme())
                for key in self.relevant_params_list
            }
        self.current_params = self.ref_params.replace(**new_physical_params)

    def precompute_static_pixels(self, dataloader_sim, target):
        logger.info("Precomputing static pixel lists for track alignment...")
        self._batch_fixed_pixels = {}
        self._batch_target_pixels = {}
        self._batch_target_expected_adc = {}
        self._batch_padded_roi = {}
        
        unpadded_pixels = {}
        
        # 1. Collect pixels for each batch
        for i in range(len(dataloader_sim)):
            # Load guess tracks
            tracks_sim_bt = dataloader_sim[i].reshape(-1, len(self.sim_track_fields))
            selected_tracks_sim = jax.device_put(tracks_sim_bt)
            if self.fit_chain_positions:
                self._batch_chain_contexts[i] = self._build_chain_contexts_for_batch(
                    i, np.asarray(tracks_sim_bt))

            # Store true target midpoint positions for per-iteration position residual.
            # Use arc-length interpolation per track to avoid batch-row index mismatch
            # between sim and target (they have different fine-step counts).
            if self.fit_chain_positions and not self.read_target:
                _tgt_raw = np.asarray(target[i].reshape(-1, len(self.tgt_track_fields)))
                _tf = self.tgt_track_fields
                _xi, _yi, _zi = _tf.index('x'), _tf.index('y'), _tf.index('z')
                _ev_i = _tf.index('eventID') if 'eventID' in _tf else None
                _tr_i = _tf.index('trackID') if 'trackID' in _tf else None
                _track_pos_dict = {}
                for _ctx in self._batch_chain_contexts.get(i, []):
                    if _ev_i is None or _tr_i is None:
                        break
                    _ev_id, _tr_id = _ctx.track_id
                    _mask = (np.abs(_tgt_raw[:, _ev_i] - _ev_id) < 0.5) & \
                            (np.abs(_tgt_raw[:, _tr_i] - _tr_id) < 0.5)
                    _rows = _tgt_raw[_mask]
                    if len(_rows) < 2:
                        continue
                    _pts = _rows[:, [_xi, _yi, _zi]].astype(np.float32)
                    _diffs   = np.diff(_pts, axis=0)
                    _seg_len = np.sqrt((_diffs ** 2).sum(axis=1))
                    _cum     = np.concatenate([[0.], np.cumsum(_seg_len)])
                    _total   = float(_cum[-1])
                    if _total < 1e-4:
                        continue
                    _alphas  = (_cum / _total).astype(np.float32)
                    _query   = np.asarray(_ctx.alpha_mid)
                    _idx     = np.searchsorted(_alphas, _query, side='right') - 1
                    _idx     = np.clip(_idx, 0, len(_alphas) - 2)
                    _t_frac  = (_query - _alphas[_idx]) / np.clip(
                        _alphas[_idx + 1] - _alphas[_idx], 1e-8, None)
                    _t_frac  = np.clip(_t_frac, 0., 1.)
                    _interp  = _pts[_idx] + _t_frac[:, None] * (_pts[_idx + 1] - _pts[_idx])
                    _track_pos_dict[_ctx.track_id] = _interp
                self._batch_true_positions[i] = _track_pos_dict

            evts_sim = self.get_batch_global_event_ids(dataloader_sim, i, selected_tracks_sim)

            # Load target
            if not self.read_target:
                selected_tracks_bt_tgt = target[i].reshape(-1, len(self.tgt_track_fields))
                this_target = jax.device_put(selected_tracks_bt_tgt)
            else:
                this_target = target
                
            ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = self.get_simulated_target(this_target, i, evts_sim, regen=False)
            
            # Simulate guess wfs
            wfs_guess, guess_pixels = simulate_wfs(self.ref_params, self.response, selected_tracks_sim, self.sim_track_fields)
            
            # Target pixels
            target_pixels_all = np.array(ref_pixel_id)
            target_adcs_all = np.array(ref_adcs)
            target_ticks_all = np.array(ref_ticks)
            
            Nticks = wfs_guess.shape[1] - 1
            valid_mask = (target_pixels_all >= 0) & (target_ticks_all >= 0) & (target_ticks_all < Nticks)
            target_pixels_clean = target_pixels_all[valid_mask]
            target_adcs_clean = target_adcs_all[valid_mask]
            target_ticks_clean = target_ticks_all[valid_mask]
            
            unique_tgt_pixels, inverse_indices = np.unique(target_pixels_clean, return_inverse=True)
            target_expected_adc_np = np.zeros((len(unique_tgt_pixels), Nticks), dtype=np.float32)
            tick_idx = np.clip(np.round(target_ticks_clean).astype(int), 0, Nticks - 1)
            np.add.at(target_expected_adc_np, (inverse_indices, tick_idx), target_adcs_clean)
            
            target_adc_sum = np.sum(target_expected_adc_np, axis=-1)
            valid_adc_mask = target_adc_sum > 0.05
            
            batch_target_pixels = unique_tgt_pixels[valid_adc_mask]
            batch_target_expected_adc = target_expected_adc_np[valid_adc_mask]
            
            self._batch_target_pixels[i] = jnp.array(batch_target_pixels)
            self._batch_target_expected_adc[i] = jnp.array(batch_target_expected_adc)
            
            # Combined unique pixels for static simulation grid
            combined = np.unique(np.concatenate([np.array(batch_target_pixels), np.array(guess_pixels[guess_pixels >= 0])]))
            combined = np.append(combined, -1)
            unpadded_pixels[i] = combined
            
            # Compute padded ROI counts
            wfs_guess_padded = pad_to_closest_multiple(wfs_guess, dims_to_pad=(0,), multiple=128, pad_value=0.0, pad_front=True)
            nb_small, nb_large = get_roi_counts(self.ref_params, wfs_guess_padded)
            padded_small_nb = int(((int(nb_small) + 127) // 128) * 128)
            padded_large_nb = int(((int(nb_large) + 127) // 128) * 128)
            self._batch_padded_roi[i] = (padded_small_nb, padded_large_nb)

        # Normalize all batches to the global max ROI sizes so every batch shares
        # the same (padded_small_nb, padded_large_nb) pair per pixel-count level.
        # This collapses all JIT signatures to at most one per unique Npix, preventing
        # per-epoch recompilation of rarely-seen batch shapes under XLA cache pressure.
        if self._batch_padded_roi:
            _max_s = max(v[0] for v in self._batch_padded_roi.values())
            _max_l = max(v[1] for v in self._batch_padded_roi.values())
            self._batch_padded_roi = {k: (_max_s, _max_l) for k in self._batch_padded_roi}
            logger.info(
                f"Global ROI sizes: padded_small={_max_s}, padded_large={_max_l} "
                f"(applied to all {len(self._batch_padded_roi)} batches)"
            )

        # 2. Determine global maximum length and pad
        max_len = max(len(c) for c in unpadded_pixels.values())
        padded_len = int(((max_len + 127) // 128) * 128)
        logger.info(f"Global max active pixels: {max_len}, padding all static pixel lists to: {padded_len}")
        
        for i in range(len(dataloader_sim)):
            fixed_pixels_np = pad_to_closest_multiple(unpadded_pixels[i], multiple=padded_len, pad_value=-1, pad_front=False)
            fixed_pixels_np = np.sort(fixed_pixels_np)
            self._batch_fixed_pixels[i] = jnp.array(fixed_pixels_np)

    def _build_chain_contexts_for_batch(self, batch_idx, tracks_np):
        """Build per-track chain contexts from a (N_fine, n_fields) numpy array."""
        fld = self.sim_track_fields
        ev_idx  = fld.index('eventID');  tr_idx  = fld.index('trackID')
        x_idx   = fld.index('x');        y_idx   = fld.index('y');   z_idx   = fld.index('z')
        xs_idx  = fld.index('x_start');  ys_idx  = fld.index('y_start'); zs_idx = fld.index('z_start')
        xe_idx  = fld.index('x_end');    ye_idx  = fld.index('y_end');   ze_idx = fld.index('z_end')

        evs = tracks_np[:, ev_idx]
        trs = tracks_np[:, tr_idx]
        seen = {}
        for ev, tr in zip(evs.tolist(), trs.tolist()):
            k = (ev, tr)
            if k not in seen:
                seen[k] = None
        unique_tracks = list(seen.keys())

        ctxs = []
        for (ev, tr) in unique_tracks:
            mask = (evs == ev) & (trs == tr)
            idxs = np.where(mask)[0]
            if len(idxs) < 2:
                continue

            mids   = tracks_np[idxs][:, [x_idx,  y_idx,  z_idx ]]
            starts = tracks_np[idxs][:, [xs_idx, ys_idx, zs_idx]]
            ends   = tracks_np[idxs][:, [xe_idx, ye_idx, ze_idx]]

            entry_pt = starts[0].astype(np.float32)
            exit_pt  = ends[-1].astype(np.float32)
            d = exit_pt - entry_pt
            total_len = float(np.linalg.norm(d))
            if total_len < 1e-4:
                continue

            n_chain  = max(3, int(total_len / self._chain_step_len))
            step_len = total_len / n_chain
            sigma_mcs = _highland_sigma_mcs(step_len, self._chain_momentum_GeV)

            u = d / total_len
            theta0 = float(np.arccos(np.clip(float(u[2]), -1., 1.)))
            phi0   = float(np.arctan2(float(u[1]), float(u[0])))

            def _alpha_proj(pts, _entry=entry_pt, _u=u, _L=total_len):
                rel = pts - _entry
                a = np.dot(rel, _u) / _L
                return np.clip(a, 0., 1.).astype(np.float32)

            ctxs.append(_ChainTrackCtx(
                track_id=(ev, tr),
                idxs=idxs.astype(np.int32),
                n_fine=len(idxs),
                n_chain=n_chain,
                step_len=step_len,
                total_len=total_len,
                x0_fixed=entry_pt,
                theta0_i=theta0,
                phi0_i=phi0,
                sigma_mcs=sigma_mcs,
                alpha_mid=_alpha_proj(mids),
                alpha_start=_alpha_proj(starts),
                alpha_end=_alpha_proj(ends),
            ))
        logger.info(f"Batch {batch_idx}: built chain contexts for {len(ctxs)} tracks")
        return ctxs

    def profile_local_geometry(self, batch_idx, tracks, target, global_params):
        fixed_pixels = self._batch_fixed_pixels[batch_idx]
        padded_small_nb, padded_large_nb = self._batch_padded_roi[batch_idx]
        target_pixels = self._batch_target_pixels[batch_idx]
        target_expected_adc = self._batch_target_expected_adc[batch_idx]

        # Cache val_and_grad_fn by (pixel shape, roi sizes) so JAX only recompiles
        # when the static arguments change — not on every call.
        cache_key = (fixed_pixels.shape, padded_small_nb, padded_large_nb)
        if not hasattr(self, '_local_geom_jit_cache'):
            self._local_geom_jit_cache = {}
        if cache_key not in self._local_geom_jit_cache:
            # Build a JIT-compiled function that takes all dynamic data explicitly.
            # Captures padded_small_nb / padded_large_nb as Python ints so JAX
            # sees them as compile-time constants (static), not traced values.
            _psn = padded_small_nb
            _pln = padded_large_nb
            response = self.response
            sim_track_fields = self.sim_track_fields

            def _local_loss(geom_vars, tracks_, global_params_, fixed_pixels_,
                            target_pixels_, target_expected_adc_):
                translation = geom_vars[0:3]
                rotation = geom_vars[3:6] * 0.1
                transformed = apply_transformation(tracks_, sim_track_fields, translation, rotation)

                wfs = simulate_wfs_static(global_params_, response, transformed, sim_track_fields, fixed_pixels_)
                wfs_padded = pad_to_closest_multiple(wfs, dims_to_pad=(0,), multiple=128, pad_value=0.0, pad_front=True)

                adcs_distrib, pixel_x, pixel_y, ticks_prob, event = simulate_probabilistic(
                    global_params_, wfs_padded, fixed_pixels_,
                    padded_small_nb=_psn, padded_large_nb=_pln
                )
                expected_adc_raw = jnp.sum(jnp.exp(ticks_prob) * adcs_distrib, axis=1)

                indices = jnp.searchsorted(fixed_pixels_, target_pixels_)
                indices_safe = jnp.clip(indices, 0, fixed_pixels_.shape[0] - 1)
                matched_mask = (jnp.take(fixed_pixels_, indices_safe) == target_pixels_)[:, None]
                guess_expected_adc = jnp.where(matched_mask, jnp.take(expected_adc_raw, indices_safe, axis=0), 0.0)

                return jnp.mean((guess_expected_adc - target_expected_adc_)**2)

            self._local_geom_jit_cache[cache_key] = jax.jit(
                jax.value_and_grad(_local_loss)
            )
            logger.info(f"profile_local_geometry: compiled new JIT trace for key {cache_key}")

        val_and_grad_fn = self._local_geom_jit_cache[cache_key]

        def scipy_loss(x):
            l, g = val_and_grad_fn(
                jnp.array(x, dtype=jnp.float32),
                tracks, global_params, fixed_pixels,
                target_pixels, target_expected_adc
            )
            return float(l), np.array(g, dtype=np.float64)

        x0 = np.zeros(6, dtype=np.float32)
        bounds = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14)]
        res = minimize(scipy_loss, x0, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': self.track_max_iter})

        aligned_tracks = apply_transformation(tracks, self.sim_track_fields, res.x[0:3], res.x[3:6]*0.1)
        return aligned_tracks

    def make_target_sim(self):
        np.random.seed(self.target_seed)
        logger.info("Constructing target param simulation")

        self.target_params = {}

        if self.target_val_dict is not None:
            if set(self.relevant_params_list + self.shift_no_fit) != set(self.target_val_dict.keys()):
                logger.debug(set(self.relevant_params_list + self.shift_no_fit))
                logger.debug(set(self.target_val_dict.keys()))
                raise ValueError("Must specify all parameters if explicitly setting target")

            logger.info("Explicitly setting targets:")
            for param in self.target_val_dict.keys():
                param_val = self.target_val_dict[param]
                logger.info(f'{param}, target: {param_val}, init {getattr(self.current_params, param)}')    
                self.target_params[param] = param_val
        else:
            for param in self.relevant_params_list + self.shift_no_fit:
                if self.target_fixed_range is not None:
                    param_val = np.random.uniform(low=ranges[param]['nom']*(1.-self.target_fixed_range), 
                                                high=ranges[param]['nom']*(1.+self.target_fixed_range))
                else:
                    param_val = np.random.uniform(low=ranges[param]['down'], 
                                                high=ranges[param]['up'])

                logger.info(f'{param}, target: {param_val}, init {getattr(self.current_params, param)}')    
                self.target_params[param] = param_val
        self.target_params = self.ref_params.replace(**self.target_params)
        # Target generation must always use original per-segment dEdx (no dE/dx density propagation).
        self.target_params = self.target_params.replace(
            use_dedx_density=False,
            dedx_density_mode="histogram",
        )
        logger.info("Target simulation forced to use_dedx_density=False (original dEdx path)")
        if not self.readout_noise_target:
            logger.info("Not simulating electronics noise for target")
            self.target_params = remove_noise_from_params(self.target_params)

    def get_simulated_target(self, target, i, evts_sim, regen=False):
        #Reading the reference
        if self.read_target:
            with open(target, 'rb') as f:
                loaded = jnp.load(f, allow_pickle=True)
                if 'event_id' in loaded:
                    mask = jnp.isin(loaded['event_id'], evts_sim)
                    ref_event = loaded['event_id'][mask]
                elif 'event' in loaded:
                    mask = jnp.isin(loaded['event'], evts_sim)
                    ref_event = loaded['event'][mask]
                else:
                    raise ValueError("No event_id or event in the target file")

                if not 'adcs' in loaded:
                    raise ValueError("No adcs in the target file")

                ref_adcs = loaded['adcs'][mask]
                if 'Q' in loaded:
                    ref_Q = loaded['Q'][mask]
                else:
                    ref_Q = adc2charge(ref_adcs, self.current_params)
    
                # switch x and z, to make z the drift
                # as x is the drift in data, and z is the drift in sim
                ref_pixel_x = loaded['z'][mask]
                ref_pixel_y = loaded['y'][mask]
                ref_pixel_z = loaded['x'][mask]
                ref_ticks = loaded['ticks'][mask]
                ref_hit_prob = jnp.ones(ref_adcs.shape[0]) #Assuming all hits are valid
        else:
            #Simulating the reference during the first epoch
            fname = os.path.join(self.target_dir, f'batch{i}_target.npz')
            if regen or not os.path.exists(fname):
                
                # Target generation should always produce "hits" (stochastic),
                # even if the fitting strategy is probabilistic.
                target_strategy = self.sim_strategy
                if isinstance(self.sim_strategy, LUTProbabilisticSimulation):
                    if self.probabilistic_sampling_target:
                        # Draw the target from the probabilistic distribution (fake-stochastic).
                        # The guess is still the full probability distribution, so gradients
                        # flow through the probabilistic model via ProbabilisticLossStrategy.
                        target_strategy = LUTProbabilisticSamplingSimulation(self.response)
                    else:
                        # Default: fall back to the real stochastic FEE to generate a
                        # proper hit-list target.
                        target_strategy = LUTSimulation(self.response)
                # LUTProbabilisticSamplingSimulation already produces hit-list output,
                # so we let it use itself as its own target generator.
                
                prediction = target_strategy.predict(self.target_params, target, self.tgt_track_fields, rngkey=i+1)
                
                ref_adcs = prediction['adcs']
                ref_pixel_x = prediction['pixel_x']
                ref_pixel_y = prediction['pixel_y']
                ref_pixel_z = prediction['pixel_z']
                ref_ticks = prediction['ticks']
                ref_hit_prob = prediction['hit_prob']
                ref_event = prediction['event']
                ref_pixel_id = prediction.get('hit_pixels', prediction.get('unique_pixels'))

                # embed_target = embed_adc_list(self.sim_target, target, pix_target, ticks_list_targ)
                #Saving the target for the batch
                #TODO: See if we have to do this for each event
                
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                with open(fname, 'wb') as f:
                    jnp.savez(f, adcs=ref_adcs, pixel_x=ref_pixel_x, pixel_y=ref_pixel_y, pixel_z=ref_pixel_z, ticks=ref_ticks, hit_prob=ref_hit_prob, event=ref_event, pixel_id=ref_pixel_id)
                    if self.keep_in_memory:
                        self.targets[i] = (ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id)

            else:
                #Loading the target
                if self.keep_in_memory:
                    ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = self.targets[i]
                else:
                    with open(fname, 'rb') as f:
                        loaded = jnp.load(f)
                        ref_adcs = loaded['adcs']
                        ref_pixel_x = loaded['pixel_x']
                        ref_pixel_y = loaded['pixel_y']
                        ref_pixel_z = loaded['pixel_z']
                        ref_ticks = loaded['ticks']
                        ref_event = loaded['event']
                        ref_hit_prob = loaded['hit_prob']
                        ref_pixel_id = loaded['pixel_id'] if 'pixel_id' in loaded else jnp.zeros_like(ref_adcs, dtype=int) # Fallback

        return ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id

    def get_batch_global_event_ids(self, dataloader, batch_idx, selected_tracks=None):
        dataset = getattr(dataloader, 'dataset', None)
        if dataset is not None and hasattr(dataset, 'get_batch_global_event_ids'):
            return jnp.asarray(dataset.get_batch_global_event_ids(batch_idx), dtype=jnp.int64)

        if selected_tracks is None:
            return jnp.empty((0,), dtype=jnp.int64)

        evt_idx = self.sim_track_fields.index(self.evt_id)
        return jnp.unique(selected_tracks[:, evt_idx]).astype(jnp.int64)

    def validate_track_batch_event_ids(self, track_batch, track_fields, context):
        evt_idx = track_fields.index(self.evt_id)
        event_ids = np.asarray(track_batch[:, evt_idx], dtype=np.int64)
        validate_local_event_ids(event_ids, context=context)
        validate_event_ids_for_packing(self.current_params, event_ids, kind="pixel", context=context)
        validate_event_ids_for_packing(self.current_params, event_ids, kind="bin", context=context)

    def load_checkpoint(self, checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            loaded = pickle.load(f)

        if not isinstance(loaded, dict):
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain a history dict")

        self.training_history = loaded
        self.total_iter = int(loaded.get('total_iter', 0))
        ckpt_norm_scheme = loaded.get('normalization_scheme')
        if ckpt_norm_scheme is not None and ckpt_norm_scheme != self.normalization_scheme:
            raise ValueError(
                f"Checkpoint normalization_scheme ({ckpt_norm_scheme}) does not match current setting "
                f"({self.normalization_scheme})"
            )

        norm_state = loaded.get('norm_params_state')
        current_state = loaded.get('current_params_state')
        opt_state = loaded.get('opt_state')

        if norm_state is None:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has no norm_params_state; "
                "old history-only pickle cannot be resumed exactly"
            )

        self.norm_params = restore_param_state(self.ref_params, norm_state)

        if current_state is not None:
            self.current_params = restore_param_state(self.ref_params, current_state)
        else:
            self.update_params()

        if opt_state is None:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has no opt_state; "
                "old history-only pickle cannot resume Adam/SGD state"
            )

        self.opt_state = opt_state
        self.resumed = True

        dedx_cache = loaded.get('dedx_cache')
        if dedx_cache is not None and hasattr(self, '_dedx_cache'):
            self._dedx_cache = dedx_cache
        batch_parent_ids = loaded.get('batch_parent_ids')
        if batch_parent_ids is not None and hasattr(self, '_batch_parent_ids'):
            self._batch_parent_ids = batch_parent_ids
        true_dedx_cache = loaded.get('true_dedx_cache')
        if true_dedx_cache is not None and hasattr(self, '_batch_true_dedx'):
            self._batch_true_dedx = true_dedx_cache

        # If resuming past the activation point, mark the Adam reset as already done
        if hasattr(self, '_dedx_adam_reset_done') and self.total_iter >= self.dedx_start_iter:
            self._dedx_adam_reset_done = True
        if hasattr(self, '_dedx_freeze_adam_reset_done') and self.dedx_freeze_iter is not None and self.total_iter >= self.dedx_freeze_iter:
            self._dedx_freeze_adam_reset_done = True

    def save_checkpoint(self, total_iter):
        """Persist history plus resume-critical optimizer/parameter state."""
        self.total_iter = int(total_iter)
        self.training_history['checkpoint_version'] = 1
        self.training_history['total_iter'] = self.total_iter
        self.training_history['normalization_scheme'] = self.normalization_scheme
        self.training_history['normalization_scale_sigmoid'] = self.normalization_scale_sigmoid
        self.training_history['normalization_scale_exp_log'] = self.normalization_scale_exp_log
        self.training_history['norm_params_state'] = serialize_param_state(
            self.norm_params, self.relevant_params_list
        )
        self.training_history['current_params_state'] = serialize_param_state(
            self.current_params, self.relevant_params_list
        )
        if hasattr(self, 'opt_state'):
            self.training_history['opt_state'] = self.opt_state
        if hasattr(self, '_dedx_cache') and self._dedx_cache:
            # Convert JAX arrays to plain numpy so pickle round-trips cleanly
            dedx_cache_np = {
                int(k): {'log_dedx': np.asarray(v['log_dedx']), 'opt_state': v['opt_state']}
                for k, v in self._dedx_cache.items()
            }
            true_dedx_np = {int(k): np.asarray(v) for k, v in self._batch_true_dedx.items()}
            self.training_history['dedx_cache'] = dedx_cache_np
            self.training_history['batch_parent_ids'] = {
                int(k): np.asarray(v) for k, v in self._batch_parent_ids.items()
            }
            self.training_history['true_dedx_cache'] = true_dedx_np
            self.training_history['dedx_student_nu']    = float(self.dedx_student_nu)
            self.training_history['dedx_student_loc']   = float(self.dedx_student_loc)
            self.training_history['dedx_student_scale'] = float(self.dedx_student_scale)
            self.training_history['dedx_freeze_iter']   = self.dedx_freeze_iter
        if hasattr(self, '_chain_cache') and self._chain_cache:
            self.training_history['chain_cache'] = {
                int(bi): [{'angles': np.asarray(s['angles'])} for s in states]
                for bi, states in self._chain_cache.items()
            }
            if hasattr(self, '_batch_chain_contexts') and self._batch_chain_contexts:
                self.training_history['chain_contexts'] = {
                    int(bi): [{'track_id': ctx.track_id, 'idxs': ctx.idxs,
                               'n_chain': ctx.n_chain, 'step_len': ctx.step_len,
                               'total_len': ctx.total_len, 'x0_fixed': ctx.x0_fixed,
                               'theta0_i': ctx.theta0_i, 'phi0_i': ctx.phi0_i,
                               'alpha_mid': ctx.alpha_mid}
                              for ctx in ctxs]
                    for bi, ctxs in self._batch_chain_contexts.items()
                }
            if hasattr(self, '_batch_true_positions') and self._batch_true_positions:
                self.training_history['true_positions'] = {
                    int(bi): {str(k): v for k, v in track_dict.items()}
                    for bi, track_dict in self._batch_true_positions.items()
                }

        if self.resume_from is not None:
            self.training_history['resumed_from'] = self.resume_from

        with open(f'fit_result/{self.test_name}/history_iter{self.total_iter}_{self.out_label}.pkl', "wb") as f_history:
            pickle.dump(self.training_history, f_history)

    def compute_loss(self, tracks, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, with_loss=True, with_grad=True, with_hess=False, epoch=0, use_physical_params=False, log_dedx=None, parent_ids=None, chain_angles_per_track=None, return_loss_closure=False, unique_size=None, roi_override=None):
        """Compute the simulation loss and (optionally) gradients.

        When ``log_dedx`` and ``parent_ids`` are provided the function jointly
        differentiates w.r.t. both the global normalised physics parameters and
        the per-segment log-dEdx values.  The per-segment dEdx regularisation
        (Student-t NLL) is included in the loss automatically.

        Returns
        -------
        (loss_val, grads_norm, grads_dedx, grads_chain_per_track, aux, hess, aux_hess)
            ``grads_dedx`` is ``None`` unless ``log_dedx`` was supplied.
            ``grads_chain_per_track`` is ``None`` unless ``chain_angles_per_track`` was supplied.
        """
        if self.probabilistic_sim and not self.probabilistic_sampling_sim:
            rngkey = None
        else:
            if self.sim_seed_strategy == "same":
                rngkey = i + 1
            elif self.sim_seed_strategy == "different":
                rngkey = -i - 1 #Need some offset otherwise batch 0 has same seed
            elif self.sim_seed_strategy == "different_epoch":
                rngkey = -i - 1 - epoch * 10000
            elif self.sim_seed_strategy == "random":
                rngkey = np.random.randint(0, 1000000)
            elif self.sim_seed_strategy == "constant":
                rngkey = 0
            else:
                raise ValueError("Unknown sim_seed_strategy. Must be same, different or random")

        assert(with_loss or with_grad)
        loss_val, grads, grads_dedx, grads_chain_per_track, aux, hess, aux_hess = None, None, None, None, None, None, None

        # Precompute padded small and large ROI counts for JAX static shapes.
        # This runs simulate_wfs ONCE per unique batch (cached), using the stable
        # ref_params. The ROI structure depends primarily on track geometry, not
        # the physics parameters being fitted, so computing with ref_params is safe.
        # The 128-multiple rounding provides headroom for small waveform changes
        # across iterations as physics parameters evolve.
        from optimize.strategies import LUTProbabilisticSimulation, LUTProbabilisticSamplingSimulation
        if isinstance(self.sim_strategy, (LUTProbabilisticSimulation, LUTProbabilisticSamplingSimulation)):
            from larndsim.sim_jax import get_roi_counts, simulate_wfs, pad_to_closest_multiple
            # Use (batch_index, track_count) as cache key. This is stable across
            # epochs even with shuffling (track count is a good proxy for identity).
            cache_key = (i, tracks.shape[0])
            if not hasattr(self, '_roi_padded_cache'):
                self._roi_padded_cache = {}
            
            if cache_key not in self._roi_padded_cache:
                wfs_eval, _ = simulate_wfs(
                    self.ref_params, self.sim_strategy.response, tracks, self.sim_track_fields
                )
                wfs_eval = pad_to_closest_multiple(
                    wfs_eval, dims_to_pad=(0,), multiple=128, pad_value=0.0, pad_front=True
                )
                nb_small, nb_large = get_roi_counts(self.ref_params, wfs_eval)
                padded_small_nb = int(((int(nb_small) + 127) // 128) * 128)
                padded_large_nb = int(((int(nb_large) + 127) // 128) * 128)
                self._roi_padded_cache[cache_key] = (padded_small_nb, padded_large_nb)
                # Track running global max so all batches can be normalized once seen.
                if not hasattr(self, '_roi_global_max'):
                    self._roi_global_max = (0, 0)
                gs, gl = self._roi_global_max
                self._roi_global_max = (max(gs, padded_small_nb), max(gl, padded_large_nb))
                logger.info(
                    f"Batch {i}: computed ROI counts: {int(nb_small)} small "
                    f"(padded→{padded_small_nb}), {int(nb_large)} large "
                    f"(padded→{padded_large_nb})"
                )

            padded_small_nb, padded_large_nb = self._roi_padded_cache[cache_key]
            # Apply global max so every batch uses the same ROI sizes, collapsing
            # all JIT signatures to one per unique Npix and eliminating recompilation.
            if hasattr(self, '_roi_global_max'):
                gs, gl = self._roi_global_max
                padded_small_nb = max(padded_small_nb, gs)
                padded_large_nb = max(padded_large_nb, gl)
            self.sim_strategy.padded_small_nb = padded_small_nb
            self.sim_strategy.padded_large_nb = padded_large_nb

        target_data = {
            'adcs': ref_adcs,
            'pixel_x': ref_pixel_x,
            'pixel_y': ref_pixel_y,
            'pixel_z': ref_pixel_z,
            'ticks': ref_ticks,
            'hit_prob': ref_hit_prob,
            'event': ref_event,
            'pixel_id': ref_pixel_id
        }

        if use_physical_params:
            def loss_wrapper(physical_params):
                prediction = self.sim_strategy.predict(physical_params, tracks, self.sim_track_fields, rngkey)
                return self.loss_strategy.compute(physical_params, prediction, target_data)

            if with_loss and with_grad:
                (loss_val, aux), grads = value_and_grad(loss_wrapper, has_aux=True)(self.current_params)
            elif with_loss:
                loss_val, aux = loss_wrapper(self.current_params)
            elif with_grad:
                grads, aux = grad(loss_wrapper, has_aux=True)(self.current_params)

            if with_hess:
                hess, aux_hess = jax.jacfwd(jax.jacrev(loss_wrapper, has_aux=True), has_aux=True)(self.current_params)

        elif (log_dedx is not None and parent_ids is not None) or (self.fit_chain_positions and chain_angles_per_track is not None):
            # ── Joint fit: norm_params (global) + optionally log_dedx (per-segment) + optionally chain angles ──
            dedx_idx = self.sim_track_fields.index('dEdx')
            dE_idx   = self.sim_track_fields.index('dE')
            dx_idx   = self.sim_track_fields.index('dx')

            # Pre-resolve field indices for chain warping (used when fit_chain_positions=True)
            _chain_ctxs = self._batch_chain_contexts.get(i, []) if self.fit_chain_positions else []
            _chain_meta = None   # must always be bound: loss_wrapper_combined reads it
            _col_map = None      # unconditionally (dEdx-without-chain fits crashed otherwise)
            if self.fit_chain_positions and chain_angles_per_track is not None:
                _x_idx  = self.sim_track_fields.index('x')
                _y_idx  = self.sim_track_fields.index('y')
                _z_idx  = self.sim_track_fields.index('z')
                _xs_idx = self.sim_track_fields.index('x_start')
                _ys_idx = self.sim_track_fields.index('y_start')
                _zs_idx = self.sim_track_fields.index('z_start')
                _xe_idx = self.sim_track_fields.index('x_end')
                _ye_idx = self.sim_track_fields.index('y_end')
                _ze_idx = self.sim_track_fields.index('z_end')
                _col_map = {'x': _x_idx, 'y': _y_idx, 'z': _z_idx,
                            'xs': _xs_idx, 'ys': _ys_idx, 'zs': _zs_idx,
                            'xe': _xe_idx, 'ye': _ye_idx, 'ze': _ze_idx, 'dx': dx_idx}
                if i not in self._batch_chain_meta:
                    if self._chain_basis == 'spline':
                        self._batch_chain_meta[i] = _build_chain_meta_spline(_chain_ctxs, self._chain_spline_knot_cm)
                    else:
                        self._batch_chain_meta[i] = _build_chain_meta(_chain_ctxs)
                _chain_meta = self._batch_chain_meta[i]

            def loss_wrapper_combined(norm_params_input, log_dedx_in, chain_angles_pt=None, data_over=None):
                # data_over: optional {'tracks', 'target', 'meta'} of TRACED arrays overriding the
                # closure-baked batch data. Passing batch data as traced args lets a single jitted
                # program serve every batch with the same shape signature (data-as-args pattern) —
                # without it, each batch bakes its data as XLA constants and forces a fresh compile.
                # Map global params: unconstrained → physical
                if self.normalization_scheme == "divide":
                    new_phys = {
                        key: getattr(norm_params_input, key) * getattr(self.params_normalization, key)
                        for key in self.relevant_params_list
                    }
                else:
                    new_phys = {
                        key: map_norm_to_phys(getattr(norm_params_input, key), key, scheme=self.normalization_scheme, scale=self._norm_scale_for_scheme())
                        for key in self.relevant_params_list
                    }
                physical_params = self.ref_params.replace(**new_phys)

                # Optionally warp track geometry using per-track chain angles.
                # Each track's segments are repositioned by interpolating along the
                # deflection chain's node positions.
                t = tracks if data_over is None else data_over['tracks']
                _meta_use = _chain_meta if data_over is None else data_over['meta']
                _chain_thetas = None
                _chain_phis = None
                _chain_warp_aux = None
                if chain_angles_pt is not None and _meta_use is not None:
                    if _meta_use.get('basis') == 'spline':
                        t, _chain_warp_aux = _chain_warp_spline(t, chain_angles_pt, _meta_use, _col_map)
                    else:
                        t, _chain_thetas, _chain_phis = _chain_warp_vectorized(
                            t, chain_angles_pt, _meta_use, _col_map)

                # Reconstruct per-step dEdx from per-segment log-mean.
                # Padding rows have parent_id == -1; preserve their original track dEdx
                # so they do not inject spurious gradients into log_dedx[0].
                if log_dedx_in is not None and parent_ids is not None:
                    valid_mask  = (parent_ids >= 0)
                    safe_ids    = jnp.clip(parent_ids, 0, log_dedx_in.shape[0] - 1)
                    dedx_fitted = jnp.maximum(jnp.exp(jnp.take(log_dedx_in, safe_ids)), 1e-3)
                    dedx_sample = jnp.where(valid_mask, dedx_fitted, t[:, dedx_idx])
                    t = t.at[:, dedx_idx].set(dedx_sample)
                    t = t.at[:, dE_idx].set(dedx_sample * t[:, dx_idx])
                else:
                    dedx_sample = t[:, dedx_idx]
                    valid_mask  = jnp.ones(t.shape[0], dtype=bool)

                prediction = self.sim_strategy.predict(physical_params, t, self.sim_track_fields, rngkey, unique_size=unique_size, roi_override=roi_override)
                _target_use = target_data if data_over is None else data_over['target']
                hit_loss, hit_aux = self.loss_strategy.compute(physical_params, prediction, _target_use)

                # Read length scale factor if present, otherwise default to 1.0
                if 'length_scale_factor' in self.sim_track_fields:
                    scale_factor_idx = self.sim_track_fields.index('length_scale_factor')
                    length_scale_factor = t[:, scale_factor_idx]
                else:
                    length_scale_factor = 1.0

                # Student-t prior on per-step dEdx values.
                # Weighted by physical step length (dx) to ensure that the regularization
                # strength is invariant to the sampling resolution.
                # The location (expected dEdx) is scaled by length_scale_factor.
                if self.dedx_use_split_t:
                    dedx_prior = split_student_t_nll(
                        dedx_sample,
                        self.dedx_student_nu_l,
                        self.dedx_student_nu_r,
                        self.dedx_student_loc * length_scale_factor,
                        self.dedx_student_scale_l,
                        self.dedx_student_scale_r,
                        weights=t[:, dx_idx]
                    ) * self.dedx_prior_weight
                else:
                    dedx_prior = student_t_nll(
                        dedx_sample,
                        self.dedx_student_nu,
                        self.dedx_student_loc * length_scale_factor,
                        self.dedx_student_scale,
                        weights=t[:, dx_idx]
                    ) * self.dedx_prior_weight

                # One-sided soft barrier penalty.
                # Weighted by physical step length (dx) and mask out invalid segments (dx <= 0)
                dx_weights = t[:, dx_idx]
                valid_step_mask = (dx_weights > 0)
                dedx_barrier = self.dedx_soft_barrier_weight * jnp.sum(
                    dx_weights * valid_step_mask * jnp.square(jnp.maximum(0., dedx_sample - self.dedx_soft_barrier_threshold))
                )

                # Length-weighted mean dEdx constraint
                segment_valid = valid_mask & valid_step_mask
                total_dx = jnp.sum(dx_weights * segment_valid)
                total_dedx_len = jnp.sum(dedx_sample * dx_weights * segment_valid)
                mean_dedx = total_dedx_len / jnp.maximum(total_dx, 1e-6)

                # Scale the mean constraint target by the length-weighted mean of scale factors
                if 'length_scale_factor' in self.sim_track_fields:
                    mean_scale_factor = jnp.sum(length_scale_factor * dx_weights * segment_valid) / jnp.maximum(total_dx, 1e-6)
                    scaled_mean_target = self.dedx_mean_constraint_target * mean_scale_factor
                else:
                    scaled_mean_target = self.dedx_mean_constraint_target

                dedx_mean_penalty = self.dedx_mean_constraint_weight * jnp.square(mean_dedx - scaled_mean_target)

                # Drift-profile penalty: per-segment dEdx must be UNCORRELATED with the drift
                # coordinate. Landau fluctuations are drift-independent; only lifetime
                # attenuation produces a drift-monotone (to leading order linear) profile, so
                # a linear trend of log(dEdx) vs |z| is exactly the dEdx<->lifetime degenerate
                # mode (measured: 400cm ceiling lifetime +18.5% with fitted dEdx vs -1.5% with
                # true dEdx). Penalize the dimensionless dx-weighted WLS trend (slope * sigma_z)^2.
                dedx_drift_penalty = jnp.float32(0.)
                if self.dedx_drift_profile_weight > 0 and log_dedx_in is not None:
                    _zc = jnp.abs(t[:, self.sim_track_fields.index('z')])
                    _w = dx_weights * segment_valid
                    _W = jnp.sum(_w) + 1e-9
                    _y = jnp.log(jnp.maximum(dedx_sample, 1e-3))
                    _zb = jnp.sum(_w * _zc) / _W
                    _yb = jnp.sum(_w * _y) / _W
                    _cov = jnp.sum(_w * (_zc - _zb) * (_y - _yb)) / _W
                    _var = jnp.sum(_w * (_zc - _zb) ** 2) / _W + 1e-9
                    # slope * sigma_z = fractional log-dEdx change across one sigma of drift
                    _trend = _cov / jnp.sqrt(_var)
                    dedx_drift_penalty = self.dedx_drift_profile_weight * jnp.square(_trend)

                total_loss = hit_loss + dedx_prior + dedx_barrier + dedx_mean_penalty + dedx_drift_penalty

                # Gaussian MCS prior on chain deflection angles (vectorized)
                if chain_angles_pt is not None and _meta_use is not None:
                    if _meta_use.get('basis') == 'spline':
                        _mcs_loss = _chain_mcs_spline(_chain_warp_aux, _meta_use, self._mcs_prior_weight)
                    else:
                        _mcs_loss = _chain_mcs_vectorized(
                            _chain_thetas, _chain_phis, _meta_use, self._mcs_prior_weight)
                    total_loss = total_loss + _mcs_loss
                else:
                    _mcs_loss = jnp.float32(0.)

                aux_out = dict(hit_aux) if hit_aux is not None else {}
                aux_out['dedx_prior'] = dedx_prior
                aux_out['dedx_barrier'] = dedx_barrier
                aux_out['dedx_mean_penalty'] = dedx_mean_penalty
                aux_out['dedx_drift_penalty'] = dedx_drift_penalty
                aux_out['mean_dedx'] = mean_dedx
                aux_out['mcs_prior'] = _mcs_loss
                return total_loss, aux_out

            _chain_angles_active = (self.fit_chain_positions and chain_angles_per_track is not None)
            _dedx_active = (log_dedx is not None)
            if return_loss_closure and _chain_angles_active:
                # Return a pure, differentiable loss(chain_angles_pt[, norm_params])->scalar for this
                # batch, with dEdx held fixed. norm_params is an OPTIONAL traced arg so a jitted
                # value_and_grad uses the *current* calibration (not the compile-time value) — required
                # for the alternating L-BFGS geometry block. Omit it to use the current self.norm_params.
                def _chain_only_loss(chain_angles_pt, norm_params_in=None, data_over=None):
                    _np = self.norm_params if norm_params_in is None else norm_params_in
                    return loss_wrapper_combined(_np, log_dedx, chain_angles_pt, data_over=data_over)[0]
                return _chain_only_loss
            if with_loss and with_grad:
                if _chain_angles_active and _dedx_active:
                    # Joint: global params + per-seg dEdx + chain angles
                    (loss_val, aux), (grads, grads_dedx, grads_chain_per_track) = value_and_grad(
                        loss_wrapper_combined, argnums=(0, 1, 2), has_aux=True
                    )(self.norm_params, log_dedx, chain_angles_per_track)
                elif _chain_angles_active and not _dedx_active:
                    # Position-only: global params + chain angles (no dEdx)
                    def _wrap_no_dedx(np_, ca_):
                        return loss_wrapper_combined(np_, None, ca_)
                    (loss_val, aux), (_g_np, grads_chain_per_track) = value_and_grad(
                        _wrap_no_dedx, argnums=(0, 1), has_aux=True
                    )(self.norm_params, chain_angles_per_track)
                    grads = _g_np
                    grads_dedx = None
                else:
                    (loss_val, aux), (grads, grads_dedx) = value_and_grad(
                        loss_wrapper_combined, argnums=(0, 1), has_aux=True
                    )(self.norm_params, log_dedx, None)
                    grads_chain_per_track = None
            elif with_loss:
                loss_val, aux = loss_wrapper_combined(self.norm_params, log_dedx,
                                                      chain_angles_per_track if _chain_angles_active else None)
            elif with_grad:
                if _chain_angles_active and _dedx_active:
                    (grads, grads_dedx, grads_chain_per_track), aux = grad(
                        loss_wrapper_combined, argnums=(0, 1, 2), has_aux=True
                    )(self.norm_params, log_dedx, chain_angles_per_track)
                elif _chain_angles_active and not _dedx_active:
                    def _wrap_no_dedx(np_, ca_):
                        return loss_wrapper_combined(np_, None, ca_)
                    (_g_np, grads_chain_per_track), aux = grad(
                        _wrap_no_dedx, argnums=(0, 1), has_aux=True
                    )(self.norm_params, chain_angles_per_track)
                    grads = _g_np
                    grads_dedx = None
                else:
                    (grads, grads_dedx), aux = grad(
                        loss_wrapper_combined, argnums=(0, 1), has_aux=True
                    )(self.norm_params, log_dedx, None)
                    grads_chain_per_track = None

            if with_hess:
                # Hessian w.r.t. the GLOBAL norm params only, with the per-segment dEdx and
                # chain angles held FIXED — the nuisance-frozen curvature needed for the GN
                # calibration polish on a joint-fit checkpoint.
                def _np_only(np_):
                    return loss_wrapper_combined(
                        np_, log_dedx if _dedx_active else None,
                        chain_angles_per_track if _chain_angles_active else None)
                hess, aux_hess = jax.jacfwd(jax.jacrev(_np_only, has_aux=True), has_aux=True)(self.norm_params)

        else:
            def loss_wrapper(norm_params_input):
                # Map from unconstrained space to physical space internally
                if self.normalization_scheme == "divide":
                    new_phys = {
                        key: getattr(norm_params_input, key) * getattr(self.params_normalization, key)
                        for key in self.relevant_params_list
                    }
                else:
                    new_phys = {
                        key: map_norm_to_phys(getattr(norm_params_input, key), key, scheme=self.normalization_scheme, scale=self._norm_scale_for_scheme())
                        for key in self.relevant_params_list
                    }
                physical_params = self.ref_params.replace(**new_phys)
                prediction = self.sim_strategy.predict(physical_params, tracks, self.sim_track_fields, rngkey)
                return self.loss_strategy.compute(physical_params, prediction, target_data)

            # Differentiate with respect to norm_params
            if with_loss and with_grad:
                (loss_val, aux), grads = value_and_grad(loss_wrapper, has_aux=True)(self.norm_params)
            elif with_loss:
                loss_val, aux = loss_wrapper(self.norm_params)
            elif with_grad:
                grads, aux = grad(loss_wrapper, has_aux=True)(self.norm_params)
 
            if with_hess:
                hess, aux_hess = jax.jacfwd(jax.jacrev(loss_wrapper, has_aux=True), has_aux=True)(self.norm_params)

        return loss_val, grads, grads_dedx, grads_chain_per_track, aux, hess, aux_hess

    
    def prepare_fit(self):
        # make a folder for the pixel target

        target_dir = self.target_dir
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, ignore_errors=True)
        os.makedirs(target_dir, exist_ok=True)

        if not os.path.exists(f'fit_result/{self.test_name}'):
            os.makedirs(f'fit_result/{self.test_name}')

        if self.resume_from is not None and self.resumed:
            return

        for param in self.relevant_params_list:
            self.training_history[param] = []
            self.training_history[param+"_grad"] = []
            self.training_history[param+"_iter"] = []
        self.training_history['step_time'] = []
        self.training_history['losses_iter'] = []
        self.training_history['aux_iter'] = []
        if self.fit_dedx:
            self.training_history['dedx_prior_iter'] = []
            self.training_history['dedx_barrier_iter'] = []
            self.training_history['dedx_mean_penalty_iter'] = []
            self.training_history['mean_dedx_iter'] = []
            self.training_history['dedx_mae_iter'] = []

        # Include initial value in training history (if haven't loaded a checkpoint)
        for param in self.relevant_params_list:
            if len(self.training_history[param]) == 0:
                self.training_history[param].append(getattr(self.current_params, param))
                if not self.read_target:
                    self.training_history[param+'_target'].append(getattr(self.target_params, param))
            if len(self.training_history[param+"_iter"]) == 0:
                self.training_history[param+"_iter"].append(getattr(self.current_params, param))
        if not self.read_target:
            for param in self.shift_no_fit:
                if len(self.training_history[param+'_target']) == 0:
                    self.training_history[param+'_target'].append(getattr(self.target_params, param))

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Fit method not implemented. Use a derived class")

class GradientDescentFitter(ParamFitter):
    def __init__(self, optimizer_fn="Adam", max_clip_norm_val=False, clip_from_range=False,
                 lr_scheduler=None, optimizer=None, lr=None, lr_kw=None, epoch_size=1, **kwargs):
        super().__init__(**kwargs)

        if optimizer_fn == "Adam":
            self.optimizer_fn = optax.adam
        elif optimizer_fn == "SGD":
            self.optimizer_fn = optax.sgd
        else:
            raise NotImplementedError("Only SGD and Adam supported")
        self.optimizer_fn_name = optimizer_fn

        self.max_clip_norm_val = max_clip_norm_val
        if self.max_clip_norm_val is not None:
            logger.info(f"Will clip gradient norm at {self.max_clip_norm_val}")

        self.clip_from_range = clip_from_range

        self.learning_rates = {}
        if lr_scheduler is not None and lr_kw is not None:
            lr_scheduler_fn = getattr(optax, lr_scheduler)
            logger.info(f"Using learning rate scheduler {lr_scheduler}")
        else:
            lr_scheduler_fn = optax.constant_schedule
            lr_kw = {}

        if self.relevant_params_dict is None:
            if lr is None:
                raise ValueError("Need to specify lr for params")
            else:
                if lr_scheduler_fn == optax.constant_schedule:
                    self.learning_rates = {par: lr_scheduler_fn(lr) for par in self.relevant_params_list}
                elif lr_scheduler_fn == optax.exponential_decay:
                    self.learning_rates = {par: lr_scheduler_fn(lr, transition_steps=epoch_size, staircase=True, **lr_kw) for par in self.relevant_params_list}
                elif lr_scheduler_fn == optax.warmup_exponential_decay_schedule:
                    self.learning_rates = {par: lr_scheduler_fn(peak_value=lr, transition_steps=epoch_size, staircase=True, **lr_kw) for par in self.relevant_params_list}
                else:
                    raise ValueError("The specified optimizer schedules are not yet supported")
        else:
            if lr_scheduler_fn == optax.constant_schedule:
                self.learning_rates = {key: lr_scheduler_fn(float(value)) for key, value in self.relevant_params_dict.items()}
            elif lr_scheduler_fn == optax.exponential_decay:
                self.learning_rates = {key: lr_scheduler_fn(float(value), transition_steps=epoch_size, staircase=True, **lr_kw) for key, value in self.relevant_params_dict.items()}
            elif lr_scheduler_fn == optax.warmup_exponential_decay_schedule:
                self.learning_rates = {key: lr_scheduler_fn(peak_value=float(value), transition_steps=epoch_size, staircase=True, **lr_kw) for key, value in self.relevant_params_dict.items()}
            else:
                raise ValueError("The specified optimizer schedules are not yet supported")

        
        # Set up optimizer -- can pass in directly, or construct as SGD from relevant params and/or lr

        if optimizer is None:
            if self.max_clip_norm_val is not None:
                self.optimizer = optax.chain(
                    optax.clip_by_global_norm(self.max_clip_norm_val),
                    optax.multi_transform({key: self.optimizer_fn(value) for key, value in self.learning_rates.items()},
                                {key: key for key in self.relevant_params_list})
                )
            else:
                self.optimizer = optax.chain(
                    optax.multi_transform({key: self.optimizer_fn(value) for key, value in self.learning_rates.items()},
                                {key: key for key in self.relevant_params_list})
                )
        else:
            raise ValueError("Passing directly optimizer is not supported")
    
        self.opt_state = self.optimizer.init(extract_relevant_params(self.norm_params, self.relevant_params_list))

        # ── Per-segment dEdx fitting state ──────────────────────────────────────
        # _dedx_cache: batch_idx → {'log_dedx': jnp.array, 'opt_state': optax_state}
        self._dedx_cache: dict = {}
        self._batch_parent_ids: dict = {}
        self._batch_true_dedx: dict = {}   # batch_idx → np.array of true (input) dEdx per orig segment
        _dedx_lr = self.dedx_lr if self.dedx_lr is not None else lr
        self._dedx_optimizer = optax.adam(_dedx_lr) if self.fit_dedx else None
        # Flag to reset calibration Adam state exactly once when dEdx activates
        self._dedx_adam_reset_done = False
        # Flag to reset calibration Adam state exactly once when dEdx is frozen
        self._dedx_freeze_adam_reset_done = False

        # ── Per-track chain geometry fitting state ───────────────────────────
        # _chain_cache: batch_idx → list of {'angles': jnp.array(2*n_chain,), 'opt_state'}
        self._chain_cache: dict = {}
        self._chain_optimizer = optax.adam(self._chain_lr) if self.fit_chain_positions else None
        # _batch_true_positions: batch_idx → (N_fine, 3) float32 array of true midpoints
        self._batch_true_positions: dict = {}

        if self.resume_from is not None:
            logger.info(f"Resuming from checkpoint {self.resume_from}")
            self.load_checkpoint(self.resume_from)

        self.training_history['optimizer_fn_name'] = self.optimizer_fn_name

    def clip_values(self, mini=0.01, maxi=100):
        cur_norm_values = extract_relevant_params(self.norm_params, self.relevant_params_list)
        cur_norm_values = {key: jnp.array(max(mini, min(maxi, val))) for key, val in cur_norm_values.items()}
        self.norm_params = self.norm_params.replace(**cur_norm_values)

    def clip_values_from_range(self):
        self.norm_params = self.norm_params.replace(
            **{key: jnp.array(max(ranges[key]['down']/getattr(self.params_normalization, key), min(ranges[key]['up']/getattr(self.params_normalization, key), getattr(self.norm_params, key)))) for key in self.relevant_params_list}
            )

    def apply_updates(self, params, update):
        setattr(self, params, getattr(self, params).replace(**{key: getattr(getattr(self, params), key) + val for key, val in update.items()}))

    def process_grads(self, grads):
        # We extract only the relevant parameters to pass to the optimizer
        relevant_grads = extract_relevant_params(grads, self.relevant_params_list)

        leaves = jax.tree_util.tree_leaves(relevant_grads)
        if any(jnp.isnan(leaf).any() for leaf in leaves):
            logger.warning("Got NaN gradients! Skipping update for this batch")
            return relevant_grads

        if self.normalization_scheme == "divide":
            relevant_grads = {
                key: relevant_grads[key] * getattr(self.params_normalization, key)
                for key in self.relevant_params_list
            }

        # Update the optimizer state and get the updates for norm_params
        updates, self.opt_state = self.optimizer.update(relevant_grads, self.opt_state)
        
        # Apply the updates directly to the unconstrained norm_params
        self.apply_updates('norm_params', updates)

        if self.normalization_scheme == "divide" and self.clip_from_range:
            self.clip_values_from_range()

        # Re-map unconstrained values to physical values for the next simulation step
        self.update_params()
        
        return relevant_grads

    # def process_grads(self, grads):
    #     scaled_grads = {key: getattr(grads, key)*getattr(self.params_normalization, key) for key in self.relevant_params_list}
    #     leaves = jax.tree_util.tree_leaves(grads)
    #     hasNaN = any(jnp.isnan(leaf).any() for leaf in leaves)
    #     if hasNaN:
    #         logger.warning("Got NaN gradients! Skipping update for this batch")
    #     else:
    #         updates, self.opt_state = self.optimizer.update(scaled_grads, self.opt_state)
    #         # self.current_params = update_params(self.norm_params, updates)
    #         self.apply_updates('norm_params', updates)
    #         if self.clip_from_range:
    #             #Clipping param values
    #             self.clip_values_from_range()
    #         self.update_params()
    #     return scaled_grads

    def add_grads(self, g1, g2):
        return jax.tree_util.tree_map(lambda a, b: a + b, g1, g2)

    def _get_or_init_dedx_state(self, batch_idx, n_orig_segs):
        """Lazily initialise per-segment log-dEdx and its Adam state for *batch_idx*.

        All segments are initialised to log(dedx_student_loc) — the prior centre.
        No true dEdx information is used.
        """
        if batch_idx not in self._dedx_cache:
            log_dedx = jnp.log(jnp.full((n_orig_segs,), float(self.dedx_student_loc)))
            opt_state = self._dedx_optimizer.init(log_dedx)
            self._dedx_cache[batch_idx] = {'log_dedx': log_dedx, 'opt_state': opt_state}
        entry = self._dedx_cache[batch_idx]
        return entry['log_dedx'], entry['opt_state']

    def _process_dedx_grads(self, batch_idx, grads_dedx, current_log_dedx):
        """Apply one Adam step for per-segment log-dEdx of *batch_idx*."""
        _, opt_state = self._get_or_init_dedx_state(batch_idx, len(current_log_dedx))
        updates, new_opt_state = self._dedx_optimizer.update(grads_dedx, opt_state)
        new_log_dedx = current_log_dedx + updates
        self._dedx_cache[batch_idx] = {'log_dedx': new_log_dedx, 'opt_state': new_opt_state}
        return new_log_dedx

    def _get_or_init_chain_state(self, batch_idx):
        """Lazily initialise per-track chain angles and Adam state for batch_idx.

        Angles layout (absolute per-segment directions):
          angles[:n_chain] = theta_0, theta_1, ..., theta_{n-1}   (global polar angle of each segment)
          angles[n_chain:] = phi_0,   phi_1,   ..., phi_{n-1}     (global azimuthal angle of each segment)
        Shape: (2 * n_chain,).  Initialised so all segments point in the initial track direction.
        MCS prior penalises jnp.diff(thetas) and jnp.diff(phis) — consecutive direction changes.
        """
        if batch_idx not in self._chain_cache:
            ctxs = self._batch_chain_contexts.get(batch_idx, [])
            states = []
            for ctx in ctxs:
                if self._chain_basis == 'spline':
                    # 2K spline coefficients, initialised to zero = nominal (linear-guess) track
                    K = _spline_K(ctx.total_len, self._chain_spline_knot_cm)
                    params = jnp.zeros(2 * K, dtype=jnp.float32)
                else:
                    n_c = ctx.n_chain
                    angles = np.empty(2 * n_c, dtype=np.float32)
                    angles[:n_c] = ctx.theta0_i   # all segments start aligned with initial direction
                    angles[n_c:] = ctx.phi0_i
                    params = jnp.array(angles)
                opt_state = self._chain_optimizer.init(params)
                states.append({'angles': params, 'opt_state': opt_state})
            self._chain_cache[batch_idx] = states
        return self._chain_cache[batch_idx]

    def _chain_lr_scale(self, total_iter):
        """Exponential decay factor for the chain-position learning rate.

        Returns 1.0 when chain_decay_rate>=1 (no decay). Otherwise decays as
        chain_decay_rate**(total_iter - chain_decay_start) once past the start,
        so the fixed-LR Adam step shrinks over the run and the positions settle.
        """
        if self._chain_decay_rate >= 1.0:
            return 1.0
        t = max(0, int(total_iter) - self._chain_decay_start)
        return float(self._chain_decay_rate ** t)

    def _process_chain_grads(self, batch_idx, grads_chain_per_track, total_iter=0):
        """Apply one (decayed) Adam step for per-track chain angles of batch_idx.

        The NaN guard is done on-device with jnp.where (identical result to skipping the
        update on NaN) instead of a Python ``if jnp.any(isnan)`` — the latter forces a
        device->host sync on every track (~33 blocking syncs/step, a large host cost).
        """
        states = self._get_or_init_chain_state(batch_idx)
        lr_scale = self._chain_lr_scale(total_iter)
        new_states = []
        for state, g in zip(states, grads_chain_per_track):
            bad = jnp.any(jnp.isnan(g))                       # scalar, stays on-device (no sync)
            g_clean = jnp.where(jnp.isnan(g), 0.0, g)         # avoid poisoning Adam moments
            updates, new_opt_state = self._chain_optimizer.update(g_clean, state['opt_state'])
            cand_angles = state['angles'] + lr_scale * updates
            new_angles = jnp.where(bad, state['angles'], cand_angles)
            new_opt = jax.tree_util.tree_map(lambda o, n: jnp.where(bad, o, n), state['opt_state'], new_opt_state)
            new_states.append({'angles': new_angles, 'opt_state': new_opt})
        self._chain_cache[batch_idx] = new_states

    def _solve_chain_gn(self, batch_idx, tracks, ref, log_dedx, parent_ids, max_iter=8):
        """Gauss-Newton / Levenberg-Marquardt solve on this batch's chain coefficients,
        with calibration + dEdx held fixed. Intended for the SPLINE basis where each track
        is only 2K (~6) parameters, so the whole batch's chain vector is small (~6-30) and
        the exact Hessian (jacfwd o jacrev through the sim) + a direct LM step converge in a
        handful of iterations — vs thousands of Adam steps. NOT recommended for the angle
        basis (2*n_chain params/track -> Hessian too large)."""
        from jax.flatten_util import ravel_pytree
        states = self._get_or_init_chain_state(batch_idx)
        coeffs = [s['angles'] for s in states]
        # Static-size unique + fixed ROI so the loss is jittable (as for the L-BFGS geom block).
        _usize = self._geom_usize(batch_idx, tracks)
        _roi = self._batch_padded_roi.get(batch_idx, (256, 256))
        # pure loss(chain_coeffs_list) with calibration + dEdx fixed (data baked in this closure)
        loss_fn = self.compute_loss(tracks, batch_idx, *ref, log_dedx=log_dedx, parent_ids=parent_ids,
                                    chain_angles_per_track=coeffs, return_loss_closure=True,
                                    unique_size=_usize, roi_override=_roi)
        flat0, unflat = ravel_pytree(coeffs)
        n = int(flat0.shape[0])

        def f(x):
            return loss_fn(unflat(x))

        key = (batch_idx, n)
        if not hasattr(self, '_chain_gn_vgh'):
            self._chain_gn_vgh = {}
        if key not in self._chain_gn_vgh:
            _vg = jax.jit(jax.value_and_grad(f))
            # Hessian-vector product: one memory-bounded 2nd-order pass per column. jax.hessian
            # would stack n tangents through the huge FEE tape at once (OOM); assembling the
            # Hessian column-by-column via HVPs keeps peak memory at a single pass.
            _hvp = jax.jit(lambda x, v: jax.jvp(jax.grad(f), (x,), (v,))[1])
            self._chain_gn_vgh[key] = (_vg, _hvp)
        _vg, _hvp = self._chain_gn_vgh[key]

        def _val_grad_hess(x):
            lo, gr = _vg(x)
            eye = np.eye(n, dtype=np.float32)
            Hc = np.stack([np.asarray(_hvp(x, jnp.asarray(eye[i]))) for i in range(n)], axis=1)
            return float(lo), np.asarray(gr), Hc

        x = jnp.asarray(flat0, dtype=jnp.float32)
        loss, g, H = _val_grad_hess(x)
        lam = 1e-2
        for it in range(max_iter):
            Hs = 0.5 * (H + H.T)
            dsc = np.maximum(np.abs(np.diag(Hs)), 1e-9 * max(np.abs(np.diag(Hs)).max(), 1.0))
            accepted = False
            for _bt in range(6):
                try:
                    step = -np.linalg.solve(Hs + lam * np.diag(dsc), g)
                except np.linalg.LinAlgError:
                    lam *= 4.0; continue
                xt = x + jnp.asarray(step, dtype=jnp.float32)
                lt = float(_vg(xt)[0])                      # trial: loss only (cheap)
                if np.isfinite(lt) and lt < loss:
                    x = xt; lam = max(lam / 3.0, 1e-9); accepted = True
                    break
                lam *= 4.0
            if not accepted or np.linalg.norm(step) < 1e-6:
                break
            loss, g, H = _val_grad_hess(x)               # refresh grad+Hessian at accepted point
        new_coeffs = unflat(x)
        self._chain_cache[batch_idx] = [{'angles': new_coeffs[ti]} for ti in range(len(coeffs))]
        return loss

    def _chain_col_map(self):
        fld = self.sim_track_fields
        return {'x': fld.index('x'), 'y': fld.index('y'), 'z': fld.index('z'),
                'xs': fld.index('x_start'), 'ys': fld.index('y_start'), 'zs': fld.index('z_start'),
                'xe': fld.index('x_end'), 'ye': fld.index('y_end'), 'ze': fld.index('z_end'),
                'dx': fld.index('dx')}

    def _solve_chain_ggn(self, batch_idx, tracks, ref, log_dedx, parent_ids, max_iter=6):
        """Matrix-free FIRST-ORDER Gauss-Newton (Fisher/GGN) solve on the chain parameters,
        for BOTH the spline coefficients AND the high-dim segment (angle) chain.

        The llhd loss is a marked Poisson point process, so its Fisher is
            F = sum_c exp(z_c) Jz[c] Jz[c]^T  +  sum_c exp(z_c)/sigma^2 Jq[c] Jq[c]^T
        with z = log-intensity field, q = expected charge. The damped normal equations
        (F + lambda I) dtheta = -g are solved MATRIX-FREE with conjugate gradient, where each
        CG iteration is one GGN-vector product = one jvp + one vjp through the sim (first
        order, no 2nd-order tape, no materialized Jacobian). CG converges in ~(number of
        data-constrained modes) iterations, NOT in n -> works even for the segment basis
        (n = 2*n_chain per track, hundreds of params) where assembling the n x n Fisher is
        impossible. dEdx + calibration held fixed; PSD by construction."""
        from jax.flatten_util import ravel_pytree
        from larndsim.sim_jax import simulate_wfs as _sim_wfs
        states = self._get_or_init_chain_state(batch_idx)
        coeffs = [s['angles'] for s in states]
        meta = self._batch_chain_meta.get(batch_idx)
        if meta is None:
            return  # nothing to solve
        basis = meta.get('basis', 'angle')
        col_map = self._chain_col_map()
        # ref_params keeps the readout-noise fields (nonzero even under --no-noise, which the
        # marginalized FEE needs for 1/sigma); overwrite only the calibration params with the
        # current values (mirrors loss_wrapper_combined). Using current_params directly would
        # carry noise=0 -> ZeroDivisionError.
        phys = self.ref_params.replace(**{k: getattr(self.current_params, k)
                                          for k in self.relevant_params_list})
        sigma = self.loss_strategy.sigma_charge / 1000.0
        _, _up = _sim_wfs(self.ref_params, self.sim_strategy.response, tracks, self.sim_track_fields)
        usize = int(((int(_up.shape[0]) + 127) // 128) * 128)
        roi = self._batch_padded_roi.get(batch_idx, (256, 256))
        dedx_idx = self.sim_track_fields.index('dEdx'); dE_idx = self.sim_track_fields.index('dE')
        dx_idx = self.sim_track_fields.index('dx')

        flat0, unflat = ravel_pytree(coeffs)
        n = int(flat0.shape[0])

        def fields(xflat):
            cl = unflat(xflat)
            if basis == 'spline':
                t, _ = _chain_warp_spline(tracks, cl, meta, col_map)
            else:
                t, _, _ = _chain_warp_vectorized(tracks, cl, meta, col_map)
            if log_dedx is not None and parent_ids is not None:
                safe = jnp.clip(parent_ids, 0, log_dedx.shape[0] - 1)
                dd = jnp.where(parent_ids >= 0, jnp.maximum(jnp.exp(jnp.take(log_dedx, safe)), 1e-3), t[:, dedx_idx])
                t = t.at[:, dedx_idx].set(dd)
                t = t.at[:, dE_idx].set(dd * t[:, dx_idx])
            pred = self.sim_strategy.predict(phys, t, self.sim_track_fields, None,
                                             unique_size=usize, roi_override=roi)
            return pred['hit_prob'], adc2charge(pred['adcs_distrib'], phys)

        def ggn_vp(xflat, v):
            # matrix-free Fisher/GGN-vector product: F v = Jz^T (w . Jz v) + Jq^T (w/sig^2 . Jq v),
            # via jvp (Jz v, Jq v) then vjp (Jz^T .). Never materializes the field Jacobian.
            (z, q), (Jzv, Jqv) = jax.jvp(fields, (xflat,), (v,))
            w = jax.lax.stop_gradient(jnp.exp(z))
            _, vjp_fn = jax.vjp(fields, xflat)
            (gv,) = vjp_fn((w * Jzv, (w / sigma ** 2) * Jqv))
            return gv

        # full-loss value+grad (data + MCS prior) via the existing closure
        loss_fn = self.compute_loss(tracks, batch_idx, *ref, log_dedx=log_dedx, parent_ids=parent_ids,
                                    chain_angles_per_track=coeffs, return_loss_closure=True,
                                    unique_size=usize, roi_override=roi)
        def f(xflat):
            return loss_fn(unflat(xflat))

        _cg_maxiter = int(min(max(20, n // 4), 60))   # CG iters ~ effective rank, capped
        _precond = os.environ.get('LARND_GN_PRECOND', '1') != '0'
        _hutch_k = 6                                    # Hutchinson probes for diag(F)

        def cg_step(xflat, gflat, lam):
            # solve (F + lam I) dtheta = -g, matrix-free, entirely on-device (jitted CG)
            def A(v):
                return ggn_vp(xflat, v) + lam * v
            if _precond:
                # Jacobi (diagonal) preconditioner. CG ran to maxiter (ill-conditioned, not
                # low-rank) so diagonal scaling should cut iterations a lot. diag(F) estimated
                # matrix-free via Hutchinson: E[v ⊙ (F v)] over Rademacher v (k GGN-vps).
                keys = jax.random.split(jax.random.PRNGKey(0), _hutch_k)
                def _probe(acc, k):
                    v = jax.random.rademacher(k, (xflat.shape[0],), dtype=xflat.dtype)
                    return acc + v * ggn_vp(xflat, v), None   # scan (sequential) = memory-bounded
                diag, _ = jax.lax.scan(_probe, jnp.zeros_like(xflat), keys)
                diag = diag / _hutch_k
                Minv = 1.0 / jnp.maximum(jnp.abs(diag) + lam, 1e-8)
                M = lambda r: Minv * r
                dx, _ = jax.scipy.sparse.linalg.cg(A, -gflat, tol=1e-4, maxiter=_cg_maxiter, M=M)
            else:
                dx, _ = jax.scipy.sparse.linalg.cg(A, -gflat, tol=1e-4, maxiter=_cg_maxiter)
            return dx

        key = (batch_idx, n, 'ggn_cg')
        if not hasattr(self, '_chain_ggn_fns'):
            self._chain_ggn_fns = {}
        if key not in self._chain_ggn_fns:
            self._chain_ggn_fns[key] = (jax.jit(jax.value_and_grad(f)),
                                        jax.jit(cg_step, static_argnums=()))
        _vg, _cg = self._chain_ggn_fns[key]

        x = jnp.asarray(flat0, dtype=jnp.float32)

        if os.environ.get('LARND_GN_TIMING'):
            import time as _t
            _fresh = key not in getattr(self, '_gn_timed_keys', set())
            # value_and_grad: 1st call (compile+exec) vs 2nd (exec only)
            t0 = _t.time(); l0, g0 = _vg(x); jax.block_until_ready((l0, g0)); t_vg1 = _t.time() - t0
            t0 = _t.time(); l0, g0 = _vg(x); jax.block_until_ready((l0, g0)); t_vg2 = _t.time() - t0
            # cg_step: 1st (compile+exec) vs 2nd (exec only)
            t0 = _t.time(); d0 = _cg(x, g0, jnp.float32(1e-2)); jax.block_until_ready(d0); t_cg1 = _t.time() - t0
            t0 = _t.time(); d0 = _cg(x, g0, jnp.float32(1e-2)); jax.block_until_ready(d0); t_cg2 = _t.time() - t0
            logger.info(f"[GN-TIMING] batch {batch_idx} n={n} usize={usize} cg_maxiter={_cg_maxiter} fresh_compile={_fresh}: "
                        f"vg compile+exec {t_vg1:.1f}s / exec {t_vg2:.2f}s | "
                        f"cg compile+exec {t_cg1:.1f}s / exec {t_cg2:.2f}s | "
                        f"vg_compile~{t_vg1-t_vg2:.1f}s cg_compile~{t_cg1-t_cg2:.1f}s")
            if not hasattr(self, '_gn_timed_keys'):
                self._gn_timed_keys = set()
            self._gn_timed_keys.add(key)

        loss, g = _vg(x); loss = float(loss)
        lam = 1e-2
        for it in range(max_iter):
            accepted = False
            for _bt in range(6):
                dx = _cg(x, g, jnp.float32(lam))
                xt = x + dx
                lt, gt = _vg(xt); lt = float(lt)
                snorm = float(jnp.linalg.norm(dx))
                if np.isfinite(lt) and lt < loss:
                    x, loss, g = xt, lt, gt; lam = max(lam / 3.0, 1e-9); accepted = True
                    break
                lam *= 4.0
            if not accepted or snorm < 1e-6:
                break
        nc2 = unflat(x)
        self._chain_cache[batch_idx] = [{'angles': nc2[ti]} for ti in range(len(coeffs))]
        return loss

    def _compute_pos_residual_batch(self, batch_idx):
        """Mean 3D distance (cm) between current fitted chain positions and true target positions."""
        ctxs           = self._batch_chain_contexts.get(batch_idx, [])
        states         = self._chain_cache.get(batch_idx, [])
        track_pos_dict = self._batch_true_positions.get(batch_idx, {})
        if not ctxs or not states:
            return None
        dists = []
        for ctx, state in zip(ctxs, states):
            true_pts = track_pos_dict.get(ctx.track_id)
            if true_pts is None:
                continue
            params = np.asarray(state['angles'])
            nc = ctx.n_chain
            if self._chain_basis == 'spline':
                K = _spline_K(ctx.total_len, self._chain_spline_knot_cm)
                u0 = _dir_from_angles(ctx.theta0_i, ctx.phi0_i)
                e1, e2 = _transverse_frame(u0)
                ks = np.arange(nc + 1)
                B = np.sin(np.outer(ks / nc, np.pi * np.arange(1, K + 1)))
                d1 = B @ params[:K]; d2 = B @ params[K:2 * K]
                nominal = ctx.x0_fixed[None, :] + (ks[:, None] * ctx.step_len) * u0[None, :]
                pos_nodes = nominal + d1[:, None] * e1[None, :] + d2[:, None] * e2[None, :]
            else:
                thetas = jnp.array(params[:nc]); phis = jnp.array(params[nc:])
                pos_nodes, _ = _absolute_chain_forward(
                    jnp.array(ctx.x0_fixed), thetas, phis, ctx.step_len)
                pos_nodes = np.asarray(pos_nodes)
            for j, alpha in enumerate(ctx.alpha_mid):
                k = int(np.clip(int(np.floor(alpha * nc)), 0, nc - 1))
                frac = float(alpha * nc - k)
                fitted = pos_nodes[k] + frac * (pos_nodes[k + 1] - pos_nodes[k])
                true   = true_pts[j]  # pre-interpolated at alpha_mid[j]
                dists.append(float(np.linalg.norm(fitted - true)))
        return float(np.mean(dists)) if dists else None

    def validate_static_unique(self, dataloader_sim, batch_idx=0, unique_size=768):
        """Phase-0 validation: confirm static-size unique (jittable) is numerically identical to the
        default dynamic path on a real batch, and that it jits."""
        i = int(batch_idx)
        tracks = jax.device_put(dataloader_sim[i].reshape(-1, len(self.sim_track_fields)))
        gp = self.current_params
        wfs_d, up_d = simulate_wfs(gp, self.response, tracks, self.sim_track_fields)                       # dynamic
        wfs_s, up_s = simulate_wfs(gp, self.response, tracks, self.sim_track_fields, unique_size=unique_size)  # static
        jfn = jax.jit(lambda t: simulate_wfs(gp, self.response, t, self.sim_track_fields, unique_size=unique_size))
        wfs_j, up_j = jfn(tracks); jax.block_until_ready(wfs_j)                                            # jittable?
        up_d = np.asarray(up_d); up_s = np.asarray(up_s); wfs_d = np.asarray(wfs_d); wfs_s = np.asarray(wfs_s)
        d_map = {int(p): wfs_d[k] for k, p in enumerate(up_d) if p >= 0}
        max_diff = 0.0; n_matched = 0; scale = float(np.max(np.abs(wfs_d))) + 1e-9
        for k, p in enumerate(up_s):
            if p >= 0 and int(p) in d_map:
                max_diff = max(max_diff, float(np.max(np.abs(wfs_s[k] - d_map[int(p)])))); n_matched += 1
        n_d = int((up_d >= 0).sum()); n_s = int((up_s >= 0).sum()); rel = max_diff/scale
        logger.info(f"[STATIC-UNIQUE-VAL] batch {i}: default {n_d} pix | static {n_s} pix (size {unique_size}) | "
                    f"matched {n_matched}/{n_d} | max wfs |diff| {max_diff:.3e} (rel {rel:.2e}, wfs scale {scale:.1f}) | "
                    f"jitted: OK ({'IDENTICAL' if (n_matched == n_d and rel < 1e-4) else 'MISMATCH'})")
        return {'n_default': n_d, 'n_static': n_s, 'matched': n_matched, 'max_diff': max_diff, 'rel': rel}

    def _build_chain_geom_vg(self, batch_idx):
        """Jitted value_and_grad of the fixed-pixel geometry loss w.r.t. per-track chain angles.

        Reuses the profile_local_geometry pattern (simulate_wfs_static -> simulate_probabilistic ->
        MSE vs target expected-ADC), but warps the tracks by the chain angles (_chain_warp_vectorized)
        + MCS prior, instead of a rigid 6-DOF transform. No dynamic jnp.unique -> fully jittable.
        Cached by (pixel shape, roi sizes, T, max_nc) so it recompiles only on shape change.
        """
        if not hasattr(self, '_chain_geom_vg_cache'):
            self._chain_geom_vg_cache = {}
        fixed_pixels = self._batch_fixed_pixels[batch_idx]
        psn, pln = self._batch_padded_roi[batch_idx]
        ctxs = self._batch_chain_contexts[batch_idx]
        if batch_idx not in self._batch_chain_meta:
            self._batch_chain_meta[batch_idx] = _build_chain_meta(ctxs)
        meta = self._batch_chain_meta[batch_idx]
        f = self.sim_track_fields
        col_map = {'x': f.index('x'), 'y': f.index('y'), 'z': f.index('z'),
                   'xs': f.index('x_start'), 'ys': f.index('y_start'), 'zs': f.index('z_start'),
                   'xe': f.index('x_end'), 'ye': f.index('y_end'), 'ze': f.index('z_end'), 'dx': f.index('dx')}
        key = (fixed_pixels.shape, int(psn), int(pln), meta['T'], meta['max_nc'])
        if key in self._chain_geom_vg_cache:
            return self._chain_geom_vg_cache[key]
        response = self.response; sim_fields = self.sim_track_fields
        mcs_w = float(self._mcs_prior_weight); _psn = int(psn); _pln = int(pln)

        def _loss(chain_angles_list, tracks_, gp_, fixed_, tgt_pix_, tgt_adc_):
            warped, thetas, phis = _chain_warp_vectorized(tracks_, chain_angles_list, meta, col_map)
            wfs = simulate_wfs_static(gp_, response, warped, sim_fields, fixed_)
            wfs_padded = pad_to_closest_multiple(wfs, dims_to_pad=(0,), multiple=128, pad_value=0.0, pad_front=True)
            adcs_distrib, _px, _py, tp, _ev = simulate_probabilistic(
                gp_, wfs_padded, fixed_, padded_small_nb=_psn, padded_large_nb=_pln)
            expected = jnp.sum(jnp.exp(tp) * adcs_distrib, axis=1)
            idx = jnp.searchsorted(fixed_, tgt_pix_); idxs = jnp.clip(idx, 0, fixed_.shape[0]-1)
            matched = (jnp.take(fixed_, idxs) == tgt_pix_)[:, None]
            guess = jnp.where(matched, jnp.take(expected, idxs, axis=0), 0.0)
            mse = jnp.mean((guess - tgt_adc_)**2)
            mcs = _chain_mcs_vectorized(thetas, phis, meta, mcs_w)
            return mse + mcs

        vg = jax.jit(jax.value_and_grad(_loss))
        self._chain_geom_vg_cache[key] = vg
        return vg

    def solve_chain_geometry_lbfgs(self, batch_idx, tracks, gp, max_iter=60, resid_fn=None):
        """Optimise one batch's chain angles with scipy L-BFGS-B on the jitted geometry gradient
        (calibration + dEdx fixed). Writes the result to _chain_cache. Returns (scipy_result, resid_hist)."""
        from jax.flatten_util import ravel_pytree
        vg = self._build_chain_geom_vg(batch_idx)
        fixed_pixels = self._batch_fixed_pixels[batch_idx]
        tgt_pix = self._batch_target_pixels[batch_idx]
        tgt_adc = self._batch_target_expected_adc[batch_idx]
        init_angles = [jnp.asarray(s['angles']) for s in self._get_or_init_chain_state(batch_idx)]
        x0, unflat = ravel_pytree(init_angles); x0 = np.asarray(x0, dtype=np.float64)
        hist = []
        def scipy_loss(x):
            angles = unflat(jnp.asarray(x, dtype=jnp.float32))
            l, g = vg(angles, tracks, gp, fixed_pixels, tgt_pix, tgt_adc)
            gflat = np.asarray(ravel_pytree(g)[0], dtype=np.float64)
            if resid_fn is not None:
                hist.append((float(l), resid_fn(angles)))
            return float(l), gflat
        res = minimize(scipy_loss, x0, jac=True, method='L-BFGS-B', options={'maxiter': max_iter})
        final = unflat(jnp.asarray(res.x, dtype=jnp.float32))
        self._chain_cache[batch_idx] = [{'angles': final[ti]} for ti in range(len(init_angles))]
        return res, hist

    def test_jitted_geom_lbfgs(self, dataloader_sim, target, batch_idx=0, unique_size=768, max_iter=60, out_path=None):
        """Jitted LLHD geometry L-BFGS on one batch: the PROVEN loss (compute_loss) made jittable via
        static-size unique, driven by scipy L-BFGS-B. Records residual convergence + timing."""
        import time as _time, pickle as _pickle
        from jax.flatten_util import ravel_pytree
        self.precompute_static_pixels(dataloader_sim, target)
        i = int(batch_idx)
        selected = jax.device_put(dataloader_sim[i].reshape(-1, len(self.sim_track_fields)))
        evts = self.get_batch_global_event_ids(dataloader_sim, i, selected)
        this_tgt = jax.device_put(target[i].reshape(-1, len(self.tgt_track_fields)))
        ref = self.get_simulated_target(this_tgt, i, evts, regen=False)
        ctxs = self._batch_chain_contexts[i]; true_pos = self._batch_true_positions.get(i, {})
        init_angles = [jnp.asarray(s['angles']) for s in self._get_or_init_chain_state(i)]
        _ok = [ti for ti, ctx in enumerate(ctxs) if ctx.track_id in true_pos]
        def _sag(ctx):
            tr = np.asarray(true_pos[ctx.track_id]); ch = (tr[-1]-tr[0])/(np.linalg.norm(tr[-1]-tr[0])+1e-9)
            t = tr - tr[0]; return float((np.linalg.norm(t-np.outer(t@ch, ch), axis=1)).max()*1e4)
        sags = [_sag(ctxs[ti]) for ti in _ok]
        def resid_fn(angles):
            out = []
            for ti in _ok:
                ctx = ctxs[ti]; nc = ctx.n_chain; a = np.asarray(angles[ti]); th = a[:nc]; ph = a[nc:]
                u = np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)], 1); u /= (np.linalg.norm(u, axis=1, keepdims=True)+1e-12)
                x0 = np.asarray(ctx.x0_fixed); nodes = np.concatenate([x0[None], x0[None] + ctx.step_len*np.cumsum(u, 0)], 0)
                al = np.asarray(ctx.alpha_mid); k = np.clip(np.floor(al*nc).astype(int), 0, nc-1); fr = al*nc - k
                fit = nodes[k] + fr[:, None]*(nodes[k+1]-nodes[k]); tr = np.asarray(true_pos[ctx.track_id]); m = min(len(fit), len(tr))
                out.append(float(np.mean(np.linalg.norm(fit[:m]-tr[:m], axis=1))*1e4))
            return out
        # jittable proven LLHD loss (static unique)
        loss_fn = self.compute_loss(selected, i, *ref, log_dedx=None, parent_ids=None,
                                    chain_angles_per_track=init_angles, return_loss_closure=True, unique_size=int(unique_size))
        _ = float(loss_fn(init_angles))                     # eager warm-up -> caches predict ROI counts
        vg = jax.jit(jax.value_and_grad(loss_fn))
        _t0 = _time.time(); jax.block_until_ready(vg(init_angles)); compile_t = _time.time() - _t0   # compile
        x0, unflat = ravel_pytree(init_angles); x0 = np.asarray(x0, dtype=np.float64); hist = []
        def scipy_loss(x):
            angles = unflat(jnp.asarray(x, dtype=jnp.float32))
            l, g = vg(angles); gflat = np.asarray(ravel_pytree(g)[0], dtype=np.float64)
            hist.append((float(l), resid_fn(angles)))
            return float(l), gflat
        t0 = _time.time(); res = minimize(scipy_loss, x0, jac=True, method='L-BFGS-B', options={'maxiter': max_iter}); dt = _time.time() - t0
        final = unflat(jnp.asarray(res.x, dtype=jnp.float32))
        self._chain_cache[i] = [{'angles': final[ti]} for ti in range(len(init_angles))]
        out = {'batch': i, 'sagittas_um': sags, 'hist': hist, 'solve_time_s': dt, 'compile_s': compile_t,
               'n_iter': int(res.nit), 'n_feval': int(res.nfev), 'unique_size': int(unique_size)}
        out_path = out_path or f'fit_result/jitted_geom_lbfgs_b{i}.pkl'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f: _pickle.dump(out, f)
        r0 = hist[0][1] if hist else [float('nan')]; r1 = hist[-1][1] if hist else [float('nan')]
        logger.info(f"[JITTED-GEOM-LBFGS] batch {i}: {res.nit} iters ({res.nfev} feval) solve {dt:.1f}s (compile {compile_t:.1f}s) | "
                    f"sagittas {['%.0f'%s for s in sags]} | resid {['%.0f'%x for x in r0]}->{['%.0f'%x for x in r1]} µm. Saved {out_path}")
        return out

    def profile_geom_compile(self, dataloader_sim, target, batch_idx=0):
        """Localise the geometry-loss compile cost: time forward vs value_and_grad compile across
        unique_size (Nvalues = fee_paths_scaling * Npix), and report the actual active-pixel count."""
        import time as _time
        self.precompute_static_pixels(dataloader_sim, target)
        i = int(batch_idx)
        selected = jax.device_put(dataloader_sim[i].reshape(-1, len(self.sim_track_fields)))
        evts = self.get_batch_global_event_ids(dataloader_sim, i, selected)
        this_tgt = jax.device_put(target[i].reshape(-1, len(self.tgt_track_fields)))
        ref = self.get_simulated_target(this_tgt, i, evts, regen=False)
        init_angles = [jnp.asarray(s['angles']) for s in self._get_or_init_chain_state(i)]
        _roi = self._batch_padded_roi.get(i, (256, 256))
        _wfs, _up = simulate_wfs(self.current_params, self.response, selected, self.sim_track_fields)
        n_active = int((np.asarray(_up) >= 0).sum())
        try: scaling = int(self.current_params.fee_paths_scaling)
        except Exception: scaling = -1
        logger.info(f"[COMPILE-PROFILE] batch {i}: n_active_pixels={n_active}, fixed_pixels={int(self._batch_fixed_pixels[i].shape[0])}, fee_paths_scaling={scaling}")
        for usize in [128, 256, 512, 1024]:
            if usize < n_active + 8:
                continue
            loss_fn = self.compute_loss(selected, i, *ref, log_dedx=None, parent_ids=None,
                                        chain_angles_per_track=init_angles, return_loss_closure=True,
                                        unique_size=usize, roi_override=_roi)
            jf = jax.jit(loss_fn)
            t0 = _time.time(); jax.block_until_ready(jf(init_angles)); tf = _time.time() - t0
            vg = jax.jit(jax.value_and_grad(loss_fn))
            t0 = _time.time(); jax.block_until_ready(vg(init_angles)); tvg = _time.time() - t0
            logger.info(f"[COMPILE-PROFILE] unique_size={usize} (Nvalues={scaling*usize}): forward {tf:.1f}s | value_and_grad {tvg:.1f}s | backward +{tvg-tf:.1f}s")
        return None

    _GEOM_TARGET_KEYS = ('adcs', 'pixel_x', 'pixel_y', 'pixel_z', 'ticks',
                         'hit_prob', 'event', 'pixel_id')
    _CHAIN_META_STATIC = ('T', 'max_nc', 'ncs_np')

    def _geom_usize(self, i, tracks):
        """Static unique-pixel size for batch i: the MEASURED unique count rounded up to a
        128 multiple (the old 2x-fixed_pixels heuristic oversized Nvalues and caused OOM)."""
        if self._geom_lbfgs_unique_size > 0:
            return int(self._geom_lbfgs_unique_size)
        if not hasattr(self, '_geom_usize_cache'):
            self._geom_usize_cache = {}
        if i not in self._geom_usize_cache:
            from larndsim.sim_jax import simulate_wfs as _sim_wfs
            _, up = _sim_wfs(self.ref_params, self.sim_strategy.response, tracks, self.sim_track_fields)
            self._geom_usize_cache[i] = int(((int(up.shape[0]) + 127) // 128) * 128)
        return self._geom_usize_cache[i]

    def _geom_padded_target(self, ref):
        """Pad the 8 target-ref arrays to a 128-multiple bucket so batches with similar hit
        counts share a jit signature. Pad rows carry pixel_id = -1 and are masked out by the
        LLHD loss (is_relevant_target_hit / pixel_match_valid)."""
        n = int(ref[0].shape[0])
        L = int(((n + 127) // 128) * 128)
        out = {}
        for k, arr in zip(self._GEOM_TARGET_KEYS, ref):
            fill = -1 if k == 'pixel_id' else 0
            a = np.asarray(arr)
            if L > n:
                a = np.pad(a, [(0, L - n)] + [(0, 0)] * (a.ndim - 1),
                           mode='constant', constant_values=fill)
            out[k] = jax.device_put(a)
        return out, L

    def _solve_geom_block(self, i, tracks, ref, init_angles, max_iter):
        """Jitted L-BFGS geometry solve for batch i (dEdx fixed; calibration is a traced arg so
        the jitted grad uses the CURRENT norm_params). Returns the solved per-track angle list.

        Data-as-args: tracks, the (padded) target arrays and the chain-meta arrays are TRACED
        arguments, so the jitted value_and_grad is cached by shape SIGNATURE, not by batch —
        every batch with the same (usize, roi, shapes, chain structure) shares one compiled
        program instead of paying a fresh ~170 s XLA compile per batch."""
        from jax.flatten_util import ravel_pytree
        usize = self._geom_usize(i, tracks)
        _roi = self._batch_padded_roi.get(i, (256, 256))

        # Chain meta: static structure (ints / python-side arrays) + traced data arrays
        if i not in self._batch_chain_meta:
            self._batch_chain_meta[i] = _build_chain_meta(self._batch_chain_contexts.get(i, []))
        meta = self._batch_chain_meta[i]
        meta_static = {k: meta[k] for k in self._CHAIN_META_STATIC}
        meta_arrays = {k: v for k, v in meta.items() if k not in self._CHAIN_META_STATIC}

        tgt, tgt_len = self._geom_padded_target(ref)
        ncs = tuple(int(x) for x in meta['ncs_np'])
        sig = (usize, tuple(_roi), tuple(tracks.shape), tgt_len, ncs,
               int(meta_arrays['seg_rows'].shape[0]))

        vg = self._geom_vg_cache.get(sig)
        if vg is None:
            loss_fn = self.compute_loss(tracks, i, *ref, log_dedx=None, parent_ids=None,
                                        chain_angles_per_track=init_angles, return_loss_closure=True,
                                        unique_size=usize, roi_override=_roi)

            def _raw(angles, normp, tracks_in, tgt_in, meta_arr_in):
                meta_full = dict(meta_static)
                meta_full.update(meta_arr_in)
                return loss_fn(angles, normp,
                               data_over={'tracks': tracks_in, 'target': tgt_in, 'meta': meta_full})

            vg = jax.jit(jax.value_and_grad(_raw, argnums=0))
            self._geom_vg_cache[sig] = vg
            logger.info(f"[GEOM-LBFGS] new jit signature {sig} (total {len(self._geom_vg_cache)})")
            if len(self._geom_vg_cache) == 1:
                # One-time numerical identity check: eager baked-closure loss vs the jitted
                # data-as-args path must agree (padding rows are masked by the loss).
                l_ref = float(loss_fn(init_angles, self.norm_params))
                l_jit = float(vg(init_angles, self.norm_params, tracks, tgt, meta_arrays)[0])
                logger.info(f"[GEOM-LBFGS] identity check: eager {l_ref:.6f} vs data-as-args "
                            f"{l_jit:.6f} (rel {abs(l_jit-l_ref)/max(abs(l_ref),1e-9):.2e})")

        x0, unflat = ravel_pytree(init_angles); x0 = np.asarray(x0, dtype=np.float64)
        def sl(x):
            a = unflat(jnp.asarray(x, dtype=jnp.float32))
            l, g = vg(a, self.norm_params, tracks, tgt, meta_arrays)
            return float(l), np.asarray(ravel_pytree(g)[0], dtype=np.float64)
        res = minimize(sl, x0, jac=True, method='L-BFGS-B', options={'maxiter': max_iter})
        final = unflat(jnp.asarray(res.x, dtype=jnp.float32))
        return [final[ti] for ti in range(len(init_angles))]

    def test_chain_geometry_lbfgs(self, dataloader_sim, target, batch_idx=0, max_iter=60, out_path=None):
        """Standalone validation of the jitted-loss + scipy-L-BFGS-B chain geometry solver on one batch.
        Records residual convergence + wall time (compare vs the eager L-BFGS benchmark)."""
        import time as _time, pickle as _pickle
        self.precompute_static_pixels(dataloader_sim, target)
        i = int(batch_idx)
        tracks = jax.device_put(dataloader_sim[i].reshape(-1, len(self.sim_track_fields)))
        gp = self.current_params
        ctxs = self._batch_chain_contexts[i]; true_pos = self._batch_true_positions.get(i, {})
        _ok = [ti for ti, ctx in enumerate(ctxs) if ctx.track_id in true_pos]
        def _sag(ctx):
            tr = np.asarray(true_pos[ctx.track_id]); ch = (tr[-1]-tr[0])/(np.linalg.norm(tr[-1]-tr[0])+1e-9)
            t = tr - tr[0]; return float((np.linalg.norm(t-np.outer(t@ch, ch), axis=1)).max()*1e4)
        sags = [_sag(ctxs[ti]) for ti in _ok]
        def resid_fn(angles):
            out = []
            for ti in _ok:
                ctx = ctxs[ti]; nc = ctx.n_chain; a = np.asarray(angles[ti]); th = a[:nc]; ph = a[nc:]
                u = np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)], 1); u /= (np.linalg.norm(u, axis=1, keepdims=True)+1e-12)
                x0 = np.asarray(ctx.x0_fixed); nodes = np.concatenate([x0[None], x0[None] + ctx.step_len*np.cumsum(u, 0)], 0)
                al = np.asarray(ctx.alpha_mid); k = np.clip(np.floor(al*nc).astype(int), 0, nc-1); fr = al*nc - k
                fit = nodes[k] + fr[:, None]*(nodes[k+1]-nodes[k]); tr = np.asarray(true_pos[ctx.track_id]); m = min(len(fit), len(tr))
                out.append(float(np.mean(np.linalg.norm(fit[:m]-tr[:m], axis=1))*1e4))
            return out
        t0 = _time.time()
        res, hist = self.solve_chain_geometry_lbfgs(i, tracks, gp, max_iter=max_iter, resid_fn=resid_fn)
        dt = _time.time() - t0
        out = {'batch': i, 'sagittas_um': sags, 'hist': hist, 'time_s': dt,
               'n_iter': int(res.nit), 'n_feval': int(res.nfev), 'success': bool(res.success)}
        out_path = out_path or f'fit_result/chain_geom_lbfgs_b{i}.pkl'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f: _pickle.dump(out, f)
        r0 = hist[0][1] if hist else [float('nan')]; r1 = hist[-1][1] if hist else [float('nan')]
        logger.info(f"[CHAIN-LBFGS-TEST] batch {i}: {res.nit} iters ({res.nfev} feval) in {dt:.1f}s | "
                    f"sagittas {['%.0f'%s for s in sags]} | resid {['%.0f'%x for x in r0]}->{['%.0f'%x for x in r1]} µm. Saved {out_path}")
        return out

    def benchmark_chain_optimizers(self, dataloader_sim, target, batch_idx=0, n_steps=400, adam_lr=None, out_path=None):
        """Benchmark chain-angle optimizers (Adam vs L-BFGS) on ONE batch, deterministically.

        Position-only: calibration and dEdx are held fixed; only the chain angles move.
        Records per-step loss, geometry residual (µm, vs true positions), and wall time for
        both optimizers. Use sim_seed_strategy='same' so the loss is deterministic (required
        for L-BFGS line search). Implements method #4 (second-order optimizer) for comparison.
        """
        import optax
        import time as _time
        import pickle as _pickle
        self.precompute_static_pixels(dataloader_sim, target)
        i = int(batch_idx)
        tracks_np = dataloader_sim[i].reshape(-1, len(self.sim_track_fields))
        selected_tracks_sim = jax.device_put(tracks_np)
        evts_sim = self.get_batch_global_event_ids(dataloader_sim, i, selected_tracks_sim)
        this_target = jax.device_put(target[i].reshape(-1, len(self.tgt_track_fields)))
        (ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks,
         ref_hit_prob, ref_event, ref_pixel_id) = self.get_simulated_target(this_target, i, evts_sim, regen=False)
        ctxs = self._batch_chain_contexts[i]
        true_pos = self._batch_true_positions.get(i, {})
        init_angles = [jnp.asarray(s['angles']) for s in self._get_or_init_chain_state(i)]

        # Pure differentiable loss(chain_angles)->scalar for this batch (calib + dEdx fixed).
        loss_fn = self.compute_loss(
            selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks,
            ref_hit_prob, ref_event, ref_pixel_id, log_dedx=None, parent_ids=None,
            chain_angles_per_track=init_angles, return_loss_closure=True)
        # Eager (like the normal fit path): the inner sim functions are already @jit'd, but
        # the outer loss must NOT be jitted whole (the sim's jnp.unique needs a dynamic size).
        loss_j = loss_fn
        vg_j = jax.value_and_grad(loss_fn)

        def geom_resid(angles):
            ds = []
            for ti, ctx in enumerate(ctxs):
                nc = ctx.n_chain
                a = np.asarray(angles[ti]); th = a[:nc]; ph = a[nc:]
                u = np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)], 1)
                u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
                x0 = np.asarray(ctx.x0_fixed)
                nodes = np.concatenate([x0[None], x0[None] + ctx.step_len*np.cumsum(u, 0)], 0)
                al = np.asarray(ctx.alpha_mid); k = np.clip(np.floor(al*nc).astype(int), 0, nc-1); fr = al*nc - k
                fit = nodes[k] + fr[:, None]*(nodes[k+1]-nodes[k])
                if ctx.track_id not in true_pos: continue
                tr = np.asarray(true_pos[ctx.track_id]); m = min(len(fit), len(tr))
                ds.append(np.mean(np.linalg.norm(fit[:m]-tr[:m], axis=1)))
            return float(np.mean(ds))*1e4 if ds else float('nan')

        # per-track residual + sagitta (to show curved-track convergence specifically)
        _ok = [ti for ti, ctx in enumerate(ctxs) if ctx.track_id in true_pos]
        def _sag(ctx):
            tr = np.asarray(true_pos[ctx.track_id]); ch = (tr[-1]-tr[0])/(np.linalg.norm(tr[-1]-tr[0])+1e-9)
            t = tr - tr[0]; return float((np.linalg.norm(t-np.outer(t@ch, ch), axis=1)).max()*1e4)
        _sags = [_sag(ctxs[ti]) for ti in _ok]
        def per_track_resid(angles):
            out = []
            for ti in _ok:
                ctx = ctxs[ti]; nc = ctx.n_chain; a = np.asarray(angles[ti]); th = a[:nc]; ph = a[nc:]
                u = np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)], 1)
                u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
                x0 = np.asarray(ctx.x0_fixed); nodes = np.concatenate([x0[None], x0[None] + ctx.step_len*np.cumsum(u, 0)], 0)
                al = np.asarray(ctx.alpha_mid); k = np.clip(np.floor(al*nc).astype(int), 0, nc-1); fr = al*nc - k
                fit = nodes[k] + fr[:, None]*(nodes[k+1]-nodes[k]); tr = np.asarray(true_pos[ctx.track_id]); m = min(len(fit), len(tr))
                out.append(float(np.mean(np.linalg.norm(fit[:m]-tr[:m], axis=1))*1e4))
            return out

        jax.block_until_ready(vg_j(init_angles))  # warm-up / compile
        results = {'batch': i, 'per_track_sagitta_um': _sags}

        # ---- Adam (current method) ----
        lr = adam_lr if adam_lr is not None else self._chain_lr
        adam = optax.adam(lr); params = init_angles; st = adam.init(params); hist = []; hist_pt = []
        for step in range(n_steps):
            t0 = _time.time()
            loss, grad = vg_j(params)
            upd, st = adam.update(grad, st, params)
            params = optax.apply_updates(params, upd)
            jax.block_until_ready(params)
            hist.append((step, float(loss), geom_resid(params), _time.time()-t0)); hist_pt.append(per_track_resid(params))
        results['adam'] = {'lr': float(lr), 'hist': hist, 'hist_pt': hist_pt}

        # ---- L-BFGS (method #4), hand-rolled with an EAGER Armijo line search ----
        # optax.lbfgs jit-traces the objective in its line search, which the sim's dynamic
        # jnp.unique forbids; a plain eager L-BFGS avoids that entirely.
        from jax.flatten_util import ravel_pytree
        x, unflat = ravel_pytree(init_angles)
        x = np.asarray(x, dtype=np.float64)
        def _f(xn):
            return float(loss_fn(unflat(jnp.asarray(xn, dtype=jnp.float32))))
        def _fg(xn):
            v, g = vg_j(unflat(jnp.asarray(xn, dtype=jnp.float32)))
            return float(v), np.asarray(ravel_pytree(g)[0], dtype=np.float64)
        m_hist = 10; s_list = []; y_list = []; rho_list = []
        f_x, g_x = _fg(x); hist = []; hist_pt = []
        for step in range(n_steps):
            t0 = _time.time()
            # two-loop recursion (use _j, not i: i is the batch index in this scope)
            q = g_x.copy(); alpha = [0.0]*len(s_list)
            for _j in range(len(s_list)-1, -1, -1):
                alpha[_j] = rho_list[_j]*np.dot(s_list[_j], q); q = q - alpha[_j]*y_list[_j]
            gamma = (np.dot(s_list[-1], y_list[-1])/np.dot(y_list[-1], y_list[-1])) if s_list else 1.0
            r = gamma*q
            for _j in range(len(s_list)):
                beta = rho_list[_j]*np.dot(y_list[_j], r); r = r + s_list[_j]*(alpha[_j]-beta)
            p = -r
            gtp = float(np.dot(g_x, p))
            if gtp >= 0:  # not a descent direction -> steepest descent fallback
                p = -g_x; gtp = float(np.dot(g_x, p))
            # Armijo backtracking
            t = 1.0; c1 = 1e-4; x_new = x; f_new = f_x
            for _ls in range(25):
                x_new = x + t*p; f_new = _f(x_new)
                if f_new <= f_x + c1*t*gtp: break
                t *= 0.5
            f_n, g_new = _fg(x_new)
            s = x_new - x; yv = g_new - g_x; ys = float(np.dot(yv, s))
            if ys > 1e-10:
                s_list.append(s); y_list.append(yv); rho_list.append(1.0/ys)
                if len(s_list) > m_hist:
                    s_list.pop(0); y_list.pop(0); rho_list.pop(0)
            x, f_x, g_x = x_new, f_n, g_new
            _ang = unflat(jnp.asarray(x, dtype=jnp.float32))
            hist.append((step, float(f_x), geom_resid(_ang), _time.time()-t0)); hist_pt.append(per_track_resid(_ang))
        results['lbfgs'] = {'hist': hist, 'hist_pt': hist_pt}

        results['track_sagitta'] = []
        for ti, ctx in enumerate(ctxs):
            if ctx.track_id in true_pos:
                tr = np.asarray(true_pos[ctx.track_id]); chord = tr[-1]-tr[0]
                ch = chord/(np.linalg.norm(chord)+1e-9); t = tr - tr[0]
                results['track_sagitta'].append((ti, float(np.linalg.norm(t-np.outer(t@ch, ch), axis=1).max()*1e4), int(ctx.n_chain)))

        out_path = out_path or f'fit_result/chain_opt_bench_b{i}.pkl'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            _pickle.dump(results, f)
        a_t = sum(h[3] for h in results['adam']['hist']); l_t = sum(h[3] for h in results['lbfgs']['hist'])
        logger.info(f"[BENCH] batch {i}: Adam resid {results['adam']['hist'][0][2]:.0f}->{results['adam']['hist'][-1][2]:.0f}µm in {a_t:.1f}s | "
                    f"L-BFGS resid {results['lbfgs']['hist'][0][2]:.0f}->{results['lbfgs']['hist'][-1][2]:.0f}µm in {l_t:.1f}s. Saved {out_path}")
        return results

    def _lbfgs_minimize(self, loss_fn, init_pytree, max_steps=80, grad_tol=1e-4,
                        plateau_tol=1e-6, plateau_k=8, resid_fn=None):
        """Eager L-BFGS (two-loop recursion + Armijo backtracking) with early stopping.
        Returns (final_pytree, per-step history, n_steps). No jit (sim uses dynamic jnp.unique)."""
        import time as _time
        from jax.flatten_util import ravel_pytree
        vg = jax.value_and_grad(loss_fn)
        x, unflat = ravel_pytree(init_pytree); x = np.asarray(x, dtype=np.float64)
        def _f(xn):  return float(loss_fn(unflat(jnp.asarray(xn, dtype=jnp.float32))))
        def _fg(xn):
            v, g = vg(unflat(jnp.asarray(xn, dtype=jnp.float32)))
            return float(v), np.asarray(ravel_pytree(g)[0], dtype=np.float64)
        m = 10; s_list = []; y_list = []; rho_list = []; hist = []
        f_x, g_x = _fg(x); f_prev = f_x; n_plat = 0
        for step in range(max_steps):
            t0 = _time.time()
            if float(np.linalg.norm(g_x)) < grad_tol: break
            q = g_x.copy(); alpha = [0.0]*len(s_list)
            for i in range(len(s_list)-1, -1, -1):
                alpha[i] = rho_list[i]*np.dot(s_list[i], q); q = q - alpha[i]*y_list[i]
            gamma = (np.dot(s_list[-1], y_list[-1])/np.dot(y_list[-1], y_list[-1])) if s_list else 1.0
            r = gamma*q
            for i in range(len(s_list)):
                beta = rho_list[i]*np.dot(y_list[i], r); r = r + s_list[i]*(alpha[i]-beta)
            p = -r; gtp = float(np.dot(g_x, p))
            if gtp >= 0: p = -g_x; gtp = float(np.dot(g_x, p))
            t = 1.0; c1 = 1e-4; x_new = x; f_new = f_x
            for _ls in range(25):
                x_new = x + t*p; f_new = _f(x_new)
                if f_new <= f_x + c1*t*gtp: break
                t *= 0.5
            f_n, g_new = _fg(x_new); s = x_new - x; yv = g_new - g_x; ys = float(np.dot(yv, s))
            if ys > 1e-10:
                s_list.append(s); y_list.append(yv); rho_list.append(1.0/ys)
                if len(s_list) > m: s_list.pop(0); y_list.pop(0); rho_list.pop(0)
            x, f_x, g_x = x_new, f_n, g_new
            rr = resid_fn(unflat(jnp.asarray(x, dtype=jnp.float32))) if resid_fn else float('nan')
            hist.append((step, float(f_x), rr, _time.time()-t0))
            n_plat = n_plat + 1 if abs(f_prev - f_x) < plateau_tol*max(1.0, abs(f_prev)) else 0
            f_prev = f_x
            if n_plat >= plateau_k: break
        return unflat(jnp.asarray(x, dtype=jnp.float32)), hist, len(hist)

    def run_lbfgs_position_fit(self, dataloader_sim, target, max_steps=80, out_path=None):
        """End-to-end position-only geometry fit with L-BFGS + early stopping (one pass over
        batches, calibration fixed). Reports total time and geometry residual vs Adam."""
        import time as _time, pickle as _pickle
        self.precompute_static_pixels(dataloader_sim, target)
        def batch_resid(i, angles_list):
            ds = []
            ctxs = self._batch_chain_contexts[i]; tp = self._batch_true_positions.get(i, {})
            for ti, ctx in enumerate(ctxs):
                nc = ctx.n_chain; a = np.asarray(angles_list[ti]); th = a[:nc]; ph = a[nc:]
                u = np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)], 1)
                u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
                x0 = np.asarray(ctx.x0_fixed); nodes = np.concatenate([x0[None], x0[None] + ctx.step_len*np.cumsum(u, 0)], 0)
                al = np.asarray(ctx.alpha_mid); k = np.clip(np.floor(al*nc).astype(int), 0, nc-1); fr = al*nc - k
                fit = nodes[k] + fr[:, None]*(nodes[k+1]-nodes[k])
                if ctx.track_id not in tp: continue
                tr = np.asarray(tp[ctx.track_id]); m = min(len(fit), len(tr))
                ds.append((np.mean(np.linalg.norm(fit[:m]-tr[:m], axis=1)), (np.linalg.norm((tr[:m]-tr[0])-np.outer((tr[:m]-tr[0])@((tr[-1]-tr[0])/(np.linalg.norm(tr[-1]-tr[0])+1e-9)), (tr[-1]-tr[0])/(np.linalg.norm(tr[-1]-tr[0])+1e-9)), axis=1)).max()))
            return ds
        t0_all = _time.time(); per_batch = []; init_ds = []; final_ds = []
        for i in range(len(dataloader_sim)):
            if i not in self._batch_chain_contexts: continue
            selected = jax.device_put(dataloader_sim[i].reshape(-1, len(self.sim_track_fields)))
            evts = self.get_batch_global_event_ids(dataloader_sim, i, selected)
            this_tgt = jax.device_put(target[i].reshape(-1, len(self.tgt_track_fields)))
            ref = self.get_simulated_target(this_tgt, i, evts, regen=False)
            init_angles = [jnp.asarray(s['angles']) for s in self._get_or_init_chain_state(i)]
            loss_fn = self.compute_loss(selected, i, *ref, log_dedx=None, parent_ids=None,
                                        chain_angles_per_track=init_angles, return_loss_closure=True)
            final_angles, hist, nst = self._lbfgs_minimize(loss_fn, init_angles, max_steps=max_steps)
            self._chain_cache[i] = [{'angles': final_angles[ti]} for ti in range(len(init_angles))]
            fd = [(d[0]*1e4, d[1]*1e4) for d in batch_resid(i, final_angles)]
            final_ds.extend(fd)
            per_batch.append((i, nst, sum(h[3] for h in hist)))
            logger.info(f"[LBFGS-FIT] batch {i}: {nst} steps, {sum(h[3] for h in hist):.1f}s")
            # incremental save (robust to timeout)
            _ip = out_path or 'fit_result/lbfgs_position_fit.pkl'
            os.makedirs(os.path.dirname(_ip), exist_ok=True)
            _R = np.array([d[0] for d in final_ds]); _S = np.array([d[1] for d in final_ds]); _c = _S > 500
            with open(_ip, 'wb') as _pf:
                _pickle.dump({'total_time': _time.time()-t0_all, 'per_batch': per_batch, 'partial': True,
                              'median': float(np.median(_R)), 'curved': float(_R[_c].mean()) if _c.any() else 0.0,
                              'straight': float(_R[~_c].mean()), 'n_tracks': int(len(_R))}, _pf)
        total_t = _time.time() - t0_all
        R = np.array([d[0] for d in final_ds]); S = np.array([d[1] for d in final_ds]); c = S > 500
        out = {'total_time': total_t, 'per_batch': per_batch,
               'median': float(np.median(R)), 'curved': float(R[c].mean()) if c.any() else 0.0,
               'straight': float(R[~c].mean()), 'n_tracks': int(len(R))}
        out_path = out_path or 'fit_result/lbfgs_position_fit.pkl'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f: _pickle.dump(out, f)
        logger.info(f"[LBFGS-FIT] DONE {out['n_tracks']} tracks in {total_t:.0f}s | median={out['median']:.0f}µm "
                    f"curved={out['curved']:.0f}µm straight={out['straight']:.0f}µm. Saved {out_path}")
        return out

    def fit(self, dataloader_sim, target, epochs=300, iterations=None, save_freq=10, print_freq=1):

        self.prepare_fit()

        if self.fit_track_positions or self.fit_chain_positions:
            self.precompute_static_pixels(dataloader_sim, target)

        start_iter = self.total_iter if self.resume_from is not None else 0

        if iterations is not None:
            pbar_total = iterations
        else:
            pbar_total = len(dataloader_sim) * epochs

        total_iter = start_iter
        target_iter = start_iter + iterations if iterations is not None else None

        # self.prepare_fit()

        # if iterations is not None:
        #     pbar_total = iterations
        # else:
        #     pbar_total = len(dataloader_sim) * epochs

        if not self.read_target:
            # If explicit number of iterations, scale epochs accordingly
            if len(dataloader_sim) != len(target):
                raise Exception("Sim and target inputs do not match in size. Panic.")

        if iterations is not None:
            epochs = iterations // len(dataloader_sim) + 1

        # Pre-fetch per-segment parent IDs for dEdx fitting
        if self.fit_dedx:
            _ds = getattr(dataloader_sim, 'dataset', None)
            if _ds is not None and hasattr(_ds, 'get_parent_segment_ids'):
                logger.info("Pre-fetching parent segment IDs for per-segment dEdx fitting")
                _dedx_field_idx = self.sim_track_fields.index('dEdx') if 'dEdx' in self.sim_track_fields else None
                for _bi in range(len(dataloader_sim)):
                    _batch_raw = dataloader_sim[_bi]  # populate TracksDataset cache, shape [max_steps, n_fields]
                    _pids = _ds.get_parent_segment_ids(_bi)
                    self._batch_parent_ids[_bi] = np.asarray(_pids, dtype=np.int32)
                    # Collect true (input) dEdx: per original segment, take the first sub-step's value
                    if _dedx_field_idx is not None:
                        _batch_flat = np.asarray(_batch_raw).reshape(-1, len(self.sim_track_fields))
                        _raw_pids = np.asarray(_pids)
                        _n_orig = int(_raw_pids[_raw_pids >= 0].max()) + 1 if np.any(_raw_pids >= 0) else 0
                        _true_dedx = np.zeros(_n_orig, dtype=np.float32)
                        for _seg in range(_n_orig):
                            _rows = np.where(_raw_pids == _seg)[0]
                            if len(_rows) > 0:
                                _true_dedx[_seg] = float(_batch_flat[_rows[0], _dedx_field_idx])
                        self._batch_true_dedx[_bi] = _true_dedx
            else:
                logger.warning("TracksDataset does not support get_parent_segment_ids; disabling fit_dedx")
                self.fit_dedx = False

        # The training loop
        # total_iter = 0
        terminate_fit = False
        with tqdm(total=pbar_total) as pbar:
            for epoch in range(epochs):
                if terminate_fit:
                    break
                logger.info(f"epoch {epoch}")

                # shuffle batches
                indices = np.arange(len(dataloader_sim))
                if self.shuffle_bt:
                    rng = np.random.default_rng(self.shuffle_seed + epoch)
                    rng.shuffle(indices)

                for i_bt, i in enumerate(indices):
                    start_time = time()

                    # sim
                    selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.sim_track_fields))
                    self.validate_track_batch_event_ids(selected_tracks_bt_sim, self.sim_track_fields, f"sim batch {i}")
                    selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)

                    if self.fit_track_positions:
                        if i not in self._batch_aligned_tracks:
                            self._batch_aligned_tracks[i] = selected_tracks_sim
                        
                        if total_iter % self.track_alignment_freq == 0:
                            logger.info(f"Aligning track geometry locally for batch {i} (iter {total_iter})...")
                            self._batch_aligned_tracks[i] = self.profile_local_geometry(i, self._batch_aligned_tracks[i], target, self.current_params)
                        
                        selected_tracks_sim = self._batch_aligned_tracks[i]

                    evts_sim = self.get_batch_global_event_ids(dataloader_sim, i, selected_tracks_sim)

                    # target
                    if not self.read_target:
                        selected_tracks_bt_tgt = target[i].reshape(-1, len(self.tgt_track_fields))
                        self.validate_track_batch_event_ids(selected_tracks_bt_tgt, self.tgt_track_fields, f"target batch {i}")
                        this_target = jax.device_put(selected_tracks_bt_tgt)
                    else:
                        this_target = target
                    ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = self.get_simulated_target(this_target, i, evts_sim, regen=False)

                    # dEdx local state for this batch (gated by dedx_start_iter / dedx_freeze_iter)
                    _dedx_active  = self.fit_dedx and (total_iter >= self.dedx_start_iter)
                    _dedx_frozen  = _dedx_active and (self.dedx_freeze_iter is not None) and (total_iter >= self.dedx_freeze_iter)
                    _use_dedx     = _dedx_active  # still inject dEdx values into the loss even when frozen

                    # Reset calibration Adam state once at the exact moment dEdx activates.
                    # Guard with i_bt % sz_mini_bt == 0 so the reset never fires mid-window
                    # when mini-batch gradient averaging is in use (sz_mini_bt > 1).
                    _at_window_start = (i_bt % max(1, self.sz_mini_bt) == 0)
                    if _use_dedx and not self._dedx_adam_reset_done and _at_window_start:
                        logger.info(
                            f"dEdx activated at iteration {total_iter}: resetting calibration Adam state."
                        )
                        self.opt_state = self.optimizer.init(
                            extract_relevant_params(self.norm_params, self.relevant_params_list)
                        )
                        self._dedx_adam_reset_done = True

                    # Reset calibration Adam state once when dEdx is frozen so the
                    # optimiser adapts cleanly to the new (dEdx-free gradient) landscape.
                    if _dedx_frozen and not self._dedx_freeze_adam_reset_done and _at_window_start:
                        logger.info(
                            f"dEdx frozen at iteration {total_iter}: resetting calibration Adam state "
                            f"for fine-tuning phase."
                        )
                        self.opt_state = self.optimizer.init(
                            extract_relevant_params(self.norm_params, self.relevant_params_list)
                        )
                        self._dedx_freeze_adam_reset_done = True

                    if _use_dedx and i in self._batch_parent_ids:
                        _parent_ids = jax.device_put(jnp.asarray(self._batch_parent_ids[i], dtype=jnp.int32))
                        _n_orig = int(jnp.max(_parent_ids).item()) + 1
                        _log_dedx, _ = self._get_or_init_dedx_state(i, _n_orig)
                    else:
                        _parent_ids = None
                        _log_dedx = None

                    # Chain position local state for this batch
                    # Allow position-only mode (fit_dedx=False) as well as joint mode
                    _chain_active = (self.fit_chain_positions and
                                     total_iter >= self._chain_start_iter and
                                     total_iter % self._chain_update_freq == 0 and
                                     i in self._batch_chain_contexts and
                                     (_log_dedx is not None or not self.fit_dedx))
                    if _chain_active:
                        _chain_states   = self._get_or_init_chain_state(i)
                        _chain_angles_pt = [s['angles'] for s in _chain_states]
                    else:
                        _chain_angles_pt = None

                    # During the warm-up phase (fit_dedx=True but not yet active),
                    # replace the dEdx and dE columns with the prior centre so no
                    # true dEdx leaks into the calibration gradient.
                    if self.fit_dedx and not _use_dedx:
                        _dedx_idx = self.sim_track_fields.index('dEdx')
                        _dE_idx   = self.sim_track_fields.index('dE')
                        _dx_idx   = self.sim_track_fields.index('dx')
                        _loc      = float(self.dedx_student_loc)
                        selected_tracks_sim = selected_tracks_sim.at[:, _dedx_idx].set(_loc)
                        selected_tracks_sim = selected_tracks_sim.at[:, _dE_idx].set(
                            _loc * selected_tracks_sim[:, _dx_idx]
                        )

                    if self.fit_dedx and not _use_dedx and total_iter == 0:
                        logger.info(f"dEdx fitting deferred: will activate at iteration {self.dedx_start_iter}")

                    # loss
                    loss_val, grads, _grads_dedx, _grads_chain, aux, _, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, epoch=epoch, with_loss=True, with_grad=True, log_dedx=_log_dedx, parent_ids=_parent_ids, chain_angles_per_track=_chain_angles_pt)

                    # Apply per-segment dEdx gradient immediately after every step.
                    if _grads_dedx is not None and _log_dedx is not None and not _dedx_frozen:
                        self._process_dedx_grads(i, _grads_dedx, _log_dedx)

                    # Geometry block: per-track chain-angle update. Default = Adam (unchanged path);
                    # 'lbfgs' = jitted L-BFGS geometry solve (calibration held at current, as a traced
                    # arg) alternating with the calibration gradient step applied below.
                    if _chain_active and _grads_chain is not None:
                        if self._geom_optimizer == 'lbfgs':
                            _ref = (ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id)
                            _init = [s['angles'] for s in self._chain_cache[i]]
                            _new = self._solve_geom_block(i, selected_tracks_sim, _ref, _init, self._geom_lbfgs_steps)
                            self._chain_cache[i] = [{'angles': a} for a in _new]
                        elif self._geom_optimizer in ('gn', 'ggn'):
                            # per-batch Gauss-Newton on the (low-dim, spline) chain coeffs:
                            # 'gn' = exact Hessian (2nd-order, slow), 'ggn' = Fisher (1st-order, fast)
                            _ref = (ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id)
                            _solver = self._solve_chain_ggn if self._geom_optimizer == 'ggn' else self._solve_chain_gn
                            _solver(i, selected_tracks_sim, _ref, _log_dedx, _parent_ids,
                                    max_iter=self._geom_lbfgs_steps)
                        else:
                            self._process_chain_grads(i, _grads_chain, total_iter)
                        if total_iter % self._pos_residual_freq == 0:
                            _pos_res = self._compute_pos_residual_batch(i)
                            if _pos_res is not None:
                                self.training_history.setdefault('pos_residual_iter', []).append(_pos_res)
                        # env-gated: snapshot this batch's chain angles for oscillation viz
                        if os.environ.get('LARND_SNAPSHOT'):
                            self.training_history.setdefault('chain_snapshots', []).append(
                                (int(total_iter), int(i),
                                 [np.asarray(_s['angles']).tolist() for _s in self._chain_cache[i]]))

                    # Accumulate gradients if sz_mini_bt > 1
                    if self.sz_mini_bt > 1:
                        # Initialize at the start of each mini-batch window
                        if i_bt % self.sz_mini_bt == 0:
                            summed_grads = jax.tree_util.tree_map(jnp.zeros_like, grads)
                            bt_loss = []
                            window_start_time = time()

                        summed_grads = self.add_grads(summed_grads, grads)
                        bt_loss.append(loss_val.item())

                        # Perform update at the end of the window
                        if (i_bt + 1) % self.sz_mini_bt == 0 or (i_bt == len(indices) - 1):
                            n_samples = len(bt_loss)
                            update_grads = jax.tree_util.tree_map(lambda x: x / n_samples, summed_grads)
                            avg_loss = np.mean(bt_loss)
 
                            modified_grads = self.process_grads(update_grads)
                            stop_time = time()

                            for param in self.relevant_params_list:
                                self.training_history[param+"_grad"].append(modified_grads[param].item())
                            self.training_history['step_time'].append(stop_time - window_start_time)

                            self.training_history['losses_iter'].append(avg_loss) # type: ignore
                            aux_serializable = {k: serialize_value(v) for k, v in aux.items()} if aux is not None else {}
                            self.training_history['aux_iter'].append(aux_serializable) # type: ignore
                            for param in self.relevant_params_list:
                                if type(getattr(self.current_params, param)) == float:
                                    self.training_history[param + '_iter'].append(getattr(self.current_params, param))
                                else:
                                    self.training_history[param + '_iter'].append(getattr(self.current_params, param).item())

                            self.training_history['size_history'].append(get_size_history())

                            if jax.devices()[0].platform == 'gpu':
                                self.training_history['memory'].append(jax.devices("gpu")[0].memory_stats())

                            if iterations is not None:
                                if total_iter % print_freq == 0:
                                    for param in self.relevant_params_list:
                                        logger.info(f"Iter {total_iter}: {param} {getattr(self.current_params,param)} {modified_grads[param]}")

                            total_iter += 1
                            self.total_iter = total_iter
                            pbar.update(1)

                            # Checkpointing and termination should only happen after an update
                            if iterations is not None:
                                if total_iter > 0 and total_iter % save_freq == 0:
                                    self.save_checkpoint(total_iter)
                                    if os.path.exists(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl'):
                                        os.remove(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl')
                            
                            if target_iter is not None:
                                if total_iter >= target_iter:
                                    self.save_checkpoint(total_iter)
                                    if os.path.exists(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl'):
                                        os.remove(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl')
                                    if os.path.exists(self.target_dir):
                                        shutil.rmtree(self.target_dir, ignore_errors=True)
                                    terminate_fit = True
                                    break

                    else:
                        modified_grads = self.process_grads(grads)
                        stop_time = time()

                        for param in self.relevant_params_list:
                            self.training_history[param+"_grad"].append(modified_grads[param].item())
                        self.training_history['step_time'].append(stop_time - start_time)

                        self.training_history['losses_iter'].append(loss_val.item()) # type: ignore
                        aux_serializable = {k: serialize_value(v) for k, v in aux.items()} if aux is not None else {}
                        self.training_history['aux_iter'].append(aux_serializable) # type: ignore
                        for param in self.relevant_params_list:
                            if type(getattr(self.current_params, param)) == float:
                                self.training_history[param + '_iter'].append(getattr(self.current_params, param))
                            else:
                                self.training_history[param + '_iter'].append(getattr(self.current_params, param).item())

                        self.training_history['size_history'].append(get_size_history())
                        if jax.devices()[0].platform == 'gpu':
                            self.training_history['memory'].append(jax.devices("gpu")[0].memory_stats())

                        if iterations is not None:
                            if total_iter % print_freq == 0:
                                logger.info(f"Iter {total_iter}: {param} {getattr(self.current_params,param)} {modified_grads[param]}")
                        
                        total_iter += 1
                        self.total_iter = total_iter
                        pbar.update(1)

                        # Checkpointing and termination should only happen after an update
                        if iterations is not None:
                            if total_iter > 0 and total_iter % save_freq == 0:
                                self.save_checkpoint(total_iter)
                                if os.path.exists(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl'):
                                    os.remove(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl')
                        
                        if target_iter is not None:
                            if total_iter >= target_iter:
                                self.save_checkpoint(total_iter)
                                if os.path.exists(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl'):
                                    os.remove(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl')
                                if os.path.exists(self.target_dir):
                                    shutil.rmtree(self.target_dir, ignore_errors=True)
                                terminate_fit = True
                                break
                            
                    # Log dEdx diagnostics unconditionally every inner step
                    if self.fit_dedx:
                        _aux_s = {k: serialize_value(v) for k, v in aux.items()} if aux is not None else {}
                        _dedx_prior = float(_aux_s.get('dedx_prior', 0.0))
                        self.training_history['dedx_prior_iter'].append(_dedx_prior)
                        _dedx_barrier = float(_aux_s.get('dedx_barrier', 0.0))
                        self.training_history['dedx_barrier_iter'].append(_dedx_barrier)
                        _dedx_mean_penalty = float(_aux_s.get('dedx_mean_penalty', 0.0))
                        self.training_history['dedx_mean_penalty_iter'].append(_dedx_mean_penalty)
                        _mean_dedx = float(_aux_s.get('mean_dedx', 0.0))
                        self.training_history['mean_dedx_iter'].append(_mean_dedx)
                        if i in self._dedx_cache:
                            _cur_dedx = np.asarray(jnp.exp(self._dedx_cache[i]['log_dedx']))
                            _true = self._batch_true_dedx.get(i)
                            if _true is not None and len(_true) == len(_cur_dedx):
                                _mae = float(np.mean(np.abs(_cur_dedx - _true)))
                            else:
                                _mae = float(np.mean(np.abs(_cur_dedx - float(self.dedx_student_loc))))
                            self.training_history['dedx_mae_iter'].append(_mae)
                            if _dedx_frozen:
                                logger.debug(f"dEdx FROZEN (iter {total_iter}): MAE vs truth = {_mae:.4f}")
                        else:
                            # Warm-up phase: dEdx not yet active — keep lists aligned with NaN
                            self.training_history['dedx_mae_iter'].append(float('nan'))

            self.save_checkpoint(total_iter)

            if os.path.exists(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl'):
                os.remove(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl')

            if os.path.exists(self.target_dir):
                shutil.rmtree(self.target_dir, ignore_errors=True)

            # libcudart.cudaProfilerStop()


class LikelihoodProfiler(ParamFitter):
    def __init__(self, scan_tgt_nom=False, **kwargs):
        self.scan_tgt_nom = scan_tgt_nom
        super().__init__(**kwargs)
        

    def make_target_sim(self, seed=2, fixed_range=None):
        if self.scan_tgt_nom:
            target_params = {}
            logger.info("Using the fitter in a gradient profile mode. Setting targets to nominal values")
            for param in self.relevant_params_list:
                target_params[param] = ranges[param]['nom']
            self.target_params = self.ref_params.replace(**target_params)
            # Keep target simulation on the original dEdx path regardless of scan mode.
            self.target_params = self.target_params.replace(
                use_dedx_density=False,
                dedx_density_mode="histogram",
            )
            logger.info("Target simulation forced to use_dedx_density=False (original dEdx path)")
            if not self.readout_noise_target:
                logger.info("Not simulating electronics noise for target")
                self.target_params = remove_noise_from_params(self.target_params)
        else:
            super().make_target_sim()

    def fit(self, dataloader_sim, target, iterations=100, **kwargs):

        self.prepare_fit()

        logger.info("Using the fitter in a gradient profile mode. The sampling will follow a regular grid.")
        logger.warning(f"Arguments {kwargs} are ignored in this mode.")

        nb_var_params = len(self.relevant_params_list)
        logger.info(f"{nb_var_params} parameters are to be scanned.")
        nb_steps = iterations
        logger.info(f"Each parameter will be scanned with {nb_steps} steps")

        self.ref_params = self.current_params

        for i in range(len(dataloader_sim)):
            logger.info(f"Batch {i}/{len(target)}")

            # sim
            selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.sim_track_fields))
            self.validate_track_batch_event_ids(selected_tracks_bt_sim, self.sim_track_fields, f"sim batch {i}")
            selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)
            evts_sim = self.get_batch_global_event_ids(dataloader_sim, i, selected_tracks_sim)

            # target
            if not self.read_target:
                selected_tracks_bt_tgt = target[i].reshape(-1, len(self.tgt_track_fields))
                self.validate_track_batch_event_ids(selected_tracks_bt_tgt, self.tgt_track_fields, f"target batch {i}")
                this_target = jax.device_put(selected_tracks_bt_tgt)
            else:
                this_target = target
            ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = self.get_simulated_target(this_target, i, evts_sim, regen=False)

            for param in self.relevant_params_list:
                lower = ranges[param]['down']
                upper = ranges[param]['up']
                param_step = (upper - lower)/(nb_steps - 1)

                for iter in tqdm(range(nb_steps)):
                    start_time = time()
                    new_param_values = {param: lower + iter*param_step}
                    self.current_params = self.ref_params.replace(**new_param_values)
                    loss_val, grads, _, _, aux, _, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, with_loss=True, with_grad=True, use_physical_params=True)

                    stop_time = time()

                    for par in self.relevant_params_list:
                        self.training_history[par+"_grad"].append(getattr(grads, par).item())
                    self.training_history['step_time'].append(stop_time - start_time)

                    self.training_history['losses_iter'].append(loss_val.item()) # type: ignore
                    aux_serializable = {k: serialize_value(v) for k, v in aux.items()} if aux is not None else {}
                    self.training_history['aux_iter'].append(aux_serializable) # type: ignore
                    for par in self.relevant_params_list:
                        #TODO: Need to check why this is not consistent
                        if type(getattr(self.current_params, par)) == float:
                            self.training_history[par + '_iter'].append(getattr(self.current_params, par))
                        else:
                            self.training_history[par + '_iter'].append(getattr(self.current_params, par).item())

                    self.training_history['size_history'].append(get_size_history())
                    if jax.devices()[0].platform == 'gpu':
                        self.training_history['memory'].append(jax.devices("gpu")[0].memory_stats())

                with open(f'fit_result/{self.test_name}/history_{param}_batch{i}_{self.out_label}.pkl', "wb") as f_history:
                    pickle.dump(self.training_history, f_history)
                if os.path.exists(f'fit_result/{self.test_name}/history_{param}_batch{i-1}_{self.out_label}.pkl'):
                    os.remove(f'fit_result/{self.test_name}/history_{param}_batch{i-1}_{self.out_label}.pkl')

        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir, ignore_errors=True)

class MinuitFitter(ParamFitter):
    def __init__(self, separate_fits=True, minimizer_strategy=1, minimizer_tol=1e-5, **kwargs):
        super().__init__(**kwargs)

        self.separate_fits = separate_fits
        self.minimizer_strategy = minimizer_strategy
        self.minimizer_tol = minimizer_tol

    def configure_minimizer(self, loss_wrapper, grad_wrapper=None,):
        self.minimizer = iminuit.Minuit(
            loss_wrapper,
            [getattr(self.current_params, param) for param in self.relevant_params_list],
            grad=grad_wrapper,
            name=self.relevant_params_list
        )
        for param in self.relevant_params_list:
            lower = ranges[param]['down']
            upper = ranges[param]['up']

            self.minimizer.limits[param] = (lower, upper)
            self.minimizer.errors[param] = (upper - lower)/10.
            self.minimizer.fixed[param] = False
        
        self.minimizer.strategy = self.minimizer_strategy  # 0, 1 or 2. Maybe, on 0, it doesn't use the grad func? Try out
        self.minimizer.errordef = 1  # definition of "1 sigma": 0.5 for NLL, 1 for chi2
        self.minimizer.tol = (  # stopping value, EDM < tol
            self.minimizer_tol / 0.002 / 1  # iminuit multiplies by default with 0.002
        )
        self.minimizer.print_level = 3  # 0, 1 or 2. Verbosity level

    def prepare_fit(self):
        self.training_history["minuit_result"] = []
        return super().prepare_fit()

    def push_minuit_result(self, result):
        result_dict = {
            "fval": result.fval,
            "edm": result.fmin.edm,
            "covariance": result.covariance,
            "params": result.values.to_dict(),
            "errors": result.errors.to_dict(),
            "valid": result.valid,
        }
        self.training_history["minuit_result"].append(result_dict)
        if jax.devices()[0].platform == 'gpu':
            self.training_history['memory'].append(jax.devices("gpu")[0].memory_stats())

    def fit(self, dataloader_sim, target, **kwargs):
        self.prepare_fit()
        logger.info("Using the fitter in a Minuit mode.")
        logger.warning(f"Arguments {kwargs} are ignored in this mode.")

        logger.info(f"Running in {'separate' if self.separate_fits else 'joint'} fit mode")

        def get_target(self, i, evts_sim, target):
            if not self.read_target:
                selected_tracks_bt_tgt = target[i].reshape(-1, len(self.tgt_track_fields))
                this_target = jax.device_put(selected_tracks_bt_tgt)
            else:
                this_target = target
            ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = self.get_simulated_target(this_target, i, evts_sim, regen=False)
            return ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id

        if self.separate_fits:
            for i in range(len(dataloader_sim)):
                logger.info(f"Batch {i}/{len(dataloader_sim)}")
                start_time = time()

                # sim
                selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.sim_track_fields))
                self.validate_track_batch_event_ids(selected_tracks_bt_sim, self.sim_track_fields, f"sim batch {i}")
                selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)
                evts_sim = self.get_batch_global_event_ids(dataloader_sim, i, selected_tracks_sim)

                # target
                if not self.read_target:
                    selected_tracks_bt_tgt = target[i].reshape(-1, len(self.tgt_track_fields))
                    self.validate_track_batch_event_ids(selected_tracks_bt_tgt, self.tgt_track_fields, f"target batch {i}")
                    this_target = jax.device_put(selected_tracks_bt_tgt)
                else:
                    this_target = target
                ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = self.get_simulated_target(this_target, i, evts_sim, regen=False)

                def loss_wrapper(args): # type: ignore
                    # Update the current params with the new values
                    self.current_params = self.current_params.replace(**{key: args[i] for i, key in enumerate(self.relevant_params_list)})
                    loss_val, _, _, _, _, _, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, with_grad=False, use_physical_params=True)
                    return loss_val

                def grad_wrapper(args): # type: ignore
                    # Update the current params with the new values
                    self.current_params = self.current_params.replace(**{key: args[i] for i, key in enumerate(self.relevant_params_list)})
                    _, grads, _, _, _, _, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, with_loss=False, use_physical_params=True)
                    return [getattr(grads, key) for key in self.relevant_params_list]

                self.configure_minimizer(loss_wrapper, grad_wrapper)
                result = self.minimizer.migrad()

                stop_time = time()
                self.training_history['step_time'].append(stop_time - start_time)
                self.push_minuit_result(result)

        else:
            # Joint fit
            def loss_wrapper(args):
                logger.debug(f"Loss wrapper called with args: {args}")
                # Update the current params with the new values

                self.current_params = self.current_params.replace(**{key: args[i] for i, key in enumerate(self.relevant_params_list)})
                tot_loss = 0
                for i in range(len(dataloader_sim)):
                    # sim
                    selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.sim_track_fields))
                    self.validate_track_batch_event_ids(selected_tracks_bt_sim, self.sim_track_fields, f"sim batch {i}")
                    selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)
                    evts_sim = self.get_batch_global_event_ids(dataloader_sim, i, selected_tracks_sim)

                    # target
                    ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = get_target(self, i, evts_sim, target)

                    loss_val, _, _, _, _, _, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, with_grad=False, with_loss=True, use_physical_params=True)
                    tot_loss += loss_val # type: ignore
                logger.debug(f"Total loss: {tot_loss}")
                return tot_loss
            
            def grad_wrapper(args):
                logger.debug(f"Grad wrapper called with args: {args}")
                # Update the current params with the new values
                self.current_params = self.current_params.replace(**{key: args[i] for i, key in enumerate(self.relevant_params_list)})
                tot_grad = [0 for _ in range(len(self.relevant_params_list))]
                for i in range(len(dataloader_sim)):
                    # sim
                    selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.sim_track_fields))
                    self.validate_track_batch_event_ids(selected_tracks_bt_sim, self.sim_track_fields, f"sim batch {i}")
                    selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)
                    evts_sim = self.get_batch_global_event_ids(dataloader_sim, i, selected_tracks_sim)

                    # target
                    ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = get_target(self, i, evts_sim, target)

                    _, grads, _, _, _, _, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, with_loss=False, use_physical_params=True)
                    tot_grad = [getattr(grads, key) + tot_grad[i] for i, key in enumerate(self.relevant_params_list)]
                logger.debug(f"Average gradient: {[g/len(dataloader_sim) for g in tot_grad]}")
                return [g for g in tot_grad]

            self.configure_minimizer(loss_wrapper, grad_wrapper)
            result = self.minimizer.migrad()
            self.push_minuit_result(result)
        
        with open(f'fit_result/{self.test_name}/history_minuit_{self.out_label}.pkl', "wb") as f_history:
            pickle.dump(self.training_history, f_history)


class HessianCalculator(ParamFitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not len(self.set_init_params) > 0:
            raise ValueError("Remember to set the initial parameter values for Hessian calculation!")

    def fit(self, dataloader_sim, target, **kwargs):

        self.prepare_fit()

        logger.info("Using the fitter for Hessian matrix calculation.")
        logger.warning(f"Arguments {kwargs} are ignored in this mode.")

        self.ref_params = self.current_params

        self.training_history['n_hit'] = []
        for i in range(len(dataloader_sim)):
            logger.info(f"Batch {i}/{len(target)}")

            # sim
            selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.sim_track_fields))
            self.validate_track_batch_event_ids(selected_tracks_bt_sim, self.sim_track_fields, f"sim batch {i}")
            selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)
            evts_sim = self.get_batch_global_event_ids(dataloader_sim, i, selected_tracks_sim)

            # target
            if not self.read_target:
                selected_tracks_bt_tgt = target[i].reshape(-1, len(self.tgt_track_fields))
                self.validate_track_batch_event_ids(selected_tracks_bt_tgt, self.tgt_track_fields, f"target batch {i}")
                this_target = jax.device_put(selected_tracks_bt_tgt)
            else:
                this_target = target
            ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = self.get_simulated_target(this_target, i, evts_sim, regen=False)
            n_hit = np.sum(ref_hit_prob)
            self.training_history['n_hit'].append(n_hit)

            if self.current_mode == 'lut':
                loss_val, grads, _, _, aux, hess, aux_hess = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, with_loss=True, with_grad=True, with_hess=True)
                self.training_history['hessian'].append(format_hessian(hess))
                self.training_history['gradient'].append(format_hessian(grads))
                self.training_history['losses_iter'].append(loss_val.item())

            else:
                hess, aux = jax.jacfwd(jax.jacrev(params_loss_parametrized, (0), has_aux=True), has_aux=True)(self.fit_params, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, selected_tracks_sim, self.sim_track_fields, rngkey=i, loss_fn=self.loss_fn, **self.loss_fn_kw)
                self.training_history['hessian'].append(format_hessian(hess))

            with open(f'fit_result/{self.test_name}/history_batch{i}_{self.out_label}.pkl', "wb") as f_history:
                pickle.dump(self.training_history, f_history)
            if os.path.exists(f'fit_result/{self.test_name}/history_batch{i-1}_{self.out_label}.pkl'):
                os.remove(f'fit_result/{self.test_name}/history_batch{i-1}_{self.out_label}.pkl')

        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir, ignore_errors=True)


class GaussNewtonCalibFitter(GradientDescentFitter):
    """Full-batch damped Newton / Levenberg-Marquardt on the global calibration params.

    Prototype to test whether a curvature-aware step beats Adam on the (small, ill-conditioned)
    5-parameter calibration block. Intended for ground-truth tracks + calibration-only fitting
    (no dEdx, no chain positions), so the only unknowns are the 5 global params.

    Each iteration accumulates loss, gradient g and Hessian H over ALL batches at the current
    normalized params, then takes a damped step  (H + lambda*diag(H)) dtheta = -g  with
    Marquardt-style adaptive damping (backtracking on the full-batch loss). Because the Hessian
    may be indefinite far from the optimum (non-smooth argsort/cummax/erf in the FEE), the
    diagonal LM damping keeps the solve well-posed.
    """

    def __init__(self, gn_max_iter=40, gn_lambda0=1e-2, gn_lambda_up=4.0,
                 gn_lambda_down=3.0, gn_tol=1e-7, gn_max_backtrack=8,
                 gn_curvature='exact', gn_validate=False, gn_degeneracy=False,
                 gn_warmup_iters=0, gn_resume_joint=None, **kwargs):
        super().__init__(**kwargs)
        self.gn_max_iter = gn_max_iter
        self.gn_lambda0 = gn_lambda0
        self.gn_lambda_up = gn_lambda_up
        self.gn_lambda_down = gn_lambda_down
        self.gn_tol = gn_tol
        self.gn_max_backtrack = gn_max_backtrack
        # 'exact'  -> full Hessian via jacfwd(jacrev) over the norm_params pytree (2nd-order graph)
        # 'ggn'    -> Fisher/Generalized-Gauss-Newton via first-order Jacobians of the
        #             (log-intensity, expected-charge) output fields; PSD by construction.
        self.gn_curvature = gn_curvature
        self.gn_validate = gn_validate
        self.gn_degeneracy = gn_degeneracy
        self.gn_warmup_iters = int(gn_warmup_iters)
        self.gn_resume_joint = gn_resume_joint
        self._fisher_cache = {}

    def _preload(self, dataloader_sim, target):
        cache = []
        for i in range(len(dataloader_sim)):
            st = jax.device_put(dataloader_sim[i].reshape(-1, len(self.sim_track_fields)))
            self.validate_track_batch_event_ids(st, self.sim_track_fields, f"sim batch {i}")
            evts = self.get_batch_global_event_ids(dataloader_sim, i, st)
            tt = jax.device_put(target[i].reshape(-1, len(self.tgt_track_fields)))
            self.validate_track_batch_event_ids(tt, self.tgt_track_fields, f"target batch {i}")
            ref = self.get_simulated_target(tt, i, evts, regen=False)
            cache.append((i, st, ref))
        return cache

    def _load_joint_checkpoint(self, path, dataloader_sim, target):
        """Restore a joint-fit (calibration + dEdx + chain positions) checkpoint and freeze
        its nuisances: per-segment log-dEdx and per-track chain angles become FIXED inputs
        to the loss, so GN polishes ONLY the 5 calibration params against the same
        objective the joint fit optimized. Chain contexts are rebuilt deterministically
        from the input data (the checkpoint stores them only partially)."""
        with open(path, 'rb') as f:
            h = pickle.load(f)
        self.norm_params = restore_param_state(self.ref_params, h['norm_params_state'])
        self.update_params()

        if self.fit_chain_positions:
            self.precompute_static_pixels(dataloader_sim, target)

        dedx = h.get('dedx_cache') or {}
        chain = h.get('chain_cache') or {}
        pids = h.get('batch_parent_ids') or {}
        self._joint_fixed = {}
        for i in sorted(set(list(dedx.keys()) + list(chain.keys()))):
            ld = jnp.asarray(dedx[i]['log_dedx']) if i in dedx else None
            pi = jnp.asarray(np.asarray(pids[i]), dtype=jnp.int32) if i in pids else None
            an = [jnp.asarray(s['angles']) for s in chain[i]] if i in chain else None
            # cross-check restored angles against rebuilt chain contexts
            if an is not None and i in self._batch_chain_contexts:
                ncs = [int(c.n_chain) for c in self._batch_chain_contexts[i]]
                got = [int(a.shape[0]) for a in an]
                if [2 * n for n in ncs] != got:
                    raise ValueError(f"batch {i}: checkpoint chain angles {got} do not match "
                                     f"rebuilt contexts {[2*n for n in ncs]} — input data/config mismatch")
            self._joint_fixed[i] = (ld, pi, an)
        logger.info(f"[GN-JOINT] restored {path}: calibration params + dEdx for {len(dedx)} "
                    f"batches + chain angles for {len(chain)} batches (all frozen)")
        logger.info("[GN-JOINT] calibration at checkpoint: "
                    + "  ".join(f"{k}={float(getattr(self.current_params,k)):.5g}" for k in self.relevant_params_list))

    def _nuis(self, i):
        """Frozen nuisances (log_dedx, parent_ids, chain angles) for batch i, if resuming
        from a joint checkpoint; empty otherwise."""
        if not hasattr(self, '_joint_fixed'):
            return {}
        ld, pi, an = self._joint_fixed.get(i, (None, None, None))
        out = {}
        if ld is not None and pi is not None:
            out['log_dedx'] = ld
            out['parent_ids'] = pi
        if an is not None:
            out['chain_angles_per_track'] = an
        return out

    def _full_batch_loss(self, cache):
        tot = 0.0
        for (i, st, ref) in cache:
            lv, _, _, _, _, _, _ = self.compute_loss(
                st, i, *ref, with_loss=True, with_grad=False, **self._nuis(i))
            tot += float(lv)
        return tot

    def _x5(self):
        return jnp.array([float(getattr(self.norm_params, k)) for k in self.relevant_params_list],
                         dtype=jnp.float32)

    def _get_fisher_fn(self, i, tracks):
        """Build (and cache) a jitted Fisher/GGN function for batch i.

        The llhd loss is a marked Poisson point process:
            NLL = sum_cells exp(z) - sum_obs z  (+ Gaussian charge marks)
        with z = log-intensity field (prediction['hit_prob']). Its Fisher information is
            F = sum_c exp(z_c) J_z[c] J_z[c]^T + sum_c exp(z_c)/sigma^2 J_q[c] J_q[c]^T
        where J_z = d z/d theta, J_q = d qhat/d theta (qhat = adc2charge(adcs_distrib)).
        Only FIRST-order Jacobians are needed (5 forward-mode passes on a 5-vector),
        the result is PSD by construction, and the graph compiles like a gradient
        rather than a Hessian.
        """
        rl = self.relevant_params_list
        n = len(rl)
        sigma = self.loss_strategy.sigma_charge / 1000.0
        roi = self._roi_padded_cache.get((i, tracks.shape[0]))
        if roi is None:
            raise RuntimeError(f"ROI cache empty for batch {i}; run compute_loss once first")
        # Use the running global-max ROI so all batches share the same static ROI sizes
        # (mirrors compute_loss), minimising the number of distinct jit signatures.
        if hasattr(self, '_roi_global_max'):
            gs, gl = self._roi_global_max
            roi = (max(roi[0], gs), max(roi[1], gl))

        # Static-size unique (jittable): measure the true unique-pixel count eagerly once,
        # then round up to a 128 multiple. Calibration params don't change the track
        # geometry, so the count is stable across iterations.
        from larndsim.sim_jax import simulate_wfs as _sim_wfs
        _, _up = _sim_wfs(self.ref_params, self.sim_strategy.response, tracks, self.sim_track_fields)
        usize = int(((int(_up.shape[0]) + 127) // 128) * 128)

        # Data-as-args: `tracks` is a TRACED argument, so every batch with the same
        # (usize, roi, tracks.shape) signature shares ONE compiled executable. dataio
        # pads all batches to a global max shape, so this collapses the per-batch
        # compile storm (32 x ~155s) to one compile per unique-pixel bucket (~2).
        cache_key = (usize, roi, tuple(tracks.shape))
        if cache_key in self._fisher_cache:
            return self._fisher_cache[cache_key]

        def to_phys(x5):
            upd = {k: x5[j] for j, k in enumerate(rl)}
            norm = self.norm_params.replace(**upd)
            if self.normalization_scheme == "divide":
                new_phys = {k: getattr(norm, k) * getattr(self.params_normalization, k) for k in rl}
            else:
                new_phys = {k: map_norm_to_phys(getattr(norm, k), k, scheme=self.normalization_scheme,
                                                scale=self._norm_scale_for_scheme()) for k in rl}
            return self.ref_params.replace(**new_phys)

        def fields_fn(x5, tracks_in):
            phys = to_phys(x5)
            pred = self.sim_strategy.predict(phys, tracks_in, self.sim_track_fields, None,
                                             unique_size=usize, roi_override=roi)
            z = pred['hit_prob']
            qhat = adc2charge(pred['adcs_distrib'], phys)
            return z, qhat

        def fisher(x5, tracks_in):
            z, _ = fields_fn(x5, tracks_in)
            Jz, Jq = jax.jacfwd(fields_fn, argnums=0)(x5, tracks_in)
            w = jnp.exp(jax.lax.stop_gradient(z)).reshape(-1)
            Jz2 = Jz.reshape(-1, n)
            Jq2 = Jq.reshape(-1, n)
            F = Jz2.T @ (w[:, None] * Jz2) + Jq2.T @ ((w / sigma**2)[:, None] * Jq2)
            return F

        fn = jax.jit(fisher)
        self._fisher_cache[cache_key] = fn
        return fn

    def _full_batch_lgh(self, cache):
        rl = self.relevant_params_list
        n = len(rl)
        tot = 0.0
        g = np.zeros(n)
        H = np.zeros((n, n))
        use_ggn = (self.gn_curvature == 'ggn')
        for (i, st, ref) in cache:
            lv, grads, _, _, _, hess, _ = self.compute_loss(
                st, i, *ref, with_loss=True, with_grad=True, with_hess=not use_ggn,
                **self._nuis(i))
            tot += float(lv)
            g += np.array([float(getattr(grads, k)) for k in rl])
            if use_ggn:
                H += np.asarray(self._get_fisher_fn(i, st)(self._x5(), st))
            else:
                H += np.array([[float(getattr(getattr(hess, ka), kb)) for kb in rl] for ka in rl])
        return tot, g, H

    def _validate_fisher(self, cache):
        """Compare exact Hessian vs Fisher/GGN (accumulated over ALL batches) at the current
        params, with per-batch timings measured on batch 0."""
        from time import time as _t
        rl = self.relevant_params_list
        n = len(rl)
        H = np.zeros((n, n)); F = np.zeros((n, n)); g = np.zeros(n)
        tH = tF_compile = tF = 0.0
        tot_loss = 0.0
        for bidx, (i, st, ref) in enumerate(cache):
            t0 = _t()
            lv, grads, _, _, _, hess, _ = self.compute_loss(
                st, i, *ref, with_loss=True, with_grad=True, with_hess=True)
            if bidx == 0:
                tH = _t() - t0
            tot_loss += float(lv)
            H += np.array([[float(getattr(getattr(hess, ka), kb)) for kb in rl] for ka in rl])
            g += np.array([float(getattr(grads, k)) for k in rl])

            fn = self._get_fisher_fn(i, st)
            t0 = _t()
            Fb = np.asarray(fn(self._x5(), st))    # includes compile on first call
            if bidx == 0:
                tF_compile = _t() - t0
                t0 = _t()
                Fb = np.asarray(fn(self._x5(), st))  # cached execution
                tF = _t() - t0
            F += Fb
        logger.info(f"[GN-VALIDATE] accumulated over {len(cache)} batches; full-batch loss {tot_loss:.2f}")

        Hs = 0.5 * (H + H.T)
        wH = np.linalg.eigvalsh(Hs)
        wF = np.linalg.eigvalsh(F)
        lam = 1e-2
        dH = -np.linalg.solve(Hs + lam * np.diag(np.abs(np.diag(Hs))), g)
        dF = -np.linalg.solve(F + lam * np.diag(np.abs(np.diag(F))), g)
        cos = float(dH @ dF / (np.linalg.norm(dH) * np.linalg.norm(dF) + 1e-30))

        logger.info(f"[GN-VALIDATE] params: {rl}")
        logger.info(f"[GN-VALIDATE] exact-H time {tH:.1f}s | fisher compile+run {tF_compile:.1f}s, cached run {tF:.1f}s")
        logger.info(f"[GN-VALIDATE] eig(H): {np.array2string(wH, precision=3)}")
        logger.info(f"[GN-VALIDATE] eig(F): {np.array2string(wF, precision=3)}")
        logger.info(f"[GN-VALIDATE] neg eig H: {int((wH<0).sum())}, neg eig F: {int((wF<0).sum())} (F must be 0)")
        logger.info(f"[GN-VALIDATE] rel Frobenius |H-F|/|H|: {np.linalg.norm(Hs-F)/np.linalg.norm(Hs):.3f}")
        logger.info(f"[GN-VALIDATE] LM-step cosine(H-step, F-step): {cos:.4f}")
        logger.info(f"[GN-VALIDATE] H diag: {np.array2string(np.diag(Hs), precision=3)}")
        logger.info(f"[GN-VALIDATE] F diag: {np.array2string(np.diag(F), precision=3)}")

        # Parameter covariance from the local curvature: cov = H^-1 (norm space),
        # converted to relative physical errors via d(phys)/d(norm) at the current point.
        try:
            cov = np.linalg.inv(Hs + 1e-9 * np.eye(len(rl)) * np.abs(np.diag(Hs)).max())
            eps = 1e-4
            x0 = np.array([float(getattr(self.norm_params, k)) for k in rl])
            dphys = np.zeros(len(rl))
            for j, k in enumerate(rl):
                p0 = float(map_norm_to_phys(x0[j] - eps, k, scheme=self.normalization_scheme,
                                            scale=self._norm_scale_for_scheme()))
                p1 = float(map_norm_to_phys(x0[j] + eps, k, scheme=self.normalization_scheme,
                                            scale=self._norm_scale_for_scheme()))
                dphys[j] = (p1 - p0) / (2 * eps)
            sig_norm = np.sqrt(np.maximum(np.diag(cov), 0.0))
            phys_now = np.array([float(getattr(self.current_params, k)) for k in rl])
            rel_sig = np.abs(sig_norm * dphys / np.where(phys_now != 0, phys_now, 1.0)) * 100
            corr = cov / np.outer(np.sqrt(np.abs(np.diag(cov))) + 1e-30,
                                  np.sqrt(np.abs(np.diag(cov))) + 1e-30)
            logger.info(f"[GN-VALIDATE] H^-1 relative sigma [%]: "
                        + "  ".join(f"{k}={rel_sig[j]:.2f}" for j, k in enumerate(rl)))
            logger.info(f"[GN-VALIDATE] H^-1 correlation matrix:\n{np.array2string(corr, precision=2)}")

            # PERFECT-KNOWLEDGE DIAGNOSTIC: at the current (= true, when seeded so) params,
            # the Newton step -H^-1 g is the displacement of the loss MINIMUM from this point.
            # If |offset| >> sigma the target's loss minimum is systematically away from truth
            # (an irreducible model/segmentation floor); if |offset| ~ sigma it is consistent
            # with statistical noise (truth IS the minimum, biases are just statistics).
            offset = -np.linalg.solve(Hs, g)                    # norm-space displacement to the min
            rel_off = offset * dphys / np.where(phys_now != 0, phys_now, 1.0) * 100
            signif = np.abs(offset) / np.maximum(sig_norm, 1e-30)
            logger.info(f"[GN-VALIDATE] loss-min OFFSET from truth [phys %]: "
                        + "  ".join(f"{k}={rel_off[j]:+.2f}" for j, k in enumerate(rl)))
            logger.info(f"[GN-VALIDATE] offset significance [|offset|/sigma]: "
                        + "  ".join(f"{k}={signif[j]:.1f}" for j, k in enumerate(rl)))
        except np.linalg.LinAlgError:
            logger.info("[GN-VALIDATE] H not invertible; no covariance")
        return H, F, g

    def _log_iter(self, loss, g):
        rl = self.relevant_params_list
        self.training_history['losses_iter'].append(float(loss))
        for j, k in enumerate(rl):
            val = float(getattr(self.current_params, k))
            self.training_history[k].append(val)
            self.training_history[k + '_iter'].append(val)
            self.training_history[k + '_grad'].append(float(g[j]))
            if len(self.training_history[k + '_target']) == 0:
                self.training_history[k + '_target'].append(float(getattr(self.target_params, k)))

    def degeneracy_study(self, cache):
        """Direct evidence for/against the lifetime-long_diff degeneracy valley.

        (A) loss along the straight (norm-space) path truth -> GN garden-path endpoint:
            flat loss + huge param change = long shallow valley.
        (C) 2D loss grid in (lifetime, long_diff) with other params at truth: a bent
            trough demonstrates the curvature that makes Newton steps overshoot.
        (D) exact Hessian at truth AND at the endpoint: rotation of the flat eigenvector
            between the two points is a second, independent curvature measure.
        Results are dumped to fit_result/<test_name>/degeneracy_<label>.npz.
        """
        rl = self.relevant_params_list
        truth = {k: float(getattr(self.target_params, k)) for k in rl}
        # GN garden-path endpoint (32-batch run, job 31145475), pulled slightly inside
        # the sigmoid bounds so the norm-space mapping stays finite.
        endpoint = {'Ab': 0.7644, 'eField': 0.5008, 'tran_diff': 5.21e-06,
                    'long_diff': 1.05e-06, 'lifetime': 5950.0}
        endpoint = {k: endpoint.get(k, truth[k]) for k in rl}

        def to_norm(vals):
            return {k: float(map_phys_to_norm(v, k, scheme=self.normalization_scheme,
                                              scale=self._norm_scale_for_scheme()))
                    for k, v in vals.items()}

        def set_norm(nvals):
            self.norm_params = self.norm_params.replace(
                **{k: jnp.array(v, dtype=jnp.float32) for k, v in nvals.items()})
            self.update_params()

        def full_H():
            n = len(rl)
            H = np.zeros((n, n)); g = np.zeros(n)
            for (i, st, ref) in cache:
                _, grads, _, _, _, hess, _ = self.compute_loss(
                    st, i, *ref, with_loss=True, with_grad=True, with_hess=True)
                H += np.array([[float(getattr(getattr(hess, ka), kb)) for kb in rl] for ka in rl])
                g += np.array([float(getattr(grads, k)) for k in rl])
            return H, g

        n_t, n_e = to_norm(truth), to_norm(endpoint)

        # (A) line scan
        ts = np.linspace(0.0, 1.0, 25)
        line = np.zeros(len(ts))
        for j, t in enumerate(ts):
            set_norm({k: (1 - t) * n_t[k] + t * n_e[k] for k in rl})
            line[j] = self._full_batch_loss(cache)
            logger.info(f"[DEGEN] line t={t:.3f} loss={line[j]:.2f} "
                        f"lifetime={float(getattr(self.current_params,'lifetime')):.0f} "
                        f"long_diff={float(getattr(self.current_params,'long_diff')):.3e}")

        # (D) Hessians at the two ends
        set_norm(n_t); H_t, g_t = full_H()
        set_norm(n_e); H_e, g_e = full_H()
        for tag, Hx in [('truth', H_t), ('endpoint', H_e)]:
            w, V = np.linalg.eigh(0.5 * (Hx + Hx.T))
            comp = ", ".join(f"{rl[k]}:{V[:,0][k]:+.2f}" for k in np.argsort(-np.abs(V[:, 0])))
            logger.info(f"[DEGEN] H@{tag}: eig={np.array2string(w, precision=3)} flat-vec: {comp}")

        # (C) 2D grid (lifetime x long_diff), others at truth
        lifs = np.linspace(600.0, 5800.0, 15)
        lds = np.linspace(1.2e-6, 9.5e-6, 15)
        grid = np.zeros((len(lifs), len(lds)))
        for a, lt in enumerate(lifs):
            for b, ld in enumerate(lds):
                v = dict(truth); v['lifetime'] = float(lt); v['long_diff'] = float(ld)
                set_norm(to_norm(v))
                grid[a, b] = self._full_batch_loss(cache)
            logger.info(f"[DEGEN] grid row {a+1}/{len(lifs)} (lifetime={lt:.0f}) done; "
                        f"min={grid[a].min():.2f}")

        out = f'fit_result/{self.test_name}/degeneracy_{self.out_label}.npz'
        np.savez(out, ts=ts, line=line, lifs=lifs, lds=lds, grid=grid,
                 H_truth=H_t, H_end=H_e, g_truth=g_t, g_end=g_e,
                 truth=np.array([truth[k] for k in rl]),
                 endpoint=np.array([endpoint[k] for k in rl]),
                 params=np.array(rl, dtype=object))
        logger.info(f"[DEGEN] saved {out}")

    def fit(self, dataloader_sim, target, **kwargs):
        self.prepare_fit()
        logger.info("Gauss-Newton / Levenberg-Marquardt calibration fitter.")
        logger.warning(f"Arguments {kwargs} are ignored in this mode.")
        # NOTE: do NOT overwrite self.ref_params with current_params here. ref_params
        # deliberately keeps the readout-noise fields intact (the marginalized FEE model
        # requires sigma > 0 even under --no-noise, which only strips current/norm copies
        # for the TARGET realization); clobbering it caused 1/sigma=1/0 at trace time.
        # The 5 fitted params are fully overwritten through the norm mapping anyway.
        rl = self.relevant_params_list

        if self.gn_resume_joint:
            if self.gn_curvature == 'ggn':
                logger.warning("[GN-JOINT] Fisher/GGN curvature does not include the frozen "
                               "nuisance terms; forcing exact Hessian for the joint polish.")
                self.gn_curvature = 'exact'
            self._load_joint_checkpoint(self.gn_resume_joint, dataloader_sim, target)

        cache = self._preload(dataloader_sim, target)
        lam = float(self.gn_lambda0)

        if getattr(self, 'gn_degeneracy', False):
            self.degeneracy_study(cache)
            return

        if self.gn_curvature == 'ggn' or self.gn_validate:
            # Pre-scan all batches once (plain loss evals) so the ROI cache and its
            # global max are FINAL before any Fisher fn is jitted — otherwise the
            # growing global max would spawn extra jit signatures on iteration 1.
            self._full_batch_loss(cache)

        if self.gn_validate:
            self._validate_fisher(cache)
            return

        if self.gn_warmup_iters > 0:
            # Hybrid mode: alternate short Adam warmups with GN takeover attempts, so GN is
            # used "as much as possible" — GN converges in O(10) full-batch iterations once
            # past the initialization plateau, but stalls ON the plateau (measured: it takes
            # near-loss-neutral steps toward the sigmoid bounds there). Adam crosses the
            # plateau; GN finishes the job. The fallback loop self-discovers the earliest
            # viable switch point.
            max_cycles = 4
            for cycle in range(max_cycles):
                self._run_adam_warmup(cache, self.gn_warmup_iters)
                status = self._run_gn_loop(cache)
                logger.info(f"GN hybrid cycle {cycle}: gn status = {status} "
                            f"(total warmup {self.gn_warmup_iters * (cycle + 1)} Adam iters)")
                if status == 'converged':
                    break
        else:
            self._run_gn_loop(cache)

        if self.gn_resume_joint:
            self._report_covariance(cache)

        with open(f'fit_result/{self.test_name}/history_iter{self.total_iter}_{self.out_label}.pkl', "wb") as fh:
            pickle.dump(self.training_history, fh)
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir, ignore_errors=True)

    def _report_covariance(self, cache):
        """H^-1 parameter errors at the current point (nuisances frozen if resuming)."""
        rl = self.relevant_params_list
        try:
            _, g, H = self._full_batch_lgh(cache)
            Hs = 0.5 * (H + H.T)
            if not np.all(np.isfinite(Hs)):
                logger.info("[GN-COV] non-finite Hessian entries (saturated point?); no covariance")
                return
            w = np.linalg.eigvalsh(Hs)
            cov = np.linalg.inv(Hs + 1e-9 * np.abs(w).max() * np.eye(len(rl)))
        except np.linalg.LinAlgError as e:
            logger.info(f"[GN-COV] covariance failed ({e}); skipping")
            return
        eps = 1e-4
        x0 = np.array([float(getattr(self.norm_params, k)) for k in rl])
        rel = np.zeros(len(rl))
        for j, k in enumerate(rl):
            p0 = float(map_norm_to_phys(x0[j] - eps, k, scheme=self.normalization_scheme,
                                        scale=self._norm_scale_for_scheme()))
            p1 = float(map_norm_to_phys(x0[j] + eps, k, scheme=self.normalization_scheme,
                                        scale=self._norm_scale_for_scheme()))
            dphys = (p1 - p0) / (2 * eps)
            phys = float(getattr(self.current_params, k))
            sig = np.sqrt(max(cov[j, j], 0.0))
            rel[j] = abs(sig * dphys / (phys if phys != 0 else 1.0)) * 100
        logger.info(f"[GN-COV] eig(H): {np.array2string(w, precision=3)}")
        logger.info("[GN-COV] conditional relative sigma [%] (nuisances frozen): "
                    + "  ".join(f"{k}={rel[j]:.2f}" for j, k in enumerate(rl)))
        self.training_history['gn_cov_rel_sigma_pct'] = {k: float(rel[j]) for j, k in enumerate(rl)}
        self.training_history['gn_hessian_final'] = Hs.tolist()

    def _run_adam_warmup(self, cache, n_iters):
        """Per-batch Adam warmup on the 5 calibration params (mirrors the production chain
        fitter's optimizer: grad-clip + Adam with warmup-exponential-decay schedule)."""
        import optax
        rl = self.relevant_params_list
        if not hasattr(self, '_warm_opt'):
            sched = optax.warmup_exponential_decay_schedule(
                init_value=0.0, peak_value=0.1,
                warmup_steps=min(500, n_iters), transition_steps=1, decay_rate=0.999)
            self._warm_opt = optax.chain(optax.clip_by_global_norm(100.0), optax.adam(sched))
            self._warm_state = self._warm_opt.init(
                extract_relevant_params(self.norm_params, rl))
            self._warm_iter = 0
        for _ in range(n_iters):
            (i, st, ref) = cache[self._warm_iter % len(cache)]
            lv, grads, _, _, _, _, _ = self.compute_loss(
                st, i, *ref, with_loss=True, with_grad=True)
            rg = extract_relevant_params(grads, rl)
            if any(jnp.isnan(leaf).any() for leaf in jax.tree_util.tree_leaves(rg)):
                logger.warning("Adam warmup: NaN gradients, skipping step")
                self._warm_iter += 1
                continue
            upd, self._warm_state = self._warm_opt.update(rg, self._warm_state)
            self.apply_updates('norm_params', upd)
            self.update_params()
            self._warm_iter += 1
            if self._warm_iter % 100 == 0:
                logger.info(f"[WARMUP] iter {self._warm_iter}: batch loss {float(lv):.2f}  "
                            + "  ".join(f"{k}={float(getattr(self.current_params,k)):.4g}" for k in rl))

    def _run_gn_loop(self, cache):
        """Damped-Newton/LM loop. Returns 'converged' (|step| < tol), 'stalled' (no decrease
        after backtracking), 'saturated' (a param ran to its sigmoid bound — the plateau
        signature; hand back to Adam), or 'budget' (gn_max_iter exhausted)."""
        rl = self.relevant_params_list
        lam = float(self.gn_lambda0)
        entry_norm_params = self.norm_params   # for do-no-harm revert on saturation
        loss, g, H = self._full_batch_lgh(cache)
        self._log_iter(loss, g)
        logger.info(f"GN init: loss {loss:.6f}  |g| {np.linalg.norm(g):.3e}  curvature={self.gn_curvature}")

        last_snorm = np.inf
        for it in range(self.gn_max_iter):
            Hs = 0.5 * (H + H.T)
            dscale = np.abs(np.diag(Hs))
            # Floor the Marquardt scaling: directions with ~zero curvature (e.g. a param
            # saturating its sigmoid bound) must still be damped, otherwise the solve
            # shoots unbounded steps along the flat direction.
            dfloor = 1e-6 * max(dscale.max(), 1.0)
            dscale = np.maximum(dscale, dfloor)

            accepted = False
            step = np.zeros(len(rl))
            for _bt in range(self.gn_max_backtrack):
                A = Hs + lam * np.diag(dscale)
                try:
                    step = -np.linalg.solve(A, g)
                except np.linalg.LinAlgError:
                    lam *= self.gn_lambda_up
                    continue
                saved = self.norm_params
                self.apply_updates('norm_params',
                                   {k: jnp.array(step[j], dtype=jnp.float32) for j, k in enumerate(rl)})
                self.update_params()
                new_loss = self._full_batch_loss(cache)
                if np.isfinite(new_loss) and new_loss < loss:
                    accepted = True
                    lam = max(lam / self.gn_lambda_down, 1e-12)
                    break
                # reject: revert and increase damping
                self.norm_params = saved
                self.update_params()
                lam *= self.gn_lambda_up

            if not accepted:
                # At the float32 loss floor LM can no longer find a decrease even though the
                # solve has converged (measured: last accepted |step| ~1e-5 at the stationary
                # point vs ~1e-1 when genuinely stuck on the init plateau). Classify by the
                # last accepted step size so the hybrid driver doesn't waste fallback cycles.
                if last_snorm < 1e-2:
                    logger.info(f"GN iter {it}: no decrease after backtracking with tiny last step "
                                f"(|step| {last_snorm:.2e}) — converged at the numerical floor.")
                    return 'converged'
                logger.info(f"GN iter {it}: no decrease after backtracking (lambda {lam:.2e}); stopping.")
                return 'stalled'

            loss, g, H = self._full_batch_lgh(cache)
            self._log_iter(loss, g)
            snorm = float(np.linalg.norm(step))
            last_snorm = snorm
            logger.info(f"GN iter {it}: loss {loss:.6f}  lambda {lam:.2e}  |step| {snorm:.3e}  "
                        + "  ".join(f"{k}={float(getattr(self.current_params,k)):.4g}" for k in rl))

            self.total_iter = getattr(self, 'total_iter', 0) + 1
            with open(f'fit_result/{self.test_name}/history_iter{self.total_iter}_{self.out_label}.pkl', "wb") as fh:
                pickle.dump(self.training_history, fh)

            # Saturation guard: a calibration param driven into its sigmoid bound is the
            # plateau-wandering signature (measured in the degeneracy study) — GN is taking
            # near-loss-neutral steps along a flat direction, not converging. Hand back to Adam.
            sat = max(abs(float(getattr(self.norm_params, k))) for k in rl)
            if sat > 3.5:
                # Do-no-harm: REVERT to the loop-entry params. Leaving the params pinned at a
                # sigmoid bound shipped a +140% long_diff in the truth-free polish (seed 1) —
                # the conditional minimum given imperfect nuisances can genuinely lie at/beyond
                # the bounds, and in that case the entry point (Adam endpoint) is the safer
                # estimate than the bound.
                logger.info(f"GN iter {it}: norm-space saturation {sat:.2f} > 3.5 (param at bound); "
                            f"REVERTING to loop-entry params and handing back to Adam.")
                self.norm_params = entry_norm_params
                self.update_params()
                # Log the reverted state so the history's LAST entry reflects the actual
                # final params (otherwise analyses reading [-1] see the saturated values).
                loss_r, g_r, _ = self._full_batch_lgh(cache)
                self._log_iter(loss_r, g_r)
                return 'saturated'

            if snorm < self.gn_tol:
                logger.info(f"GN converged at iter {it} (|step| {snorm:.2e} < {self.gn_tol:.1e}).")
                return 'converged'
        return 'budget'
