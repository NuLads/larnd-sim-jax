import os, sys

larndsim_dir=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
src_dir = os.path.join(larndsim_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if larndsim_dir not in sys.path:
    sys.path.insert(0, larndsim_dir)
import shutil
import copy
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

# ---------------------------------------------------------------------------
# Structured Newton solve for the per-segment log-dEdx block.
#
# The dEdx Hessian is (measured, see the GN report):
#     H = H_band(bandwidth 2)  +  diag(student-t + barrier + mean-constraint residual term)
#         +  q q^T           with q = sqrt(2 w_mean) * g,  g_i = l_i * dedx_i / L
# The banded part is recovered from EXACTLY 2*bw+1 = 5 Hessian-vector products regardless of N
# (structurally-orthogonal "mod-P" probing): for probe v_c with v_c[j]=1 iff j % P == c,
#     (H_band v_c)[i] = sum_{|i-j|<=bw, j==c mod P} H[i,j]
# and since {i-bw..i+bw} are P consecutive integers exactly ONE of them is == c (mod P),
# so each band entry is read off uniquely. The dense rank-1 part is subtracted analytically
# (it would otherwise break the structural orthogonality) and restored by Sherman-Morrison.
# ---------------------------------------------------------------------------
# (scale, shape) reparametrisation of the per-segment dEdx block.
#
#   dedx_i = M * s_i      with   M = exp(mu)      and    sum_i l_i s_i / L = 1
#   s_i    = exp(a_i) / Z          Z = sum_j l_j exp(a_j) / L
#   =>  log_dedx_i = mu + a_i - log Z
#   =>  mean_dedx  = sum_i l_i dedx_i / L = exp(mu)   EXACTLY, by construction.
#
# Why this matters: the mean-dEdx constraint w(mean - t)^2 becomes a function of the SINGLE
# parameter mu. Its Hessian contribution collapses to one (mu,mu) entry -- the dense rank-1
# q q^T coupling across all N segments (which sets kappa ~ 2.4e4 and cancels short-segment
# curvature, see the GN report) DISAPPEARS from the shape block entirely.
#
# The loss itself is UNCHANGED: it still receives log_dedx. Only the parametrisation and the
# gradient mapping change, which keeps the blast radius small.
#
# Gauge: a_i -> a_i + c leaves s_i (and hence the loss) exactly invariant. The induced gradient
# satisfies sum_k dL/da_k = 0 identically (see below), so the optimiser never moves along it.


def _dedx_scale_shape_forward(mu, a, seg_dx):
    """(mu, a) -> log_dedx.  seg_dx = per-segment physical length l_i."""
    l = seg_dx
    ea = jnp.exp(a - jnp.max(a))                 # shift for numerical safety; cancels in the ratio
    Zl = jnp.sum(l * ea)
    L = jnp.sum(l)
    # log Z with the max-shift undone:  Z = sum(l e^a)/L
    logZ = jnp.log(Zl) + jnp.max(a) - jnp.log(L)
    return mu + a - logZ


def _dedx_scale_shape_grads(g, a, seg_dx):
    """Map dL/dlog_dedx -> (dL/dmu, dL/da).

    log_dedx_i = mu + a_i - log Z,  Z = sum_j l_j e^{a_j} / L
      dlog_dedx_i/dmu  = 1                   -> dL/dmu  = sum_i g_i
      dlog_dedx_i/da_k = delta_ik - p_k      -> dL/da_k = g_k - p_k * sum_i g_i
    with p_k = l_k e^{a_k} / sum_j l_j e^{a_j}  (sum_k p_k = 1).
    Note sum_k dL/da_k = sum g - 1*sum g = 0 exactly: the gauge direction gets zero gradient.
    """
    l = np.asarray(seg_dx, dtype=np.float64)
    aa = np.asarray(a, dtype=np.float64)
    w = l * np.exp(aa - aa.max())
    sw = w.sum()
    p = w / sw if sw > 0 else np.zeros_like(w)
    gs = float(np.sum(g))
    return gs, np.asarray(g, dtype=np.float64) - p * gs


_DEDX_GN_BW = 2


def _dedx_probe_matrix(N, bw=_DEDX_GN_BW):
    """P x N array whose row c is the probe vector v_c."""
    P = 2 * bw + 1
    idx = np.arange(N)
    return np.stack([(idx % P == c).astype(np.float32) for c in range(P)], axis=0)


def _dedx_band_from_probes(probes, N, bw=_DEDX_GN_BW):
    """Assemble scipy 'matrix diagonal ordered form' ab from the P probe results.

    probes[c][i] == H_band[i, j] for the unique j in [i-bw, i+bw] with j % P == c.
    scipy convention: ab[u + i - j, j] == a[i, j], here u = l = bw.
    """
    P = 2 * bw + 1
    ab = np.zeros((P, N), dtype=np.float64)
    i = np.arange(N)
    for d in range(-bw, bw + 1):
        j = i + d
        ok = (j >= 0) & (j < N)
        ab[bw - d, j[ok]] = probes[j[ok] % P, i[ok]]
    return ab


def _dedx_banded_sm_solve(ab, q, rhs, lam, bw=_DEDX_GN_BW):
    """Solve (B + lam*I + q q^T) x = rhs with B banded.

    Sherman-Morrison is unconditionally stable here: the rank-1 enters with a PLUS sign
    (w_mean > 0), so the denominator 1 + q^T B^-1 q > 1 and there is no cancellation.
    Returns (x, ok).
    """
    from scipy.linalg import solve_banded
    abl = ab.copy()
    abl[bw, :] += lam
    try:
        y = solve_banded((bw, bw), abl, rhs)
        if q is None:
            return y, True
        z = solve_banded((bw, bw), abl, q)
        denom = 1.0 + float(q @ z)
        if not np.isfinite(denom) or denom <= 0:
            return None, False
        x = y - z * (float(q @ y) / denom)
        return (x, True) if np.all(np.isfinite(x)) else (None, False)
    except Exception:
        return None, False


# Change-of-variables term for the dEdx prior (see compute_loss). Default OFF -> the objective
# is bit-identical to prior behaviour. Read once at import so it stays static under jit.
_DEDX_LOG_JACOBIAN = os.environ.get('LARND_DEDX_LOG_JACOBIAN', '0') == '1'
# Evaluate the dEdx shape prior at scale-normalised dEdx so it says nothing about the mean.
_DEDX_SHAPE_PRIOR = os.environ.get('LARND_DEDX_SHAPE_PRIOR', '0') == '1'

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


# Drift axis in the fitter's coordinate convention. dataio.py swaps x<->z on load
# (swap_xz=True by default), so the drift direction in the track table that the chain warp
# writes is +z -- consistent with detsim_jax's z_anode = tpc_borders[..., 2, 0] and
# hit_z = z_anode + ticks * t_sampling * vdrift.
_DRIFT_AXIS = np.array([0., 0., 1.], np.float32)

# Frame convention for the spline transverse basis. "drift" aligns e1 with the wire plane
# so the drift-axis sensitivity is isolated in e2; "legacy" is the original arbitrary frame.
_CHAIN_FRAME = os.environ.get('LARND_CHAIN_FRAME', 'drift').strip().lower()


def _transverse_frame(u0, frame=None):
    """Two orthonormal vectors spanning the plane perpendicular to unit vector u0.

    With frame='drift' (default) the basis is chosen so that e1 lies entirely in the wire
    plane (e1 . zhat == 0 exactly) and e2 carries the whole drift-axis sensitivity
    (|e2 . zhat| = sin(angle between u0 and the drift axis)).

    Why this matters: the drift coordinate is measured by hit TIMING, not inferred by the
    position fit. In the legacy arbitrary frame the drift component is smeared across both
    basis vectors, so the geometry fit can slide charge along drift and manufacture apparent
    attenuation -- degenerate with lifetime. Measured on an S4 fit: 379 um of the fitted
    displacement lay along the drift axis (59% of its magnitude), with |e2 . zhat| = 0.83.
    Isolating it in one basis vector makes it separately penalizable (see
    _chain_drift_penalty) and decouples the penalized direction from the free one, which a
    diagonal preconditioner like Adam would otherwise fight.

    For a track (nearly) parallel to the drift axis, zhat x u0 -> 0; there BOTH transverse
    directions already lie in the wire plane, so the legacy frame is used and no drift
    penalty applies -- which is the physically correct behaviour.
    """
    if (frame or _CHAIN_FRAME) == 'drift':
        e1 = np.cross(_DRIFT_AXIS, u0)
        n1 = float(np.linalg.norm(e1))
        if n1 > 1e-3:                                  # track not parallel to drift
            e1 = (e1 / n1).astype(np.float32)
            e2 = np.cross(u0, e1); e2 /= (np.linalg.norm(e2) + 1e-12)
            return e1.astype(np.float32), e2.astype(np.float32)
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
        # Drift-axis (z) components of the two transverse basis vectors. Under the default
        # 'drift' frame e1z == 0 and |e2z| = sin(theta_to_drift); kept general so the
        # penalty is also correct in the legacy frame.
        'e1z': jnp.asarray(np.stack(e1s)[:, 2].astype(np.float32)),
        'e2z': jnp.asarray(np.stack(e2s)[:, 2].astype(np.float32)),
        'step': jnp.asarray(np.asarray([float(c.step_len) for c in ctxs], np.float32)),
        'sigma': jnp.asarray(np.asarray([float(c.sigma_mcs) for c in ctxs], np.float32)),
        'seg_rows': jnp.asarray(np.concatenate(seg_rows).astype(np.int32)),
        'seg_track': jnp.asarray(np.concatenate(seg_track)),
        'seg_am': jnp.asarray(np.concatenate(seg_am).astype(np.float32)),
        'seg_as': jnp.asarray(np.concatenate(seg_as).astype(np.float32)),
        'seg_ae': jnp.asarray(np.concatenate(seg_ae).astype(np.float32)),
    }


def _pad_spline_meta(meta, Tm, ncm, Km, Nm, zero_row):
    """Pad a spline chain meta to fixed global-max dims (Tm tracks, ncm nodes, Km modes,
    Nm segments) so every batch presents ONE jit signature -> the compiled cg_step is shared
    across all batches (data-as-args). Padded per-track slots are harmless (no real segment
    references them); padded segments point to `zero_row` (a zero-charge track row) so they
    deposit nothing and are loss-neutral."""
    T = meta['T']; nc = meta['max_nc']; K = meta['max_K']; Ns = int(meta['seg_rows'].shape[0])
    def pT(a, val=0.0):
        return jnp.pad(a, [(0, Tm - T)] + [(0, 0)] * (a.ndim - 1), constant_values=val)
    def pS(a, val=0):
        return jnp.pad(a, [(0, Nm - Ns)], constant_values=val)
    return {
        'basis': 'spline', 'T': Tm, 'max_nc': ncm, 'max_K': Km,
        'ncs_np': np.pad(meta['ncs_np'], (0, Tm - T), constant_values=1),
        'ncs': pT(meta['ncs'], 1),
        'Ks_np': np.pad(meta['Ks_np'], (0, Tm - T), constant_values=K),
        'nominal': jnp.pad(meta['nominal'], [(0, Tm - T), (0, ncm - nc), (0, 0)]),
        'Bmat': jnp.pad(meta['Bmat'], [(0, Tm - T), (0, ncm - nc), (0, Km - K)]),
        'e1': pT(meta['e1']), 'e2': pT(meta['e2']),
        'e1z': pT(meta['e1z']), 'e2z': pT(meta['e2z']),
        'step': pT(meta['step'], 1.0), 'sigma': pT(meta['sigma'], 1.0),
        'seg_rows': pS(meta['seg_rows'], zero_row), 'seg_track': pS(meta['seg_track'], 0),
        'seg_am': pS(meta['seg_am'], 0.0), 'seg_as': pS(meta['seg_as'], 0.0), 'seg_ae': pS(meta['seg_ae'], 0.0),
    }


def _pad_spline_coeffs(coeffs, Tm, Km):
    """Pad the per-track spline coeffs list to Tm tracks x 2Km (matching _pad_spline_meta).
    The warp reads c[:K] and c[K:2K] per track (Ks_np[ti]=K), so end-padding is transparent."""
    out = []
    for c in coeffs:
        K = c.shape[0] // 2
        out.append(jnp.concatenate([c[:K], jnp.zeros(Km - K, c.dtype),
                                    c[K:2 * K], jnp.zeros(Km - K, c.dtype)]))
    for _ in range(Tm - len(coeffs)):
        out.append(jnp.zeros(2 * Km, dtype=jnp.float32))
    return out


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


# ── GN curvature perf-investigation knobs (env-gated; default OFF => identical behaviour) ──
# LARND_GN_PROFILE=1: log wall time + GPU peak_bytes_in_use around each curvature/gradient
#                      sweep in _full_batch_lgh. Pure logging -- never changes computed values.
# LARND_GN_REMAT=1:   wrap the forward simulate call (predict) of the plain-calibration
#                      loss_wrapper in jax.checkpoint, trading recompute for activation memory
#                      in both the gradient and (via _hess_by_hvp's inner jax.grad) Hessian
#                      passes. Numerically identical to the unwrapped call.
_GN_PROFILE = os.environ.get('LARND_GN_PROFILE', '0') == '1'
_GN_REMAT = os.environ.get('LARND_GN_REMAT', '0') == '1'


def _gn_mem_peak():
    """Best-effort GPU peak_bytes_in_use in GiB; 0.0 if unavailable (e.g. CPU-only)."""
    try:
        stats = jax.devices('gpu')[0].memory_stats()
        return stats.get('peak_bytes_in_use', 0) / (1024.0 ** 3)
    except Exception:
        return 0.0


def _hess_by_hvp(f_with_aux, params, keys):
    """Hessian of a scalar (loss, aux) function w.r.t. the named params, built column by
    column with Hessian-vector products.

    Returns an object supporting ``hess.<ka>.<kb>`` so it is a drop-in for the
    jacfwd(jacrev(.)) result the callers already index that way.

    Why: jacfwd(jacrev(f)) evaluates all P tangents through one reverse-mode tape at once,
    so peak memory scales as P x (gradient tape). At 400 cm batches that reached a 129 GB
    allocation request. P separate jvp-of-grad calls need ~1 x gradient tape each.
    """
    from types import SimpleNamespace
    zero = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)

    def _scalar(p):
        return f_with_aux(p)[0]

    grad_fn = jax.grad(_scalar)
    cols = {}
    for kb in keys:
        tangent = zero.replace(**{kb: jnp.ones_like(getattr(params, kb))})
        _, hv = jax.jvp(grad_fn, (params,), (tangent,))
        cols[kb] = hv                       # hv.<ka> = d(grad_ka)/d(eps along kb) = H[ka, kb]
    hess = SimpleNamespace(**{
        ka: SimpleNamespace(**{kb: getattr(cols[kb], ka) for kb in keys}) for ka in keys})
    _, aux = f_with_aux(params)
    return hess, aux


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


def _chain_drift_penalty(disp, meta, weight):
    """Penalize the component of the fitted transverse displacement that lies along the
    DRIFT axis.

    The drift coordinate is reconstructed from hit timing, not from the position fit, so
    fitted motion along it is unphysical -- and worse, it is directly degenerate with charge
    attenuation: shifting a segment along drift changes how much lifetime attenuation the
    forward model applies to it, letting geometry absorb (or manufacture) apparent
    attenuation and bias the lifetime. Measured on an S4 fit: 379 um of the fitted
    displacement lay along drift, 59% of its magnitude.

    disp is (T, max_nc+1, 2) = displacement along (e1, e2); the drift component is
    d1*(e1.zhat) + d2*(e2.zhat), which reduces to d2*sin(theta) in the default drift frame.
    Units: cm^2, averaged per track (matching _chain_mcs_spline's normalization).
    """
    if weight is None or float(weight) <= 0.:
        return jnp.float32(0.)
    T, max_nc, ncs = meta['T'], meta['max_nc'], meta['ncs']
    dz = disp[..., 0] * meta['e1z'][:, None] + disp[..., 1] * meta['e2z'][:, None]
    j = jnp.arange(max_nc + 1)
    mask = (j[None, :] <= ncs[:, None]).astype(disp.dtype)
    return weight * jnp.sum(mask * dz ** 2) / max(1, T)


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
                 loss_fn=None, loss_fn_kw=None, readout_noise_target=True, readout_noise_guess=True, 
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
        # Freeze geometry after it converges so calibration+penalty fit against FIXED positions
        # (replicates the robust "ceiling" condition; a jittering geometry destabilizes lifetime).
        self._chain_freeze_iter = int(os.environ.get('LARND_CHAIN_FREEZE_ITER', '0'))
        # Gate calibration updates until this iteration (0 = off). Enables a true geometry-only
        # phase 1: with dEdx off the sim uses the input-file dEdx, and free calibration would
        # absorb that per-segment mismatch via lifetime/diffusion (wrong basin, deterministic).
        self._calib_start_iter = int(os.environ.get('LARND_CALIB_START_ITER', '0'))
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
        # Weight on the drift-projected component of the fitted geometry displacement.
        # 0 (default) reproduces previous behaviour; see _chain_drift_penalty.
        self._chain_drift_weight = float(os.environ.get('LARND_CHAIN_DRIFT_W', '0') or 0.)
        if self._chain_drift_weight > 0. and str(chain_basis) != 'spline':
            logger.warning("LARND_CHAIN_DRIFT_W is only implemented for the spline chain "
                           "basis; ignoring it for basis=%s", chain_basis)
            self._chain_drift_weight = 0.
        if str(chain_basis) == 'spline':
            logger.info("[CHAIN] transverse frame=%s, drift penalty weight=%g",
                        _CHAIN_FRAME, self._chain_drift_weight)
        self._chain_step_len = float(chain_step_len)
        self._chain_momentum_GeV = float(chain_momentum_GeV)
        self._chain_basis = str(chain_basis)              # 'angle' | 'spline'
        self._chain_spline_knot_cm = float(chain_spline_knot_cm)
        # Diffusion-annealing (less-local loss): guess-diffusion WIDTH starts at this factor and
        # anneals to 1 over N iterations. env-driven for quick experimentation.
        self._diff_anneal_start = float(os.environ.get('LARND_DIFF_ANNEAL_START', '1.0'))
        self._diff_anneal_iters = int(os.environ.get('LARND_DIFF_ANNEAL_ITERS', '0'))
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
        # Adaptive freeze: dedx_freeze_iter == -1 => latch the freeze when dEdx MAE plateaus
        # (robust across seeds/batch-sizes vs a fixed iteration that mistimes the co-adaptation).
        self._dedx_freeze_adaptive = (self.dedx_freeze_iter is not None and self.dedx_freeze_iter == -1)
        self._dedx_freeze_triggered = None
        self._dedx_freeze_window = int(os.environ.get('LARND_DEDX_FREEZE_WINDOW', '60'))
        self._dedx_freeze_rtol = float(os.environ.get('LARND_DEDX_FREEZE_RTOL', '0.02'))
        self._dedx_freeze_min_iter = int(os.environ.get('LARND_DEDX_FREEZE_MIN_ITER', '0'))
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
        # PROVENANCE (reviewer #9): LARND_* env vars change fit behaviour but were invisible in
        # checkpoints; git SHA lets a result be tied to the code that produced it.
        try:
            import subprocess as _sp, socket as _sock, sys as _sys, time as _time
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _sha = _sp.check_output(['git', 'rev-parse', 'HEAD'], cwd=_root, stderr=_sp.DEVNULL).decode().strip()
            _dirty = bool(_sp.check_output(['git', 'status', '--porcelain'], cwd=_root, stderr=_sp.DEVNULL).decode().strip())
        except Exception:
            _sha, _dirty = None, None
        self.training_history['env_overrides'] = {k: v for k, v in os.environ.items() if k.startswith('LARND_')}
        self.training_history['provenance'] = dict(
            git_sha=_sha, git_dirty=_dirty, hostname=__import__('socket').gethostname(),
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
            argv=list(sys.argv), timestamp=__import__('time').time())
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
            # NOTE: for fit_type=chain/gn_calib the loss builds physical_params from ref_params
            # (noise fields INTACT) — the probabilistic guess divides by sigma, so a zeroed sigma
            # would be 1/0. This strip therefore affects only paths that use current_params
            # (LikelihoodProfiler/Minuit/Hessian). The guess in the main fit KEEPS its noise model.
            logger.info("Stripping noise from current_params (NOTE: chain/gn_calib guess still uses "
                        "ref_params noise — this flag does not disable the guess noise model there)")
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
        # In (scale, shape) mode the live state is (mu, a); mirror the DERIVED log_dedx into
        # _dedx_cache so every existing consumer (checkpoint format, MAE diagnostics, the
        # analysis scripts) keeps working unchanged.
        if getattr(self, '_dedx_param_mode', 'perseg') == 'scale_shape' and getattr(self, '_dedx_ss', None):
            for _b, _st in self._dedx_ss.items():
                _sd = self._batch_seg_dx.get(_b)
                if _sd is None:
                    continue
                _ld = _dedx_scale_shape_forward(_st['mu'], _st['a'],
                                                jnp.asarray(np.asarray(_sd)[:len(_st['a'])], dtype=jnp.float32))
                self._dedx_cache[_b] = {'log_dedx': _ld, 'opt_state': _st['opt_state']}
            self.training_history['dedx_ss_mu'] = {int(k): float(v['mu']) for k, v in self._dedx_ss.items()}
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
            # The arc-length-matched TARGET reference, kept separate from true_dedx_cache (which is
            # the SIM input) so an offline reader can tell the two apart.
            if getattr(self, '_batch_true_dedx_matched', None):
                self.training_history['true_dedx_matched_cache'] = {
                    int(k): np.asarray(v) for k, v in self._batch_true_dedx_matched.items()
                }
            if getattr(self, '_batch_seg_dx', None):
                self.training_history['seg_dx_cache'] = {
                    int(k): np.asarray(v) for k, v in self._batch_seg_dx.items()
                }
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

    def compute_loss(self, tracks, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, with_loss=True, with_grad=True, with_hess=False, epoch=0, use_physical_params=False, log_dedx=None, parent_ids=None, chain_angles_per_track=None, return_loss_closure=False, unique_size=None, roi_override=None, diff_scale=1.0):
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
                # wfs_eval.shape[0] is the dynamic-path 128-padded unique-pixel count: using it as the
                # static unique_size reproduces the dynamic path's shapes EXACTLY (jit-loss path).
                self._batch_usize_cache = getattr(self, '_batch_usize_cache', {})
                self._batch_usize_cache[cache_key] = int(wfs_eval.shape[0])
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
                if getattr(self, '_hess_via_hvp', False):
                    hess, aux_hess = _hess_by_hvp(loss_wrapper, self.current_params,
                                                  self.relevant_params_list)
                else:
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

                # LESS-LOCAL LOSS via diffusion annealing: inflate the guess diffusion so charge
                # spreads over more pixels -> predicted & target intensities overlap even when the
                # track is off -> long-range position gradient (widens the capture basin). diff_scale
                # is a param-space factor (width ~ sqrt(param)), annealed 1 over iterations.
                if diff_scale != 1.0:
                    physical_params = physical_params.replace(
                        long_diff=physical_params.long_diff * diff_scale,
                        tran_diff=physical_params.tran_diff * diff_scale)

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

                # SHAPE-ONLY prior (LARND_DEDX_SHAPE_PRIOR=1, default OFF).
                # Problem it solves: MAP shrinkage toward the split-t MODE (1.8740) biases the
                # length-weighted MEAN low, because the prior's gradient reaches the scale as well
                # as the shape. Evaluating the prior at SCALE-NORMALISED dEdx --
                #     dedx_shape_i = dedx_i * (M_ref / mean_dedx)
                # -- makes it exactly invariant to the overall scale, so it carries NO information
                # about the mean and its gradient w.r.t. the scale is identically zero. The scale
                # is then set solely by the mean constraint, which is a proper prior on M whose
                # width should come from the measured batch-to-batch spread (see the GN report).
                # Works in BOTH parametrisations: it only uses dedx and dx.
                # NB the soft barrier is deliberately NOT normalised -- it is a delta-ray cap on
                # the PHYSICAL dEdx (8.5 MeV/cm), not a statement about shape.
                _dedx_for_prior = dedx_sample
                if _DEDX_SHAPE_PRIOR:
                    _dxw0 = t[:, dx_idx]
                    _sv0 = valid_mask & (_dxw0 > 0)
                    _tot_dx0 = jnp.sum(_dxw0 * _sv0)
                    _mean0 = jnp.sum(dedx_sample * _dxw0 * _sv0) / jnp.maximum(_tot_dx0, 1e-6)
                    _dedx_for_prior = dedx_sample * (self._dedx_shape_prior_ref
                                                     / jnp.maximum(_mean0, 1e-6))

                # Student-t prior on per-step dEdx values.
                # Weighted by physical step length (dx) to ensure that the regularization
                # strength is invariant to the sampling resolution.
                # The location (expected dEdx) is scaled by length_scale_factor.
                if self.dedx_use_split_t:
                    dedx_prior = split_student_t_nll(
                        _dedx_for_prior,
                        self.dedx_student_nu_l,
                        self.dedx_student_nu_r,
                        self.dedx_student_loc * length_scale_factor,
                        self.dedx_student_scale_l,
                        self.dedx_student_scale_r,
                        weights=t[:, dx_idx]
                    ) * self.dedx_prior_weight
                else:
                    dedx_prior = student_t_nll(
                        _dedx_for_prior,
                        self.dedx_student_nu,
                        self.dedx_student_loc * length_scale_factor,
                        self.dedx_student_scale,
                        weights=t[:, dx_idx]
                    ) * self.dedx_prior_weight

                # One-sided soft barrier penalty.
                # Weighted by physical step length (dx) and mask out invalid segments (dx <= 0)
                dx_weights = t[:, dx_idx]
                valid_step_mask = (dx_weights > 0)

                # Optional change-of-variables (Jacobian) term -- LARND_DEDX_LOG_JACOBIAN=1,
                # default OFF (bit-identical to prior behaviour).
                # The split-t is evaluated on the LINEAR dEdx while the fitted parameter is
                # log_dedx, so -log p_d(e^theta) is NOT a proper prior in theta: it omits
                # -log|d(dedx)/dtheta| = -theta. Omitting it biases the MAP DOWN; including it
                # moves the mode 1.8740 -> 1.8780, i.e. ~38% of the way to the prior's mean
                # (1.8847), which is the gap dedx_mean_constraint exists to repair.
                # Weighted identically to the split-t itself (dx * mask * prior_weight) so the
                # two stay on the same footing.
                if _DEDX_LOG_JACOBIAN:
                    dedx_prior = dedx_prior - self.dedx_prior_weight * jnp.sum(
                        jnp.log(_dedx_for_prior) * dx_weights * valid_step_mask
                    )

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
                        _drift_loss = _chain_drift_penalty(
                            _chain_warp_aux, _meta_use, self._chain_drift_weight)
                        _mcs_loss = _mcs_loss + _drift_loss
                    else:
                        _drift_loss = jnp.float32(0.)
                        _mcs_loss = _chain_mcs_vectorized(
                            _chain_thetas, _chain_phis, _meta_use, self._mcs_prior_weight)
                    total_loss = total_loss + _mcs_loss
                else:
                    _mcs_loss = jnp.float32(0.)
                    _drift_loss = jnp.float32(0.)

                aux_out = dict(hit_aux) if hit_aux is not None else {}
                aux_out['dedx_prior'] = dedx_prior
                aux_out['dedx_barrier'] = dedx_barrier
                aux_out['dedx_mean_penalty'] = dedx_mean_penalty
                aux_out['dedx_drift_penalty'] = dedx_drift_penalty
                aux_out['mean_dedx'] = mean_dedx
                aux_out['mcs_prior'] = _mcs_loss
                # Logged separately from mcs_prior (which now includes it) so the
                # drift-projected geometry component is directly diagnosable.
                aux_out['chain_drift_penalty'] = _drift_loss
                # Saturation guard for the static-unique jit path: number of genuinely active pixels.
                # If this reaches unique_size, jnp.unique(size=...) may be silently truncating.
                aux_out['n_active_pixels'] = jnp.sum(prediction['unique_pixels'] >= 0)
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
            if return_loss_closure and _dedx_active and not _chain_angles_active:
                # Additive branch (dEdx-Hessian investigation): pure, differentiable
                # loss(log_dedx_pt[, norm_params])->scalar for this batch, with chain angles
                # inactive (position fitting off). Mirrors _chain_only_loss above. Never taken by
                # any existing call site: every current return_loss_closure=True caller passes a
                # non-None chain_angles_per_track (so _chain_angles_active is True and the branch
                # above returns first) -- verified by grep over all return_loss_closure=True sites.
                def _dedx_only_loss(log_dedx_pt, norm_params_in=None, data_over=None):
                    _np = self.norm_params if norm_params_in is None else norm_params_in
                    return loss_wrapper_combined(_np, log_dedx_pt, None, data_over=data_over)[0]
                return _dedx_only_loss
            # ── Optional jitted value_and_grad (LARND_JIT_LOSS=1) ──
            # The eager path re-traces the whole loss every iteration and runs it op-by-op with all
            # intermediates materialized. Jitting requires static shapes: unique_size is set to the
            # dynamic path's own padded pixel count (cached at ROI evaluation, identical layout after
            # sort) and roi_override to the cached ROI sizes — so the jitted program computes exactly
            # the same arrays. The jitted closure is cached per batch and reused across iterations;
            # everything it closes over (tracks, target, parent_ids, rngkey, meta) is batch-static.
            # Cache key includes rngkey/diff_scale so strategies where they vary just recompile.
            _jit_loss_on = (os.environ.get('LARND_JIT_LOSS', '0') == '1'
                            and not use_physical_params and not with_hess)
            _jit_usize = getattr(self, '_batch_usize_cache', {}).get((i, tracks.shape[0]))
            if _jit_loss_on and unique_size is None and _jit_usize is not None:
                unique_size = _jit_usize  # read by loss_wrapper_combined via closure
                roi_override = (self.sim_strategy.padded_small_nb, self.sim_strategy.padded_large_nb)

            def _get_jit_vg(mode, argnums):
                self._jit_vg_cache = getattr(self, '_jit_vg_cache', {})
                _shapes = tuple(a.shape for a in chain_angles_per_track) if chain_angles_per_track is not None else None
                key = (i, mode, argnums, tracks.shape, _shapes, unique_size, roi_override,
                       int(rngkey) if rngkey is not None else None, float(diff_scale))
                fn = self._jit_vg_cache.get(key)
                if fn is None:
                    logger.info(f"[JIT-LOSS] compiling batch {i} mode={mode} usize={unique_size} roi={roi_override}")
                    if mode == 'joint':
                        fn = jax.jit(value_and_grad(loss_wrapper_combined, argnums=argnums, has_aux=True))
                    else:  # 'pos'
                        def _wrap_no_dedx_j(np_, ca_):
                            return loss_wrapper_combined(np_, None, ca_)
                        fn = jax.jit(value_and_grad(_wrap_no_dedx_j, argnums=argnums, has_aux=True))
                    self._jit_vg_cache[key] = fn
                return fn

            if with_loss and with_grad:
                if _chain_angles_active and _dedx_active:
                    # Joint: global params + per-seg dEdx + chain angles
                    _vg = (_get_jit_vg('joint', (0, 1, 2)) if _jit_loss_on and unique_size is not None
                           else value_and_grad(loss_wrapper_combined, argnums=(0, 1, 2), has_aux=True))
                    (loss_val, aux), (grads, grads_dedx, grads_chain_per_track) = _vg(
                        self.norm_params, log_dedx, chain_angles_per_track)
                elif _chain_angles_active and not _dedx_active:
                    # Position-only: global params + chain angles (no dEdx)
                    def _wrap_no_dedx(np_, ca_):
                        return loss_wrapper_combined(np_, None, ca_)
                    _vg = (_get_jit_vg('pos', (0, 1)) if _jit_loss_on and unique_size is not None
                           else value_and_grad(_wrap_no_dedx, argnums=(0, 1), has_aux=True))
                    (loss_val, aux), (_g_np, grads_chain_per_track) = _vg(
                        self.norm_params, chain_angles_per_track)
                    grads = _g_np
                    grads_dedx = None
                else:
                    _vg = (_get_jit_vg('joint', (0, 1)) if _jit_loss_on and unique_size is not None
                           else value_and_grad(loss_wrapper_combined, argnums=(0, 1), has_aux=True))
                    (loss_val, aux), (grads, grads_dedx) = _vg(self.norm_params, log_dedx, None)
                    grads_chain_per_track = None
                # Saturation guard: if active pixels reach the static size, unique() may truncate
                if _jit_loss_on and unique_size is not None and aux is not None and 'n_active_pixels' in aux:
                    _nact = int(aux['n_active_pixels'])
                    if _nact >= int(unique_size):
                        logger.warning(f"[JIT-LOSS] batch {i}: active pixels {_nact} saturate unique_size {unique_size} — "
                                       f"falling back to dynamic path for this batch")
                        self._batch_usize_cache.pop((i, tracks.shape[0]), None)
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
                if getattr(self, '_hess_via_hvp', False):
                    # Column-by-column via Hessian-vector products. jacfwd(jacrev(.)) pushes all
                    # P tangents through the reverse-mode tape SIMULTANEOUSLY (~P x gradient
                    # memory), which OOMs on a 400 cm batch (measured: 129 GB request for one
                    # batch). P sequential jvp-of-grad passes cost ~1 x gradient memory each.
                    hess, aux_hess = _hess_by_hvp(_np_only, self.norm_params,
                                                  self.relevant_params_list)
                else:
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
                if _GN_REMAT:
                    # Wrap the forward sim in jax.checkpoint: don't retain its internal
                    # activations on the tape, recompute them on demand during the backward
                    # (and, for _hess_by_hvp, the forward-over-reverse) pass. Trades compute
                    # for the dominant (Npix, MAX_ADC_VALUES, Nticks) activation memory.
                    # Numerically identical to the unwrapped call; only the memory/time
                    # profile changes.
                    def _predict_call(pp):
                        return self.sim_strategy.predict(pp, tracks, self.sim_track_fields, rngkey)
                    prediction = jax.checkpoint(_predict_call)(physical_params)
                else:
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
                if getattr(self, '_hess_via_hvp', False):
                    hess, aux_hess = _hess_by_hvp(loss_wrapper, self.norm_params,
                                                  self.relevant_params_list)
                else:
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
            # MAE against the TARGET file, arc-length matched. dedx_mae_iter is against the SIM
            # input, so it only equals a truth error when sim file == target file (S1/S2).
            self.training_history['dedx_mae_truth_iter'] = []

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
        self._batch_seg_dx: dict = {}      # batch_idx → np.array of per-segment physical dx (cm)
        # batch_idx → per-parent TRUE dEdx matched from the TARGET by arc length (see
        # _build_true_dedx_matched). Distinct from _batch_true_dedx, which is the SIM input.
        self._batch_true_dedx_matched: dict = {}
        # Structured Newton for the dEdx block (LARND_DEDX_OPT=gn). Default 'adam' = unchanged.
        self._dedx_opt_mode   = os.environ.get('LARND_DEDX_OPT', 'adam').lower()
        # (scale, shape) reparametrisation: 'perseg' (default, unchanged) | 'scale_shape'
        self._dedx_param_mode = os.environ.get('LARND_DEDX_PARAM', 'perseg').lower()
        self._dedx_mu_lr_mult = float(os.environ.get('LARND_DEDX_MU_LR_MULT', '1.0'))
        self._dedx_ss: dict = {}   # batch_idx -> {'mu':float, 'a':array, 'opt_state':...}
        self._dedx_min_dx = float(os.environ.get('LARND_DEDX_MIN_DX', '0.0'))
        self._dedx_frozen_mask: dict = {}
        # reference mean the shape prior normalises to; default = the mean-constraint target,
        # so the shape prior and the scale prior agree about where the mean should be.
        self._dedx_shape_prior_ref = float(os.environ.get(
            'LARND_DEDX_SHAPE_PRIOR_REF', self.dedx_mean_constraint_target))
        self._dedx_gn_lam0    = float(os.environ.get('LARND_DEDX_GN_LAMBDA0', '1.0'))
        self._dedx_gn_step_cap= float(os.environ.get('LARND_DEDX_GN_STEP_CAP', '0.5'))
        self._dedx_gn_max_bt  = int(os.environ.get('LARND_DEDX_GN_MAXBT', '4'))
        self._dedx_gn_lam: dict = {}
        self._dedx_gn_stats   = {'ok': 0, 'fallback': 0}
        self._dedx_gn_validate = os.environ.get('LARND_DEDX_GN_VALIDATE', '0') == '1'
        self._dedx_gn_validated = False
        self._dedx_gn_chunk   = int(os.environ.get('LARND_DEDX_GN_CHUNK', '1'))
        self._dedx_gn_refresh = int(os.environ.get('LARND_DEDX_GN_REFRESH', '1'))
        self._dedx_gn_band: dict = {}
        # Study 2 (see _eval_truth_point): evaluate the objective at constructed points instead
        # of fitting. Unset = normal fit.
        self._truthpoint_ckpt = os.environ.get('LARND_TRUTHPOINT', '').strip() or None
        self._tp_loaded = None
        self._truthpoint_done: set = set()

        _dedx_lr = self.dedx_lr if self.dedx_lr is not None else lr
        # ── dEdx learning-rate schedule (LARND_DEDX_LR_FINAL_FRAC) ────────────────────────────
        # HISTORICALLY THE dEdx BLOCK NEVER ANNEALED. Calibration decays per EPOCH and geometry
        # per ITERATION, but this was a bare optax.adam(constant) -- the one block that co-adapts
        # with lifetime was the one that never cooled. LARND_DEDX_LR_FINAL_FRAC=f anneals it
        # exponentially to f x its initial value by the END of the run. Unset => constant, i.e.
        # bit-identical to the previous behaviour.
        #
        # TWO TRAPS, both avoided deliberately:
        #  1. transition_steps=1, NOT epoch_size. The dEdx Adam state is stored PER BATCH
        #     (_dedx_cache[batch]['opt_state']) and stepped once per visit, so its internal count
        #     already advances once per EPOCH. Copying the calibration call, which uses
        #     transition_steps=epoch_size against a global count, would decay 1/epoch_size as
        #     fast -- a silent near-no-op.
        #  2. The rate is derived from the ACTUAL epoch count rather than being given directly,
        #     so the requested annealing is invariant to max_nbatch and iterations. The
        #     calibration schedule is specified as a per-epoch rate and that is exactly why
        #     raising max_nbatch silently un-annealed it and cost us a whole 4x-data run.
        _dedx_sched = _dedx_lr
        _ff_raw = os.environ.get('LARND_DEDX_LR_FINAL_FRAC', '').strip()
        if _ff_raw and self.fit_dedx:
            _ff = float(_ff_raw)
            if not (0.0 < _ff <= 1.0):
                raise ValueError(f"LARND_DEDX_LR_FINAL_FRAC must be in (0, 1], got {_ff}")
            _cfg = self.training_history.get('config')
            _iters = int(getattr(_cfg, 'iterations', 0) or 0)
            _esz = max(1, int(epoch_size))
            _nep = max(1, int(np.ceil(_iters / _esz))) if _iters > 0 else 1
            if _iters <= 0:
                logger.warning("[dEdx-LR] iterations unknown; cannot derive a decay rate — "
                               "leaving the dEdx LR CONSTANT")
            elif _ff == 1.0:
                logger.info("[dEdx-LR] final fraction 1.0 requested — LR stays constant")
            else:
                _rate = float(_ff ** (1.0 / _nep))
                _dedx_sched = optax.exponential_decay(
                    init_value=_dedx_lr, transition_steps=1, decay_rate=_rate, staircase=False)
                logger.info(f"[dEdx-LR] annealing {_dedx_lr:g} -> {_ff:g}x over {_nep} epochs "
                            f"({_iters} iters / epoch_size {_esz}); decay_rate={_rate:.6f} per "
                            f"epoch, applied per-batch (transition_steps=1)")
        self._dedx_optimizer = optax.adam(_dedx_sched) if self.fit_dedx else None
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

    def _reinit_opt_state_keep_schedule(self):
        """Reset Adam moments but PRESERVE the LR-schedule step count.

        optax stores ScaleByScheduleState(count) in the same pytree as the moments, so a bare
        optimizer.init() rewinds the schedule to step 0 — restarting warmup and the decay clock
        mid-run. That silently confounds every staging/freeze experiment (the LR jumps back to
        peak exactly at the freeze point)."""
        import optax as _optax
        old = self.opt_state
        new = self.optimizer.init(extract_relevant_params(self.norm_params, self.relevant_params_list))

        def _copy_counts(o, n):
            if isinstance(o, _optax.ScaleByScheduleState) and isinstance(n, _optax.ScaleByScheduleState):
                return n._replace(count=o.count)
            if isinstance(o, tuple) and isinstance(n, tuple) and type(o) is type(n) and len(o) == len(n):
                vals = [_copy_counts(a, b) for a, b in zip(o, n)]
                try:
                    return type(n)(*vals)
                except TypeError:
                    return tuple(vals)
            if isinstance(o, dict) and isinstance(n, dict):
                return {k: (_copy_counts(o[k], v) if k in o else v) for k, v in n.items()}
            return n

        try:
            return _copy_counts(old, new)
        except Exception as e:
            logger.warning(f"schedule-count preservation failed ({e}); falling back to plain init")
            return new

    def process_grads(self, grads):
        # We extract only the relevant parameters to pass to the optimizer
        relevant_grads = extract_relevant_params(grads, self.relevant_params_list)

        # Calibration-start gate (phase-1 geometry-only): skip the update entirely
        if getattr(self, '_calib_start_iter', 0) > 0 and getattr(self, '_cur_total_iter', 0) < self._calib_start_iter:
            return relevant_grads

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

    # ── Study 2: is the minimum misplaced, or merely unreachable? ────────────────────────────
    #
    # LARND_TRUTHPOINT=<a completed history_*.pkl> evaluates the objective at a handful of
    # constructed points instead of fitting, and writes them to training_history['truthpoint'].
    #
    #   P0 fit    : calib/dEdx/geometry all taken from that checkpoint  (where the fit ended up)
    #   P1 truth  : all three at the BEST their parameterisation can represent given truth
    #   P2/P3/P4  : exactly ONE block moved to truth, the other two left at the fit
    #
    # Reading:  loss(P1) < loss(P0) => the minimum is in the right place and we fail to reach it
    #           (an optimiser problem).  loss(P1) > loss(P0) => the fit found a genuinely better
    #           point than truth, so the minimum is displaced (an objective problem) and no
    #           optimiser change can fix it. P2-P4 say WHICH block the objective disagrees about.
    #
    # The full aux decomposition is stored at every point, so the LIKELIHOOD-ONLY comparison
    # (log_likelihood_tick + no_match_penalty + expected_total_hits) can be made offline
    # alongside the total objective. That distinction matters: if truth loses on the total but
    # wins on the likelihood, the defect is in our PRIORS, not in the physics model. The
    # decomposition was verified exact to 4.5e-7 relative, so this needs no separate prior-off run.
    def _truth_spline_coeffs(self, batch_idx):
        """Least-squares projection of the TRUE positions onto the spline basis.

        This is the best our geometry parameterisation can do with perfect knowledge -- the
        REPRESENTATION FLOOR. Identical construction to LARND_INIT_FROM_TRUTH.
        """
        ctxs = self._batch_chain_contexts.get(batch_idx, [])
        tp = self._batch_true_positions.get(batch_idx, {})
        out = []
        for ctx in ctxs:
            K = _spline_K(ctx.total_len, self._chain_spline_knot_cm)
            params = np.zeros(2 * K, dtype=np.float32)
            true_pts = tp.get(ctx.track_id)
            if true_pts is not None:
                u0 = _dir_from_angles(ctx.theta0_i, ctx.phi0_i)
                e1, e2 = _transverse_frame(u0)
                L = float(ctx.total_len)
                al = np.asarray(ctx.alpha_mid, float)
                tr = np.asarray(true_pts, float)
                m = min(len(al), len(tr))
                al, tr = al[:m], tr[:m]
                nom = np.asarray(ctx.x0_fixed)[None, :] + al[:, None] * L * u0[None, :]
                d = tr - nom
                B = np.sin(np.outer(al, np.pi * np.arange(1, K + 1)))
                b1 = np.linalg.lstsq(B, d @ e1, rcond=None)[0]
                b2 = np.linalg.lstsq(B, d @ e2, rcond=None)[0]
                params = np.concatenate([b1, b2]).astype(np.float32)
            out.append(jnp.asarray(params))
        return out

    def _load_truthpoint_ckpt(self):
        """Fitted geometry / dEdx / calibration from a completed run."""
        if getattr(self, '_tp_loaded', None) is not None:
            return self._tp_loaded
        with open(self._truthpoint_ckpt, 'rb') as f:
            ck = pickle.load(f)
        # Geometry keyed by TRACK_ID, never by list position: position matching silently
        # transplants another track's geometry whenever batching or ordering differs between
        # runs (the trap LARND_CHAIN_INIT_FROM already documents).
        cc, cx = ck.get('chain_cache', {}), ck.get('chain_contexts', {})
        geom = {}
        for bi, states in cc.items():
            ctxs = cx.get(bi, [])
            geom[int(bi)] = ({str(c['track_id']): np.asarray(s['angles'])
                              for c, s in zip(ctxs, states)} if ctxs else None)
        dedx = {int(bi): np.asarray(v['log_dedx'])
                for bi, v in ck.get('dedx_cache', {}).items() if 'log_dedx' in v}
        calib = {p: float(np.ravel(ck[p + '_iter'])[-1]) for p in self.relevant_params_list}
        truth = {p: float(np.ravel(ck[p + '_target'])[0]) for p in self.relevant_params_list}
        logger.info(f"[TRUTHPOINT] loaded {self._truthpoint_ckpt}: {len(geom)} geometry batches, "
                    f"{len(dedx)} dEdx batches; fit calib {calib}; target {truth}")
        self._tp_loaded = (geom, dedx, calib, truth)
        return self._tp_loaded

    def _set_calib_physical(self, phys):
        """Write PHYSICAL calibration values into norm_params, the exact inverse of
        update_params(). The joint branch of compute_loss reads norm_params, not
        current_params, so setting current_params directly would be a silent no-op."""
        if self.normalization_scheme == "divide":
            upd = {k: float(v) / float(getattr(self.params_normalization, k))
                   for k, v in phys.items()}
        else:
            upd = {k: float(map_phys_to_norm(float(v), k, scheme=self.normalization_scheme,
                                             scale=self._norm_scale_for_scheme()))
                   for k, v in phys.items()}
        self.norm_params = self.norm_params.replace(**upd)
        self.update_params()

    def _eval_truth_point(self, i, tracks, ref, log_dedx_live, parent_ids, epoch):
        """Evaluate the objective at the fit point, the truth point, and the three one-block
        hybrids, for batch *i*. Appends one record to training_history['truthpoint']."""
        geom_fit, dedx_fit, calib_fit, calib_truth = self._load_truthpoint_ckpt()
        ctxs = self._batch_chain_contexts.get(i, [])

        # geometry: fit (from the checkpoint, matched by track_id) and truth-projected
        g_truth = self._truth_spline_coeffs(i)
        g_map = geom_fit.get(i)
        if g_map:
            g_fit, missing = [], 0
            for k, ctx in enumerate(ctxs):
                a = g_map.get(str(ctx.track_id))
                if a is None:
                    missing += 1
                    a = np.zeros_like(np.asarray(g_truth[k]))
                g_fit.append(jnp.asarray(np.asarray(a, dtype=np.float32)))
            if missing:
                logger.warning(f"[TRUTHPOINT] batch {i}: {missing}/{len(ctxs)} tracks absent from "
                               f"the checkpoint geometry; using the nominal straight line there")
        else:
            logger.warning(f"[TRUTHPOINT] batch {i}: no checkpoint geometry, using nominal")
            g_fit = [jnp.zeros_like(c) for c in g_truth]

        # dEdx: fit (checkpoint) and truth (arc-length-matched target)
        d_live = np.asarray(log_dedx_live) if log_dedx_live is not None else None
        d_fit = dedx_fit.get(i)
        if d_fit is None or (d_live is not None and len(d_fit) != len(d_live)):
            logger.warning(f"[TRUTHPOINT] batch {i}: checkpoint dEdx missing or mis-shaped; "
                           f"falling back to the live state")
            d_fit = d_live
        tm = self._batch_true_dedx_matched.get(i)
        d_truth = None
        if tm is not None and d_fit is not None and len(tm) == len(d_fit):
            tm = np.asarray(tm, dtype=np.float64)
            # matched truth can carry zeros where no target segment matched; keep the fit there
            # rather than taking log(0).
            good = tm > 0
            d_truth = np.asarray(d_fit, dtype=np.float32).copy()
            d_truth[good] = np.log(tm[good]).astype(np.float32)
            if not good.all():
                logger.info(f"[TRUTHPOINT] batch {i}: {int((~good).sum())}/{len(tm)} segments have "
                            f"no matched truth dEdx; left at the fit value")
        elif tm is not None:
            logger.warning(f"[TRUTHPOINT] batch {i}: matched-truth dEdx length {len(tm)} != "
                           f"{len(d_fit) if d_fit is not None else None}; skipping dEdx-truth points")
        d_fit_j = jnp.asarray(d_fit) if d_fit is not None else None
        d_truth_j = jnp.asarray(d_truth) if d_truth is not None else None

        # representation floor: how well can the spline basis represent truth AT ALL?
        floor_um = None
        _saved = self._chain_cache.get(i)
        try:
            self._chain_cache[i] = [{'angles': c} for c in g_truth]
            _f = self._compute_pos_residual_batch(i)
            floor_um = float(_f) * 1e4 if _f is not None else None
        except Exception as e:
            logger.warning(f"[TRUTHPOINT] batch {i}: representation floor failed: {e}")
        finally:
            if _saved is not None:
                self._chain_cache[i] = _saved
            else:
                self._chain_cache.pop(i, None)

        POINTS = [('P0_fit',        calib_fit,   d_fit_j,   g_fit),
                  ('P1_truth',      calib_truth, d_truth_j, g_truth),
                  ('P2_calib_true', calib_truth, d_fit_j,   g_fit),
                  ('P3_dedx_true',  calib_fit,   d_truth_j, g_fit),
                  ('P4_geom_true',  calib_fit,   d_fit_j,   g_truth)]
        rec = {'batch': int(i), 'repr_floor_um': floor_um,
               'n_tracks': len(ctxs), 'points': {}}
        _norm_saved, _cur_saved = self.norm_params, self.current_params
        try:
            for name, cal, ld, gg in POINTS:
                if ld is None:
                    logger.warning(f"[TRUTHPOINT] batch {i}: point {name} unavailable, skipped")
                    continue
                # NOTE use_physical_params MUST stay False. That flag selects a branch which
                # ignores log_dedx and chain_angles_per_track entirely -- passing them alongside
                # it silently evaluates the nominal geometry and the input-file dEdx, which
                # would make every point here identical apart from the calibration values.
                self._set_calib_physical(cal)
                lv, _, _, _, aux, _, _ = self.compute_loss(
                    tracks, i, *ref, epoch=epoch, with_loss=True, with_grad=False,
                    use_physical_params=False, log_dedx=ld, parent_ids=parent_ids,
                    chain_angles_per_track=gg)
                rec['points'][name] = {
                    'loss': float(lv),
                    'aux': {k: serialize_value(v) for k, v in (aux or {}).items()},
                    'calib': dict(cal)}
                logger.info(f"[TRUTHPOINT] batch {i:3d} {name:14s} loss = {float(lv):.4f}")
        finally:
            self.norm_params, self.current_params = _norm_saved, _cur_saved
        p0 = rec['points'].get('P0_fit', {}).get('loss')
        p1 = rec['points'].get('P1_truth', {}).get('loss')
        if p0 is not None and p1 is not None:
            rec['delta_truth_minus_fit'] = p1 - p0
            logger.info(f"[TRUTHPOINT] batch {i:3d} SUMMARY  truth - fit = {p1 - p0:+.4f}  "
                        f"({'truth is BETTER -> optimiser-limited' if p1 < p0 else 'fit is BETTER -> objective displaced'})"
                        f"  repr_floor = {floor_um if floor_um is None else round(floor_um, 1)} um")
        self.training_history.setdefault('truthpoint', []).append(rec)

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

    def _get_or_init_dedx_ss(self, batch_idx, n_orig_segs, seg_dx):
        """State for the (scale, shape) parametrisation: scalar mu plus shape vector a.

        Initialised at mu = log(dedx_student_loc), a = 0  ->  log_dedx == log(loc) for every
        segment, i.e. bit-identical to the per-segment initialisation.
        """
        if batch_idx not in self._dedx_ss:
            mu = jnp.asarray(np.log(float(self.dedx_student_loc)), dtype=jnp.float32)
            a = jnp.zeros((n_orig_segs,), dtype=jnp.float32)
            flat = jnp.concatenate([mu[None], a])
            self._dedx_ss[batch_idx] = {'mu': mu, 'a': a,
                                        'opt_state': self._dedx_optimizer.init(flat)}
        return self._dedx_ss[batch_idx]

    def _dedx_ss_log_dedx(self, batch_idx, n_orig_segs, seg_dx):
        """Current log_dedx implied by (mu, a) for this batch."""
        st = self._get_or_init_dedx_ss(batch_idx, n_orig_segs, seg_dx)
        return _dedx_scale_shape_forward(st['mu'], st['a'], jnp.asarray(seg_dx, dtype=jnp.float32))

    def _process_dedx_grads_ss(self, batch_idx, grads_dedx, seg_dx):
        """Adam step in (mu, a) coordinates.

        The loss is untouched; only the chain rule changes. mu carries the length-weighted MEAN
        (exactly), a carries the SHAPE -- so the mean constraint acts on mu alone and no longer
        couples the shape parameters (see _dedx_scale_shape_* and test_dedx_reparam.py).
        """
        st = self._dedx_ss[batch_idx]
        gmu, ga = _dedx_scale_shape_grads(np.asarray(grads_dedx, dtype=np.float64),
                                          np.asarray(st['a'], dtype=np.float64),
                                          np.asarray(seg_dx, dtype=np.float64))
        flat_g = jnp.concatenate([jnp.asarray([gmu], dtype=jnp.float32),
                                  jnp.asarray(ga, dtype=jnp.float32)])
        updates, new_opt = self._dedx_optimizer.update(flat_g, st['opt_state'])
        # Scale mu's UPDATE, not its gradient. Adam is scale-invariant per coordinate
        # (it divides by the running second moment), so multiplying the GRADIENT is a no-op --
        # that was a real defect in the first version of this code.
        mu_new = st['mu'] + updates[0] * self._dedx_mu_lr_mult
        a_new = st['a'] + updates[1:]
        # fix the gauge (a -> a + c is exactly flat): re-centre so sum_i l_i a_i = 0.
        l = jnp.asarray(seg_dx, dtype=jnp.float32)
        a_new = a_new - jnp.sum(l * a_new) / jnp.maximum(jnp.sum(l), 1e-6)
        self._dedx_ss[batch_idx] = {'mu': mu_new, 'a': a_new, 'opt_state': new_opt}
        return _dedx_scale_shape_forward(mu_new, a_new, l)

    def _process_dedx_grads(self, batch_idx, grads_dedx, current_log_dedx):
        """Apply one Adam step for per-segment log-dEdx of *batch_idx*."""
        _, opt_state = self._get_or_init_dedx_state(batch_idx, len(current_log_dedx))
        updates, new_opt_state = self._dedx_optimizer.update(grads_dedx, opt_state)
        new_log_dedx = current_log_dedx + updates
        self._dedx_cache[batch_idx] = {'log_dedx': new_log_dedx, 'opt_state': new_opt_state}
        return new_log_dedx

    def _dedx_gn_step(self, batch_idx, log_dedx, loss_fn, grads_dedx, seg_dx, total_iter=0):
        """One structured Newton step for the per-segment log-dEdx of *batch_idx*.

        H = band(bw 2) + q q^T. The band comes from 5 HVPs via mod-P probing (cost independent
        of N); the rank-1 mean-constraint term is analytic and is subtracted from the probes,
        then restored by Sherman-Morrison. Levenberg-Marquardt damping with backtracking.

        Returns (new_log_dedx, info dict). Falls back to returning None on any failure so the
        caller can use the Adam path instead.
        """
        N = int(log_dedx.shape[0])
        if N < 2 * _DEDX_GN_BW + 1:
            return None, {'why': 'N too small'}
        x0 = np.asarray(log_dedx, dtype=np.float64)
        g = np.asarray(grads_dedx, dtype=np.float64)
        if not np.all(np.isfinite(g)):
            return None, {'why': 'non-finite gradient'}

        # ---- analytic rank-1 of the mean constraint:  q = sqrt(2w) * l_i exp(x_i) / L ----
        w = float(self.dedx_mean_constraint_weight)
        q = None
        if w > 0 and seg_dx is not None:
            l = np.asarray(seg_dx, dtype=np.float64)
            if l.shape[0] >= N:
                l = l[:N]
                L = float(l.sum())
                if L > 1e-6:
                    q = np.sqrt(2.0 * w) * (l * np.exp(x0) / L)

        # ---- band via 2*bw+1 HVPs (independent of N) ----
        grad_fn = jax.grad(loss_fn)
        V = _dedx_probe_matrix(N)
        Vj = jnp.asarray(V)
        # All 2*bw+1 probes in ONE vmapped forward-over-reverse pass: XLA then shares the primal
        # forward/backward work across the probes instead of redoing it 5x (measured ~5x faster
        # than the eager loop). Only 5 tangents, so this is nothing like the jacfwd(jacrev) blowup.
        # HVPs. vmapping all P tangents at once is ~5x faster but holds P copies of the reverse
        # tape (measured: OOM, 8.08 GiB request on an 11 GiB card) -- the same (P+1)x blowup that
        # rules out jacfwd(jacrev) here. Default chunk=1 = sequential, ~1x tape. Raise only if the
        # card has room.
        # Do NOT jit-and-cache across iterations: loss_fn closes over THIS iteration's
        # tracks/target/noise realisation, so a cached jit would silently reuse a stale loss.
        cached = self._dedx_gn_band.get(batch_idx)
        reuse = (cached is not None and self._dedx_gn_refresh > 1 and
                 (total_iter - cached[1]) < self._dedx_gn_refresh and cached[0].shape[1] == N)
        if reuse:
            ab = cached[0]
        else:
            ch = max(1, int(self._dedx_gn_chunk))
            probes = np.empty((V.shape[0], N), dtype=np.float64)
            for c0 in range(0, V.shape[0], ch):
                Vc = Vj[c0:c0 + ch]
                if Vc.shape[0] == 1:
                    hv = jax.jvp(grad_fn, (log_dedx,), (Vc[0],))[1][None, :]
                else:
                    hv = jax.vmap(lambda v: jax.jvp(grad_fn, (log_dedx,), (v,))[1])(Vc)
                probes[c0:c0 + Vc.shape[0]] = np.asarray(hv, dtype=np.float64)
            # contamination for probe c at row i is q_i * (q . V[c])  ->  shape (P, N).
            # np.outer(q, V@q) would be (N, P): transposed. Guarded by a unit test.
            probes = probes - (np.outer(V @ q, q) if q is not None else 0.0)
            if not np.all(np.isfinite(probes)):
                return None, {'why': 'non-finite HVP'}
            ab = _dedx_band_from_probes(probes, N)
            self._dedx_gn_band[batch_idx] = (ab, total_iter)

        # ---- optional in-situ validation against the DENSE Hessian (LARND_DEDX_GN_VALIDATE=1) ----
        # Costs N extra HVPs, so only for small-N diagnostic runs. This is the check that the
        # bandwidth-2 assumption actually holds on REAL data at the REAL operating point --
        # everything else only validates the linear algebra.
        if self._dedx_gn_validate and not self._dedx_gn_validated:
            Hd = np.empty((N, N), dtype=np.float64)
            for k in range(N):
                ek = np.zeros(N, dtype=np.float32); ek[k] = 1.0
                _, hv = jax.jvp(grad_fn, (log_dedx,), (jnp.asarray(ek),))
                Hd[:, k] = np.asarray(hv, dtype=np.float64)
            Hb = Hd - (np.outer(q, q) if q is not None else 0.0)   # remove analytic rank-1
            dref = np.abs(np.diag(Hb)).max()
            iu = np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
            offband = np.abs(Hb[iu > _DEDX_GN_BW]).max() if (iu > _DEDX_GN_BW).any() else 0.0
            rec = np.zeros((N, N))
            ii = np.arange(N)
            for d in range(-_DEDX_GN_BW, _DEDX_GN_BW + 1):
                jj = ii + d; ok2 = (jj >= 0) & (jj < N)
                rec[ii[ok2], jj[ok2]] = ab[_DEDX_GN_BW - d, jj[ok2]]
            banderr = np.abs(rec[iu <= _DEDX_GN_BW] - Hb[iu <= _DEDX_GN_BW]).max()
            asym = np.abs(Hb - Hb.T).max()
            logger.info(
                f"[DEDX-GN-VALIDATE] N={N} diag_max={dref:.4g} | "
                f"max|off-band|={offband:.4g} (rel {offband/max(dref,1e-30):.3e}) | "
                f"band recovery err={banderr:.4g} (rel {banderr/max(dref,1e-30):.3e}) | "
                f"asym={asym:.3g} | 5 probes vs {N} dense HVPs")
            self._dedx_gn_validated = True

        # ---- LM loop with backtracking ----
        f0 = float(loss_fn(log_dedx))
        lam = float(self._dedx_gn_lam.get(batch_idx, self._dedx_gn_lam0))
        cap = self._dedx_gn_step_cap
        best = None
        for _bt in range(self._dedx_gn_max_bt + 1):
            step, ok = _dedx_banded_sm_solve(ab, q, -g, lam)
            if ok:
                mx = np.abs(step).max()
                if cap > 0 and mx > cap:
                    step = step * (cap / mx)
                xn = x0 + step
                fn = float(loss_fn(jnp.asarray(xn, dtype=log_dedx.dtype)))
                if np.isfinite(fn) and fn < f0:
                    best = (xn, fn, lam, mx, _bt)
                    break
            lam *= 4.0
        if best is None:
            self._dedx_gn_lam[batch_idx] = min(lam, 1e8)
            return None, {'why': 'no descent', 'lam': lam, 'f0': f0}

        xn, fn, lam_used, mx, nbt = best
        self._dedx_gn_lam[batch_idx] = max(lam_used / 3.0, 1e-6)
        return (jnp.asarray(xn, dtype=log_dedx.dtype),
                {'f0': f0, 'f1': fn, 'dloss': f0 - fn, 'lam': lam_used,
                 'step_max': mx, 'backtracks': nbt, 'N': N,
                 'reused': bool(reuse)})

    def _get_or_init_chain_state(self, batch_idx):
        """Lazily initialise per-track chain angles and Adam state for batch_idx.

        Angles layout (absolute per-segment directions):
          angles[:n_chain] = theta_0, theta_1, ..., theta_{n-1}   (global polar angle of each segment)
          angles[n_chain:] = phi_0,   phi_1,   ..., phi_{n-1}     (global azimuthal angle of each segment)
        Shape: (2 * n_chain,).  Initialised so all segments point in the initial track direction.
        MCS prior penalises jnp.diff(thetas) and jnp.diff(phis) — consecutive direction changes.
        """
        if batch_idx not in self._chain_cache:
            # TWO-PASS (LARND_CHAIN_INIT_FROM=<checkpoint.pkl>): initialise chain coefficients from
            # a previous fit's chain_cache — combined with LARND_CHAIN_FREEZE_ITER=1 this fits
            # calib+dEdx against FROZEN fitted geometry (pass 2). Batches must match (same config).
            _init_ckpt = None
            _ckpt_path = os.environ.get('LARND_CHAIN_INIT_FROM')
            if _ckpt_path:
                if not hasattr(self, '_chain_init_ckpt'):
                    with open(_ckpt_path, 'rb') as _f:
                        _ck = pickle.load(_f)
                    # key by TRACK_ID (not list position): position matching silently transplants
                    # another track's geometry if batching/ordering differs between passes.
                    _cc, _cx = _ck.get('chain_cache', {}), _ck.get('chain_contexts', {})
                    self._chain_init_ckpt = {}
                    for _bi, _states in _cc.items():
                        _ctxs = _cx.get(_bi, [])
                        self._chain_init_ckpt[_bi] = {
                            str(_c['track_id']): _s['angles']
                            for _c, _s in zip(_ctxs, _states)} if _ctxs else None
                    logger.info(f"[TWO-PASS] chain init from {_ckpt_path} ({len(self._chain_init_ckpt)} batches, keyed by track_id)")
                _init_ckpt = self._chain_init_ckpt.get(batch_idx)
            ctxs = self._batch_chain_contexts.get(batch_idx, [])
            states = []
            for _ci, ctx in enumerate(ctxs):
                if self._chain_basis == 'spline':
                    # 2K spline coefficients, initialised to zero = nominal (linear-guess) track
                    K = _spline_K(ctx.total_len, self._chain_spline_knot_cm)
                    params = jnp.zeros(2 * K, dtype=jnp.float32)
                    # DIAGNOSTIC (LARND_INIT_FROM_TRUTH): seed at the least-squares spline fit of the
                    # TRUE positions. If the fit then STAYS near truth -> the loss min is fine and the
                    # linear-guess init/landscape was the accuracy limiter; if it DRIFTS away -> the
                    # loss min is genuinely offset from truth (loss bias).
                    if os.environ.get('LARND_INIT_FROM_TRUTH') and hasattr(self, '_batch_true_positions'):
                        tp = self._batch_true_positions.get(batch_idx, {})
                        true_pts = tp.get(ctx.track_id)
                        if true_pts is not None:
                            u0 = _dir_from_angles(ctx.theta0_i, ctx.phi0_i)
                            e1, e2 = _transverse_frame(u0)
                            L = float(ctx.total_len); al = np.asarray(ctx.alpha_mid, float)
                            tr = np.asarray(true_pts, float); m = min(len(al), len(tr)); al, tr = al[:m], tr[:m]
                            nom = np.asarray(ctx.x0_fixed)[None, :] + al[:, None] * L * u0[None, :]
                            d = tr - nom
                            B = np.sin(np.outer(al, np.pi * np.arange(1, K + 1)))
                            b1 = np.linalg.lstsq(B, d @ e1, rcond=None)[0]
                            b2 = np.linalg.lstsq(B, d @ e2, rcond=None)[0]
                            params = jnp.asarray(np.concatenate([b1, b2]).astype(np.float32))
                else:
                    n_c = ctx.n_chain
                    angles = np.empty(2 * n_c, dtype=np.float32)
                    angles[:n_c] = ctx.theta0_i   # all segments start aligned with initial direction
                    angles[n_c:] = ctx.phi0_i
                    params = jnp.array(angles)
                if _init_ckpt:
                    _tk = str(ctx.track_id)
                    if _tk not in _init_ckpt:
                        logger.warning(f"[TWO-PASS] batch {batch_idx}: track_id {_tk} absent in source — default init")
                    else:
                        _loaded = jnp.asarray(np.asarray(_init_ckpt[_tk], dtype=np.float32))
                        if _loaded.shape == params.shape:
                            params = _loaded
                        else:
                            logger.warning(f"[TWO-PASS] batch {batch_idx} track {_tk}: shape mismatch "
                                           f"{_loaded.shape} vs {params.shape} — default init")
                opt_state = self._chain_optimizer.init(params)
                states.append({'angles': params, 'opt_state': opt_state})
            self._chain_cache[batch_idx] = states
        return self._chain_cache[batch_idx]

    def _build_true_dedx_matched(self, batch_idx, batch_flat, raw_pids, n_orig, target):
        """Per-parent-segment TRUE dE/dx, matched from the TARGET file by arc length.

        WHY THIS EXISTS. `_batch_true_dedx` is read from `dataloader_sim`, i.e. from
        `--input_file_sim`. Whenever that is not the target file -- which is every S3/S4 run, where
        the sim side is the straight-line guess -- the logged `dedx_mae_iter` is NOT an error
        against truth. It is |fitted - starting values|, i.e. how far the fit has moved from its own
        initialisation. Since the guess file's dE/dx is known to be bad (40% of the true spread,
        mean -0.44%), MOVING AWAY from it is what a working fit should do, and the long-standing
        "the dE/dx MAE ends worse than it starts" observation may be measuring exactly that.
        This builds the reference the diagnostic should have used.

        THE MATCHING PROBLEM. Sim and target have different segmentations (10.08M vs 3.03M
        segments), so there is no row correspondence and no per-segment index to join on. What is
        shared is the trajectory. So for each track we compare in normalised arc length:

          * sim side:    parent segment p occupies [a0, a1] of the track, from cumsum of the
                         per-parent lengths already computed in `_batch_seg_dx`.
          * target side: build cumulative length s(a) and cumulative energy E(a) over the true
                         segments, then
                             dEdx_true(p) = [E(a1) - E(a0)] / [(a1 - a0) * L_true].

        Integrating dE over the span (rather than point-sampling dEdx at a midpoint) is the
        physically correct reduction: dE/dx is an average over a finite length, and the spans differ
        by ~3.3x between the two files. Point sampling would alias the true fluctuations instead of
        averaging them, and would report spurious MAE driven purely by segmentation.

        Falls back to leaving the batch absent (rather than guessing) whenever the target lacks the
        id fields or a track cannot be matched -- a missing entry is skipped downstream, which is
        safer than silently comparing against the wrong thing, the very bug this method fixes.
        """
        if self.read_target or target is None:
            return
        tf = self.tgt_track_fields
        if 'eventID' not in tf or 'trackID' not in tf or 'dEdx' not in tf or 'dx' not in tf:
            return
        sf = self.sim_track_fields
        if self.evt_id not in sf or 'dx' not in sf:
            return
        trk_name = 'trackID' if 'trackID' in sf else ('traj_id' if 'traj_id' in sf else None)
        if trk_name is None:
            return
        seg_dx = self._batch_seg_dx.get(batch_idx)
        if seg_dx is None:
            return

        s_ev, s_tr = sf.index(self.evt_id), sf.index(trk_name)
        t_ev, t_tr = tf.index('eventID'), tf.index('trackID')
        t_dedx, t_dx = tf.index('dEdx'), tf.index('dx')

        valid = raw_pids >= 0
        rows = np.flatnonzero(valid)
        uniq, first = np.unique(raw_pids[valid], return_index=True)
        # (event, track) of each parent segment, taken from its first sub-step
        p_ev = np.zeros(n_orig); p_tr = np.zeros(n_orig)
        p_ev[uniq] = batch_flat[rows[first], s_ev]
        p_tr[uniq] = batch_flat[rows[first], s_tr]

        tgt = np.asarray(target[batch_idx].reshape(-1, len(tf)))
        out = np.full(n_orig, np.nan, dtype=np.float32)

        for ev, tr in {(float(a), float(b)) for a, b in zip(p_ev[uniq], p_tr[uniq])}:
            sel = np.flatnonzero((np.abs(p_ev - ev) < 0.5) & (np.abs(p_tr - tr) < 0.5))
            sel = np.array([p for p in sel if p in set(uniq.tolist())], dtype=int)
            if len(sel) == 0:
                continue
            sel = np.sort(sel)                       # parent index order == along-track order
            L = seg_dx[sel].astype(np.float64)
            if L.sum() <= 0:
                continue
            edges = np.concatenate([[0.0], np.cumsum(L)]) / L.sum()      # normalised spans

            m = (np.abs(tgt[:, t_ev] - ev) < 0.5) & (np.abs(tgt[:, t_tr] - tr) < 0.5)
            tdx = tgt[m, t_dx].astype(np.float64)
            tde = tgt[m, t_dedx].astype(np.float64) * tdx
            if len(tdx) < 2 or tdx.sum() <= 0:
                continue
            a_true = np.concatenate([[0.0], np.cumsum(tdx)]) / tdx.sum()
            E_true = np.concatenate([[0.0], np.cumsum(tde)])
            Ltot = tdx.sum()
            E0 = np.interp(edges[:-1], a_true, E_true)
            E1 = np.interp(edges[1:], a_true, E_true)
            span = (edges[1:] - edges[:-1]) * Ltot
            good = span > 1e-9
            out[sel[good]] = ((E1 - E0)[good] / span[good]).astype(np.float32)

        if np.isfinite(out).any():
            self._batch_true_dedx_matched[batch_idx] = out

    def _freq_gate(self, tag, batch_idx, freq):
        """Every-`freq`-VISITS gate, counted PER BATCH.

        Replaces `total_iter % freq == 0`, which is a global-iteration gate and is broken under
        sequential batching: batch b is always visited at total_iter == b (mod nbatch), so
        `total_iter % freq == 0` is a FIXED function of the batch index. With freq > 1 a fixed
        subset of batches receives EVERY update and the rest receive NONE, ever -- and shuffling
        the batch order accidentally repairs it, which makes the two orderings incomparable.
        Counting each batch's own visits makes the gate mean what it says and removes the
        order dependence.

        freq <= 1 fires on every visit, identical to the old behaviour (which was already correct
        at freq == 1, the default -- so no existing run changes).
        """
        if freq is None or freq <= 1:
            return True
        if not hasattr(self, '_freq_counters'):
            self._freq_counters = {}
        key = (tag, int(batch_idx))
        n = self._freq_counters.get(key, 0)
        self._freq_counters[key] = n + 1
        return n % int(freq) == 0

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

    def _solve_chain_ggn_shared(self, batch_idx, tracks, ref, log_dedx, parent_ids, meta, coeffs, col_map, max_iter):
        """Data-as-args spline GGN solve: pad every batch to the running global-max shape and
        pass tracks/target/meta/coeffs as TRACED args, so all batches reuse ONE compiled
        cg_step/vg (the spline is ~99% compile-bound; kills the per-batch compile storm)."""
        from jax.flatten_util import ravel_pytree
        from larndsim.sim_jax import simulate_wfs as _sim_wfs
        sigma = self.loss_strategy.sigma_charge / 1000.0
        phys = self.ref_params.replace(**{k: getattr(self.current_params, k) for k in self.relevant_params_list})

        # GLOBAL geometry spec, precomputed ONCE from all batch contexts (available after
        # precompute_static_pixels) so batch 0 already uses the final shape -> one shared compile
        # (a running max would keep growing across varied batches and recompile through epoch 1).
        # SINGLE global-max shape = fewest compile signatures. Compile (~4min) dominates, so
        # minimising distinct shapes beats per-batch sizing (bucketing measured WORSE: more
        # compiles). All geometry dims + usize + nhits precomputed from contexts/pixel-counts so
        # batch 0 already uses the final shape -> ideally ONE shared compile for the whole run.
        if not hasattr(self, '_ggn_geom'):
            Tm = ncm = Km = Nm = rs = rl = usg0 = 0
            for i, ctxs in self._batch_chain_contexts.items():
                if not ctxs:
                    continue
                Tm = max(Tm, len(ctxs)); ncm = max(ncm, max(int(c.n_chain) for c in ctxs))
                Km = max(Km, max(_spline_K(c.total_len, self._chain_spline_knot_cm) for c in ctxs))
                Nm = max(Nm, sum(int(np.asarray(c.idxs).shape[0]) for c in ctxs))
                r = self._batch_padded_roi.get(i, (256, 256)); rs = max(rs, int(r[0])); rl = max(rl, int(r[1]))
                fp = self._batch_fixed_pixels.get(i)
                if fp is not None:
                    usg0 = max(usg0, int(fp.shape[0]))
                tp = self._batch_target_pixels.get(i)          # proxy upper-bound for target hits
                if tp is not None:
                    Nm = Nm  # (Nseg already covers sim segments)
            usg = int(((usg0 + 256 + 127) // 128) * 128) if usg0 else 768
            self._ggn_geom = dict(T=Tm, nc=ncm, K=Km, Nseg=Nm, rs=rs, rl=rl, usg=usg)
        gs = self._ggn_geom
        Tm, ncm, Km, Nm, roig, usg = gs['T'], gs['nc'], gs['K'], gs['Nseg'], (gs['rs'], gs['rl']), gs['usg']
        # nhits: round to next POWER OF 2 (>=512) + running-max -> log-few distinct values even
        # across the wide 400cm range (512-fixed buckets gave 14 recompiles there).
        _nh0 = int(np.asarray(ref[0]).shape[0])
        nhits = 1 << max(9, (max(_nh0, 1) - 1).bit_length())
        if not hasattr(self, '_ggn_rmax'):
            self._ggn_rmax = dict(nh=0)
        self._ggn_rmax['nh'] = max(self._ggn_rmax['nh'], nhits)
        nhm = self._ggn_rmax['nh']

        # pad: tracks (+1 guaranteed zero-charge row for padded segments), coeffs, meta, target
        zero_row = int(tracks.shape[0])
        tracks_p = jnp.concatenate([tracks, jnp.zeros((1, tracks.shape[1]), tracks.dtype)], axis=0)
        # parent_ids must grow with tracks_p (the extra zero-charge row): the dEdx reconstruction
        # in loss_wrapper_combined broadcasts parent_ids against t -> shape mismatch otherwise.
        # -1 marks it invalid so it keeps its (zero) track dEdx and injects no dEdx gradient.
        if parent_ids is not None:
            parent_ids = jnp.concatenate(
                [jnp.asarray(parent_ids).ravel(),
                 -jnp.ones(1, dtype=jnp.asarray(parent_ids).dtype)], axis=0)
        meta_p = _pad_spline_meta(meta, Tm, ncm, Km, Nm, zero_row)
        coeffs_p = _pad_spline_coeffs(coeffs, Tm, Km)
        tkeys = ['adcs', 'pixel_x', 'pixel_y', 'pixel_z', 'ticks', 'hit_prob', 'event', 'pixel_id']
        def _padh(a, val):
            a = jnp.asarray(a)
            return jnp.pad(a, [(0, nhm - a.shape[0])] + [(0, 0)] * (a.ndim - 1), constant_values=val)
        target_p = tuple(_padh(r, -1 if k == 'pixel_id' else 0) for k, r in zip(tkeys, ref))

        marr_keys = ['ncs', 'nominal', 'Bmat', 'e1', 'e2', 'step', 'sigma',
                     'seg_rows', 'seg_track', 'seg_am', 'seg_as', 'seg_ae']
        meta_static = {'basis': 'spline', 'T': Tm, 'max_nc': ncm, 'max_K': Km,
                       'ncs_np': meta_p['ncs_np'], 'Ks_np': meta_p['Ks_np']}
        meta_arr = tuple(meta_p[k] for k in marr_keys)
        flat0, unflat = ravel_pytree(coeffs_p)

        def _meta_full(marr):
            d = dict(meta_static); d.update({k: marr[i] for i, k in enumerate(marr_keys)}); return d

        def fields(xflat, tracks_in, marr):
            t, _ = _chain_warp_spline(tracks_in, unflat(xflat), _meta_full(marr), col_map)
            pred = self.sim_strategy.predict(phys, t, self.sim_track_fields, None,
                                             unique_size=usg, roi_override=roig)
            return pred['hit_prob'], adc2charge(pred['adcs_distrib'], phys)

        def ggn_vp(xflat, v, tracks_in, marr):
            ff = lambda xx: fields(xx, tracks_in, marr)
            (z, q), (Jzv, Jqv) = jax.jvp(ff, (xflat,), (v,))
            w = jax.lax.stop_gradient(jnp.exp(z))
            _, vjp_fn = jax.vjp(ff, xflat)
            (gv,) = vjp_fn((w * Jzv, (w / sigma ** 2) * Jqv))
            return gv

        # loss closure (data + MCS prior) reused across batches via data_over override
        if not hasattr(self, '_ggn_shared_lossfn'):
            self._ggn_shared_lossfn = self.compute_loss(
                tracks_p, batch_idx, *ref, log_dedx=log_dedx, parent_ids=parent_ids,
                chain_angles_per_track=coeffs, return_loss_closure=True, unique_size=usg, roi_override=roig)
        loss_fn = self._ggn_shared_lossfn
        def f(xflat, tracks_in, target_in, marr):
            td = {k: target_in[i] for i, k in enumerate(tkeys)}
            return loss_fn(unflat(xflat), data_over={'tracks': tracks_in, 'target': td, 'meta': _meta_full(marr)})

        npad = int(flat0.shape[0])
        _cg_maxiter = int(min(max(20, npad // 4), 60))
        def cg_step(xflat, gflat, lam, tracks_in, marr):
            def A(v):
                return ggn_vp(xflat, v, tracks_in, marr) + lam * v
            dx, _ = jax.scipy.sparse.linalg.cg(A, -gflat, tol=1e-4, maxiter=_cg_maxiter)
            return dx

        key = (Tm, ncm, Km, Nm, usg, roig, nhm)          # constant once nhits settles -> one compile
        if not hasattr(self, '_ggn_shared_fns'):
            self._ggn_shared_fns = {}
        if key not in self._ggn_shared_fns:
            self._ggn_shared_fns = {key: (jax.jit(jax.value_and_grad(f)), jax.jit(cg_step))}  # drop stale
        _vg, _cg = self._ggn_shared_fns[key]

        x = jnp.asarray(flat0, dtype=jnp.float32)
        loss, g = _vg(x, tracks_p, target_p, meta_arr); loss = float(loss)
        lam = 1e-2
        for it in range(max_iter):
            accepted = False
            for _bt in range(6):
                dx = _cg(x, g, jnp.float32(lam), tracks_p, meta_arr)
                xt = x + dx
                lt, gt = _vg(xt, tracks_p, target_p, meta_arr); lt = float(lt)
                snorm = float(jnp.linalg.norm(dx))
                if np.isfinite(lt) and lt < loss:
                    x, loss, g = xt, lt, gt; lam = max(lam / 3.0, 1e-9); accepted = True
                    break
                lam *= 4.0
            if not accepted or snorm < 1e-6:
                break
        # unpad: recover each real track's 2K coeffs from the padded (2Km) layout
        padded = unflat(x)
        new = []
        for ti, c in enumerate(coeffs):
            Kt = c.shape[0] // 2
            pc = padded[ti]
            new.append(jnp.concatenate([pc[:Kt], pc[Km:Km + Kt]]))
        self._chain_cache[batch_idx] = [{'angles': new[ti]} for ti in range(len(coeffs))]
        return float(loss)

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
        # Spline: use the shared-jit (data-as-args) path so all batches reuse ONE compiled
        # cg_step (spline is 99% compile-bound; execution is 2-21s). Angle keeps the per-batch
        # path (n too varied to share usefully).
        if basis == 'spline' and os.environ.get('LARND_GN_SHARED', '1') == '1':
            return self._solve_chain_ggn_shared(batch_idx, tracks, ref, log_dedx, parent_ids,
                                                meta, coeffs, col_map, max_iter)
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
        _precond = os.environ.get('LARND_GN_PRECOND', '0') == '1'
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

    def analyze_dedx_hessian(self, dataloader_sim, target, batch_idx=0, n_cols=12,
                              diag_chunk=8, max_diag=None, out_path=None):
        """Measure diagonal dominance of the per-segment log-dEdx Hessian on ONE batch.

        Position fitting must be off (self.fit_chain_positions False) and self.fit_dedx True.
        Calibration (norm_params) is held FIXED at self.norm_params (its current/initial value);
        only the per-segment log-dEdx array is differentiated.

        Computes, via forward-over-reverse HVPs (jax.jvp(jax.grad(f), (x,), (e_k,)), the same
        pattern as the module-level _hess_by_hvp helper but for a flat array param instead of a
        calibration pytree):
          - the FULL diagonal H_ii of the TOTAL loss (hit_loss + all active dEdx regularizers)
            w.r.t. log-dEdx, for the first `max_diag` segments (all of them if None)
          - full Hessian COLUMNS at `n_cols` indices spread across the segment range
          - the same two things for the PRIOR-ONLY loss (student-t + soft barrier + mean
            constraint [+ drift-profile penalty if its weight > 0]), so the analytic-prior
            curvature can be separated from the data/likelihood curvature.

        HVPs are computed diag_chunk columns at a time via jax.vmap (shares the primal
        forward/backward pass across the chunk) to bound memory -- doing all columns in one
        vmap would be the OOM-prone jacfwd(jacrev(.)) this whole investigation is trying to
        avoid computing at full 4000-parameter scale.

        Results (numpy arrays + metadata) are pickled to `out_path`
        (default fit_result/dedx_hessian_b{batch_idx}.pkl) and returned.
        """
        import pickle as _pickle
        i = int(batch_idx)
        tracks_np = dataloader_sim[i].reshape(-1, len(self.sim_track_fields))
        selected_tracks_sim = jax.device_put(tracks_np)
        evts_sim = self.get_batch_global_event_ids(dataloader_sim, i, selected_tracks_sim)
        this_target = jax.device_put(target[i].reshape(-1, len(self.tgt_track_fields)))
        (ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks,
         ref_hit_prob, ref_event, ref_pixel_id) = self.get_simulated_target(this_target, i, evts_sim, regen=False)

        _ds = getattr(dataloader_sim, 'dataset', None)
        if _ds is None or not hasattr(_ds, 'get_parent_segment_ids'):
            raise RuntimeError("dataloader_sim.dataset has no get_parent_segment_ids; cannot run dEdx-Hessian analysis")
        pids = _ds.get_parent_segment_ids(i)
        parent_ids = jax.device_put(jnp.asarray(pids, dtype=jnp.int32))
        n_orig = int(jnp.max(parent_ids).item()) + 1
        log_dedx, _ = self._get_or_init_dedx_state(i, n_orig)

        dx_idx = self.sim_track_fields.index('dx')
        total_len_cm = float(jnp.sum(jnp.where(parent_ids >= 0, selected_tracks_sim[:, dx_idx], 0.0)))
        logger.info(f"[DEDX-HESS] batch {i}: n_segments={n_orig} n_steps={selected_tracks_sim.shape[0]} "
                    f"total_track_len_cm={total_len_cm:.2f}")

        loss_fn = self.compute_loss(
            selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks,
            ref_hit_prob, ref_event, ref_pixel_id, log_dedx=log_dedx, parent_ids=parent_ids,
            chain_angles_per_track=None, return_loss_closure=True)
        # loss_fn(log_dedx_pt) -> scalar TOTAL loss (hit_loss + dEdx regularizers), calib fixed.

        N = n_orig
        N_eff = min(N, int(max_diag)) if max_diag is not None else N
        n_cols = min(int(n_cols), N_eff)
        col_indices = sorted(set(int(round(x)) for x in np.linspace(0, N_eff - 1, n_cols)))

        def _sweep(fn, tag):
            grad_fn = jax.grad(fn)
            diag = np.full(N_eff, np.nan, dtype=np.float64)
            cols = {}
            t0 = time()
            for start in range(0, N_eff, int(diag_chunk)):
                end = min(start + int(diag_chunk), N_eff)
                idxs = np.arange(start, end)

                def _hvp_e(k):
                    e = jnp.zeros((N,), dtype=log_dedx.dtype).at[k].set(1.0)
                    _, hv = jax.jvp(grad_fn, (log_dedx,), (e,))
                    return hv

                hv_batch = np.asarray(jax.vmap(_hvp_e)(jnp.asarray(idxs)))
                for li, gi in enumerate(idxs):
                    diag[gi] = hv_batch[li, gi]
                    if int(gi) in col_indices:
                        cols[int(gi)] = hv_batch[li]
                logger.info(f"[DEDX-HESS] {tag}: chunk {start}:{end}/{N_eff} done ({time()-t0:.1f}s elapsed)")
            return diag, cols

        # Prior-only closure: re-evaluate JUST the analytic dEdx regularization terms as a
        # function of log_dedx, reconstructing dedx_sample exactly as loss_wrapper_combined does.
        # (norm_params / hit_loss do not enter here at all.)
        dedx_idx = self.sim_track_fields.index('dEdx')
        t_fields = self.sim_track_fields
        _valid_mask = (parent_ids >= 0)
        _safe_ids = jnp.clip(parent_ids, 0, N - 1)
        _dx = selected_tracks_sim[:, dx_idx]
        _lsf = (selected_tracks_sim[:, t_fields.index('length_scale_factor')]
                if 'length_scale_factor' in t_fields else None)

        def prior_only_fn(x):
            dedx_fitted = jnp.maximum(jnp.exp(jnp.take(x, _safe_ids)), 1e-3)
            dedx_sample = jnp.where(_valid_mask, dedx_fitted, selected_tracks_sim[:, dedx_idx])
            length_scale_factor = _lsf if _lsf is not None else 1.0
            if self.dedx_use_split_t:
                dedx_prior = split_student_t_nll(
                    dedx_sample, self.dedx_student_nu_l, self.dedx_student_nu_r,
                    self.dedx_student_loc * length_scale_factor,
                    self.dedx_student_scale_l, self.dedx_student_scale_r, weights=_dx
                ) * self.dedx_prior_weight
            else:
                dedx_prior = student_t_nll(
                    dedx_sample, self.dedx_student_nu, self.dedx_student_loc * length_scale_factor,
                    self.dedx_student_scale, weights=_dx
                ) * self.dedx_prior_weight
            dx_weights = _dx
            valid_step_mask = (dx_weights > 0)
            dedx_barrier = self.dedx_soft_barrier_weight * jnp.sum(
                dx_weights * valid_step_mask * jnp.square(jnp.maximum(0., dedx_sample - self.dedx_soft_barrier_threshold)))
            segment_valid = _valid_mask & valid_step_mask
            total_dx = jnp.sum(dx_weights * segment_valid)
            total_dedx_len = jnp.sum(dedx_sample * dx_weights * segment_valid)
            mean_dedx = total_dedx_len / jnp.maximum(total_dx, 1e-6)
            if _lsf is not None:
                mean_scale_factor = jnp.sum(length_scale_factor * dx_weights * segment_valid) / jnp.maximum(total_dx, 1e-6)
                scaled_mean_target = self.dedx_mean_constraint_target * mean_scale_factor
            else:
                scaled_mean_target = self.dedx_mean_constraint_target
            dedx_mean_penalty = self.dedx_mean_constraint_weight * jnp.square(mean_dedx - scaled_mean_target)
            dedx_drift_penalty = jnp.float32(0.)
            if self.dedx_drift_profile_weight > 0 and 'z' in t_fields:
                _zc = jnp.abs(selected_tracks_sim[:, t_fields.index('z')])
                _w = dx_weights * segment_valid
                _W = jnp.sum(_w) + 1e-9
                _y = jnp.log(jnp.maximum(dedx_sample, 1e-3))
                _zb = jnp.sum(_w * _zc) / _W
                _yb = jnp.sum(_w * _y) / _W
                _cov = jnp.sum(_w * (_zc - _zb) * (_y - _yb)) / _W
                _var = jnp.sum(_w * (_zc - _zb) ** 2) / _W + 1e-9
                _trend = _cov / jnp.sqrt(_var)
                dedx_drift_penalty = self.dedx_drift_profile_weight * jnp.square(_trend)
            return dedx_prior + dedx_barrier + dedx_mean_penalty + dedx_drift_penalty

        logger.info(f"[DEDX-HESS] batch {i}: computing TOTAL-loss diagonal/columns "
                    f"({N_eff} of {N} segments, chunk={diag_chunk}, n_cols={len(col_indices)})...")
        diag_total, cols_total = _sweep(loss_fn, 'total')
        logger.info(f"[DEDX-HESS] batch {i}: computing PRIOR-only diagonal/columns...")
        diag_prior, cols_prior = _sweep(prior_only_fn, 'prior')

        results = {
            'batch': i, 'n_segments': N, 'n_eff': N_eff, 'col_indices': col_indices,
            'total_track_len_cm': total_len_cm,
            'diag_total': diag_total, 'cols_total': cols_total,
            'diag_prior': diag_prior, 'cols_prior': cols_prior,
            'dedx_prior_weight': self.dedx_prior_weight,
            'dedx_soft_barrier_weight': self.dedx_soft_barrier_weight,
            'dedx_soft_barrier_threshold': self.dedx_soft_barrier_threshold,
            'dedx_mean_constraint_weight': self.dedx_mean_constraint_weight,
            'dedx_drift_profile_weight': self.dedx_drift_profile_weight,
        }
        out_path = out_path or f'fit_result/dedx_hessian_b{i}.pkl'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            _pickle.dump(results, f)
        logger.info(f"[DEDX-HESS] batch {i}: saved {out_path}")
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
                        # Vectorized first-occurrence lookup (identical to the per-segment loop, ~100x faster)
                        _valid = _raw_pids >= 0
                        _uniq, _first = np.unique(_raw_pids[_valid], return_index=True)
                        _true_dedx[_uniq] = _batch_flat[np.flatnonzero(_valid)[_first], _dedx_field_idx].astype(np.float32)
                        self._batch_true_dedx[_bi] = _true_dedx
                        # Per-segment physical length: sum of the sub-step dx. Needed to
                        # reconstruct the LENGTH-WEIGHTED mean dEdx offline -- that is the
                        # quantity dedx_mean_constraint acts on, and without it the true mean
                        # cannot be compared against the constraint target.
                        _dx_field_idx = (self.sim_track_fields.index('dx')
                                         if 'dx' in self.sim_track_fields else None)
                        if _dx_field_idx is not None:
                            self._batch_seg_dx[_bi] = np.bincount(
                                _raw_pids[_valid],
                                weights=_batch_flat[np.flatnonzero(_valid), _dx_field_idx],
                                minlength=_n_orig).astype(np.float32)
                            # Arc-length-matched truth reference, so the dEdx diagnostic can be
                            # read against the TARGET rather than against the sim input file.
                            try:
                                self._build_true_dedx_matched(_bi, _batch_flat, _raw_pids, _n_orig, target)
                            except Exception as _e:
                                logger.warning(f"true-dEdx matching failed for batch {_bi}: {_e}")
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
                    self._cur_total_iter = total_iter  # for the calibration-start gate in process_grads
                    _dedx_active  = self.fit_dedx and (total_iter >= self.dedx_start_iter)
                    # Adaptive freeze: latch once the dEdx MAE history plateaus (mean over the last
                    # window vs the previous window changes < rtol) -> robust across seeds/batch sizes.
                    if self._dedx_freeze_adaptive and _dedx_active and self._dedx_freeze_triggered is None and total_iter >= self._dedx_freeze_min_iter:
                        _mh = [m for m in self.training_history.get('dedx_mae_iter', []) if m == m]
                        _W = self._dedx_freeze_window
                        if len(_mh) >= 2 * _W:
                            _recent = float(np.mean(_mh[-_W:])); _prev = float(np.mean(_mh[-2*_W:-_W]))
                            if _prev > 0 and abs(_prev - _recent) / _prev < self._dedx_freeze_rtol:
                                self._dedx_freeze_triggered = total_iter
                                logger.info(f"Adaptive dEdx freeze triggered at iter {total_iter} (MAE plateau: {_prev:.4f}->{_recent:.4f})")
                    _eff_freeze = self._dedx_freeze_triggered if self._dedx_freeze_adaptive else self.dedx_freeze_iter
                    _dedx_frozen  = _dedx_active and (_eff_freeze is not None) and (_eff_freeze >= 0) and (total_iter >= _eff_freeze)
                    _use_dedx     = _dedx_active  # still inject dEdx values into the loss even when frozen

                    # Reset calibration Adam state once at the exact moment dEdx activates.
                    # Guard with i_bt % sz_mini_bt == 0 so the reset never fires mid-window
                    # when mini-batch gradient averaging is in use (sz_mini_bt > 1).
                    _at_window_start = (i_bt % max(1, self.sz_mini_bt) == 0)
                    if _use_dedx and not self._dedx_adam_reset_done and _at_window_start:
                        logger.info(
                            f"dEdx activated at iteration {total_iter}: resetting calibration Adam state."
                        )
                        self.opt_state = self._reinit_opt_state_keep_schedule()
                        self._dedx_adam_reset_done = True

                    # Reset calibration Adam state once when dEdx is frozen so the
                    # optimiser adapts cleanly to the new (dEdx-free gradient) landscape.
                    if _dedx_frozen and not self._dedx_freeze_adam_reset_done and _at_window_start:
                        logger.info(
                            f"dEdx frozen at iteration {total_iter}: resetting calibration Adam state "
                            f"for fine-tuning phase."
                        )
                        self.opt_state = self._reinit_opt_state_keep_schedule()
                        self._dedx_freeze_adam_reset_done = True

                    if _use_dedx and i in self._batch_parent_ids:
                        _parent_ids = jax.device_put(jnp.asarray(self._batch_parent_ids[i], dtype=jnp.int32))
                        _n_orig = int(jnp.max(_parent_ids).item()) + 1
                        # MINIMUM-LENGTH CUT (LARND_DEDX_MIN_DX, default 0.0 = off).
                        # Both the likelihood and the student-t prior are dx-weighted, so a
                        # segment's curvature scales with its length: a 0.036 cm segment was
                        # measured to have 1044x less curvature than the median and ran away to
                        # -82% of truth, dominating the per-segment MAE. Such segments are not
                        # identifiable; freeze them at the prior centre instead of fitting them.
                        if self._dedx_min_dx > 0 and i not in self._dedx_frozen_mask:
                            _sd_all = self._batch_seg_dx.get(i)
                            if _sd_all is not None:
                                _sd = np.asarray(_sd_all)[:_n_orig]
                                _frozen = _sd < self._dedx_min_dx
                                self._dedx_frozen_mask[i] = jnp.asarray(_frozen)
                                if _frozen.any():
                                    logger.info(
                                        f"[DEDX-MINDX] batch {i}: freezing {int(_frozen.sum())}/{_n_orig} "
                                        f"segments with dx < {self._dedx_min_dx} cm "
                                        f"({100*_sd[_frozen].sum()/max(_sd.sum(),1e-9):.3f}% of the track length)")
                        if self._dedx_param_mode == 'scale_shape' and self._batch_seg_dx.get(i) is not None:
                            _sdx = np.asarray(self._batch_seg_dx[i])[:_n_orig]
                            _log_dedx = self._dedx_ss_log_dedx(i, _n_orig, _sdx)
                        else:
                            _log_dedx, _ = self._get_or_init_dedx_state(i, _n_orig)
                    else:
                        _parent_ids = None
                        _log_dedx = None

                    # Chain position local state for this batch
                    # Allow position-only mode (fit_dedx=False) as well as joint mode
                    # WARP-vs-UPDATE separation (all gates): the warp must be APPLIED on every
                    # iteration once a chain state exists, otherwise calibration gradients are
                    # computed against the NOMINAL straight line (silent geometry revert). Only the
                    # chain UPDATE is gated by chain_update_freq / chain_start_iter / freeze.
                    _chain_available = (self.fit_chain_positions and
                                        i in self._batch_chain_contexts and
                                        (_log_dedx is not None or not self.fit_dedx
                                         or total_iter < self.dedx_start_iter))
                    _chain_active = (_chain_available and
                                     total_iter >= self._chain_start_iter and
                                     self._freq_gate('chain', i, self._chain_update_freq))
                    _chain_frozen_now = (self._chain_freeze_iter > 0 and total_iter >= self._chain_freeze_iter)
                    if _chain_available:
                        _chain_states   = self._get_or_init_chain_state(i)
                        _chain_angles_pt = [s['angles'] for s in _chain_states]
                    else:
                        _chain_angles_pt = None

                    # ── Study 2 hook: evaluate the objective at the truth point vs the fit
                    # point for this batch, then move on WITHOUT taking a step. Placed here
                    # because _log_dedx / _parent_ids / _chain_angles_pt / selected_tracks_sim /
                    # ref_* are all in scope and built exactly as a real fit builds them --
                    # which is the whole point of hooking the loop rather than reimplementing it.
                    if self._truthpoint_ckpt is not None:
                        if i not in self._truthpoint_done:
                            self._eval_truth_point(
                                i, selected_tracks_sim,
                                (ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z,
                                 ref_ticks, ref_hit_prob, ref_event, ref_pixel_id),
                                _log_dedx, _parent_ids, epoch)
                            self._truthpoint_done.add(i)
                            _ndone, _ntot = len(self._truthpoint_done), len(dataloader_sim)
                            if _ndone >= _ntot:
                                logger.info(f"[TRUTHPOINT] all {_ntot} batches evaluated; saving")
                                self.save_checkpoint(total_iter)
                                terminate_fit = True
                        if terminate_fit:
                            break
                        continue

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

                    # Diffusion-annealing schedule (less-local loss): width factor decays start->1
                    # over diff_anneal_iters; param-space scale = factor**2 (width ~ sqrt(param)).
                    _diff_scale = 1.0
                    if getattr(self, '_diff_anneal_start', 1.0) > 1.0 and total_iter < getattr(self, '_diff_anneal_iters', 0):
                        _frac = 1.0 - total_iter / float(self._diff_anneal_iters)
                        _wf = 1.0 + (self._diff_anneal_start - 1.0) * _frac
                        _diff_scale = _wf * _wf

                    # loss
                    loss_val, grads, _grads_dedx, _grads_chain, aux, _, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, epoch=epoch, with_loss=True, with_grad=True, log_dedx=_log_dedx, parent_ids=_parent_ids, chain_angles_per_track=_chain_angles_pt, diff_scale=_diff_scale)

                    # Apply per-segment dEdx update immediately after every step.
                    # Default = Adam (unchanged). LARND_DEDX_OPT=gn uses the structured Newton
                    # step (banded + Sherman-Morrison); it falls back to Adam whenever the solve
                    # fails to find descent, so the run can never stall on it.
                    if (_grads_dedx is not None and _log_dedx is not None
                            and self._dedx_min_dx > 0 and i in self._dedx_frozen_mask):
                        # frozen segments keep their initial value: kill their gradient entirely
                        _fm = self._dedx_frozen_mask[i]
                        if _fm.shape[0] == _grads_dedx.shape[0]:
                            _grads_dedx = jnp.where(_fm, 0.0, _grads_dedx)
                    if _grads_dedx is not None and _log_dedx is not None and not _dedx_frozen:
                        _did_gn = False
                        if self._dedx_opt_mode == 'gn' and not _chain_available:
                            _lf = self.compute_loss(
                                selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y,
                                ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id,
                                epoch=epoch, log_dedx=_log_dedx, parent_ids=_parent_ids,
                                chain_angles_per_track=None, return_loss_closure=True)
                            _newx, _info = self._dedx_gn_step(
                                i, _log_dedx, _lf, _grads_dedx, self._batch_seg_dx.get(i),
                                total_iter=total_iter)
                            if _newx is not None:
                                _, _ost = self._get_or_init_dedx_state(i, len(_log_dedx))
                                self._dedx_cache[i] = {'log_dedx': _newx, 'opt_state': _ost}
                                self._dedx_gn_stats['ok'] += 1
                                _did_gn = True
                                if total_iter % 50 == 0:
                                    logger.info(
                                        f"[DEDX-GN] it{total_iter} b{i} N={_info['N']} "
                                        f"dloss={_info['dloss']:.4g} lam={_info['lam']:.3g} "
                                        f"|step|max={_info['step_max']:.3g} bt={_info['backtracks']}")
                            else:
                                self._dedx_gn_stats['fallback'] += 1
                        if not _did_gn:
                            if (self._dedx_param_mode == 'scale_shape'
                                    and self._batch_seg_dx.get(i) is not None):
                                self._process_dedx_grads_ss(
                                    i, _grads_dedx, np.asarray(self._batch_seg_dx[i])[:len(_log_dedx)])
                            else:
                                self._process_dedx_grads(i, _grads_dedx, _log_dedx)

                    # Geometry block: per-track chain-angle update. Default = Adam (unchanged path);
                    # 'lbfgs' = jitted L-BFGS geometry solve (calibration held at current, as a traced
                    # arg) alternating with the calibration gradient step applied below.
                    if _chain_active and _grads_chain is not None and not _chain_frozen_now:
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
                        if self._freq_gate('posres', i, self._pos_residual_freq):
                            _pos_res = self._compute_pos_residual_batch(i)
                            if _pos_res is not None:
                                self.training_history.setdefault('pos_residual_iter', []).append(_pos_res)
                                # Record WHICH batch this residual belongs to. pos_residual is a
                                # per-batch quantity and batches differ enormously (881 um median
                                # vs individual batches spanning 62-3051 um), so a bare time series
                                # is only interpretable if the visit order is known. Without this,
                                # a --shuffle_bt run cannot be compared against a sequential one at
                                # all -- which is exactly how the SHUFFLE-vs-ANNEALLONG comparison
                                # went wrong. Additive; costs one int per recorded residual.
                                self.training_history.setdefault('pos_residual_batch', []).append(int(i))
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

                            if total_iter % print_freq == 0:
                                # deepcopy: get_size_history() returns a global MUTABLE dict; appending it raw
                                # aliases every entry to the same object (useless history, growing pickle)
                                self.training_history['size_history'].append(copy.deepcopy(get_size_history()))
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
                                        (None if os.environ.get('LARND_KEEP_TARGETS') else shutil.rmtree(self.target_dir, ignore_errors=True))
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

                        if total_iter % print_freq == 0:
                            self.training_history['size_history'].append(copy.deepcopy(get_size_history()))
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
                                    (None if os.environ.get('LARND_KEEP_TARGETS') else shutil.rmtree(self.target_dir, ignore_errors=True))
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
                            _tm = self._batch_true_dedx_matched.get(i)
                            if _tm is not None and len(_tm) == len(_cur_dedx):
                                _ok = np.isfinite(_tm)
                                self.training_history['dedx_mae_truth_iter'].append(
                                    float(np.mean(np.abs(_cur_dedx[_ok] - _tm[_ok]))) if _ok.any() else float('nan'))
                            else:
                                self.training_history['dedx_mae_truth_iter'].append(float('nan'))
                            if _dedx_frozen:
                                logger.debug(f"dEdx FROZEN (iter {total_iter}): MAE vs truth = {_mae:.4f}")
                        else:
                            # Warm-up phase: dEdx not yet active — keep lists aligned with NaN
                            self.training_history['dedx_mae_iter'].append(float('nan'))
                            self.training_history['dedx_mae_truth_iter'].append(float('nan'))

            self.save_checkpoint(total_iter)

            if os.path.exists(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl'):
                os.remove(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl')

            if os.path.exists(self.target_dir):
                (None if os.environ.get('LARND_KEEP_TARGETS') else shutil.rmtree(self.target_dir, ignore_errors=True))

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

            # LARND_SCAN_2D="pA,pB" -> scan the TWO named parameters on a joint grid instead of
            # doing the usual independent 1-D sweeps. Unset = previous behaviour exactly.
            #
            # WHY THIS MATTERS. Every scan so far is a 1-D SLICE with the other four parameters
            # frozen, which is why they manufacture a ~25x sensitivity of lifetime to any charge
            # normalisation error: lifetime is the only knob left that can change total charge, so
            # it absorbs the whole thing. In a real fit A_b is free and absorbs it instead. A 2-D
            # grid fixes this WITHOUT needing a minimiser: minimising the loss over pB at each pA
            # gives the PROFILE likelihood in pA, which is the quantity a displaced minimum has to
            # be measured on before it can be called a bias.
            #
            # Cost is nb_steps^2 x nbatch (vs nb_steps x 5 x nbatch), so use a smaller grid.
            _scan2d = os.environ.get('LARND_SCAN_2D', '').strip()
            if _scan2d:
                # NOTE: accept ':' as well as ','. `sbatch --export` is itself a comma-separated
                # list, so a value containing a comma is silently split in half before it ever
                # reaches python -- use LARND_SCAN_2D="lifetime:Ab" from SLURM.
                _norm = _scan2d.replace(';', ',').replace(':', ',')
                _parts = [x.strip() for x in _norm.split(',') if x.strip()]
                if len(_parts) < 2:
                    raise ValueError(
                        f"LARND_SCAN_2D needs two parameter names, got {_scan2d!r}. "
                        f"From sbatch use a colon (LARND_SCAN_2D='lifetime:Ab') -- --export "
                        f"splits on commas.")
                _pA, _pB = _parts[:2]
                _w2 = float(os.environ.get('LARND_SCAN_WINDOW', 0) or 0)
                def _axis(pn):
                    lo, hi = ranges[pn]['down'], ranges[pn]['up']
                    if _w2 > 0:
                        nm = ranges[pn]['nom']
                        lo = max(nm * (1.0 - _w2), ranges[pn]['down'])
                        hi = min(nm * (1.0 + _w2), ranges[pn]['up'])
                    return np.linspace(lo, hi, nb_steps)
                _gA, _gB = _axis(_pA), _axis(_pB)
                logger.info(f"[SCAN-2D] batch {i}: {_pA} x {_pB} on a {nb_steps}x{nb_steps} grid "
                            f"({_pA} {_gA[0]:.4g}..{_gA[-1]:.4g}, {_pB} {_gB[0]:.4g}..{_gB[-1]:.4g})")
                for _ia in tqdm(range(nb_steps)):
                    for _ib in range(nb_steps):
                        start_time = time()
                        self.current_params = self.ref_params.replace(
                            **{_pA: float(_gA[_ia]), _pB: float(_gB[_ib])})
                        loss_val, grads, _, _, aux, _, _ = self.compute_loss(
                            selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y,
                            ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id,
                            with_loss=True, with_grad=True, use_physical_params=True)
                        stop_time = time()
                        for par in self.relevant_params_list:
                            self.training_history[par+"_grad"].append(getattr(grads, par).item())
                        self.training_history['step_time'].append(stop_time - start_time)
                        self.training_history['losses_iter'].append(loss_val.item())
                        _auxs = {k: serialize_value(v) for k, v in aux.items()} if aux is not None else {}
                        self.training_history['aux_iter'].append(_auxs)
                        # every parameter is logged each step, so the (pA, pB) grid position is
                        # recoverable offline from <param>_iter alone -- no new storage format.
                        for par in self.relevant_params_list:
                            _v = getattr(self.current_params, par)
                            self.training_history[par + '_iter'].append(
                                _v if type(_v) == float else _v.item())
                        self.training_history['size_history'].append(copy.deepcopy(get_size_history()))
                        if jax.devices()[0].platform == 'gpu':
                            self.training_history['memory'].append(jax.devices("gpu")[0].memory_stats())
                with open(f'fit_result/{self.test_name}/history_scan2d_batch{i}_{self.out_label}.pkl', "wb") as f_history:
                    pickle.dump(self.training_history, f_history)
                if os.path.exists(f'fit_result/{self.test_name}/history_scan2d_batch{i-1}_{self.out_label}.pkl'):
                    os.remove(f'fit_result/{self.test_name}/history_scan2d_batch{i-1}_{self.out_label}.pkl')
                continue

            for param in self.relevant_params_list:
                lower = ranges[param]['down']
                upper = ranges[param]['up']
                # LARND_SCAN_WINDOW (default unset = full ranges.py interval, bit-identical to
                # before) narrows the grid to nom*(1 +/- w) so the SAME number of steps resolves
                # the minimum far better. Needed because the full-range grid is ~11x coarser than
                # the likelihood width (lifetime step 10.2% of truth vs a 0.88% Fisher sigma), so
                # the parabola vertex is an extrapolation and shifts ~2.4 points between a 3- and
                # a 5-point fit. Cost is unchanged: it scales as steps x batches, not as the span.
                _w = float(os.environ.get('LARND_SCAN_WINDOW', 0) or 0)
                if _w > 0:
                    nom = ranges[param]['nom']
                    lower = max(nom * (1.0 - _w), ranges[param]['down'])
                    upper = min(nom * (1.0 + _w), ranges[param]['up'])
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

                    self.training_history['size_history'].append(copy.deepcopy(get_size_history()))
                    if jax.devices()[0].platform == 'gpu':
                        self.training_history['memory'].append(jax.devices("gpu")[0].memory_stats())

                with open(f'fit_result/{self.test_name}/history_{param}_batch{i}_{self.out_label}.pkl', "wb") as f_history:
                    pickle.dump(self.training_history, f_history)
                if os.path.exists(f'fit_result/{self.test_name}/history_{param}_batch{i-1}_{self.out_label}.pkl'):
                    os.remove(f'fit_result/{self.test_name}/history_{param}_batch{i-1}_{self.out_label}.pkl')

        if os.path.exists(self.target_dir):
            (None if os.environ.get('LARND_KEEP_TARGETS') else shutil.rmtree(self.target_dir, ignore_errors=True))

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
            (None if os.environ.get('LARND_KEEP_TARGETS') else shutil.rmtree(self.target_dir, ignore_errors=True))


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
        # Assemble the per-batch Hessian column-by-column via HVPs. Numerically identical to
        # jacfwd(jacrev(.)) (unit-tested to 0.0 max abs difference) but ~P times less peak
        # memory; the jacfwd path requested 129 GB for a single 400 cm batch.
        self._hess_via_hvp = True
        # Optional: restrict the GN step to the STIFF eigen-subspace of the Hessian, i.e.
        # discard components along the flattest `gn_soft_modes` directions. Measured on the
        # 60-step runaway of job 33123555: 99.3% of the total displacement lay in the 2-D soft
        # plane, and every individual step was a median 98% soft. 0 disables (default).
        self.gn_soft_modes = int(os.environ.get('LARND_GN_SOFT_MODES', '0'))
        # Curvature refresh cadence: recompute the Hessian only every Nth accepted step and
        # reuse it in between (loss and gradient are still recomputed every step). The
        # standalone prototype has shipped this as `--refresh` (default 5) since the start; it
        # was never ported here, so this fitter pays a full Hessian sweep over every batch
        # after every accepted step. Default 1 == previous behaviour exactly.
        self.gn_refresh = max(1, int(os.environ.get('LARND_GN_REFRESH', '1')))
        # --- Diagnosis (job gnA_*): the first LM step overshoots the sigmoid bounds by ~10x,
        # WORST when starting closest to truth. Root cause: Marquardt/diagonal damping
        # `A = H + lam*diag(|H_ii|)` regularizes each parameter by its OWN curvature, but the
        # softest eigen-directions here are a MIXED (lifetime, long_diff) plane (not axis
        # aligned) with eigenvalues spanning 3.3e3..1.3e10 (condition ~4e6). Diagonal damping
        # leaves the off-diagonal correlation that produces the near-null combination almost
        # untouched, so the damped solve can still return a huge step along it even though
        # every individual diag(H_ii) looks large. All three knobs below default OFF/'diag'
        # == exactly the pre-existing behaviour.
        #   LARND_GN_DAMP_MODE=identity  : A = H + lam*scale*I  (scale = lambda_max(H)), i.e.
        #                                   isotropic damping that bounds ALL directions
        #                                   uniformly instead of per-axis.
        #   LARND_GN_DAMP_MODE=abseig    : saddle-free Newton — replace H by
        #                                   V diag(max(|lambda|,floor)) V^T before damping, so a
        #                                   negative-curvature direction becomes a descent
        #                                   direction of the same magnitude instead of an ascent
        #                                   one. Addresses the measured INDEFINITE Hessian at the
        #                                   init point (eig -2.51 at 1% offset), which plain
        #                                   Marquardt damping cannot fix without also destroying
        #                                   the step. Diagonal damping is then applied on top.
        #   LARND_GN_LAMBDA0_SCALE=1     : seed lam at loop start as gn_lambda0 * lambda_max(H)
        #                                   instead of the bare fitter-wide constant, so the
        #                                   initial damping tracks the Hessian's own scale
        #                                   (which varies by >6 orders of magnitude with offset).
        #   LARND_GN_STEP_CAP=<float>    : hard trust-region cap on the norm-space step length
        #                                   (post soft-mode projection); step is rescaled, not
        #                                   the iteration rejected. 0 = off.
        self.gn_damp_mode = os.environ.get('LARND_GN_DAMP_MODE', 'diag')
        self.gn_lambda0_scale = os.environ.get('LARND_GN_LAMBDA0_SCALE', '0') not in ('0', '', 'false', 'False')
        self.gn_step_cap = float(os.environ.get('LARND_GN_STEP_CAP', '0'))
        # --- Bound-aware stepping (jobs gnD_*). The measured failure is NOT that LM takes a
        # bad step: the oversized step DOES decrease the loss, so LM accepts it, and only then
        # does the post-hoc saturation guard fire and revert the whole run. Backtracking never
        # engages because it only triggers on a loss INCREASE. The sigmoid box |theta| <= 3.5
        # is therefore an unmodelled constraint. Two standard bound-constrained-Newton pieces,
        # both default OFF (0 == exactly the pre-existing behaviour):
        #   LARND_GN_BOUND_TR=<tau>  fraction-to-boundary: scale the step by the largest
        #                            alpha <= 1 that keeps every |theta_j + alpha*step_j| <=
        #                            tau*bound. Unlike LARND_GN_STEP_CAP this is scale-free and
        #                            caps exactly the quantity the guard measures. Typical 0.95.
        #   LARND_GN_ACTIVE_SET=1    active-set reduction: a coordinate already pressed against
        #                            its bound whose step pushes it further OUT is removed from
        #                            the solve and the Newton system re-solved on the remaining
        #                            free block. Without this, fraction-to-boundary drives
        #                            alpha->0 the moment one param parks at a bound and stalls
        #                            the whole step. This is the saturation-aware replacement
        #                            for rank-based LARND_GN_SOFT_MODES: it targets the
        #                            directions that are actually blocked rather than the ones
        #                            that merely have small eigenvalues (measured at the init
        #                            point, those two sets coincide only within ~5% of truth).
        #   LARND_GN_BOUND=<x>       the box half-width; must match the saturation guard (3.5).
        # LARND_GN_STALL_PATIENCE=<K>: how many CONSECUTIVE iterations may fail to find a loss
        # decrease before the loop gives up. 1 == exactly the pre-existing behaviour (terminate on
        # the first stall). >1 refreshes loss/grad/Hessian and retries with lambda knocked back
        # down, which recovers runs killed by a noise-level stall — see the long comment at the
        # stall site for the two-run measurement that motivates this.
        self.gn_stall_patience = max(1, int(os.environ.get('LARND_GN_STALL_PATIENCE', '1')))
        self.gn_bound_tr = float(os.environ.get('LARND_GN_BOUND_TR', '0'))
        self.gn_active_set = os.environ.get('LARND_GN_ACTIVE_SET', '0') not in ('0', '', 'false', 'False')
        self.gn_bound = float(os.environ.get('LARND_GN_BOUND', '3.5'))

    def _bounded_step(self, A, g, theta):
        """Solve A step = -g subject to the sigmoid box |theta| <= gn_bound.

        Returns (step, n_blocked). With both knobs off this is exactly `-solve(A, g)`.

        Active set: iteratively drop coordinates that sit at a bound and whose Newton
        component points further outward, re-solving the reduced system each time (at most
        len(g) drops, so the loop always terminates). Then fraction-to-boundary rescales
        the surviving step so it stops just short of the box face.
        """
        n = len(g)
        free = np.ones(n, dtype=bool)
        step = np.zeros(n)
        lim = self.gn_bound
        at_bound = np.abs(theta) >= (self.gn_bound_tr if self.gn_bound_tr > 0 else 1.0) * lim

        for _ in range(n + 1):
            step = np.zeros(n)
            idx = np.where(free)[0]
            if idx.size == 0:
                break
            step[idx] = -np.linalg.solve(A[np.ix_(idx, idx)], g[idx])
            if not self.gn_active_set:
                break
            # Blocked = free, already against the bound, and pushed further out.
            blocked = free & at_bound & (np.sign(step) == np.sign(theta)) & (step != 0.0)
            if not blocked.any():
                break
            free = free & ~blocked

        return step, int(n - free.sum())

    def _frac_to_boundary(self, step, theta):
        """Rescale `step` by the largest alpha <= 1 keeping every |theta_j + alpha*step_j|
        inside gn_bound_tr*gn_bound. Applied LAST — after soft-mode projection and the norm
        cap — since either of those can push an already-safe step back out of the box.
        Returns `step` unchanged when the knob is off."""
        if self.gn_bound_tr <= 0:
            return step
        step = np.array(step, dtype=float, copy=True)
        edge = self.gn_bound_tr * self.gn_bound
        _pre = float(np.linalg.norm(step))
        # A coordinate that is ALREADY at/outside the edge and is pushed further out admits no
        # positive alpha. Scaling the whole step by that alpha (<=0) would zero every other
        # coordinate too — a silent stall. Zero only the offending component and keep the rest
        # of the Newton direction. (With LARND_GN_ACTIVE_SET=1 these are dropped from the solve
        # already, so this is the standalone-safety branch.)
        _pinned = []
        for j in range(len(step)):
            if step[j] == 0.0:
                continue
            if abs(theta[j]) >= edge and np.sign(step[j]) == np.sign(theta[j]):
                step[j] = 0.0
                _pinned.append(j)
        alpha = 1.0
        for j in range(len(step)):
            if step[j] == 0.0:
                continue
            if abs(theta[j] + step[j]) > edge:
                face = edge if step[j] > 0 else -edge
                _a = (face - theta[j]) / step[j]
                if _a < alpha:
                    alpha = max(_a, 0.0)
        if alpha < 1.0:
            step = step * alpha
        if alpha < 1.0 or _pinned:
            logger.info(f"[GN-BOUNDTR] fraction-to-boundary alpha={alpha:.4g} (edge {edge:.3g})"
                        + (f", pinned coords {_pinned}" if _pinned else "")
                        + f": |step| {_pre:.4g} -> {np.linalg.norm(step):.4g}")
        return step

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
            # `parent_ids` is stored padded to max(batch_nsteps) over the batch set that the
            # ORIGINAL run loaded (dataio.py:419). A run that loads fewer/shorter batches pads
            # to a smaller max, so the stored vector is longer than this run's tracks array and
            # the gather at ~line 1905 fails to broadcast. Batch composition is deterministic and
            # independent of max_nbatch -- only the pad length differs -- so truncating is exact.
            # The assertion refuses to drop anything that is not padding.
            if pi is not None and i < len(dataloader_sim):
                _cur = int(dataloader_sim[i].shape[0])
                if _cur < int(pi.shape[0]):
                    _dropped_real = int(jnp.sum(pi[_cur:] >= 0))
                    if _dropped_real:
                        raise ValueError(
                            f"batch {i}: truncating parent_ids {int(pi.shape[0])}->{_cur} would "
                            f"drop {_dropped_real} REAL segments - refusing (config mismatch)")
                    pi = pi[:_cur]
            an = [jnp.asarray(s['angles']) for s in chain[i]] if i in chain else None
            # cross-check restored coefficients against rebuilt chain contexts. The expected
            # count is basis-dependent: the angle basis stores 2*n_chain (theta,phi per node),
            # the spline basis stores 2*K sine-mode coefficients with K=_spline_K(total_len).
            if an is not None and i in self._batch_chain_contexts:
                ctxs = self._batch_chain_contexts[i]
                if self._chain_basis == 'spline':
                    exp = [2 * _spline_K(c.total_len, self._chain_spline_knot_cm) for c in ctxs]
                else:
                    exp = [2 * int(c.n_chain) for c in ctxs]
                got = [int(a.shape[0]) for a in an]
                if exp != got:
                    raise ValueError(f"batch {i}: checkpoint chain coefficients {got} do not match "
                                     f"rebuilt contexts {exp} for basis '{self._chain_basis}' "
                                     f"— input data/config mismatch")
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

    def _full_batch_lgh(self, cache, with_hess=True):
        """Accumulate loss, gradient and (optionally) Hessian over all cached batches.

        with_hess=False returns H=None and skips the curvature work entirely -- used by the
        refresh cadence to reuse a stale Hessian between recomputes. Default True preserves
        the original behaviour for every existing caller.
        """
        rl = self.relevant_params_list
        n = len(rl)
        tot = 0.0
        g = np.zeros(n)
        H = np.zeros((n, n)) if with_hess else None
        use_ggn = (self.gn_curvature == 'ggn')
        _t0 = time() if _GN_PROFILE else None
        for (i, st, ref) in cache:
            _tb0 = time() if _GN_PROFILE else None
            lv, grads, _, _, _, hess, _ = self.compute_loss(
                st, i, *ref, with_loss=True, with_grad=True,
                with_hess=(with_hess and not use_ggn),
                **self._nuis(i))
            tot += float(lv)
            g += np.array([float(getattr(grads, k)) for k in rl])
            if with_hess:
                if use_ggn:
                    H += np.asarray(self._get_fisher_fn(i, st)(self._x5(), st))
                else:
                    H += np.array([[float(getattr(getattr(hess, ka), kb)) for kb in rl] for ka in rl])
            if _GN_PROFILE:
                logger.info(f"[GN-PROF] batch {i}: {time() - _tb0:.2f}s  "
                            f"peak_mem {_gn_mem_peak():.2f} GiB  with_hess={with_hess and not use_ggn}")
        if _GN_PROFILE:
            logger.info(f"[GN-PROF] full sweep (with_hess={with_hess}, curvature={self.gn_curvature}, "
                        f"nbatch={len(cache)}): {time() - _t0:.2f}s total, "
                        f"peak_mem {_gn_mem_peak():.2f} GiB")
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
            (None if os.environ.get('LARND_KEEP_TARGETS') else shutil.rmtree(self.target_dir, ignore_errors=True))

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
        # Diagnostics (cheap, always on): spectrum of the raw Hessian at the loop's entry
        # point, to characterise the ill-conditioning / mixed-mode structure driving the
        # oversized first step (see __init__ docstring for the env knobs that address it).
        _w0 = np.linalg.eigvalsh(0.5 * (H + H.T))
        logger.info(f"GN init Hessian: eig {np.array2string(_w0, precision=3)}  "
                    f"diag {np.array2string(np.diag(H), precision=3)}  "
                    f"cond {(_w0.max()/max(abs(_w0.min()),1e-300)):.3e}")
        if self.gn_lambda0_scale:
            lam = float(self.gn_lambda0) * float(np.abs(_w0).max())
            logger.info(f"[GN-LAMBDA0] scaling lam0 by lambda_max(H)={np.abs(_w0).max():.3e} -> lam={lam:.3e}")

        last_snorm = np.inf
        _stall = 0          # consecutive non-improving iterations (see gn_stall_patience)
        for it in range(self.gn_max_iter):
            Hs = 0.5 * (H + H.T)
            if self.gn_damp_mode == 'abseig':
                # Saddle-free Newton: replace Hs by V diag(max(|lambda|, floor)) V^T. The exact
                # Hessian at the toy's init point is INDEFINITE (measured eigenvalue -2.51 at 1%
                # offset), so along its negative-curvature eigenvector the Newton step ASCENDS.
                # Marquardt damping can only cure that by growing lam until the sign flips, which
                # is also the regime where the step is too short to be useful — hence damping
                # alone cannot separate "wrong direction" from "too long". Flipping the sign of
                # negative eigenvalues keeps the magnitude of the curvature information (so the
                # step stays Newton-scaled) while guaranteeing a descent direction.
                _w, _V = np.linalg.eigh(Hs)
                _flr = 1e-8 * max(float(np.abs(_w).max()), 1.0)
                _wf = np.maximum(np.abs(_w), _flr)
                _nneg = int((_w < 0).sum())
                Hs = (_V * _wf) @ _V.T
                if _nneg:
                    logger.info(f"[GN-ABSEIG] flipped {_nneg} negative eigenvalue(s) "
                                f"(raw eig {np.array2string(_w, precision=3)})")
            dscale = np.abs(np.diag(Hs))
            # Floor the Marquardt scaling: directions with ~zero curvature (e.g. a param
            # saturating its sigmoid bound) must still be damped, otherwise the solve
            # shoots unbounded steps along the flat direction.
            dfloor = 1e-6 * max(dscale.max(), 1.0)
            dscale = np.maximum(dscale, dfloor)
            # Isotropic alternative to Marquardt/diagonal damping: add lam*lambda_max(Hs)*I
            # instead of lam*diag(|H_ii|). Diagonal damping regularizes each parameter by its
            # OWN curvature and leaves near-null MIXED eigen-directions (large off-diagonal
            # correlation, e.g. the measured (lifetime, long_diff) plane) almost untouched even
            # when every individual H_ii looks large; isotropic damping bounds every direction,
            # including mixed ones, by construction. See __init__ for the env knob.
            if self.gn_damp_mode == 'identity':
                _iso_scale = max(float(np.abs(np.linalg.eigvalsh(Hs)).max()), dfloor)

            accepted = False
            step = np.zeros(len(rl))
            for _bt in range(self.gn_max_backtrack):
                if self.gn_damp_mode == 'identity':
                    A = Hs + (lam * _iso_scale) * np.eye(len(rl))
                else:
                    A = Hs + lam * np.diag(dscale)
                try:
                    _theta = np.array([float(getattr(self.norm_params, k)) for k in rl])
                    step, _nblk = self._bounded_step(A, g, _theta)
                    if _nblk:
                        logger.info(f"[GN-ACTIVESET] {_nblk} coordinate(s) blocked at the bound; "
                                    f"solved on the free {len(rl) - _nblk}-D block")
                except np.linalg.LinAlgError:
                    lam *= self.gn_lambda_up
                    continue
                if self.gn_soft_modes > 0:
                    # Remove the components of the step along the flattest eigenvectors of the
                    # (symmetrised) Hessian. These are the near-degenerate directions along
                    # which the loss decreases almost freely, so LM's loss-based accept test
                    # cannot police them.
                    _w, _V = np.linalg.eigh(0.5 * (Hs + Hs.T))
                    _soft = _V[:, :self.gn_soft_modes]              # eigh returns ascending
                    _before = float(np.linalg.norm(step))
                    step = step - _soft @ (_soft.T @ step)
                    logger.info(f"[GN-PROJ] dropped {self.gn_soft_modes} soft mode(s) "
                                f"(lambda_soft={_w[:self.gn_soft_modes]}): "
                                f"|step| {_before:.4g} -> {np.linalg.norm(step):.4g}")
                if self.gn_step_cap > 0:
                    # Hard trust-region cap on the norm-space step length: rescale (don't
                    # reject) an oversized step. Backstops the damping solve directly against
                    # the measured failure mode (a single LM step landing at norm-space
                    # distance ~9-46 when the sigmoid bound is at |theta|~3.5) regardless of
                    # which damping scheme produced it.
                    _snorm_pre = float(np.linalg.norm(step))
                    if _snorm_pre > self.gn_step_cap:
                        step = step * (self.gn_step_cap / _snorm_pre)
                        logger.info(f"[GN-STEPCAP] |step| {_snorm_pre:.4g} -> "
                                    f"{np.linalg.norm(step):.4g} (cap {self.gn_step_cap:.4g})")
                step = self._frac_to_boundary(step, _theta)
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
                # A STALL IS NOT PROOF OF CONVERGENCE. Measured on two runs of an identical
                # config at offset 0.05: at iter 1 the available improvement (~2e-4) was the same
                # size as the loss's run-to-run reproducibility (~5e-4). One run got a favourable
                # rounding, accepted a |step|=2.8e-07 no-op, and went on to reach loss 1995; the
                # other found no decrease, escalated lam 8x to 2.18e+02 and terminated for good
                # with 5584 -> 1939 of progress still on the table. Once lam is that large the
                # step is tiny and the loss delta sits under the noise, so further backtracking
                # cannot help — but a FRESH curvature/gradient evaluation can (that is precisely
                # what the surviving run stumbled into). Retry up to gn_stall_patience times,
                # re-solving from a re-evaluated landscape with lam knocked back down.
                _stall += 1
                if _stall < self.gn_stall_patience:
                    lam = max(lam / (self.gn_lambda_up ** self.gn_max_backtrack), float(self.gn_lambda0))
                    loss, g, H = self._full_batch_lgh(cache)
                    logger.info(f"GN iter {it}: no decrease after backtracking; STALL "
                                f"{_stall}/{self.gn_stall_patience} — refreshing curvature and "
                                f"retrying with lambda {lam:.2e} (loss {loss:.6f}).")
                    continue
                logger.info(f"GN iter {it}: no decrease after backtracking (lambda {lam:.2e}); stopping.")
                return 'stalled'
            _stall = 0

            if self.gn_refresh > 1 and ((it + 1) % self.gn_refresh) != 0:
                loss, g, _ = self._full_batch_lgh(cache, with_hess=False)   # reuse stale H
            else:
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
