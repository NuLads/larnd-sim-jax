"""
Standalone script: multi-track deflection chain fit.
Equivalent to local_deflection_chain_fit.ipynb run top-to-bottom.
"""
import sys, os, dataclasses, pickle, time
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/src')
os.chdir('/sdf/group/neutrino/pgranger/larnd-sim-jax')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax
from scipy.optimize import minimize

from optimize.dataio import TracksDataset, TgtTracksDataset
from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from larndsim.sim_jax import (
    simulate_wfs, simulate_probabilistic, pad_to_closest_multiple, get_roi_counts,
    simulate_drift_new, simulate_signals_new,
)
from optimize.dedx_utils import (
    split_student_t_nll,
    DEDX_SPLIT_NU_L, DEDX_SPLIT_NU_R,
    DEDX_SPLIT_SCALE_L, DEDX_SPLIT_SCALE_R,
    DEDX_SPLIT_LOC,
)

print('JAX devices:', jax.devices())

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_SIM = '/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5'
INPUT_TGT = '/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5'
MAX_BATCHES = int(os.environ.get('MAX_BATCHES', '6'))   # override via env
FIT_MODE    = os.environ.get('FIT_MODE', 'full')        # 'full' | 'no_pos'
MIN_TRACK_LEN_CM = 5.0
N_FINE_PER_SEG = 20
mean_dedx = 1.887
_X0_LAR_CM = 14.0
CHECKPOINT_FILE = f'chain_fit_{FIT_MODE}_b{MAX_BATCHES}.pkl'
EARLY_STOP_GNORM    = 5e-6   # grad-norm threshold for early stopping
EARLY_STOP_PATIENCE = 15     # consecutive steps below threshold before exit
os.makedirs('plots', exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
print('\n=== Loading data ===')
dataset_sim = TracksDataset(
    filename=INPUT_SIM, nevents=-1, max_nbatch=MAX_BATCHES, random_nevents=True,
    data_seed=1, track_len_sel=2.0, max_abs_costheta_sel=0.966,
    min_abs_segz_sel=15.0, track_z_bound=28.0, max_batch_len=200.0,
    chopped=True, pad=True, electron_sampling_resolution=0.01)
dataset_target = TgtTracksDataset(
    filename=INPUT_TGT, dataset_sim=dataset_sim,
    chopped=True, pad=True, electron_sampling_resolution=0.01)
fields = tuple(dataset_sim.get_track_fields())
tgt_fields = list(dataset_target.tgt_track_fields)
event_idx = fields.index('eventID')
track_idx = fields.index('trackID')
print(f'Loaded {len(dataset_sim)} batch(es)')

ds_raw = TracksDataset(
    filename=INPUT_TGT, nevents=-1, max_nbatch=MAX_BATCHES, random_nevents=True,
    data_seed=1, track_len_sel=2.0, max_abs_costheta_sel=0.966,
    min_abs_segz_sel=15.0, track_z_bound=28.0, max_batch_len=200.0,
    chopped=False, pad=False, electron_sampling_resolution=0.01)
fields_raw = list(ds_raw.get_track_fields())
x_s_r = fields_raw.index('x_start'); y_s_r = fields_raw.index('y_start'); z_s_r = fields_raw.index('z_start')
x_e_r = fields_raw.index('x_end');   y_e_r = fields_raw.index('y_end');   z_e_r = fields_raw.index('z_end')
ev_idx_r = fields_raw.index('eventID'); trk_idx_r = fields_raw.index('trackID')

raw_endpoint_map = {}
for bi in range(len(ds_raw)):
    rb = ds_raw[bi].reshape(-1, len(fields_raw))
    rb = rb[rb[:, ev_idx_r] >= 0]
    for ev_tr in np.unique(np.column_stack([rb[:, ev_idx_r], rb[:, trk_idx_r]]), axis=0):
        ev, tr = int(ev_tr[0]), int(ev_tr[1])
        mask = (rb[:, ev_idx_r] == ev) & (rb[:, trk_idx_r] == tr)
        segs = rb[mask]
        entry_pt = np.array([segs[0, x_s_r],  segs[0, y_s_r],  segs[0, z_s_r]])
        exit_pt  = np.array([segs[-1, x_e_r], segs[-1, y_e_r], segs[-1, z_e_r]])
        raw_endpoint_map[(ev, tr)] = (entry_pt, exit_pt, segs)

all_track_specs = []
for bi in range(len(dataset_sim)):
    batch = dataset_sim[bi].reshape(-1, len(fields))
    parent_ids = dataset_sim.get_parent_segment_ids(bi)
    real = batch[parent_ids >= 0]
    if len(real) == 0: continue
    for ev_tr in np.unique(np.column_stack([real[:, event_idx], real[:, track_idx]]), axis=0):
        ev, tr = int(ev_tr[0]), int(ev_tr[1])
        if (ev, tr) not in raw_endpoint_map: continue
        entry_pt, exit_pt, raw_segs = raw_endpoint_map[(ev, tr)]
        total_len = float(np.linalg.norm(exit_pt - entry_pt))
        if total_len < MIN_TRACK_LEN_CM: continue
        all_track_specs.append({'batch_idx': bi, 'ev': ev, 'tr': tr,
                                 'entry_pt': entry_pt, 'exit_pt': exit_pt,
                                 'total_len': total_len, 'raw_segs': raw_segs})
print(f'Found {len(all_track_specs)} tracks:')
for s in all_track_specs:
    print(f'  batch={s["batch_idx"]} ev={s["ev"]} tr={s["tr"]} L={s["total_len"]:.1f} cm')

# ── Detector ─────────────────────────────────────────────────────────────────
print('\n=== Loading detector ===')
Params = build_params_class(['Ab', 'kb'])
ref_params = load_detector_properties(
    Params, 'src/larndsim/detector_properties/module0.yaml',
    'src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml')
response, ref_params = load_lut('src/larndsim/detector_properties/response_44.npy', ref_params)
ab_nominal = 0.8348813503927325
kb_nominal  = 0.04860
params_true = ref_params.replace(Ab=jnp.array(ab_nominal), kb=jnp.array(kb_nominal))
print('Detector loaded.')

# ── Chain kinematics ─────────────────────────────────────────────────────────
def highland_sigma_mcs(step_cm, momentum_GeV=3.0):
    r = step_cm / _X0_LAR_CM
    return float((0.0136 / momentum_GeV) * np.sqrt(r) * (1.0 + 0.038 * np.log(r)))

def _build_local_basis(u):
    cy = jnp.linalg.cross(u, jnp.array([0.,1.,0.]))
    cx = jnp.linalg.cross(u, jnp.array([1.,0.,0.]))
    c  = jnp.where(jnp.linalg.norm(cy) > 1e-4, cy, cx)
    v  = c / jnp.linalg.norm(c)
    return v, jnp.linalg.cross(u, v)

def deflection_chain_forward(x0, theta0, phi0, dtheta, dphi, step_len):
    u0 = jnp.array([jnp.sin(theta0)*jnp.cos(phi0),
                    jnp.sin(theta0)*jnp.sin(phi0),
                    jnp.cos(theta0)])
    u0 = u0 / jnp.linalg.norm(u0)
    def step(carry, ang):
        pos, u = carry
        v, w = _build_local_basis(u)
        ct, st = jnp.cos(ang[0]), jnp.sin(ang[0])
        cp, sp = jnp.cos(ang[1]), jnp.sin(ang[1])
        u2 = ct*u + st*cp*v + st*sp*w
        u2 = u2 / jnp.linalg.norm(u2)
        return (pos + step_len*u, u2), (pos + step_len*u, u2)
    (pl, ul), (pm, um) = jax.lax.scan(step, (x0, u0), jnp.stack([dtheta, dphi], axis=1))
    positions  = jnp.concatenate([x0[None], pm, (pl + step_len*ul)[None]], axis=0)
    directions = jnp.concatenate([u0[None], um], axis=0)
    return positions, directions

def chain_to_fine_segments(pos, direc, dE_per_seg, fields, nf=N_FINE_PER_SEG):
    n  = pos.shape[0] - 1
    dE = dE_per_seg[:n]
    ps, pe = pos[:-1], pos[1:]
    ts = jnp.linspace(0., 1., nf+1)[:-1]
    te = ts + 1./nf
    dp = (pe - ps)[:, None, :]
    pfs = (ps[:,None,:] + ts[None,:,None]*dp).reshape(n*nf, 3)
    pfe = (ps[:,None,:] + te[None,:,None]*dp).reshape(n*nf, 3)
    pfm = 0.5*(pfs+pfe)
    sl = jnp.sqrt(jnp.sum((pfe-pfs)**2, axis=-1) + 1e-10)
    dEf = jnp.repeat(dE/nf, nf)
    out = jnp.zeros((n*nf, len(fields)), dtype=jnp.float32)
    fm  = {'x_start':pfs[:,0],'y_start':pfs[:,1],'z_start':pfs[:,2],
           'x_end':pfe[:,0],'y_end':pfe[:,1],'z_end':pfe[:,2],
           'x':pfm[:,0],'y':pfm[:,1],'z':pfm[:,2],
           'dx':sl,'dE':dEf,'dEdx':dEf/jnp.maximum(sl,1e-10),
           'eventID':jnp.zeros(n*nf),'trackID':jnp.zeros(n*nf),
           'pdg_id':jnp.full(n*nf,13.),'pixel_plane':jnp.zeros(n*nf),
           't0':jnp.zeros(n*nf)}
    for k, v in fm.items():
        if k in fields:
            out = out.at[:, fields.index(k)].set(v)
    return out

def make_straight_chain_init(entry, exit_, n):
    d = exit_ - entry; L = float(np.linalg.norm(d)); u = d/L
    return (float(np.arccos(np.clip(u[2],-1.,1.))),
            float(np.arctan2(u[1],u[0])),
            np.zeros(n-1,dtype=np.float32),
            np.zeros(n-1,dtype=np.float32),
            L/n, L)

# ── Simulation helper ────────────────────────────────────────────────────────
def simulate_wfs_static(params, tracks, fixed_px, fields):
    (_, pixels, nelectrons, t0_after_diff, long_diff,
     currents_idx, pIDs_neigh, currents_idx_neigh,
     nelectrons_neigh, t0_neigh) = simulate_drift_new(params, tracks, fields)
    pr = jnp.searchsorted(fixed_px, pIDs_neigh.ravel(), method='sort')
    mask = (pr < fixed_px.size) & (fixed_px[pr] == pIDs_neigh.ravel())
    pr = jnp.where(mask, pr, 0)
    wfs = simulate_signals_new(params, fixed_px, pixels, t0_after_diff, response,
                               nelectrons, long_diff, currents_idx,
                               nelectrons_neigh, pr, t0_neigh, currents_idx_neigh)
    return wfs[:, 1:]

# ── TrackContext ──────────────────────────────────────────────────────────────
@dataclasses.dataclass
class TrackContext:
    x0_fixed: jnp.ndarray; step_len: float; total_len: float
    sigma_mcs: float; n_chain: int; theta0_i: float; phi0_i: float
    dtheta_i: np.ndarray; dphi_i: np.ndarray; x0_chain: np.ndarray
    bounds_chain: list; n_chain_params: int; target_pixels: jnp.ndarray
    target_adc: jnp.ndarray; fixed_pixels: jnp.ndarray; psn: int; pln: int

def build_per_track_context(entry_pt, exit_pt, tgt_fine, sim_fine, target_step_len=1.5):
    total_len = float(np.linalg.norm(exit_pt - entry_pt))
    n_chain   = max(4, min(30, round(total_len / target_step_len)))
    theta0_i, phi0_i, dtheta_i, dphi_i, step_len, total_len = \
        make_straight_chain_init(entry_pt, exit_pt, n_chain)
    phi0_i   = float(phi0_i % (2*np.pi))
    sigma_mcs = highland_sigma_mcs(step_len)
    x0_fixed  = jnp.array(entry_pt, dtype=jnp.float32)

    dE_init = np.full(n_chain, mean_dedx*step_len, dtype=np.float32)
    pos_i, dir_i = deflection_chain_forward(x0_fixed, theta0_i, phi0_i,
                    jnp.array(dtheta_i), jnp.array(dphi_i), step_len)
    track_fine_i = chain_to_fine_segments(pos_i, dir_i, jnp.array(dE_init), fields)

    tgt_jax = jnp.array(tgt_fine, dtype=jnp.float32)
    wfs_t, up_t = simulate_wfs(params_true, response, tgt_jax, fields)
    up_t_p  = pad_to_closest_multiple(up_t, multiple=128, pad_value=-1, pad_front=True)
    wfs_t_p = pad_to_closest_multiple(wfs_t, dims_to_pad=(0,), multiple=128, pad_value=0., pad_front=True)
    nb_s, nb_l = get_roi_counts(params_true, wfs_t_p)
    psn = int(((int(nb_s)+127)//128)*128)
    pln = int(((int(nb_l)+127)//128)*128)
    adcs_d, _, _, ticks_p, _ = simulate_probabilistic(params_true, wfs_t_p, up_t_p, psn, pln)
    tgt_adc = jnp.sum(jnp.exp(ticks_p)*adcs_d, axis=1)
    valid   = (up_t_p >= 0) & (jnp.sum(tgt_adc, axis=-1) > 0.05)
    target_pixels = up_t_p[valid]; target_adc = tgt_adc[valid]

    _, guess_pix = simulate_wfs(params_true, response, track_fine_i, fields)
    combined = np.unique(np.concatenate([np.array(target_pixels), np.array(guess_pix[guess_pix>=0])]))
    combined = np.append(combined, -1)
    fixed_np = np.sort(np.array(pad_to_closest_multiple(jnp.array(combined), multiple=128, pad_value=-1, pad_front=False)))
    fixed_pixels = jnp.array(fixed_np)

    n_chain_params = 2 + 2*(n_chain-1)
    x0_chain  = np.concatenate([[theta0_i, phi0_i], dtheta_i, dphi_i]).astype(np.float64)
    bounds    = [(0.01, np.pi-0.01), (0., 2*np.pi)]
    bounds   += [(0., np.pi/4)]*(n_chain-1) + [(-np.pi, np.pi)]*(n_chain-1)
    return TrackContext(x0_fixed=x0_fixed, step_len=step_len, total_len=total_len,
        sigma_mcs=sigma_mcs, n_chain=n_chain, theta0_i=theta0_i, phi0_i=phi0_i,
        dtheta_i=dtheta_i, dphi_i=dphi_i, x0_chain=x0_chain, bounds_chain=bounds,
        n_chain_params=n_chain_params, target_pixels=target_pixels, target_adc=target_adc,
        fixed_pixels=fixed_pixels, psn=psn, pln=pln)

def unpack_chain(x, n):
    return x[0], x[1], x[2:2+n-1], x[2+n-1:2+2*(n-1)]

def _to_f32(v): return jnp.array(v, dtype=jnp.float32)
def _gc(*a): return np.concatenate([np.atleast_1d(np.array(g, dtype=np.float64)) for g in a])

# ── Loss factory ─────────────────────────────────────────────────────────────
def make_chain_loss_fn(ctx):
    x0_, sl_, nc_ = ctx.x0_fixed, ctx.step_len, ctx.n_chain
    fp_, psn_, pln_ = ctx.fixed_pixels, ctx.psn, ctx.pln
    tpx_, tadc_ = ctx.target_pixels, ctx.target_adc
    def loss(theta0, phi0, dtheta, dphi, dedx_log=None, ab_scale=None, mcs_sigma=None):
        pos, dir_ = deflection_chain_forward(x0_, theta0, phi0, dtheta, dphi, sl_)
        dedx_v = jnp.maximum(jnp.exp(dedx_log), 1e-3) if dedx_log is not None \
                 else jnp.full(nc_, mean_dedx, dtype=jnp.float32)
        tracks_s = chain_to_fine_segments(pos, dir_, dedx_v*sl_, fields)
        pr = params_true.replace(Ab=jnp.array(ab_nominal*ab_scale, dtype=jnp.float32)) \
             if ab_scale is not None else params_true
        wfs   = simulate_wfs_static(pr, tracks_s, fp_, fields)
        wfs_p = pad_to_closest_multiple(wfs, dims_to_pad=(0,), multiple=128, pad_value=0., pad_front=True)
        adcs_d, _, _, ticks_p, _ = simulate_probabilistic(pr, wfs_p, fp_, psn_, pln_)
        exp_adc = jnp.sum(jnp.exp(ticks_p)*adcs_d, axis=1)
        idx   = jnp.clip(jnp.searchsorted(fp_, tpx_), 0, fp_.shape[0]-1)
        hit   = (jnp.take(fp_, idx) == tpx_)[:, None]
        guess = jnp.where(hit, jnp.take(exp_adc, idx, axis=0), 0.)
        mse   = jnp.mean((guess - tadc_)**2)
        if mcs_sigma is not None:
            return mse + jnp.mean(dtheta**2) / (2.*mcs_sigma**2)
        return mse
    return loss

def make_stage_fns(ctx, clf):
    sig = jnp.array(ctx.sigma_mcs, dtype=jnp.float32)
    def s1(th,ph,dt,dp): return clf(th,ph,dt,dp,mcs_sigma=sig)
    def s2(th,ph,dt,dp,ld):
        mse = clf(th,ph,dt,dp,dedx_log=ld,mcs_sigma=sig)
        pr  = split_student_t_nll(jnp.maximum(jnp.exp(ld),1e-3),
                DEDX_SPLIT_NU_L,DEDX_SPLIT_NU_R,DEDX_SPLIT_LOC,
                DEDX_SPLIT_SCALE_L,DEDX_SPLIT_SCALE_R)
        return mse + 0.1*pr/ctx.n_chain
    def s3(th,ph,dt,dp,ld,ab):
        mse = clf(th,ph,dt,dp,dedx_log=ld,ab_scale=ab,mcs_sigma=sig)
        pr  = split_student_t_nll(jnp.maximum(jnp.exp(ld),1e-3),
                DEDX_SPLIT_NU_L,DEDX_SPLIT_NU_R,DEDX_SPLIT_LOC,
                DEDX_SPLIT_SCALE_L,DEDX_SPLIT_SCALE_R)
        return mse + 0.1*pr/ctx.n_chain
    return (jax.jit(jax.value_and_grad(s1,argnums=(0,1,2,3))),
            jax.jit(jax.value_and_grad(s2,argnums=(0,1,2,3,4))),
            jax.jit(jax.value_and_grad(s3,argnums=(0,1,2,3,4,5))))

@dataclasses.dataclass
class OptResult:
    x: np.ndarray
    fun: float

def _optax_run(vg_fn, params, n_pos, n_steps, lr_pos, lr_other, label):
    """Adam with per-group learning rates.
    params[:n_pos] use lr_pos (position angles); params[n_pos:] use lr_other (dEdx/Ab).
    Gradients are sliced explicitly to avoid optax.masked pass-through bug in 0.2.3."""
    opt_p = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_pos))
    n = len(params)
    has_other = n > n_pos

    if has_other:
        opt_o = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_other))
        s_p = opt_p.init(params[:n_pos])
        s_o = opt_o.init(params[n_pos:])

        @jax.jit
        def step(params, s_p, s_o):
            loss, grads = vg_fn(*params)
            up, ns_p = opt_p.update(grads[:n_pos], s_p)
            uo, ns_o = opt_o.update(grads[n_pos:], s_o)
            new_params = optax.apply_updates(params, up + uo)
            gnorm = jnp.sqrt(sum(jnp.sum(g**2) for g in grads))
            return new_params, ns_p, ns_o, loss, gnorm

        hist = []; t0 = time.time(); patience = 0
        for i in range(n_steps):
            params, s_p, s_o, loss, gnorm = step(params, s_p, s_o)
            hist.append({'loss': float(loss), 'grad_norm': float(gnorm)})
            if float(gnorm) < EARLY_STOP_GNORM:
                patience += 1
                if patience >= EARLY_STOP_PATIENCE:
                    break
            else:
                patience = 0
    else:
        s = opt_p.init(params)

        @jax.jit
        def step(params, s):
            loss, grads = vg_fn(*params)
            updates, ns = opt_p.update(grads, s)
            gnorm = jnp.sqrt(sum(jnp.sum(g**2) for g in grads))
            return optax.apply_updates(params, updates), ns, loss, gnorm

        hist = []; t0 = time.time(); patience = 0
        for i in range(n_steps):
            params, s, loss, gnorm = step(params, s)
            hist.append({'loss': float(loss), 'grad_norm': float(gnorm)})
            if float(gnorm) < EARLY_STOP_GNORM:
                patience += 1
                if patience >= EARLY_STOP_PATIENCE:
                    break
            else:
                patience = 0

    e = time.time() - t0
    hist[-1]['t'] = e
    actual = len(hist)
    print(f'  {label}: {actual:4d}/{n_steps} steps  loss={hist[-1]["loss"]:.4e}  t={e:.1f}s')
    return params, hist

# n_pos=4 always means (th, ph, dt, dp); remaining params are dEdx/Ab
LR_POS  = 3e-5   # position angles — small to prevent Adam overshoot on flat loss
LR_DEDX = 1e-3   # log-dEdx — normal Adam scale
LR_AB   = 3e-4   # Ab recombination scale

def run_stage1(ctx, vg, n_steps=500):
    params = (
        jnp.array(ctx.theta0_i, dtype=jnp.float32),
        jnp.array(ctx.phi0_i,   dtype=jnp.float32),
        jnp.array(ctx.dtheta_i, dtype=jnp.float32),
        jnp.array(ctx.dphi_i,   dtype=jnp.float32),
    )
    params, hist = _optax_run(vg, params, n_pos=4, n_steps=n_steps,
                               lr_pos=LR_POS, lr_other=LR_POS, label='S1')
    th, ph, dt, dp = [np.array(p) for p in params]
    x = np.concatenate([th.ravel(), ph.ravel(), dt, dp])
    return OptResult(x=x, fun=hist[-1]['loss']), hist

def run_stage2(ctx, vg, res1, n_steps=500):
    nc = ctx.n_chain
    th = jnp.array(res1.x[0],                           dtype=jnp.float32)
    ph = jnp.array(res1.x[1],                           dtype=jnp.float32)
    dt = jnp.array(res1.x[2:2+nc-1],                    dtype=jnp.float32)
    dp = jnp.array(res1.x[2+nc-1:ctx.n_chain_params],   dtype=jnp.float32)
    ld = jnp.full(nc, float(np.log(DEDX_SPLIT_LOC)),    dtype=jnp.float32)
    params, hist = _optax_run(vg, (th, ph, dt, dp, ld), n_pos=4, n_steps=n_steps,
                               lr_pos=LR_POS, lr_other=LR_DEDX, label='S2')
    th, ph, dt, dp, ld = [np.array(p) for p in params]
    x = np.concatenate([th.ravel(), ph.ravel(), dt, dp, ld])
    return OptResult(x=x, fun=hist[-1]['loss']), hist

def run_stage3(ctx, vg, res2, n_steps=300):
    nc = ctx.n_chain
    th = jnp.array(res2.x[0],                           dtype=jnp.float32)
    ph = jnp.array(res2.x[1],                           dtype=jnp.float32)
    dt = jnp.array(res2.x[2:2+nc-1],                    dtype=jnp.float32)
    dp = jnp.array(res2.x[2+nc-1:ctx.n_chain_params],   dtype=jnp.float32)
    ld = jnp.array(res2.x[ctx.n_chain_params:],          dtype=jnp.float32)
    ab = jnp.array(1.,                                   dtype=jnp.float32)
    params, hist = _optax_run(vg, (th, ph, dt, dp, ld, ab), n_pos=4, n_steps=n_steps,
                               lr_pos=LR_POS, lr_other=LR_DEDX, label='S3')
    th, ph, dt, dp, ld, ab = [np.array(p) for p in params]
    x = np.concatenate([th.ravel(), ph.ravel(), dt, dp, ld, ab.ravel()])
    return OptResult(x=x, fun=hist[-1]['loss']), hist

def run_joint(ctx, vg, n_steps=800):
    """Single-stage joint optimization of position + dEdx + Ab from a cold start."""
    nc = ctx.n_chain
    th = jnp.array(ctx.theta0_i,                         dtype=jnp.float32)
    ph = jnp.array(ctx.phi0_i,                           dtype=jnp.float32)
    dt = jnp.array(ctx.dtheta_i,                         dtype=jnp.float32)
    dp = jnp.array(ctx.dphi_i,                           dtype=jnp.float32)
    ld = jnp.full(nc, float(np.log(DEDX_SPLIT_LOC)),     dtype=jnp.float32)
    ab = jnp.array(1.,                                   dtype=jnp.float32)
    params, hist = _optax_run(vg, (th, ph, dt, dp, ld, ab), n_pos=4, n_steps=n_steps,
                               lr_pos=LR_POS, lr_other=LR_DEDX, label='Joint')
    th, ph, dt, dp, ld, ab = [np.array(p) for p in params]
    x = np.concatenate([th.ravel(), ph.ravel(), dt, dp, ld, ab.ravel()])
    return OptResult(x=x, fun=hist[-1]['loss']), hist

# ── Residuals helper ─────────────────────────────────────────────────────────
def track_residuals(nodes, tgt_xyz):
    dists = []
    for p in tgt_xyz:
        best = np.inf
        for i in range(len(nodes)-1):
            a,b = nodes[i], nodes[i+1]; ab=b-a
            t = np.clip(np.dot(p-a,ab)/(np.dot(ab,ab)+1e-12),0.,1.)
            best = min(best, np.linalg.norm(p-(a+t*ab)))
        dists.append(best)
    return np.array(dists)

def chain_nodes_from_res(ctx, res):
    th,ph,dt,dp = unpack_chain(res.x[:ctx.n_chain_params], ctx.n_chain)
    pos,_ = deflection_chain_forward(ctx.x0_fixed, th, ph,
              _to_f32(dt), _to_f32(dp), ctx.step_len)
    return np.array(pos)

# ── Plotting ─────────────────────────────────────────────────────────────────
COLORS = {'init':'gray','s1':'royalblue','s2':'seagreen','s3':'tomato'}
LABELS = {'init':'Initial','s1':'Stage 1 (pos)','s2':'Stage 2 (+dEdx)','s3':'Stage 3 (+Ab)'}

def plot_track_diagnostics(key, result):
    ctx = result['ctx']
    h1,h2,h3 = result['hist_s1'],result['hist_s2'],result['hist_s3']
    r1,r2,r3 = result['res_s1'],result['res_s2'],result['res_s3']
    cn = result['chain_nodes']
    bi,ev,tr = key
    _, _, raw_segs = raw_endpoint_map[(ev,tr)]

    # Use x_start/x_end if 'x' not available, else midpoint
    def get_coord(segs, name):
        if name in fields_raw: return segs[:, fields_raw.index(name)]
        s = name+'_start'; e2 = name+'_end'
        return 0.5*(segs[:,fields_raw.index(s)]+segs[:,fields_raw.index(e2)])
    tgt_xyz = np.stack([get_coord(raw_segs,c) for c in ('x','y','z')], axis=1)

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle(f'Track Fit  batch={bi} ev={ev} tr={tr}  '
                 f'L={ctx.total_len:.1f} cm  N_chain={ctx.n_chain}',
                 fontsize=13, fontweight='bold')

    # Row 1: geometry
    for ci, (da,db,xl,yl,ttl) in enumerate([
        (0,1,'X [cm]','Y [cm]','XY Projection'),
        (0,2,'X [cm]','Z [cm]','XZ Projection'),
    ]):
        ax = axes[0,ci]
        ax.scatter(tgt_xyz[:,da], tgt_xyz[:,db], c='black', s=12, alpha=0.6,
                   label='True', zorder=5)
        for sn,nodes in cn.items():
            ax.plot(nodes[:,da], nodes[:,db], color=COLORS[sn],
                    ls='--' if sn=='init' else '-',
                    lw=1.0 if sn=='init' else 1.8, label=LABELS[sn])
        ax.scatter([float(ctx.x0_fixed[da])],[float(ctx.x0_fixed[db])],
                   marker='*',s=200,c='gold',zorder=6,label='Entry')
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0,2]
    init_loss = h1[0]['loss'] if h1 else float('nan')
    _stage_labels=['Init','S1','S2','S3']
    _stage_vals=[init_loss,r1.fun,r2.fun,r3.fun]
    _stage_colors=['gray','royalblue','seagreen','tomato']
    # filter out nan stages (e.g. no_pos mode has nan S1)
    _ok = [(lbl,v,c) for lbl,v,c in zip(_stage_labels,_stage_vals,_stage_colors)
           if np.isfinite(v)]
    if _ok:
        _lbls = [lbl for lbl,_,_ in _ok]; _vs=[v for _,v,_ in _ok]; _cs=[c for _,_,c in _ok]
        bars=ax.bar(_lbls,_vs,color=_cs,alpha=0.8)
        for bar,v in zip(bars,_vs):
            ax.text(bar.get_x()+bar.get_width()/2, v*1.05,
                    f'{v:.2e}',ha='center',va='bottom',fontsize=8)
        ax.set_yscale('log')
    ax.set_ylabel('Loss'); ax.set_title('Loss by Stage')
    ax.grid(True,alpha=0.3,axis='y')

    # Row 2: convergence + dEdx
    for ci,(hist,lbl,col) in enumerate([
        (h1,'Stage 1 (pos+MCS)','royalblue'),
        (h2,'Stage 2 (pos+dEdx)','seagreen'),
        (h3,'Stage 3 (+Ab)','tomato'),
    ]):
        ax = axes[1,ci]
        if not hist:
            ax.text(0.5,0.5,'no data',transform=ax.transAxes,ha='center'); continue
        losses=[h['loss'] for h in hist]; gnorms=[h['grad_norm'] for h in hist]
        finite_losses = [v for v in losses if np.isfinite(v)]
        if not finite_losses:
            ax.text(0.5,0.5,'skipped (no_pos)',transform=ax.transAxes,ha='center',fontsize=9)
            ax.set_title(lbl); continue
        ax.plot(losses, color=col, lw=1.5, label='Loss')
        ax.set_yscale('log'); ax.set_xlabel('Iteration'); ax.set_ylabel('Loss')
        ax2=ax.twinx(); ax2.plot(gnorms,color=col,lw=1,ls='--',alpha=0.55)
        ax2.set_yscale('log'); ax2.set_ylabel('‖∇‖',color='gray',fontsize=8)
        ax.set_title(f'{lbl}  {len(hist)} iter  final={losses[-1]:.3e}')
        ax.grid(True,alpha=0.3); ax.legend(fontsize=8)

    # Row 3: dEdx, deflections, residuals
    arc = np.arange(ctx.n_chain)*ctx.step_len + ctx.step_len/2
    ax = axes[2,0]
    for sn,res,col in [('s2',r2,'seagreen'),('s3',r3,'tomato')]:
        if len(res.x)>ctx.n_chain_params:
            ld=res.x[ctx.n_chain_params:ctx.n_chain_params+ctx.n_chain]
            ax.plot(arc, np.exp(ld), color=col, lw=1.8, marker='o', ms=4,
                    label=LABELS[sn])
    ax.axhline(mean_dedx, color='gray', ls='--', lw=1.5,
               label=f'mean {mean_dedx} MeV/cm')
    ax.set_xlabel('Arc length [cm]'); ax.set_ylabel('dE/dx [MeV/cm]')
    ax.set_title('Fitted dE/dx Profile'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    ax = axes[2,1]
    joints = np.arange(ctx.n_chain-1)
    for sn,res,col in [('s1',r1,'royalblue'),('s2',r2,'seagreen'),('s3',r3,'tomato')]:
        _,_,dt,_ = unpack_chain(res.x[:ctx.n_chain_params],ctx.n_chain)
        ax.plot(joints, np.array(dt)*1e3, color=col, lw=1.5, marker='o', ms=4,
                label=LABELS[sn])
    ax.axhspan(-ctx.sigma_mcs*1e3, ctx.sigma_mcs*1e3, alpha=0.12, color='gray',
               label=f'±σ_MCS {ctx.sigma_mcs*1e3:.2f} mrad')
    ax.set_xlabel('Joint index'); ax.set_ylabel('δθ [mrad]')
    ax.set_title('Angular Deflections per Joint'); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

    ax = axes[2,2]
    resids={sn:track_residuals(nodes,tgt_xyz) for sn,nodes in cn.items()}
    for sn in ['init','s1','s2','s3']:
        ax.hist(resids[sn], bins=np.linspace(0,3,60), alpha=0.5, density=True,
                color=COLORS[sn],
                label=f'{LABELS[sn]}  med={np.median(resids[sn]):.3f} cm')
    ax.set_xlabel('Distance to true [cm]'); ax.set_ylabel('Density')
    ax.set_title('Track-to-True Residuals'); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

    plt.tight_layout(rect=[0,0,1,0.97])
    return fig, resids

# ── Main fitting loop ────────────────────────────────────────────────────────
print('\n=== Fitting loop ===')
all_results = (pickle.load(open(CHECKPOINT_FILE,'rb'))
               if os.path.exists(CHECKPOINT_FILE) else {})

tgt_ev_idx  = tgt_fields.index('eventID')
tgt_trk_idx = tgt_fields.index('trackID')

for spec_idx, spec in enumerate(all_track_specs):
    key = (spec['batch_idx'], spec['ev'], spec['tr'])
    if key in all_results:
        print(f'[{spec_idx+1}/{len(all_track_specs)}] {key} already done — skipping.'); continue

    print(f'\n{"="*60}')
    print(f'[{spec_idx+1}/{len(all_track_specs)}]  batch={spec["batch_idx"]}  '
          f'ev={spec["ev"]}  tr={spec["tr"]}  L={spec["total_len"]:.1f} cm')
    print('='*60)

    bi,ev,tr = spec['batch_idx'],spec['ev'],spec['tr']
    batch_sim = dataset_sim[bi].reshape(-1,len(fields))
    parent_ids = dataset_sim.get_parent_segment_ids(bi)
    real_sim = batch_sim[parent_ids>=0]
    sim_trk  = real_sim[(real_sim[:,event_idx]==ev)&(real_sim[:,track_idx]==tr)]

    batch_tgt = dataset_target[bi].reshape(-1,len(tgt_fields))
    real_tgt  = batch_tgt[batch_tgt[:,tgt_ev_idx]>=0]
    tgt_trk   = real_tgt[(real_tgt[:,tgt_ev_idx]==ev)&(real_tgt[:,tgt_trk_idx]==tr)]
    print(f'  sim segs={len(sim_trk)}  tgt segs={len(tgt_trk)}')

    print('  Building context...')
    ctx = build_per_track_context(spec['entry_pt'], spec['exit_pt'], tgt_trk, sim_trk)
    print(f'  N_chain={ctx.n_chain}  step={ctx.step_len:.2f} cm  '
          f'σ_MCS={ctx.sigma_mcs*1e3:.2f} mrad  '
          f'fixed_px={len(ctx.fixed_pixels)}  tgt_px={len(ctx.target_pixels)}  '
          f'psn={ctx.psn}  pln={ctx.pln}')

    clf = make_chain_loss_fn(ctx)
    vg1,vg2,vg3 = make_stage_fns(ctx,clf)

    print('  JIT warm-up...')
    if FIT_MODE == 'full':
        _ = vg1(_to_f32(ctx.theta0_i),_to_f32(ctx.phi0_i),
                _to_f32(ctx.dtheta_i),_to_f32(ctx.dphi_i))
    else:
        nc_ = ctx.n_chain
        _ = vg3(_to_f32(ctx.theta0_i), _to_f32(ctx.phi0_i),
                _to_f32(ctx.dtheta_i), _to_f32(ctx.dphi_i),
                jnp.full(nc_, float(np.log(DEDX_SPLIT_LOC)), dtype=jnp.float32),
                jnp.array(1., dtype=jnp.float32))
    print('  JIT ready.')

    _dummy_x = np.concatenate([[ctx.theta0_i, ctx.phi0_i], ctx.dtheta_i, ctx.dphi_i])
    _dummy = OptResult(x=_dummy_x, fun=float('nan'))
    _empty_h = []

    if FIT_MODE == 'joint':
        r1, h1 = _dummy, _empty_h
        r2, h2 = _dummy, _empty_h
        r3, h3 = run_joint(ctx, vg3)
    elif FIT_MODE == 'no_pos':
        r1 = OptResult(x=_dummy_x, fun=float('nan'))
        h1 = [{'loss': float('nan'), 'grad_norm': 0.0, 't': 0.0}]
        r2, h2 = run_stage2(ctx, vg2, r1)
        r3, h3 = run_stage3(ctx, vg3, r2)
    else:  # 'full'
        r1, h1 = run_stage1(ctx, vg1)
        r2, h2 = run_stage2(ctx, vg2, r1)
        r3, h3 = run_stage3(ctx, vg3, r2)

    # chain nodes
    pos_i,_ = deflection_chain_forward(ctx.x0_fixed, ctx.theta0_i, ctx.phi0_i,
                _to_f32(ctx.dtheta_i), _to_f32(ctx.dphi_i), ctx.step_len)
    cn = {'init':np.array(pos_i),
          's1':chain_nodes_from_res(ctx,r1),
          's2':chain_nodes_from_res(ctx,r2),
          's3':chain_nodes_from_res(ctx,r3)}

    all_results[key] = dict(spec=spec,ctx=ctx,res_s1=r1,res_s2=r2,res_s3=r3,
                             hist_s1=h1,hist_s2=h2,hist_s3=h3,chain_nodes=cn)

    with open(CHECKPOINT_FILE,'wb') as f:
        pickle.dump(all_results,f)
    print(f'  Checkpoint saved.')

    try:
        fig,resids = plot_track_diagnostics(key, all_results[key])
        fname = f"plots/chain_b{bi}_ev{ev}_tr{tr}.png"
        fig.savefig(fname, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'  Plot saved: {fname}')
        print(f'  Resid median: init={np.median(resids["init"]):.3f} → '
              f's3={np.median(resids["s3"]):.3f} cm')
    except Exception as e:
        print(f'  Plot failed (non-fatal): {e}')

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print('SUMMARY')
print('='*60)
rows=[]
for key,r in all_results.items():
    bi,ev,tr = key
    ctx=r['ctx']; nc=ctx.n_chain
    h1,h2,h3=r['hist_s1'],r['hist_s2'],r['hist_s3']
    r1,r2,r3=r['res_s1'],r['res_s2'],r['res_s3']
    cn=r['chain_nodes']
    init_l=h1[0]['loss'] if h1 else float('nan')
    ab_fit=ab_nominal*(r3.x[ctx.n_chain_params+nc] if len(r3.x)>ctx.n_chain_params+nc else 1.)
    # prefer r3 for dEdx (joint mode stores it there); fall back to r2, then nominal
    if len(r3.x) > ctx.n_chain_params+nc:          # r3 has ab_scale → dEdx is at [n_cp:n_cp+nc]
        dedx_m = float(np.exp(r3.x[ctx.n_chain_params:ctx.n_chain_params+nc]).mean())
    elif len(r3.x) > ctx.n_chain_params:            # r3 has dEdx but no ab_scale
        dedx_m = float(np.exp(r3.x[ctx.n_chain_params:ctx.n_chain_params+nc]).mean())
    elif len(r2.x) > ctx.n_chain_params:            # r2 has dEdx (full/no_pos S2)
        dedx_m = float(np.exp(r2.x[ctx.n_chain_params:ctx.n_chain_params+nc]).mean())
    else:
        dedx_m = mean_dedx
    _,_,raw_s=raw_endpoint_map[(ev,tr)]
    def gc(segs,n):
        if n in fields_raw: return segs[:,fields_raw.index(n)]
        return 0.5*(segs[:,fields_raw.index(n+'_start')]+segs[:,fields_raw.index(n+'_end')])
    tgt_xyz=np.stack([gc(raw_s,c) for c in ('x','y','z')],axis=1)
    ri=np.median(track_residuals(cn['init'],tgt_xyz))
    r3m=np.median(track_residuals(cn['s3'],tgt_xyz))
    tot_t=sum(h[-1].get('t',0) for h in [h1,h2,h3] if h)
    rows.append({'fit_mode':FIT_MODE,'b':bi,'ev':ev,'tr':tr,'L':f'{ctx.total_len:.1f}',
                 'N_ch':nc,'L_init':f'{init_l:.3e}',
                 'L_S1':f'{r1.fun:.3e}','L_S2':f'{r2.fun:.3e}','L_S3':f'{r3.fun:.3e}',
                 'dEdx_S2':f'{dedx_m:.3f}','Ab':f'{ab_fit:.5f}',
                 'Ab_bias%':f'{(ab_fit-ab_nominal)/ab_nominal*100:+.3f}',
                 'Ri':f'{ri:.3f}','Rs3':f'{r3m:.3f}','t_s':f'{tot_t:.0f}'})
    print(f"  b={bi} ev={ev} tr={tr}: L_init={init_l:.3e} → L_S3={r3.fun:.3e} "
          f"({(init_l-r3.fun)/init_l*100:.1f}% drop)  "
          f"resid {ri:.3f}→{r3m:.3f} cm  Ab={ab_fit:.4f}  "
          f"dEdx={dedx_m:.3f} MeV/cm  t={tot_t:.0f}s")

df=pd.DataFrame(rows)
summary_csv = f'chain_fit_summary_{FIT_MODE}_b{MAX_BATCHES}.csv'
df.to_csv(summary_csv, index=False)
print(f'\nSummary written: {summary_csv}')

# Cross-track summary plot
if len(all_results) >= 2:
    cmap=plt.cm.tab10
    tcolors={k:cmap(i%10) for i,k in enumerate(all_results)}
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    fig.suptitle(f'Cross-Track — {len(all_results)} tracks',fontsize=13,fontweight='bold')
    for ax,(da,db,xl,yl,ttl) in zip(axes,[
        (0,1,'X [cm]','Y [cm]','XY'),
        (0,2,'X [cm]','Z [cm]','XZ'),
    ]):
        for k,r in all_results.items():
            bi2,ev2,tr2=k; col=tcolors[k]
            _,_,rs=raw_endpoint_map[(ev2,tr2)]
            def gc2(segs,n):
                if n in fields_raw: return segs[:,fields_raw.index(n)]
                return 0.5*(segs[:,fields_raw.index(n+'_start')]+segs[:,fields_raw.index(n+'_end')])
            ax.plot(gc2(rs,'x'),gc2(rs,['y','z'][db-1]),'-',color=col,lw=1.5,alpha=0.55,
                    label=f'ev{ev2}tr{tr2} true')
            n3=r['chain_nodes']['s3']
            ax.plot(n3[:,da],n3[:,db],'--',color=col,lw=1.5,label='S3 fit')
        ax.set_xlabel(xl);ax.set_ylabel(yl);ax.set_title(ttl)
        ax.legend(fontsize=7,ncol=2);ax.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/cross_track_summary.png',dpi=150,bbox_inches='tight')
    plt.close()
    print('Cross-track plot: plots/cross_track_summary.png')

print('\nDone.')
