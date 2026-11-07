"""Iterated joint Newton steps on the five calibration parameters and the per-segment log dE/dx (21 Oct), positions held
fixed: joint_polish.py (one step, same derivatives, same trust-region ladder) repeated from the accepted point until the
best rung no longer lowers the objective or --nsteps is reached. Question: is the one-step end point (12 same-data runs:
lifetime -6 %, same-data SD 0.9 %) the objective's minimum at fixed positions, or a truncated step on the way to it?
Added per step: the state (theta, log dE/dx per batch) saved to --state (npz, overwritten after every accepted step);
the d-only rung (du = 0, dE/dx re-profiled at theta fixed, best kappa) to split the gain into nuisance re-fit and
calibration move; per batch, the Schur gradient gS_b and its share of the calibration step -S^-1 gS_b at the best rung
(which batches pull the lifetime); the data / prior split of the objective change.
usage: python joint_polish2.py --run KN10S --sample 1 [--suffix b] --iter 10000 --out out.json --state st.npz
       [--nsteps 3] [--nb 100] [--eps 3e-3] [--batches 0,1,2] [--init_state st.npz]"""
import os, sys, glob, json, time, argparse
import numpy as np
# LARND_JAX_LRU=N: the speed bundle's import hook (copied from optimize/example_run.py, which is not imported here to keep
# its per-job disk cache off): JAX's weakref_lru_cache default maxsize 2048 -> N, set before jax is imported.
_JAX_LRU = int(os.environ.get('LARND_JAX_LRU', '0') or 0)
if _JAX_LRU:
    import importlib.abc, importlib.machinery
    assert 'jax' not in sys.modules, 'LARND_JAX_LRU: jax already imported'
    _LRU_OLD = 'def weakref_lru_cache(call: Callable, maxsize=2048,'

    class _LRULoader(importlib.abc.Loader):
        def __init__(self, origin): self.origin = origin
        def create_module(self, spec): return None
        def exec_module(self, module):
            src = open(self.origin).read(); assert src.count(_LRU_OLD) == 1, 'jax._src.util changed'
            exec(compile(src.replace(_LRU_OLD, f'def weakref_lru_cache(call: Callable, maxsize={_JAX_LRU},'), self.origin, 'exec'), module.__dict__)
            print(f'[LARND_JAX_LRU] maxsize 2048 -> {_JAX_LRU}', file=sys.stderr, flush=True)

    class _LRUFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != 'jax._src.util': return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is not None: spec.loader = _LRULoader(spec.origin)
            return spec
    sys.meta_path.insert(0, _LRUFinder())
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/eventdisplay')
from rebuild import FitRebuild
import jax, jax.numpy as jnp

ap = argparse.ArgumentParser(); ap.add_argument('--run', default='KN10S'); ap.add_argument('--iter', type=int, default=10000)
ap.add_argument('--sample', type=int, required=True); ap.add_argument('--suffix', default=''); ap.add_argument('--nb', type=int, default=100)
ap.add_argument('--out', required=True); ap.add_argument('--eps', type=float, default=3e-3); ap.add_argument('--batches', default='')
ap.add_argument('--nsteps', type=int, default=3); ap.add_argument('--state', required=True); ap.add_argument('--init_state', default='')
ap.add_argument('--geom', default='fit', choices=['fit', 'truth', 'outlier', 'frozen'])   # joint_polish3: outlier = truth only for
ap.add_argument('--thr_um', type=float, default=200.)   # tracks whose fitted mean |r| > thr_um; frozen = truth only for tracks called straight
ap.add_argument('--drop', default='none', choices=['none', 'flag', 'outlier'])   # joint_polish3: remove tracks from data AND model (dE -> 0,
ap.add_argument('--drop_q', type=float, default=0.8)   # owned target hits dropped, out of the dE/dx prior). flag = truth-free hit-offset flag
#   (hit_flag.py) above its drop_q quantile over the run; outlier = fitted |r| > thr_um (oracle)
a = ap.parse_args()
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
ckpt = sorted(glob.glob(f'{FR}/sci_full_{a.run}{a.sample}{a.suffix}/history_iter{a.iter}_*.pkl'))[0]
R = FitRebuild(ckpt); pc = R.pc; PARAMS = pc.PARAMS; ck = R.ck; cfg = ck['config']
for p in PARAMS:
    t = float(np.ravel(ck[p + '_target'])[0]); assert abs(t / pc.TRUTH[p] - 1) < 1e-6, (p, t, pc.TRUTH[p])
assert float(cfg.dedx_drift_profile_weight) == 0.0 and bool(cfg.dedx_use_split_t)
fp = R.fp; assert not fp._DEDX_SHAPE_PRIOR and not fp._DEDX_LOG_JACOBIAN, 'prior variants not implemented'
from optimize.strategies import LUTProbabilisticSimulation, ProbabilisticLossStrategy
from optimize.dedx_utils import split_student_t_nll
from larndsim.losses_jax import llhd_loss
sim = LUTProbabilisticSimulation(R.response); loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)
base = R.truth_params; fields = R.sfields
di, Ei, xi = fields.index('dEdx'), fields.index('dE'), fields.index('dx')
lsf_i = fields.index('length_scale_factor') if 'length_scale_factor' in fields else None
mk = lambda th: base.replace(**{k: jnp.ones_like(getattr(base, k)) * th[k] for k in PARAMS})
H_STEP = dict(Ab=5e-4, eField=1.5e-5, tran_diff=1e-2, long_diff=2e-2, lifetime=1e-2)   # ln-units, as newton_polish.py
th0 = dict(R.calib_fit); err = lambda th: {k: 100 * (th[k] / pc.TRUTH[k] - 1) for k in PARAMS}
f32 = lambda x: jnp.asarray(x, jnp.float32)
NU_L, NU_R, LOC, SC_L, SC_R = (f32(getattr(cfg, k)) for k in ('dedx_student_nu_l', 'dedx_student_nu_r', 'dedx_student_loc',
                                                               'dedx_student_scale_l', 'dedx_student_scale_r'))
PW, BW, BT = float(cfg.dedx_prior_weight), float(cfg.dedx_soft_barrier_weight), float(cfg.dedx_soft_barrier_threshold)
MW, MT = float(cfg.dedx_mean_constraint_weight), float(cfg.dedx_mean_constraint_target)
print(f'{a.run}{a.sample}{a.suffix} @ {a.iter}: theta_0 error %', {k: f'{v:+.3f}' for k, v in err(th0).items()},
      f'| prior: split-t x{PW}, barrier {BW}@{BT}, mean {MW}->{MT}, eps {a.eps}', flush=True)


def truth_geom(i):
    """Batch i's tracks with the spline coefficients replaced by the least-squares projection of the TRUE positions
    (fit_params._truth_spline_coeffs, the representation floor); dE/dx columns as rebuild.batch. Tracks without true
    positions keep their fitted coefficients (counted)."""
    bt = R.batch(i, check=False); guess, ctxs = bt['guess'], bt['ctxs']; tp = ck['true_positions'][i]; coeffs, nkeep = [], 0
    for c in ctxs:
        K = fp._spline_K(c.total_len, R.knot_cm); fc = np.asarray(bt['coeffs'][tuple(c.track_id)]); assert fc.shape == (2 * K,), (fc.shape, K)
        tr = tp.get(tuple(map(float, c.track_id)), tp.get(str(tuple(map(float, c.track_id)))))
        if tr is None: coeffs.append(jnp.asarray(fc)); nkeep += 1; continue
        if a.geom == 'outlier':                           # joint_polish3: oracle -- move only the badly positioned tracks
            idx = np.asarray(c.idxs)[:len(tr)]; cxyz = [R.col['x'], R.col['y'], R.col['z']]
            if 1e4 * float(np.linalg.norm(np.asarray(bt['fitted'])[idx][:, cxyz] - np.asarray(tr, float)[:len(idx)], axis=1).mean()) <= a.thr_um:
                coeffs.append(jnp.asarray(fc)); nkeep += 1; continue
        if a.geom == 'frozen' and not np.all(fc[K:] == 0):
            coeffs.append(jnp.asarray(fc)); nkeep += 1; continue
        u0 = fp._dir_from_angles(c.theta0_i, c.phi0_i); e1, e2 = fp._transverse_frame(u0)
        L = float(c.total_len); al = np.asarray(c.alpha_mid, float); tr = np.asarray(tr, float); m = min(len(al), len(tr)); al, tr = al[:m], tr[:m]
        d = tr - (np.asarray(c.x0_fixed)[None, :] + al[:, None] * L * u0[None, :]); Bm = fp._spline_basis(al, K)
        coeffs.append(jnp.asarray(np.concatenate([np.linalg.lstsq(Bm, d @ e1, rcond=None)[0], np.linalg.lstsq(Bm, d @ e2, rcond=None)[0]]).astype(np.float32)))
    tw, _ = fp._chain_warp_spline(jnp.asarray(guess), coeffs, fp._build_chain_meta_spline(ctxs, R.knot_cm), R.col)
    tw = np.asarray(tw).copy(); pid = np.asarray(ck['batch_parent_ids'][i]); ldc = np.asarray(ck['dedx_cache'][i]['log_dedx'])
    ded = np.where(pid >= 0, np.maximum(np.exp(ldc[np.clip(pid, 0, len(ldc) - 1)]), 1e-3), tw[:, di]); tw[:, di] = ded; tw[:, Ei] = ded * tw[:, xi]
    return tw, nkeep


def track_flags(i, fitted):                               # hit_flag.py: per track (owned hit mask, flag um, truth |r| um, segment idxs)
    bt = R.batch(i, check=False); tg = R.target(i); tp = ck['true_positions'][i]; f = np.asarray(fitted); cx = [R.col['x'], R.col['y'], R.col['z']]
    ev_c = fields.index('eventID'); hx = np.stack([tg['pixel_x'], tg['pixel_y'], tg['pixel_z']], 1).astype(float)
    hq = np.asarray(tg['adcs'], float); hev = np.asarray(tg['event']); res = []
    evs = [e for e in np.unique(f[:, ev_c]) if e >= 0]
    trk = [(c, f[np.asarray(c.idxs)][:, cx], float(f[np.asarray(c.idxs)[0], ev_c])) for c in bt['ctxs']]
    for le, e in enumerate(evs):
        hsel = np.where(hev == le)[0]; T = [(c, P) for c, P, ee in trk if ee == e]
        if len(hsel) == 0 or not T: continue
        H = hx[hsel]; dist = np.stack([np.min(np.linalg.norm(H[:, None, :] - P[None, ::10, :], axis=2), axis=1) for _, P in T], 1)
        own = np.argmin(dist, 1); dmin = dist[np.arange(len(H)), own]
        for k, (c, P) in enumerate(T):
            m = (own == k) & (dmin < 1.0); h, q = H[m], hq[hsel][m]; fl = float('nan')
            if m.sum() >= 8:
                u = P[-1] - P[0]; L = np.linalg.norm(u); u = u / max(L, 1e-9); nxy = np.array([-u[1], u[0], 0.]); nxy /= max(np.linalg.norm(nxy), 1e-9)
                j = np.argmin(np.linalg.norm(h[:, None, :] - P[None, ::10, :], axis=2), axis=1) * 10
                off = (h - P[j]) @ nxy; al = ((h - P[0]) @ u) / max(L, 1e-9)
                sl = [float(np.sum(q[(al >= a0) & (al < a0 + .25)] * off[(al >= a0) & (al < a0 + .25)]) / np.sum(q[(al >= a0) & (al < a0 + .25)]))
                      for a0 in (0, .25, .5, .75) if np.sum((al >= a0) & (al < a0 + .25)) >= 4]
                fl = 1e4 * float(np.max(np.abs(sl))) if sl else float('nan')
            tr = tp.get(tuple(map(float, c.track_id)), tp.get(str(tuple(map(float, c.track_id))))); rr = float('nan')
            if tr is not None:
                tr = np.asarray(tr, float); n = min(len(tr), len(P)); rr = 1e4 * float(np.linalg.norm(P[:n] - tr[:n], axis=1).mean())
            res.append(dict(hits=hsel[m], flag=fl, r=rr, idxs=np.asarray(c.idxs)))
    return res


DROP = {}                                                 # batch -> (segment keep mask, target hit keep mask, n dropped)
if a.drop != 'none':
    _all = {i: track_flags(i, R.batch(i, check=False)['fitted']) for i in range(min(a.nb, len(R.ds_sim)))}
    _fl = np.array([t['flag'] for v in _all.values() for t in v]); thr_fl = float(np.nanquantile(_fl, a.drop_q))
    for i, v in _all.items():
        nseg = np.asarray(R.batch(i, check=False)['fitted']).shape[0]; nh = len(R.target(i)['adcs'])
        ks, kh, nd = np.ones(nseg, np.float32), np.ones(nh, bool), 0
        for t in v:
            d = (t['flag'] > thr_fl) if a.drop == 'flag' else (t['r'] > a.thr_um)
            if d: ks[t['idxs']] = 0.0; kh[t['hits']] = False; nd += 1
        DROP[i] = (ks, kh, nd)
    print(f'[drop {a.drop}] flag threshold {thr_fl:.0f} um (q {a.drop_q}); dropped {sum(x[2] for x in DROP.values())} of {len(_fl)} tracks, '
          f'{sum(int((~x[1]).sum()) for x in DROP.values())} of {sum(len(x[1]) for x in DROP.values())} target hits', flush=True)


def batch_fns(i):
    if a.geom != 'fit':
        tw, nk = truth_geom(i); print(f'[geom {a.geom}] batch {i}: {len(R.batch(i, check=False)["ctxs"]) - nk} tracks moved to truth', flush=True)
    fitted = jnp.asarray(R.batch(i, check=False)['fitted'] if a.geom == 'fit' else tw); tgt = {k: jnp.asarray(v) for k, v in R.target(i).items()}
    ks = jnp.ones(fitted.shape[0], jnp.float32)
    if i in DROP:                                         # joint_polish3 --drop: the track leaves the data and the model
        ks = jnp.asarray(DROP[i][0]); tgt = {k: jnp.asarray(np.asarray(v)[DROP[i][1]]) for k, v in R.target(i).items()}
    pid = jnp.asarray(np.asarray(ck['batch_parent_ids'][i])); ld0 = np.asarray(ck['dedx_cache'][i]['log_dedx'], np.float64).ravel()
    valid = pid >= 0; safe = jnp.clip(pid, 0, ld0.shape[0] - 1)

    def tracks_of(ld):                                    # loss_wrapper_combined, "Reconstruct per-step dEdx"
        ded = jnp.where(valid, jnp.maximum(jnp.exp(jnp.take(ld, safe)), 1e-3), fitted[:, di])
        return fitted.at[:, di].set(ded).at[:, Ei].set(ded * fitted[:, xi] * ks), ded

    def prior(ld):                                        # loss_wrapper_combined: split-t + barrier + mean constraint
        t, ded = tracks_of(ld); dxw = t[:, xi] * ks; lsf = t[:, lsf_i] if lsf_i is not None else 1.0
        p = split_student_t_nll(ded, NU_L, NU_R, LOC * lsf, SC_L, SC_R, weights=dxw) * PW
        vs = dxw > 0; p = p + BW * jnp.sum(dxw * vs * jnp.square(jnp.maximum(0., ded - BT)))
        sv = valid & vs; tot = jnp.sum(dxw * sv); mean = jnp.sum(ded * dxw * sv) / jnp.maximum(tot, 1e-6)
        tgt_m = MT * (jnp.sum(lsf * dxw * sv) / jnp.maximum(tot, 1e-6) if lsf_i is not None else 1.0)
        return p + MW * jnp.square(mean - tgt_m), mean

    def data(P, ld):
        t, _ = tracks_of(ld); return loss.compute(P, sim.predict(P, t, fields, None), tgt)[0]
    vg = jax.value_and_grad(data, argnums=(0, 1))

    def grad(th, ld):                                     # -> L, g_u (5, ln units), g_d (N)
        L, (gP, gl) = vg(mk(th), f32(ld))
        return float(L), np.array([float(getattr(gP, k)) * th[k] for k in PARAMS]), np.asarray(gl, np.float64)
    val = lambda th, ld: float(data(mk(th), f32(ld)))      # forward only
    return grad, prior, ld0, val


_js = lambda o: (str(o) if not hasattr(o, 'item') else o.item())   # numpy scalars -> JSON
nb = min(a.nb, len(R.ds_sim)); todo = [int(x) for x in a.batches.split(',')] if a.batches else list(range(nb))
fns = {}                                                  # per batch: (grad, prior, ld0, val), built once
th = dict(th0); ld = {}
for i in todo:
    fns[i] = batch_fns(i); ld[i] = fns[i][2].copy()
if a.init_state:
    z = np.load(a.init_state); th = {k: float(z['th_' + k]) for k in PARAMS}
    for i in todo: ld[i] = np.asarray(z[f'ld_{i}'], np.float64)
res = dict(run=a.run, iter=a.iter, sample=a.sample, suffix=a.suffix, ckpt=ckpt, eps=a.eps, params=PARAMS, truth=pc.TRUTH,
           theta0=th0, theta_start=dict(th), init_state=a.init_state, geom=a.geom, steps=[])
KAPPAS = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, float('inf')]


def solve_abs(M, v):                                      # |M|^-1 v, saddle-free
    w, V = np.linalg.eigh(0.5 * (M + M.T)); y = V.T @ v
    return V @ (y / np.abs(w).reshape((-1,) + (1,) * (y.ndim - 1)))


def solve_pos(M, v):                                      # M^+ v on the positive-curvature subspace (jtg_KN10S1 step 2: S indefinite)
    w, V = np.linalg.eigh(0.5 * (M + M.T)); p = w > 1e-12 * max(w.max(), 1e-300); y = V[:, p].T @ v
    return V[:, p] @ (y / w[p].reshape((-1,) + (1,) * (y.ndim - 1)))


def derivs(i, th, ldi):                                   # joint_polish.py's per-batch block, at (th, ldi)
    grad, prior, _, val = fns[i]; N = ldi.shape[0]
    L0, gu, gd = grad(th, ldi)
    Huu = np.zeros((5, 5)); Hdu = np.zeros((N, 5))
    for c, k in enumerate(PARAMS):
        h = H_STEP[k]
        _, gup, gdp = grad({**th, k: th[k] * float(np.exp(h))}, ldi); _, gum, gdm = grad({**th, k: th[k] * float(np.exp(-h))}, ldi)
        Huu[:, c] = (gup - gum) / (2 * h); Hdu[:, c] = (gdp - gdm) / (2 * h)
    B = np.zeros((N, N)); probe_ud = np.zeros((5, 5)); col_ud = np.zeros((5, 5)); j = np.arange(N)
    for c in range(5):
        v = (j % 5 == c).astype(np.float64)
        _, gup, gdp = grad(th, ldi + a.eps * v); _, gum, gdm = grad(th, ldi - a.eps * v)
        Hv = (gdp - gdm) / (2 * a.eps); probe_ud[:, c] = (gup - gum) / (2 * a.eps); col_ud[:, c] = Hdu[v > 0].sum(0)
        for off in range(-2, 3):
            r = j[(j + off >= 0) & (j + off < N) & ((j + off) % 5 == c)]; B[r, r + off] = Hv[r]
    asym = float(np.abs(B - B.T).max() / max(np.abs(np.diag(B)).max(), 1e-30)); B = 0.5 * (B + B.T)
    Huu = 0.5 * (Huu + Huu.T)
    sym_ud = float(np.abs(probe_ud - col_ud).max() / max(np.abs(col_ud).max(), 1e-30))
    R0, mean0 = prior(f32(ldi)); gR = np.asarray(jax.grad(lambda x: prior(x)[0])(f32(ldi)), np.float64)
    HR = np.asarray(jax.hessian(lambda x: prior(x)[0])(f32(ldi)), np.float64)
    D = B + HR; D = 0.5 * (D + D.T); ev, Vd = np.linalg.eigh(D)
    gdt = gd + gR
    b = dict(batch=i, N=N, L0=L0, R0=float(R0), mean_dedx0=float(mean0), g_u=gu.tolist(), H_uu=Huu.tolist(),
             eig_D_min=float(ev[0]), eig_D_max=float(ev[-1]), n_neg=int((ev <= 0).sum()), band_asym=asym, sym_ud=sym_ud,
             gd_data_norm=float(np.linalg.norm(gd)), gd_total_norm=float(np.linalg.norm(gdt)))
    return b, dict(ev=ev, V=Vd, Hdu=Hdu, gdt=gdt, gu=gu, Huu=Huu)


def step(kappa, ok, keep, d_only=False):                  # trust-region rung (joint_polish.py), + per-batch Schur gradients
    if np.isinf(kappa):
        Huu = sum(keep[b['batch']]['Huu'] for b in ok); gu = sum(keep[b['batch']]['gu'] for b in ok)
        return -solve_pos(Huu, gu), {b['batch']: np.zeros(b['N']) for b in ok}, Huu, {}
    S = np.zeros((5, 5)); gS = np.zeros(5); X = {}; gSb = {}
    for b in ok:
        kp = keep[b['batch']]; ev = kp['ev']; pos = ev > 1e-9 * ev[-1]
        il = np.where(pos, 1.0 / np.where(pos, ev + kappa * np.median(np.abs(ev)), 1.0), 0.0)
        Xb = kp['V'] @ ((kp['V'].T @ np.column_stack([kp['gdt'], kp['Hdu']])) * il[:, None]); X[b['batch']] = Xb
        S += kp['Huu'] - kp['Hdu'].T @ Xb[:, 1:]; gSb[b['batch']] = kp['gu'] - kp['Hdu'].T @ Xb[:, 0]; gS += gSb[b['batch']]
    du = np.zeros(5) if d_only else -solve_pos(S, gS)
    return du, {i: -(Xb[:, 0] + Xb[:, 1:] @ du) for i, Xb in X.items()}, S, gSb


def model(du, dd, ok, keep):                              # quadratic model with the TRUE (signed) curvature
    Huu = sum(keep[b['batch']]['Huu'] for b in ok); gu = sum(keep[b['batch']]['gu'] for b in ok)
    m = float(gu @ du) + 0.5 * float(du @ Huu @ du)
    for b in ok:
        kp = keep[b['batch']]; d = dd[b['batch']]; y = kp['V'].T @ d
        m += float(kp['gdt'] @ d) + float(d @ kp['Hdu'] @ du) + 0.5 * float(np.sum(kp['ev'] * y * y))
    return m


def J_at(th, du, dd, ok):                                 # objective at (u + du, d + dd): (data, prior, theta, mean dE/dx change %)
    th1 = {k: th[k] * float(np.exp(x)) for k, x in zip(PARAMS, du)}; tL, tR, dm = 0.0, 0.0, []
    for b in ok:
        i = b['batch']; _, pr, _, val = fns[i]; ld1 = ld[i] + dd[i]
        L1 = val(th1, ld1); R1, m1 = pr(f32(ld1)); tL += L1; tR += float(R1); dm.append(100 * (float(m1) / b['mean_dedx0'] - 1))
    return tL, tR, th1, dm


def save_state(th, ld):
    np.savez(a.state, **{'th_' + k: th[k] for k in PARAMS}, **{f'ld_{i}': ld[i] for i in todo})


print(f'start error % ' + ' '.join(f'{k} {v:+.3f}' for k, v in err(th).items()), flush=True)
for it in range(a.nsteps):
    t0 = time.perf_counter(); keep, ok, failed = {}, [], []
    for i in todo:
        try:
            b, kp = derivs(i, th, ld[i]); keep[i] = kp; ok.append(b)
        except Exception as e:
            failed.append(dict(batch=i, error=repr(e)[:300])); print(f'[step {it} batch {i}] FAILED {repr(e)[:300]}', flush=True)
    if not ok: sys.exit('no batch succeeded')
    L0 = sum(b['L0'] for b in ok); R0 = sum(b['R0'] for b in ok); J0 = L0 + R0
    print(f'[step {it}] derivatives on {len(ok)}/{len(todo)} batches in {time.perf_counter() - t0:.0f} s; J0 {J0:.2f}', flush=True)
    ladder = []
    for kappa, donly in [(k, False) for k in KAPPAS] + [(k, True) for k in KAPPAS if np.isfinite(k)]:   # + d-only rungs (fix 2)
        du, dd, S, gSb = step(kappa, ok, keep, d_only=donly); L1, R1, th1, dm = J_at(th, du, dd, ok)
        e = dict(kappa=kappa if np.isfinite(kappa) else 'inf', d_only=donly, dJ=L1 + R1 - J0, dL=L1 - L0, dR=R1 - R0, model=model(du, dd, ok, keep),
                 du_pct=(100 * du).tolist(), err1=err(th1), rms_dd=float(np.sqrt(np.mean(np.concatenate(list(dd.values())) ** 2))),
                 max_abs_dd=float(max(np.abs(v).max() for v in dd.values())), dmean_dedx_pct_median=float(np.median(dm)),
                 eig_S=np.linalg.eigvalsh(0.5 * (S + S.T)).tolist())
        ladder.append(e)
        print(f'[step {it}] kappa {e["kappa"]}{" d-only" if donly else ""}: dJ {e["dJ"]:+.2f} (data {e["dL"]:+.2f} prior {e["dR"]:+.2f}; model {e["model"]:+.2f})  step % '
              + ' '.join(f'{k} {x:+.3f}' for k, x in zip(PARAMS, e['du_pct'])) + f'  | dd rms {e["rms_dd"]:.4f}', flush=True)
    best = min(ladder, key=lambda x: x['dJ'] if np.isfinite(x['dJ']) else np.inf)   # smoke 4: a NaN rung won min()
    kb = float(best['kappa']) if best['kappa'] != 'inf' else float('inf')
    rec = dict(step=it, n_ok=len(ok), failed=failed, J0=J0, L0=L0, R0=R0, err_before=err(th), ladder=ladder, best_kappa=best['kappa'], best_d_only=best['d_only'],
               accepted=bool(best['dJ'] < 0), dJ=best['dJ'], du_pct=best['du_pct'], err_after=best['err1'],
               n_batches_neg=int(sum(b['n_neg'] > 0 for b in ok)), max_band_asym=max(b['band_asym'] for b in ok),
               max_sym_ud=max(b['sym_ud'] for b in ok))
    S0 = step(0.0, ok, keep)[2]; I5 = np.eye(5)
    rec['sigma_marg_pct'] = (100 * np.sqrt(np.abs(np.diag(solve_abs(S0, I5))))).tolist()
    Huu_t = sum(keep[b['batch']]['Huu'] for b in ok)
    rec['sigma_cond_pct'] = (100 * np.sqrt(np.abs(np.diag(solve_abs(Huu_t, I5))))).tolist()
    if np.isfinite(kb):                                   # diagnostics at the best finite rung
        du, dd, S, gSb = step(kb, ok, keep); Si = solve_pos(S, I5)
        rec['per_batch'] = [dict(batch=i, gS=g.tolist(), share_pct=(-100 * Si @ g).tolist()) for i, g in gSb.items()]
        rec['theta_profile_gain'] = float(-0.5 * du @ S @ du)   # quadratic profile gain of the calibration move
        du0, dd0, _, _ = step(kb, ok, keep, d_only=True); L1, R1, _, _ = J_at(th, du0, dd0, ok)
        rec['d_only'] = dict(dJ=L1 + R1 - J0, dL=L1 - L0, dR=R1 - R0, model=model(du0, dd0, ok, keep))
        print(f'[step {it}] d-only rung (kappa {kb}): dJ {rec["d_only"]["dJ"]:+.2f}; quadratic theta-profile gain {rec["theta_profile_gain"]:+.2f}', flush=True)
    res['steps'].append(rec)
    print(f'[step {it}] best kappa {best["kappa"]}{" d-only" if best["d_only"] else ""}: dJ {best["dJ"]:+.2f}  error after % ' + ' '.join(f'{k} {v:+.3f}' for k, v in best['err1'].items())
          + '  | sigma marg % ' + ' '.join(f'{x:.3f}' for x in rec['sigma_marg_pct']) + f'  ({time.perf_counter() - t0:.0f} s)', flush=True)
    if not rec['accepted']:
        json.dump(res, open(a.out, 'w'), indent=1, default=_js); print(f'[step {it}] not accepted -> stop', flush=True); break
    du, dd, _, _ = step(kb, ok, keep, d_only=best['d_only'])
    th = {k: th[k] * float(np.exp(x)) for k, x in zip(PARAMS, du)}
    for i in dd: ld[i] = ld[i] + dd[i]
    save_state(th, ld); res['theta_final'] = dict(th); res['err_final'] = err(th)
    json.dump(res, open(a.out, 'w'), indent=1, default=_js)
res.setdefault('err_final', err(th)); res.setdefault('theta_final', dict(th))
json.dump(res, open(a.out, 'w'), indent=1, default=_js)
print('final error % ' + ' '.join(f'{k} {v:+.3f}' for k, v in res['err_final'].items()), flush=True)
print('done', flush=True)
