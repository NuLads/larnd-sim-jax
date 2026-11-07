"""Joint Newton step on the five calibration parameters AND the per-segment log dE/dx of a finished joint fit (21 Oct),
positions (spline coefficients of the final checkpoint) held fixed. Extends newton_polish.py, which held dE/dx fixed too.

Objective, summed over the batches b of one track sample, against the fit's own stored stochastic target:
  J(u, d) = sum_b [ L_b(u, d_b) + R_b(d_b) ],   u = ln theta (5, shared),  d_b = log dE/dx of batch b's segments
L_b = data loss (probabilistic model + PPP likelihood, as newton_polish.py); R_b = the fit's dE/dx prior, exactly as
loss_wrapper_combined builds it (split-t on the per-row dE/dx weighted by dx, x prior weight; soft barrier; length-
weighted mean constraint; drift-profile weight is 0 in KN10). Each d_b enters only batch b, so the Hessian is
block-arrow and d is eliminated per batch (Schur complement):
  S = sum_b (H_uu,b - H_ud,b D_b^-1 H_du,b),  gS = sum_b (g_u,b - H_ud,b D_b^-1 g_d,b),  du = -S^-1 gS,
  dd_b = -D_b^-1 (g_d,b + H_du,b du),  D_b = Hdata_dd,b + Hprior_dd,b.
Derivatives: central finite differences of the AD gradient (the forward-over-reverse HVP needs > 40 GB at 400 cm):
  5 columns from u_k +- h_k (as newton_polish.py) give H_uu and H_du exactly; 5 structurally orthogonal probes
  v_c[j] = 1 iff j % 5 == c, d +- eps v_c, give the data band |i - j| <= 2 (rows follow the segments along each track);
  the prior's Hessian is dense (rank-1 mean term) and comes from AD exactly. 21 gradients per batch.
Checks: probe-side H_ud v_c vs the column-side sum (symmetry); --dense_check B rebuilds Hdata_dd of batch B from N unit
columns and reports the band error and the off-band mass. The step is a saddle-free trust-region ladder (see the
comment at the step; the pure Newton step failed in smoke 1): kappa in 0 .. 10 and inf (= d fixed), lowest objective kept;
the objective at every rung and the quadratic model with the true signed curvature are reported.
usage: python joint_polish.py --run KN10S --sample 1 [--suffix b] --iter 10000 --out out.json [--nb 100] [--eps 1e-2]
       [--dense_check B] [--batches 0,1,2]"""
import os, sys, glob, json, time, argparse
import numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/eventdisplay')
from rebuild import FitRebuild
import jax, jax.numpy as jnp

ap = argparse.ArgumentParser(); ap.add_argument('--run', default='KN10S'); ap.add_argument('--iter', type=int, default=10000)
ap.add_argument('--sample', type=int, required=True); ap.add_argument('--suffix', default=''); ap.add_argument('--nb', type=int, default=100)
ap.add_argument('--out', required=True); ap.add_argument('--eps', type=float, default=1e-2)
ap.add_argument('--dense_check', type=int, default=-1); ap.add_argument('--batches', default='')
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


def batch_fns(i):
    fitted = jnp.asarray(R.batch(i, check=False)['fitted']); tgt = {k: jnp.asarray(v) for k, v in R.target(i).items()}
    pid = jnp.asarray(np.asarray(ck['batch_parent_ids'][i])); ld0 = np.asarray(ck['dedx_cache'][i]['log_dedx'], np.float64).ravel()
    valid = pid >= 0; safe = jnp.clip(pid, 0, ld0.shape[0] - 1)

    def tracks_of(ld):                                    # loss_wrapper_combined, "Reconstruct per-step dEdx"
        ded = jnp.where(valid, jnp.maximum(jnp.exp(jnp.take(ld, safe)), 1e-3), fitted[:, di])
        return fitted.at[:, di].set(ded).at[:, Ei].set(ded * fitted[:, xi]), ded

    def prior(ld):                                        # loss_wrapper_combined: split-t + barrier + mean constraint
        t, ded = tracks_of(ld); dxw = t[:, xi]; lsf = t[:, lsf_i] if lsf_i is not None else 1.0
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
res = dict(run=a.run, iter=a.iter, sample=a.sample, suffix=a.suffix, ckpt=ckpt, eps=a.eps, params=PARAMS, truth=pc.TRUTH,
           theta0=th0, batches=[])
nb = min(a.nb, len(R.ds_sim)); todo = [int(x) for x in a.batches.split(',')] if a.batches else list(range(nb))
keep, fwd = {}, {}           # per batch: derivatives; forward-only closures (reused by the step ladder)
for i in todo:
    t0 = time.perf_counter()
    try:
        grad, prior, ld0, val = batch_fns(i); N = ld0.shape[0]
        L0, gu, gd = grad(th0, ld0)
        Huu = np.zeros((5, 5)); Hdu = np.zeros((N, 5))
        for c, k in enumerate(PARAMS):                    # columns d(g)/d u_k
            h = H_STEP[k]
            _, gup, gdp = grad({**th0, k: th0[k] * float(np.exp(h))}, ld0); _, gum, gdm = grad({**th0, k: th0[k] * float(np.exp(-h))}, ld0)
            Huu[:, c] = (gup - gum) / (2 * h); Hdu[:, c] = (gdp - gdm) / (2 * h)
        B = np.zeros((N, N)); probe_ud = np.zeros((5, 5)); col_ud = np.zeros((5, 5)); j = np.arange(N)
        for c in range(5):                                # structurally orthogonal probes of the data band
            v = (j % 5 == c).astype(np.float64)
            _, gup, gdp = grad(th0, ld0 + a.eps * v); _, gum, gdm = grad(th0, ld0 - a.eps * v)
            Hv = (gdp - gdm) / (2 * a.eps); probe_ud[:, c] = (gup - gum) / (2 * a.eps); col_ud[:, c] = Hdu[v > 0].sum(0)
            for off in range(-2, 3):                      # entry (r, r + off) with (r + off) % 5 == c
                r = j[(j + off >= 0) & (j + off < N) & ((j + off) % 5 == c)]; B[r, r + off] = Hv[r]
        asym = float(np.abs(B - B.T).max() / max(np.abs(np.diag(B)).max(), 1e-30)); B = 0.5 * (B + B.T)
        Huu = 0.5 * (Huu + Huu.T)
        sym_ud = float(np.abs(probe_ud - col_ud).max() / max(np.abs(col_ud).max(), 1e-30))
        R0, mean0 = prior(f32(ld0)); gR = np.asarray(jax.grad(lambda x: prior(x)[0])(f32(ld0)), np.float64)
        HR = np.asarray(jax.hessian(lambda x: prior(x)[0])(f32(ld0)), np.float64)
        D = B + HR; D = 0.5 * (D + D.T); ev, Vd = np.linalg.eigh(D)
        gdt = gd + gR                                     # total d-gradient (data + prior)
        pos = ev > 1e-9 * ev[-1]                          # profile only the positive-curvature directions of D_b;
        Y = (Vd.T @ Hdu)[pos]                             # the (rare) others are held fixed (smoke 2: |eigenvalue| distorts S)
        Sb0 = Huu - Y.T @ (Y / ev[pos][:, None])
        b = dict(batch=i, N=N, L0=L0, R0=float(R0), mean_dedx0=float(mean0), g_u=gu.tolist(), H_uu=Huu.tolist(), S_b=Sb0.tolist(),
                 eig_D_min=float(ev[0]), eig_D_max=float(ev[-1]), n_neg=int((ev <= 0).sum()), band_asym=asym, sym_ud=sym_ud,
                 gd_data_norm=float(np.linalg.norm(gd)), gd_total_norm=float(np.linalg.norm(gdt)))
        if i == a.dense_check:                            # full data block from N unit columns
            Dn = np.zeros((N, N))
            for jj in range(N):
                e = np.zeros(N); e[jj] = a.eps
                Dn[:, jj] = (grad(th0, ld0 + e)[2] - grad(th0, ld0 - e)[2]) / (2 * a.eps)
            Dn = 0.5 * (Dn + Dn.T); band = np.abs(np.subtract.outer(j, j)) <= 2; sc = np.abs(np.diag(Dn)).max()
            Dd = Dn + HR; Dd = 0.5 * (Dd + Dd.T); evd, Vdd = np.linalg.eigh(Dd)
            pd_ = evd > 1e-9 * evd[-1]; Yd = (Vdd.T @ Hdu)[pd_]; Sbd = Huu - Yd.T @ (Yd / evd[pd_][:, None])
            b['dense'] = dict(band_err=float(np.abs((Dn - B)[band]).max() / sc), offband_max=float(np.abs(Dn[~band]).max() / sc),
                              offband_frac=float(np.square(Dn[~band]).sum() / np.square(Dn).sum()),
                              S_b_dense=Sbd.tolist(), eig_D_min_dense=float(evd[0]), n_neg_dense=int((evd <= 0).sum()))
            print(f'[dense {i}] band err {b["dense"]["band_err"]:.2e}  off-band max {b["dense"]["offband_max"]:.2e}  '
                  f'off-band energy {b["dense"]["offband_frac"]:.2e}  S_b rel diff '
                  f'{np.abs(Sbd - Sb0).max() / np.abs(Sb0).max():.2e}  eig(D) min dense {evd[0]:.2e} vs band {ev[0]:.2e}', flush=True)
        keep[i] = dict(ld0=ld0, ev=ev, V=Vd, Hdu=Hdu, gdt=gdt); fwd[i] = (prior, val)
        res['batches'].append(b)
        print(f'[batch {i}] N {N}  L0 {L0:.1f} R0 {float(R0):.1f}  eig(D) {ev[0]:.2e}..{ev[-1]:.2e} n_neg {int((ev <= 0).sum())}  band asym {asym:.1e}  '
              f'H_ud sym {sym_ud:.1e}  |g_d| data {np.linalg.norm(gd):.2f} total {np.linalg.norm(gdt):.2f}  {time.perf_counter() - t0:.0f} s', flush=True)
    except Exception as e:
        res['batches'].append(dict(batch=i, error=repr(e)[:300])); print(f'[batch {i}] FAILED {repr(e)[:300]}', flush=True)
    json.dump(res, open(a.out, 'w'), default=_js)

ok = [b for b in res['batches'] if 'S_b' in b]
if not ok: sys.exit('no batch succeeded')
Huu = sum(np.array(b['H_uu']) for b in ok); Huu = 0.5 * (Huu + Huu.T); gu = sum(np.array(b['g_u']) for b in ok)
J0 = sum(b['L0'] + b['R0'] for b in ok)
# Trust-region ladder (21 Oct, after smoke 1): the pure Newton step is ill-posed -- with dE/dx profiled out, a direction
# that is almost pure lifetime is ~10x softer than at fixed dE/dx, and the step leaves the quadratic region (objective
# +42508 for a predicted -1656). Each D_b is replaced by V (Lambda + kappa median|Lambda|) V^T on its positive-curvature
# subspace (Levenberg damping; the few directions with Lambda <= 0 are held fixed -- smoke 2 showed |Lambda| distorts the
# Schur complement there: batch 0, S_b(eField) < 0); kappa = inf is d fixed (newton_polish.py's step, from the same derivatives). The step with the lowest
# objective is kept.
KAPPAS = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, float('inf')]


def solve_abs(M, v):                                      # |M|^-1 v, saddle-free
    w, V = np.linalg.eigh(0.5 * (M + M.T)); y = V.T @ v
    return V @ (y / np.abs(w).reshape((-1,) + (1,) * (y.ndim - 1)))


def step(kappa):
    if np.isinf(kappa):
        return -solve_abs(Huu, gu), {b['batch']: np.zeros(b['N']) for b in ok}, Huu
    S = np.zeros((5, 5)); gS = np.zeros(5); X = {}
    for b in ok:
        kp = keep[b['batch']]; ev = kp['ev']; pos = ev > 1e-9 * ev[-1]
        il = np.where(pos, 1.0 / np.where(pos, ev + kappa * np.median(np.abs(ev)), 1.0), 0.0)   # non-positive directions: no step
        Xb = kp['V'] @ ((kp['V'].T @ np.column_stack([kp['gdt'], kp['Hdu']])) * il[:, None]); X[b['batch']] = Xb
        S += np.array(b['H_uu']) - kp['Hdu'].T @ Xb[:, 1:]; gS += np.array(b['g_u']) - kp['Hdu'].T @ Xb[:, 0]
    du = -solve_abs(S, gS)
    return du, {i: -(Xb[:, 0] + Xb[:, 1:] @ du) for i, Xb in X.items()}, S


def model(du, dd):                                        # quadratic model with the TRUE (signed) curvature
    m = float(gu @ du) + 0.5 * float(du @ Huu @ du)
    for b in ok:
        kp = keep[b['batch']]; d = dd[b['batch']]; y = kp['V'].T @ d
        m += float(kp['gdt'] @ d) + float(d @ kp['Hdu'] @ du) + 0.5 * float(np.sum(kp['ev'] * y * y))
    return m


def J_at(du, dd):                                         # objective at (u0 + du, d0 + dd), same batches
    th = {k: th0[k] * float(np.exp(x)) for k, x in zip(PARAMS, du)}; tot, dm = 0.0, []
    for b in ok:
        i = b['batch']
        if i not in fwd: _, pr, _, val = batch_fns(i); fwd[i] = (pr, val)
        pr, val = fwd[i]; ld1 = keep[i]['ld0'] + dd[i]
        L1 = val(th, ld1); R1, m1 = pr(f32(ld1)); tot += L1 + float(R1); dm.append(100 * (float(m1) / b['mean_dedx0'] - 1))
    return tot, th, dm


ladder = []
for kappa in KAPPAS:
    du, dd, S = step(kappa); J1, th1, dm = J_at(du, dd)
    e = dict(kappa=kappa if np.isfinite(kappa) else 'inf', dJ=J1 - J0, model=model(du, dd), du_pct=(100 * du).tolist(), err1=err(th1),
             max_abs_dd=float(max(np.abs(v).max() for v in dd.values())), rms_dd=float(np.sqrt(np.mean(np.concatenate(list(dd.values())) ** 2))),
             dmean_dedx_pct_median=float(np.median(dm)), eig_S=np.linalg.eigvalsh(0.5 * (S + S.T)).tolist())
    ladder.append(e)
    print(f'kappa {e["kappa"]}: dJ {e["dJ"]:+.2f} (model {e["model"]:+.2f})  step % ' + ' '.join(f'{k} {x:+.3f}' for k, x in zip(PARAMS, e['du_pct']))
          + f'  | dd rms {e["rms_dd"]:.4f} max {e["max_abs_dd"]:.3f}  mean dE/dx {e["dmean_dedx_pct_median"]:+.3f}%', flush=True)
    json.dump(dict(res, ladder=ladder), open(a.out, 'w'), default=_js)
best = min(ladder, key=lambda x: x['dJ'])
S0 = step(0.0)[2]; I5 = np.eye(5)
res['summary'] = dict(n_ok=len(ok), n_batches=len(todo), ladder=ladder, best_kappa=best['kappa'], accepted=bool(best['dJ'] < 0),
                      dJ=best['dJ'], dJ_model=best['model'], du_pct=best['du_pct'], err0=err(th0), err1=best['err1'],
                      du_cond_pct=ladder[-1]['du_pct'], err1_cond=ladder[-1]['err1'], dJ_cond=ladder[-1]['dJ'],
                      n_batches_neg=int(sum(b['n_neg'] > 0 for b in ok)), n_neg_total=int(sum(b['n_neg'] for b in ok)),
                      eig_S0=np.linalg.eigvalsh(0.5 * (S0 + S0.T)).tolist(),
                      sigma_marg_pct=(100 * np.sqrt(np.abs(np.diag(solve_abs(S0, I5))))).tolist(),
                      sigma_cond_pct=(100 * np.sqrt(np.abs(np.diag(solve_abs(Huu, I5))))).tolist(),
                      max_band_asym=max(b['band_asym'] for b in ok), max_sym_ud=max(b['sym_ud'] for b in ok))
json.dump(res, open(a.out, 'w'), indent=1, default=_js)
s = res['summary']
print(f"\n{a.run}{a.sample}{a.suffix}: {len(ok)}/{len(todo)} batches; batches with negative D curvature {s['n_batches_neg']} "
      f"({s['n_neg_total']} eigenvalues); eig(S, kappa 0) " + ' '.join(f'{x:.3e}' for x in s['eig_S0']))
print(f"  best kappa {s['best_kappa']}: dJ {s['dJ']:+.2f} (model {s['dJ_model']:+.2f}); d fixed: dJ {s['dJ_cond']:+.2f}")
print('  error before %       ' + '  '.join(f'{k} {s["err0"][k]:+.3f}' for k in PARAMS))
print('  error after (best) % ' + '  '.join(f'{k} {s["err1"][k]:+.3f}' for k in PARAMS))
print('  error after (d fixed)' + '  '.join(f'{k} {s["err1_cond"][k]:+.3f}' for k in PARAMS))
print('  sigma marginal / conditional % ' + '  '.join(f'{k} {x:.3f}/{y:.3f}' for k, x, y in zip(PARAMS, s['sigma_marg_pct'], s['sigma_cond_pct'])), flush=True)
print('done', flush=True)
