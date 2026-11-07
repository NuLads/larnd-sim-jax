"""
Ablation ladder: which production dE/dx ingredient turns the clean-setup lifetime result
(-2.6 +- 2.2%, Laplace test rung 0) into the production +8.85%?

Everything is at truth except per-unit dE/dx nuisances (geometry = the TRUE trajectories).
For each (Ab, lifetime) grid point and batch, dE/dx is Newton-fitted to convergence (MAP); the
profiled objective G(theta) = min_u [NLL + prior (+ mean constraint)] is recorded, together with the
reference NLL at the true (unit-averaged) dE/dx.

  --seg target : nuisance unit = target-file segment (rung 0 of the Laplace test)
  --seg cm1    : nuisance unit = ~1 cm piece of the TRUE trajectory (n = round(L/1cm) equal pieces per
                 trajectory, like the production guess grid, but following the true path)
  --prior binned : proper population prior (length-binned split-t, smoothed at the mode, 1 count/unit)
  --prior refit_proper : production 'refit' split-t shape, 1 count/unit, smoothed, + Jacobian
  --prior prod : EXACT production form: -w_p * sum_unit l_u * log splitt(d_u) [w_p = 1.0088, unsmoothed,
                 no Jacobian] + soft barrier sum l_u relu(d_u - 8.5)^2
  --mean_w     : production length-weighted mean constraint w (m - t)^2, t = 1.887 (0 = off)
Imports the repo read-only.
"""
import os, sys, time, argparse, json
import numpy as np

REPO = '/sdf/group/neutrino/pgranger/larnd-sim-jax'
sys.path.insert(0, REPO + '/src'); sys.path.insert(0, REPO)
os.chdir(REPO)

import jax, jax.numpy as jnp
jax.config.update('jax_platform_name', 'gpu')

from optimize.dataio import TracksDataset, TgtTracksDataset, chop_tracks, np_from_structured, remap_event_ids_to_local
from optimize.strategies import LUTProbabilisticSimulation, ProbabilisticLossStrategy
from optimize.dedx_utils import split_t_logpdf_jax
from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from larndsim.sim_jax import simulate_wfs, simulate_stochastic
from larndsim.losses_jax import llhd_loss

p = argparse.ArgumentParser()
p.add_argument('--b0', type=int, default=0)
p.add_argument('--b1', type=int, default=1)
p.add_argument('--out', required=True)
p.add_argument('--seg', default='cm1', choices=['target', 'cm1'])
p.add_argument('--piece_cm', type=float, default=1.0)
p.add_argument('--prior', default='prod', choices=['binned', 'refit_proper', 'prod'])
p.add_argument('--mean_w', type=float, default=0.0)
p.add_argument('--mean_tgt', type=float, default=1.887)
p.add_argument('--lt_grid', default='0.88,0.92,0.96,1.0,1.04,1.08,1.12,1.16')
p.add_argument('--dab_grid', default='-0.0015,0.0,0.0015,0.003,0.0045')
p.add_argument('--valley', type=float, default=20.3)
p.add_argument('--start', default='1.0,0.0015', help='(lt, dab) of the cold-start point')
p.add_argument('--hess_every', type=int, default=4)
p.add_argument('--dense_check', type=int, default=0)
p.add_argument('--rt', type=float, default=4.5)
p.add_argument('--maxit', type=int, default=25)
p.add_argument('--maxit_cold', type=int, default=60)
p.add_argument('--tol', type=float, default=5e-3)
p.add_argument('--max_nbatch', type=int, default=100)
p.add_argument('--batch_list', default='', help='colon-separated batch indices (overrides --b0/--b1)')
p.add_argument('--rungs', default='', help='NAME:prior:mean_w,... run in sequence per batch, each warm-started from the previous rung (overrides --prior/--mean_w)')
args = p.parse_args()

INPUT_SIM = '/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5'
INPUT_TGT = '/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5'
TRUTH = dict(Ab=0.83488135, eField=0.52151894, tran_diff=1.00276338e-05,
             long_diff=5.81418228e-06, lifetime=2406.44659703)
PARAMS = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
HERE = os.path.dirname(os.path.abspath(__file__))
PROD = dict(nuL=21.76, nuR=4.72, loc=1.8786, sL=0.0782, sR=0.0801, w=1.0088, barrier_thr=8.5, barrier_w=1.0)

t_start = time.time()
ds_sim = TracksDataset(filename=INPUT_SIM, nevents=600, max_nbatch=args.max_nbatch, random_nevents=True,
                       data_seed=1, track_len_sel=2., max_abs_costheta_sel=0.966, min_abs_segz_sel=15.,
                       track_z_bound=28., max_batch_len=400., print_input=False, chopped=True,
                       pad=True, electron_sampling_resolution=0.01, live_selection=False,
                       use_dedx_density=False, dedx_density_mode='histogram')
ds_tgt = TgtTracksDataset(filename=INPUT_TGT, dataset_sim=ds_sim, chopped=True, pad=True,
                          electron_sampling_resolution=0.01, print_input=False)
fields = ds_tgt.get_track_fields()
F = {k: fields.index(k) for k in ['dEdx', 'dE', 'dx', 'eventID', 'trackID', 'pixel_plane', 'x', 'y',
                                   'x_start', 'y_start', 'z_start', 'x_end', 'y_end', 'z_end', 'z']}
print(f'[DATA] {len(ds_tgt)} batches, load {time.time()-t_start:.1f}s  config {vars(args)}', flush=True)

Params = build_params_class(PARAMS)
ref = load_detector_properties(Params, 'src/larndsim/detector_properties/module0.yaml',
                               'src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml')
ref = ref.replace(electron_sampling_resolution=0.01, number_pix_neighbors=4, signal_length=150, time_window=150)
response, ref = load_lut('src/larndsim/detector_properties/response_44.npy', ref)
ref = ref.replace(diffusion_in_current_sim=False, mc_diff=False, use_dedx_density=False, dedx_density_mode='histogram')
truth = ref.replace(**TRUTH)
sim = LUTProbabilisticSimulation(response)
loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)
pb = np.load(os.path.join(HERE, 'prior_bins.npz'))
PEDGES, PPAR = pb['edges'], pb['params']

lt_grid = np.array([float(x) for x in args.lt_grid.split(',')])
dab_grid = np.array([float(x) for x in args.dab_grid.split(',')])
s_lt, s_da = [float(x) for x in args.start.split(',')]
START = (int(np.argmin(np.abs(lt_grid - s_lt))), int(np.argmin(np.abs(dab_grid - s_da))))


def theta(il, ia):
    ab = TRUTH['Ab'] * (1.0 - (lt_grid[il] - 1.0) / args.valley + dab_grid[ia])
    return ab, TRUTH['lifetime'] * lt_grid[il]


def smooth_split_t_logkernel(x, nuL, nuR, loc, sL, sR, wfrac=0.3):
    w = wfrac * jnp.minimum(sL, sR)
    t = jax.nn.sigmoid((x - loc) / w)
    s = sL + (sR - sL) * t; nu = nuL + (nuR - nuL) * t
    z = (x - loc) / s
    return -0.5 * (nu + 1.0) * jnp.log1p(z * z / nu)


def build_units(i):
    """Return tracks (padded), unit id per row (-1 = padding), unit length, true unit dE/dx,
    unit event/plane, and (x,y) sample points with owners for the coupling graph."""
    rows = ds_tgt.tracks_struct[ds_tgt.batch_row_indices[i]]
    arr = np_from_structured(rows)
    arr = remap_event_ids_to_local(arr, rows['eventID'], ds_tgt.batch_event_global_ids[i], fields)
    chopped, pid = chop_tracks(arr, fields, precision=0.01)
    tracks_np = np.asarray(ds_tgt[i]).reshape(-1, len(fields))
    nreal = chopped.shape[0]
    assert np.allclose(tracks_np[:nreal, F['dEdx']], chopped[:, F['dEdx']], rtol=1e-5), 'row mismatch'
    st = tracks_np[:nreal]
    if args.seg == 'target':
        unit = pid.astype(np.int64)
    else:
        key = st[:, F['eventID']].astype(np.int64) * 1000003 + st[:, F['trackID']].astype(np.int64)
        unit = -np.ones(nreal, dtype=np.int64); off = 0
        _, first = np.unique(key, return_index=True)
        for k in key[np.sort(first)]:
            idx = np.where(key == k)[0]
            dx = st[idx, F['dx']].astype(np.float64)
            cum = np.cumsum(dx); L = cum[-1]
            n = max(1, int(round(L / args.piece_cm))); pl = L / n
            pc = np.minimum(((cum - 0.5 * dx) / pl).astype(np.int64), n - 1)
            unit[idx] = off + pc; off += n
    N = int(unit.max()) + 1
    dx = st[:, F['dx']].astype(np.float64); dE = st[:, F['dE']].astype(np.float64)
    l = np.bincount(unit, weights=dx, minlength=N)
    d_true = np.bincount(unit, weights=dE, minlength=N) / np.maximum(l, 1e-12)
    ev = np.zeros(N); pl_ = np.zeros(N); ev[unit] = st[:, F['eventID']]; pl_[unit] = st[:, F['pixel_plane']]
    z_mid = np.bincount(unit, weights=dx * st[:, F['z']], minlength=N) / np.maximum(l, 1e-12)
    sub = np.arange(0, nreal, 10)
    pts = st[sub][:, [F['x'], F['y']]].astype(np.float64); own = unit[sub]
    ends = np.r_[np.where(np.diff(unit) != 0)[0], nreal - 1]
    pts = np.vstack([pts, st[ends][:, [F['x'], F['y']]]]); own = np.r_[own, unit[ends]]
    unit_full = -np.ones(tracks_np.shape[0], dtype=np.int32); unit_full[:nreal] = unit
    return tracks_np, unit_full, l, d_true, ev, pl_, z_mid, pts, own


def coupling(N, pts, own, ev, pl_, rt):
    from scipy.spatial import cKDTree
    A = np.eye(N, dtype=bool)
    pr = cKDTree(pts).query_pairs(rt, output_type='ndarray')
    a, b = own[pr[:, 0]], own[pr[:, 1]]
    A[a, b] = True; A[b, a] = True
    A &= (ev[:, None] == ev[None, :]) & (pl_[:, None] == pl_[None, :])
    A2 = (A.astype(np.int32) @ A.astype(np.int32)) > 0
    col = -np.ones(N, dtype=int)
    for s in np.argsort(-A2.sum(1)):
        used = set(col[A2[s]].tolist()); c = 0
        while c in used: c += 1
        col[s] = c
    return A, col


REC_KEYS = ['G', 'nll', 'prior', 'mean_pen', 'mean_dedx', 'nll_ref', 'decr', 'niter', 'fresh', 'logdet',
            'n_negeig', 'mineig', 'scale_err']
RUNGS = [tuple(r.split(':')) for r in args.rungs.split(',')] if args.rungs else [('R', args.prior, str(args.mean_w))]
RUNGS = [(n, pk, float(mw)) for n, pk, mw in RUNGS]
print('[RUNGS]', RUNGS, flush=True)
rec = {'batch': [], 'nunit': [], 'ncolor': [], 'dense_err': [], 't_batch': []}
for n, _, _ in RUNGS:
    for k in REC_KEYS + ['nfall', 't_rung']: rec[f'{n}_{k}'] = []
extra = {}
shape = (len(lt_grid), len(dab_grid))
ORDER = [START]; _last = START
_todo = {(a, b) for a in range(shape[0]) for b in range(shape[1])} - {START}
while _todo:
    _n = min(_todo, key=lambda q: (abs(q[0] - _last[0]) + abs(q[1] - _last[1]), abs(q[0] - _last[0]), q))
    ORDER.append(_n); _todo.remove(_n); _last = _n

import gc
BATCHES = [int(x) for x in args.batch_list.split(':')] if args.batch_list else list(range(args.b0, min(args.b1, len(ds_tgt))))
print('[BATCHES]', BATCHES, flush=True)
for i in BATCHES:
    tb = time.time()
    tracks_np, unit_full, l_np, d_true, ev, pl_, z_mid, pts, own = build_units(i)
    N = len(l_np)
    u_true = np.log(np.maximum(d_true, 1e-3)).astype(np.float32)
    tracks = jnp.asarray(tracks_np)
    valid = jnp.asarray(unit_full >= 0); safe = jnp.asarray(np.clip(unit_full, 0, N - 1))
    lj = jnp.asarray(l_np, dtype=jnp.float32); Ltot = float(l_np.sum())
    bi = np.clip(np.searchsorted(PEDGES, l_np, side='right') - 1, 0, len(PPAR) - 1)
    pp = jnp.asarray(PPAR[bi].T, dtype=jnp.float32)
    wfs, up = simulate_wfs(truth, response, tracks, fields)
    adcs, x, y, z, ticks, hp, evh, pix = simulate_stochastic(truth, wfs, up, rngseed=i + 1)
    tg = dict(adcs=adcs, pixel_x=x, pixel_y=y, pixel_z=z, ticks=ticks, hit_prob=hp, event=evh, pixel_id=pix)
    usize = int(np.asarray(up).shape[0])
    _ = sim.predict(truth, tracks, fields, None)
    roi = (sim.padded_small_nb, sim.padded_large_nb)
    del wfs
    A, col = coupling(N, pts, own, ev, pl_, args.rt)
    ncol = int(col.max()) + 1

    def nll_of(u, ab, lt):
        d = jnp.exp(u)
        ds = jnp.where(valid, d[safe], tracks[:, F['dEdx']])
        t = tracks.at[:, F['dEdx']].set(ds).at[:, F['dE']].set(ds * tracks[:, F['dx']])
        par = truth.replace(Ab=ab, lifetime=lt)
        pred = sim.predict(par, t, fields, None, unique_size=usize, roi_override=roi)
        return loss.compute(par, pred, tg)[0]
    nll_ref_fn = jax.jit(nll_of)

    def make_rung(prior_kind, mean_w):
        def prior_of(u):
            d = jnp.exp(u)
            if prior_kind == 'binned':
                return -jnp.sum(smooth_split_t_logkernel(d, pp[0], pp[1], pp[2], pp[3], pp[4])) - jnp.sum(u)
            if prior_kind == 'refit_proper':
                return -jnp.sum(smooth_split_t_logkernel(d, PROD['nuL'], PROD['nuR'], PROD['loc'], PROD['sL'], PROD['sR'])) - jnp.sum(u)
            lp = split_t_logpdf_jax(d, PROD['nuL'], PROD['nuR'], PROD['loc'], PROD['sL'], PROD['sR'])
            bar = PROD['barrier_w'] * jnp.sum(lj * jnp.square(jnp.maximum(0., d - PROD['barrier_thr'])))
            return -PROD['w'] * jnp.sum(lj * lp) + bar

        def G_nomean(u, ab, lt):
            return nll_of(u, ab, lt) + prior_of(u)

        def G(u, ab, lt):
            nll = nll_of(u, ab, lt); pr = prior_of(u); m = jnp.sum(lj * jnp.exp(u)) / Ltot
            mp = mean_w * jnp.square(m - args.mean_tgt)
            return nll + pr + mp, (nll, pr, mp, m)

        R = dict(vg=jax.jit(jax.value_and_grad(G, has_aux=True)), val=jax.jit(G), NF={'jit': 0, 'eager': 0, 'fd': 0})
        _hvp_raw = lambda u, v, ab, lt: jax.jvp(lambda w_: jax.grad(lambda z_: G_nomean(z_, ab, lt))(w_), (u,), (v,))[1]
        _hvp_jit = jax.jit(_hvp_raw); _gnm = jax.jit(jax.grad(G_nomean))

        def hvp(u, v, ab, lt):
            R['NF']['jit'] += 1
            r = _hvp_jit(u, v, ab, lt)
            if not bool(jnp.all(jnp.isfinite(r))):
                R['NF']['eager'] += 1; r = _hvp_raw(u, v, ab, lt)
            if not bool(jnp.all(jnp.isfinite(r))):
                R['NF']['fd'] += 1; eps = 1e-3
                r = (_gnm(u + eps * v, ab, lt) - _gnm(u - eps * v, ab, lt)) / (2 * eps)
            if not bool(jnp.all(jnp.isfinite(r))):
                raise FloatingPointError('HVP non-finite after jit, eager and finite-difference attempts')
            return r

        def mean_hess(u):
            if mean_w == 0: return 0.0
            g = l_np * np.exp(np.asarray(u, dtype=np.float64)) / Ltot; m = g.sum()
            return 2 * mean_w * np.outer(g, g) + 2 * mean_w * (m - args.mean_tgt) * np.diag(g)

        def hessian(u, ab, lt, dense=False):
            if dense:
                H = np.stack([np.asarray(hvp(u, jnp.asarray(np.eye(N, dtype=np.float32)[k]), ab, lt), dtype=np.float64)
                              for k in range(N)], 1)
            else:
                H = np.zeros((N, N))
                for c in range(ncol):
                    hv = np.asarray(hvp(u, jnp.asarray((col == c).astype(np.float32)), ab, lt), dtype=np.float64)
                    for j in np.where(col == c)[0]:
                        rws = np.where(A[:, j])[0]; H[rws, j] = hv[rws]
            H = 0.5 * (H + H.T)
            return H + mean_hess(u)
        R['hessian'] = hessian; R['mean_hess'] = mean_hess
        return R

    def newton(R, u, H, ab, lt, maxit, refresh_every=0):
        vg, val = R['vg'], R['val']
        (g0, aux), g = vg(u, ab, lt); it = 0; decr = np.inf
        for it in range(1, maxit + 1):
            if refresh_every and it % refresh_every == 0:
                H = R['hessian'](u, ab, lt)
            gn = np.asarray(g, dtype=np.float64)
            w, V = np.linalg.eigh(H)
            wabs = np.maximum(np.abs(w), 1e-6 * np.abs(w).max())
            step = -(V @ ((V.T @ gn) / wabs))
            decr = float(-gn @ step)
            if decr < args.tol:
                break
            s = 1.0; ok = False
            for _ in range(10):
                un = u + jnp.asarray(s * step, dtype=jnp.float32)
                Gn, _a = val(un, ab, lt)
                if np.isfinite(float(Gn)) and float(Gn) <= float(g0) - 1e-4 * s * decr:
                    ok = True; break
                s *= 0.5
            if not ok:
                H = R['hessian'](u, ab, lt)
                continue
            u = un
            (g0, aux), g = vg(u, ab, lt)
        return u, H, float(g0), aux, decr, it

    u_start = jnp.asarray(u_true); first_rung = True; dense_err = np.nan
    for (rname, prior_kind, mean_w) in RUNGS:
        tr = time.time()
        R = make_rung(prior_kind, mean_w)
        ab_s, lt_s = theta(*START)
        u = u_start
        H = R['hessian'](u, ab_s, lt_s)
        print(f'[BATCH {i}] rung {rname} ({prior_kind}, mean_w {mean_w}) units={N} (median {np.median(l_np):.2f} cm) '
              f'colors={ncol} usize={usize} roi={roi} first Hessian {time.time()-tr:.0f}s', flush=True)
        if args.dense_check and first_rung:
            t1 = time.time(); Hd = R['hessian'](u, ab_s, lt_s, dense=True)
            dense_err = float(np.abs(H - Hd).max() / np.abs(np.diag(Hd)).max())
            leak = float(np.abs((Hd - R['mean_hess'](u))[~A]).max() / np.abs(np.diag(Hd)).max())
            print(f'[DENSE] batch {i}: color-vs-dense {dense_err:.2e}; leak {leak:.2e}; {time.time()-t1:.0f}s', flush=True)
        out = {k: np.full(shape, np.nan) for k in REC_KEYS}
        umap = np.full(shape + (N,), np.nan, dtype=np.float32)
        solved = {}; since = 0; prev = None
        for n_pt, (il, ia) in enumerate(ORDER):
            ab, lt = theta(il, ia); tp = time.time()
            if solved:
                src = min(solved, key=lambda q: (abs(q[0] - il) + abs(q[1] - ia), q))
                u = jnp.asarray(solved[src])
                if src != prev:
                    H = R['hessian'](u, ab, lt); since = 0
            cold = (n_pt == 0)
            u, H, Gv, aux, decr, it = newton(R, u, H, ab, lt, args.maxit_cold if cold else args.maxit,
                                             refresh_every=10 if cold else 0)
            since += 1
            fresh = cold or since >= args.hess_every or decr > args.tol
            if fresh:
                H = R['hessian'](u, ab, lt); since = 0
                w_ = np.linalg.eigvalsh(H)
                gn = np.asarray(R['vg'](u, ab, lt)[1], dtype=np.float64)
                wa = np.maximum(np.abs(w_), 1e-12); _, V = np.linalg.eigh(H)
                decr = float(gn @ (V @ ((V.T @ gn) / wa)))
                out['logdet'][il, ia] = float(np.sum(np.log(wa)))
                out['n_negeig'][il, ia] = int((w_ <= 0).sum()); out['mineig'][il, ia] = float(w_.min())
            nll, pr, mp, m = [float(x_) for x_ in aux]
            out['G'][il, ia] = Gv; out['nll'][il, ia] = nll; out['prior'][il, ia] = pr
            out['mean_pen'][il, ia] = mp; out['mean_dedx'][il, ia] = m
            out['decr'][il, ia] = decr; out['niter'][il, ia] = it; out['fresh'][il, ia] = float(fresh)
            out['nll_ref'][il, ia] = float(nll_ref_fn(jnp.asarray(u_true), ab, lt)) if first_rung else np.nan
            ue = np.asarray(u, dtype=np.float64)
            out['scale_err'][il, ia] = float(np.sum(l_np * np.exp(ue)) / np.sum(l_np * d_true) - 1)
            umap[il, ia] = np.asarray(u); solved[(il, ia)] = np.asarray(u); prev = (il, ia)
            print(f'  [{rname} PT lt x{lt_grid[il]:.3f} dAb {dab_grid[ia]:+.4f}] G {Gv:.2f} nll {nll:.2f} '
                  f'mean {m:.4f} scale {100*out["scale_err"][il, ia]:+.3f}% decr {decr:.1e} it {it} '
                  f'{"F" if fresh else "-"} {time.time()-tp:.0f}s', flush=True)
        for k in REC_KEYS: rec[f'{rname}_{k}'].append(out[k])
        rec[f'{rname}_nfall'].append([R['NF']['jit'], R['NF']['eager'], R['NF']['fd']])
        rec[f'{rname}_t_rung'].append(time.time() - tr)
        extra[f'{rname}_u_map_{i}'] = umap
        u_start = jnp.asarray(solved[START]); first_rung = False
        print(f'[BATCH {i}] rung {rname} done {time.time()-tr:.0f}s  hvp jit/eager/fd {R["NF"]}', flush=True)
    rec['batch'].append(i); rec['nunit'].append(N); rec['ncolor'].append(ncol); rec['dense_err'].append(dense_err)
    rec['t_batch'].append(time.time() - tb)
    extra[f'u_true_{i}'] = u_true; extra[f'l_{i}'] = l_np; extra[f'z_{i}'] = z_mid
    print(f'[BATCH {i}] done {time.time()-tb:.0f}s', flush=True)
    # release this batch's compiled programs: every closure bakes the batch data into its own XLA
    # executables, and keeping them accumulated ~64 GB of host RAM over ~5 batches (OOM, 24 Sep)
    del R, nll_ref_fn, nll_of, tracks, tg, adcs, x, y, z, ticks, hp, evh, pix, H
    jax.clear_caches(); gc.collect()
    _rss = [l for l in open('/proc/self/status') if l.startswith('VmRSS')][0].split()[1]
    print(f'[MEM] after batch {i}: host RSS {int(_rss)/1e6:.2f} GB', flush=True)
    np.savez(args.out, lt_grid=lt_grid, dab_grid=dab_grid, valley=args.valley, config=json.dumps(vars(args)),
             rungs=json.dumps(RUNGS), truth=json.dumps(TRUTH), **{k: np.array(v) for k, v in rec.items()}, **extra)
print(f'[DONE] batches {rec["batch"]} total {time.time()-t_start:.1f}s', flush=True)
