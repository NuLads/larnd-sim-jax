"""
Cheap test of the Laplace (dE/dx-marginalised) calibration objective.

Question: the production lifetime bias is attributed to MAP-profiling ~4000 per-segment dE/dx
nuisances (Neyman-Scott). If so, adding the Laplace term 1/2 log det H_dEdx to the profiled
objective should pull the (Ab, lifetime) minimum back towards the truth-dE/dx reference.

Setup (everything except dE/dx at truth, so representation effects are excluded by construction):
  * target  = production stochastic noise-ON simulation of the TRUE target-file segments
              (rngkey i+1, bit-identical to production), seed-0 truth parameters;
  * model   = production probabilistic simulation of the SAME segments, same geometry;
  * nuisance: u_s = log dE/dx of every target-file segment s (one per segment, ~120/batch);
  * prior   : split-t per segment, parameters from the length-binned population fit
              (prior_bins.npz; one count per segment -> a proper Bayesian prior), plus the
              log-Jacobian -sum(u) so the Laplace integral is over linear dE/dx;
  * no mean constraint, no barrier.
For each grid point theta = (Ab, lifetime), per batch:
  REF : NLL(theta, true dE/dx)                           (reference, no nuisance)
  MAP : G(theta) = min_u [NLL + prior - sum u]           (profiled, = production-style)
  LAP : G(theta) + 1/2 log det H_uu(theta, u*)            (Laplace-marginalised)
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
p.add_argument('--max_nbatch', type=int, default=100)
p.add_argument('--max_batch_len', type=float, default=400.)
p.add_argument('--lt_grid', default='0.96,1.0,1.04,1.08,1.12')
p.add_argument('--dab_grid', default='-0.0015,0.0,0.0015',
               help='Ab offsets RELATIVE to the valley line Ab = Ab_t*(1 - (lt-1)/VALLEY)')
p.add_argument('--valley', type=float, default=20.3)
p.add_argument('--dense_check', type=int, default=0, help='1: also build the dense Hessian and compare')
p.add_argument('--hess_mode', default='color', choices=['color', 'dense'])
p.add_argument('--rt', type=float, default=4.5, help='transverse coupling radius [cm] for the colouring')
p.add_argument('--maxit', type=int, default=15)
p.add_argument('--tol', type=float, default=5e-3, help='Newton decrement tolerance [nats]')
p.add_argument('--use_jit', type=int, default=1)
p.add_argument('--debug', type=int, default=0)
p.add_argument('--init_glob', default='', help='warm-start u from a previous run: u_map at the same (lt, dab) as GRID[0]')
args = p.parse_args()

INPUT_SIM = '/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5'
INPUT_TGT = '/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5'
TRUTH = dict(Ab=0.83488135, eField=0.52151894, tran_diff=1.00276338e-05,
             long_diff=5.81418228e-06, lifetime=2406.44659703)
PARAMS = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
HERE = os.path.dirname(os.path.abspath(__file__))

t_start = time.time()
ds_sim = TracksDataset(filename=INPUT_SIM, nevents=600, max_nbatch=args.max_nbatch, random_nevents=True,
                       data_seed=1, track_len_sel=2., max_abs_costheta_sel=0.966, min_abs_segz_sel=15.,
                       track_z_bound=28., max_batch_len=args.max_batch_len, print_input=False, chopped=True,
                       pad=True, electron_sampling_resolution=0.01, live_selection=False,
                       use_dedx_density=False, dedx_density_mode='histogram')
ds_tgt = TgtTracksDataset(filename=INPUT_TGT, dataset_sim=ds_sim, chopped=True, pad=True,
                          electron_sampling_resolution=0.01, print_input=False)
fields = ds_tgt.get_track_fields()
F = {k: fields.index(k) for k in ['dEdx', 'dE', 'dx', 'eventID', 'pixel_plane', 'x_start', 'y_start',
                                   'z_start', 'x_end', 'y_end', 'z_end']}
print(f'[DATA] {len(ds_tgt)} batches, load {time.time()-t_start:.1f}s', flush=True)

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
PEDGES, PPAR = pb['edges'], pb['params']       # (nbin, 5): nuL nuR loc sL sR

lt_grid = [float(x) for x in args.lt_grid.split(',')]
dab_grid = [float(x) for x in args.dab_grid.split(',')]
# visit order: serpentine so each point warm-starts from a neighbour
GRID = []
for il, fl in enumerate(lt_grid):
    row = [(il, ia) for ia in range(len(dab_grid))]
    GRID += row if il % 2 == 0 else row[::-1]

def theta(il, ia):
    fl = lt_grid[il]
    ab = TRUTH['Ab'] * (1.0 - (fl - 1.0) / args.valley + dab_grid[ia])
    return ab, TRUTH['lifetime'] * fl


def seg_table(i):
    rows = ds_tgt.tracks_struct[ds_tgt.batch_row_indices[i]]
    arr = np_from_structured(rows)
    arr = remap_event_ids_to_local(arr, rows['eventID'], ds_tgt.batch_event_global_ids[i], fields)
    chopped, pid = chop_tracks(arr, fields, precision=0.01)
    return arr, chopped.shape[0], pid


def coupling_colors(seg, rt):
    """Greedy colouring of the distance-2 graph of A_ij = [same event & plane & transverse(x,y)
    distance between the two segments < rt]. Columns sharing a colour have disjoint row supports,
    so one HVP per colour recovers them all exactly IF the coupling assumption holds."""
    N = seg.shape[0]
    pts, owner = [], []
    for s in range(N):
        a = seg[s, [F['x_start'], F['y_start']]]; b = seg[s, [F['x_end'], F['y_end']]]
        n = int(np.ceil(np.linalg.norm(b - a) / 0.25)) + 1
        pts.append(a[None] + np.linspace(0, 1, n)[:, None] * (b - a)[None]); owner.append(np.full(n, s))
    pts = np.concatenate(pts); owner = np.concatenate(owner)
    from scipy.spatial import cKDTree
    A = np.eye(N, dtype=bool)
    tree = cKDTree(pts)
    for (u, v) in tree.query_pairs(rt):
        su, sv = owner[u], owner[v]
        if su != sv:
            A[su, sv] = A[sv, su] = True
    ev = seg[:, F['eventID']]; pl = seg[:, F['pixel_plane']]
    A &= (ev[:, None] == ev[None, :]) & (pl[:, None] == pl[None, :])
    A2 = (A.astype(np.int32) @ A.astype(np.int32)) > 0
    col = -np.ones(N, dtype=int)
    for s in np.argsort(-A2.sum(1)):
        used = set(col[A2[s]]); c = 0
        while c in used: c += 1
        col[s] = c
    return A, col



def smooth_split_t_logkernel(x, pp, wfrac=0.3):
    """Split-t log-kernel with the left/right (nu, scale) blended by a logistic of width
    w = wfrac*min(sL, sR) around loc. The plain split-t is only C1 at loc: its curvature jumps from
    (nu_L+1)/(nu_L sL^2) to (nu_R+1)/(nu_R sR^2), which makes log det H step discontinuously whenever
    a weakly-constrained segment's MAP crosses the mode (seen in batch 60 of the first launch).
    Normalisation is dropped: it does not depend on x, so it cancels in every comparison."""
    nuL, nuR, loc, sL, sR = pp
    w = wfrac * jnp.minimum(sL, sR)
    t = jax.nn.sigmoid((x - loc) / w)
    s = sL + (sR - sL) * t
    nu = nuL + (nuR - nuL) * t
    z = (x - loc) / s
    return -0.5 * (nu + 1.0) * jnp.log1p(z * z / nu)

rec = {k: [] for k in ['batch', 'nseg', 'ncolor', 'G_map', 'nll_map', 'logdet', 'nll_ref', 'decr',
                       'n_negeig', 'mineig', 'niter', 'u_map', 'u_true', 'seg_dx', 'seg_z', 'dense_err',
                       't_batch']}
for i in range(args.b0, min(args.b1, len(ds_tgt))):
    tb = time.time()
    seg, nreal, pid = seg_table(i)
    tracks_np = np.asarray(ds_tgt[i]).reshape(-1, len(fields))
    assert nreal <= tracks_np.shape[0]
    pid_full = -np.ones(tracks_np.shape[0], dtype=np.int32); pid_full[:nreal] = pid
    assert np.allclose(tracks_np[:nreal, F['dEdx']], seg[pid, F['dEdx']], rtol=1e-5), 'segment<->row mismatch'
    N = seg.shape[0]
    seg_dx = seg[:, F['dx']].astype(np.float64)
    u_true = np.log(np.maximum(seg[:, F['dEdx']], 1e-3)).astype(np.float32)
    bi = np.clip(np.searchsorted(PEDGES, seg_dx, side='right') - 1, 0, len(PPAR) - 1)
    pp = jnp.asarray(PPAR[bi].T, dtype=jnp.float32)          # (5, N)

    tracks = jnp.asarray(tracks_np)
    valid = jnp.asarray(pid_full >= 0)
    safe = jnp.asarray(np.clip(pid_full, 0, N - 1))
    wfs, up = simulate_wfs(truth, response, tracks, fields)
    adcs, x, y, z, ticks, hp, ev, pix = simulate_stochastic(truth, wfs, up, rngseed=i + 1)
    tg = dict(adcs=adcs, pixel_x=x, pixel_y=y, pixel_z=z, ticks=ticks, hit_prob=hp, event=ev, pixel_id=pix)
    usize = int(np.asarray(up).shape[0])
    _ = sim.predict(truth, tracks, fields, None)            # populate ROI cache
    roi = (sim.padded_small_nb, sim.padded_large_nb)
    del wfs

    def nll_of(u, ab, lt):
        d = jnp.exp(u)
        ds = jnp.where(valid, d[safe], tracks[:, F['dEdx']])
        t = tracks.at[:, F['dEdx']].set(ds).at[:, F['dE']].set(ds * tracks[:, F['dx']])
        par = truth.replace(Ab=ab, lifetime=lt)
        pred = sim.predict(par, t, fields, None, unique_size=usize, roi_override=roi)
        v, _ = loss.compute(par, pred, tg)
        return v

    def G(u, ab, lt):
        nll = nll_of(u, ab, lt)
        d = jnp.exp(u)
        prior = -jnp.sum(smooth_split_t_logkernel(d, pp))
        return nll + prior - jnp.sum(u), nll

    jit = jax.jit if args.use_jit else (lambda f: f)
    vg = jit(jax.value_and_grad(G, has_aux=True))
    val = jit(G)
    _hvp_raw = lambda u, v, ab, lt: jax.jvp(lambda w: jax.grad(lambda z: G(z, ab, lt)[0])(w), (u,), (v,))[1]
    _hvp_jit = jit(_hvp_raw)
    NFALL = [0, 0]
    def hvp(u, v, ab, lt):
        # the jit-compiled forward-over-reverse pass returns NaN on this model while the eager pass is
        # finite (debug run 38909357); fall back to eager whenever the compiled result is non-finite
        NFALL[1] += 1
        r = _hvp_jit(u, v, ab, lt)
        if not bool(jnp.all(jnp.isfinite(r))):
            NFALL[0] += 1
            r = _hvp_raw(u, v, ab, lt)
        if not bool(jnp.all(jnp.isfinite(r))):
            # second fallback: central difference of the (finite) jitted gradient; eps in log dE/dx
            NFALL.append(1)
            eps = 1e-3
            gp = vg(u + eps * v, ab, lt)[1]; gm = vg(u - eps * v, ab, lt)[1]
            r = (gp - gm) / (2 * eps)
        if not bool(jnp.all(jnp.isfinite(r))):
            raise FloatingPointError('HVP non-finite after jit, eager and finite-difference attempts')
        return r
    nll_ref_fn = jit(nll_of)

    if args.debug:
        uu = jnp.asarray(u_true); ab0, lt0 = theta(*GRID[0])
        v0 = jnp.asarray(np.eye(N, dtype=np.float32)[0])
        prior_only = lambda z: -jnp.sum(smooth_split_t_logkernel(jnp.exp(z), pp)) - jnp.sum(z)
        nll_only = lambda z: nll_of(z, ab0, lt0)
        fin = lambda a: (bool(np.all(np.isfinite(np.asarray(a)))), int(np.sum(~np.isfinite(np.asarray(a)))))
        print('[DBG] prior val', float(prior_only(uu)), 'grad', fin(jax.grad(prior_only)(uu)),
              'hvp', fin(jax.jvp(jax.grad(prior_only), (uu,), (v0,))[1]), flush=True)
        print('[DBG] pp finite', fin(pp), 'logpdf', fin(split_t_logpdf_jax(jnp.exp(uu), pp[0], pp[1], pp[2], pp[3], pp[4])), flush=True)
        print('[DBG] nll val', float(nll_only(uu)), flush=True)
        gq = jax.grad(nll_only)(uu); print('[DBG] nll grad', fin(gq), float(jnp.nanmax(jnp.abs(gq))), flush=True)
        for k in [0, N//2, N-1]:
            vk = jnp.asarray(np.eye(N, dtype=np.float32)[k])
            hq = jax.jvp(jax.grad(nll_only), (uu,), (vk,))[1]
            print('[DBG] nll hvp col', k, fin(hq), flush=True)
        # float-precision probe: same HVP by central finite difference of the gradient
        eps = 1e-3
        fd = (jax.grad(nll_only)(uu + eps * v0) - jax.grad(nll_only)(uu - eps * v0)) / (2 * eps)
        print('[DBG] fd hvp col0 finite', fin(fd), 'diag', float(fd[0]), flush=True)
        sys.exit(0)
    A, col = coupling_colors(seg, args.rt)
    ncol = int(col.max()) + 1

    def hessian(u, ab, lt, mode):
        if mode == 'dense':
            H = np.stack([np.asarray(hvp(u, jnp.asarray(np.eye(N, dtype=np.float32)[k]), ab, lt)) for k in range(N)], 1)
        else:
            H = np.zeros((N, N))
            for c in range(ncol):
                v = (col == c).astype(np.float32)
                hv = np.asarray(hvp(u, jnp.asarray(v), ab, lt), dtype=np.float64)
                for j in np.where(col == c)[0]:
                    rows = np.where(A[:, j])[0]
                    H[rows, j] = hv[rows]
        H = 0.5 * (H + H.T)
        return H.astype(np.float64)

    t0 = time.time()
    ab0, lt0 = theta(*GRID[0])
    u = jnp.asarray(u_true)
    if args.init_glob:
        import glob as _g
        for _f in sorted(_g.glob(args.init_glob)):
            _z = np.load(_f, allow_pickle=True)
            if f'u_map_{i}' in _z.files:
                _lt0, _da0 = lt_grid[GRID[0][0]], dab_grid[GRID[0][1]]
                _il = int(np.argmin(np.abs(_z['lt_grid'] - _lt0))); _ia = int(np.argmin(np.abs(_z['dab_grid'] - _da0)))
                assert abs(_z['lt_grid'][_il] - _lt0) < 1e-9 and abs(_z['dab_grid'][_ia] - _da0) < 1e-9
                u = jnp.asarray(_z[f'u_map_{i}'][_il, _ia]); print(f'[INIT] batch {i} warm-start from {_f} at lt {_lt0} dab {_da0}', flush=True)
                break
    H = hessian(u, ab0, lt0, args.hess_mode)
    t_h = time.time() - t0
    dense_err = np.nan
    if args.dense_check:
        t1 = time.time()
        Hd = hessian(u, ab0, lt0, 'dense')
        off = ~A
        dense_err = float(np.abs(H - Hd).max() / np.abs(np.diag(Hd)).max())
        leak = float(np.abs(Hd[off]).max() / np.abs(np.diag(Hd)).max()) if off.any() else 0.
        print(f'[DENSE] batch {i}: N={N} colors={ncol} color-vs-dense max rel err {dense_err:.2e}; '
              f'max |H| outside assumed support / max diag {leak:.2e}; dense {time.time()-t1:.0f}s', flush=True)
        H = Hd if args.hess_mode == 'dense' else H
    print(f'[BATCH {i}] N={N} colors={ncol} usize={usize} roi={roi} first Hessian {t_h:.0f}s  hvp eager-fallbacks {NFALL[0]}/{NFALL[1]} fd-fallbacks {len(NFALL)-2}', flush=True)

    out = {k: np.full((len(lt_grid), len(dab_grid)), np.nan) for k in
           ['G_map', 'nll_map', 'logdet', 'nll_ref', 'decr', 'n_negeig', 'mineig', 'niter']}
    umap = np.zeros((len(lt_grid), len(dab_grid), N), dtype=np.float32)
    for (il, ia) in GRID:
        ab, lt = theta(il, ia)
        tp = time.time()
        (g0, nll), g = vg(u, ab, lt)
        it = 0
        _maxit = args.maxit * (3 if (il, ia) == GRID[0] else 1)
        for it in range(1, _maxit + 1):
            gn = np.asarray(g, dtype=np.float64)
            lam = 1e-6 * np.abs(np.diag(H)).max()
            w, V = np.linalg.eigh(H)
            wabs = np.maximum(np.abs(w), lam)
            step = -(V @ ((V.T @ gn) / wabs))
            decr = float(-gn @ step)
            if decr < args.tol:
                break
            s = 1.0; accepted = False
            for _ in range(8):
                un = u + jnp.asarray(s * step, dtype=jnp.float32)
                Gn, nlln = val(un, ab, lt)
                if float(Gn) <= float(g0) - 1e-4 * s * decr:
                    accepted = True; break
                s *= 0.5
            if not accepted:
                break
            u = un
            (g0, nll), g = vg(u, ab, lt)
        H = hessian(u, ab, lt, args.hess_mode)             # fresh at the optimum: logdet + next warm start
        w = np.linalg.eigvalsh(H)
        gn = np.asarray(g, dtype=np.float64)
        wabs = np.maximum(np.abs(w), 1e-12)
        _, V = np.linalg.eigh(H)
        decr = float(gn @ (V @ ((V.T @ gn) / wabs)))
        out['G_map'][il, ia] = float(g0); out['nll_map'][il, ia] = float(nll)
        out['logdet'][il, ia] = float(np.sum(np.log(np.abs(w))))
        out['n_negeig'][il, ia] = int((w <= 0).sum()); out['mineig'][il, ia] = float(w.min())
        out['decr'][il, ia] = decr; out['niter'][il, ia] = it
        out['nll_ref'][il, ia] = float(nll_ref_fn(jnp.asarray(u_true), ab, lt))
        umap[il, ia] = np.asarray(u)
        print(f'  [PT lt x{lt_grid[il]:.3f} dAb {dab_grid[ia]:+.4f}] G {float(g0):.2f} nll {float(nll):.2f} '
              f'ref {out["nll_ref"][il, ia]:.2f} logdet {out["logdet"][il, ia]:.2f} decr {decr:.1e} '
              f'it {it} neg {int((w<=0).sum())} mineig {w.min():.2e} {time.time()-tp:.0f}s', flush=True)
    rec['batch'].append(i); rec['nseg'].append(N); rec['ncolor'].append(ncol)
    for k in out: rec[k].append(out[k])
    rec['u_map'].append(umap); rec['u_true'].append(u_true); rec['seg_dx'].append(seg_dx)
    rec['seg_z'].append(seg[:, [F['z_start'], F['z_end']]]); rec['dense_err'].append(dense_err)
    rec['t_batch'].append(time.time() - tb)
    print(f'[BATCH {i}] done {time.time()-tb:.0f}s', flush=True)
    np.savez(args.out, lt_grid=np.array(lt_grid), dab_grid=np.array(dab_grid), valley=args.valley,
             truth=json.dumps(TRUTH),
             **{k: np.array(v) for k, v in rec.items() if k not in ('u_map', 'u_true', 'seg_dx', 'seg_z')},
             **{f'{k}_{b}': v for k in ('u_map', 'u_true', 'seg_dx', 'seg_z') for b, v in zip(rec['batch'], rec[k])})
print(f'[DONE] batches {rec["batch"]} total {time.time()-t_start:.1f}s', flush=True)
