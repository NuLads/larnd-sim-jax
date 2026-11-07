"""Paper figure: how track trajectories are INITIALISED in the joint fit (subsec:muon_reco_input).

This is the figure for the "starting shape from the hits" + "straight tracks with a fixed
drift coordinate" procedure: deriving each track's starting transverse shape from its own
target hits, and freezing the drift-carrying coordinate for the ~15% of tracks whose hits
are consistent with a straight line. The paper's claim is that this -- rather than starting
every track straight and fitting it freely -- is what makes the joint calibration fit land
on the right answer instead of a biased one.

Three panels:
  (a) Worked example along the drift-carrying transverse direction e2 vs arc length, for
      track (event 2, track 4) of batch 35, data sample 8 (sci_full_STRFRZ508, the recipe
      "R2" fit): the per-pixel charge-weighted hit points (with single-hit scatter behind
      them), the straight nominal axis, the hit-derived starting shape, the true trajectory
      and the final fitted trajectory. Inset: a nearly-straight track from the same batch
      (event 2, track 0) that the hits classify as straight, so its drift coordinate is held
      fixed at the straight line for the whole fit.
  (b) Starting-shape quality over the nine recipe track samples (sci_full_STRFRZ50{1..9}):
      median distance to the true shape for the straight line vs. the hit-derived start, for
      the drift-carrying direction e2 and the pixel-plane direction e1, split into
      near-straight (<250 um true deviation) and curved (>500 um) tracks.
  (c) Final per-track drift-shape error (e2, end of a 10 000-iteration fit), complementary
      CDF on log-log axes: the naive free fit from a straight start (sci_full_DSREP{1..9})
      vs. the recipe (sci_full_STRFRZ50{1..9}), with the free fit from a hit-seeded start and
      no freezing (sci_full_HITSEED{1,2,4}) as an intermediate curve.

ALGORITHM FIDELITY. The hit-derived starting shape is NOT reimplemented here: this script
calls optimize.fit_params.GradientDescentFitter._seed_drift_from_hits itself (the exact
production method), on a lightweight stand-in object carrying a FRESH (zero) chain state --
the same pattern used by the offline tools in review_2026_09/DRIFTHITS/ (seed_eval.py,
offline_test.py, straight_select.py, test_straight_freeze.py). The relevant LARND_* env vars
(frame convention, spline K floor, freeze threshold, hit-seeding knobs) are read from the
STRFRZ508 checkpoint's own 'env_overrides' and set before optimize.fit_params is imported,
so the module-level constants that _seed_drift_from_hits reads match what produced the
checkpoint. Cross-checked against the production log: for batch 35, this script reproduces
line-for-line the "[GEOM-HITS] batch 35: 9 drift-frame tracks (skipped 0); seeding; called
straight (frozen) 2; e2 seeded on 5, e1 seeded on 4; shape error vs truth median e2 306 um
(before 294), e1 164 um (before 164)" and "[CHAIN] batch 35: drift-direction geometry FROZEN
on 2/9 tracks" lines of logs/sci_case/job-40145691_0.out (identified as the STRFRZ508 run by
its finish time matching the checkpoint's mtime to the second).

The final-fit shape error (panel c, and the "fitted" curve of panel a) needs no re-seeding at
all: it is read directly from each finished checkpoint's own 'chain_cache' (the fitted spline
coefficients) and 'true_positions' (the simulation truth kept at fit time), exactly as
review_2026_09/DRIFTHITS/shape_err.py does.

CPU only; JAX_PLATFORMS is forced to 'cpu' before jax/optax/fit_params are imported (fit_params
imports jax at module scope). No SLURM, no GPU. Caches the (small) derived arrays/numbers under
CACHE_DIR; pass --force to recompute from the checkpoints and target files.

Usage:
    python fig_initialisation.py [--force]
Writes:
    /sdf/group/neutrino/pgranger/NuLads_draft/figures/initialisation.pdf
    /sdf/group/neutrino/pgranger/NuLads_draft/figures/initialisation.png
and prints the numbers used in the caption to stdout.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # must be set before fit_params pulls in jax
import sys
import glob
import json
import pickle
import types
import argparse

import numpy as np

REPO = '/sdf/group/neutrino/pgranger/larnd-sim-jax'
sys.path.insert(0, REPO)

CACHE_DIR = '/sdf/scratch/neutrino/pgranger/review_2026_09/paperfig_cache'
N_INIT = int(os.environ.get('NINIT', '16'))  # recipe/naive track samples 1..N (9 for the 8 Oct version)
# recipe arm and its iteration: STRFRZ50 at 1e4 (K = 3, the paper until 18 Oct) or KN10C at 1.4e4 (10 cm knots)
ARM = os.environ.get('INIT_ARM', 'STRFRZ50')
IT = int(os.environ.get('INIT_ITER', '10000'))
_SFX = ('' if N_INIT == 9 else f'_n{N_INIT}') + ('' if ARM == 'STRFRZ50' else f'_{ARM}{IT}')
CACHE_NPZ = f'{CACHE_DIR}/fig_initialisation_arrays{_SFX}.npz'
CACHE_JSON = f'{CACHE_DIR}/fig_initialisation_numbers{_SFX}.json'
OUT_DIR = os.environ.get('INIT_OUT_DIR', '/sdf/group/neutrino/pgranger/NuLads_draft/figures')

FR = f'{REPO}/fit_result'
TEMPLATE_CKPT = (f'{FR}/sci_full_{ARM}8/'
                 f'history_iter{IT}_sci_full_b100_len400_mw2500.0_refit_seed0.pkl')

# ----------------------------------------------------------------------------------------
# Set the LARND_* env vars the checkpoint was produced with, BEFORE importing fit_params
# (every one of these is read once, as a module-level constant, at import time).
# ----------------------------------------------------------------------------------------
with open(TEMPLATE_CKPT, 'rb') as _f:
    _TEMPLATE_ENV = pickle.load(_f)['env_overrides']
for _k in ('LARND_CHAIN_FRAME', 'LARND_SPLINE_KMIN', 'LARND_FREEZE_STRAIGHT_CL',
           'LARND_DRIFT_FROM_HITS', 'LARND_CHAIN_FREEZE_DRIFT', 'LARND_HITS_SEED_E1',
           'LARND_HITS_SEED_E2_CL', 'LARND_HITS_SEED_E1_AMP_UM', 'LARND_DRIFT_FROM_HITS_K'):
    _v = _TEMPLATE_ENV.get(_k)
    if _v not in (None, ''):
        os.environ[_k] = str(_v)

import logging
logging.disable(logging.INFO)  # silence the (very verbose, one line/batch) [GEOM-HITS]/[CHAIN] logging
import optax  # noqa: E402
from optimize import fit_params as fp  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------------------------
# Shared paper style (Okabe-Ito palette; see review_2026_09/scripts/paperfigs/fig_performance.py)
# ----------------------------------------------------------------------------------------
C_TRUE = '#000000'
C_FIT = '#D55E00'
C_HITSTART = '#009E73'
C_STRAIGHT = '#999999'
C_HITS = '#0072B2'

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
})

AG = np.linspace(0.0, 1.0, 101)  # normalised arc-length grid used for all RMS-shape metrics
WORKED_SAMPLE, WORKED_BATCH = 8, 35
WORKED_CURVED_TID = (2.0, 4.0)     # event 2, track 4 -- the paper's worked example
WORKED_STRAIGHT_TID = (2.0, 0.0)   # event 2, track 0 -- a frozen (straight-classified) track, same batch

RECIPE_SAMPLES = list(range(1, N_INIT + 1))  # sci_full_STRFRZ50{1..N}: all samples the paper validates
NAIVE_SAMPLES = list(range(1, N_INIT + 1))   # sci_full_DSREP{1..N}: same samples, free fit from a straight start
HITSEED_SAMPLES = [1, 2, 4]             # sci_full_HITSEED{1,2,4}: only arm available, free fit, hit-seeded start


# ==========================================================================================
# Low-level helpers (shared by every panel)
# ==========================================================================================
def ckpt_path(run):
    cands = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter*.pkl'),
                    key=lambda x: int(x.split('iter')[-1].split('_')[0]))
    it_max = IT if run.startswith(ARM) else 10000
    hits = [c for c in cands if int(c.split('iter')[-1].split('_')[0]) <= it_max]
    return (hits or cands)[-1]


def load_ckpt(run):
    with open(ckpt_path(run), 'rb') as f:
        return pickle.load(f)


def target_npz(sample, batch, run=ARM):
    d = (f'{REPO}/target_sci_full_{run}{sample}_sci_full_b100_len400_mw2500.0_refit_seed0_'
         'Ab_eField_tran_diff_long_diff_lifetime')
    return np.load(f'{d}/batch{batch}_target.npz')


def parse_true_positions(tp_raw):
    """true_positions[batch] is keyed by str(tuple(float,float)) -- parse back to float tuples."""
    out = {}
    for k, v in tp_raw.items():
        if isinstance(k, str):
            k = tuple(float(x) for x in k.strip('()').split(','))
        out[tuple(float(x) for x in k)] = v
    return out


def rms_um(K, coeffs):
    """RMS, over a dense 101-point arc-length grid, of a K-sine-mode shape -- in micrometres."""
    Bg = np.sin(np.outer(AG, np.pi * np.arange(1, K + 1)))
    return float(np.sqrt(np.mean((Bg @ coeffs) ** 2))) * 1e4


def true_shape_coeffs(ctx, tp, K):
    """Least-squares projection of a track's true (fine-segment) positions onto the K-mode
    sine basis, along e1 (pixel plane) and e2 (drift-carrying) -- same recipe as
    DRIFTHITS/shape_err.py and scripts/eventdisplay/event_display.py."""
    u0 = fp._dir_from_angles(ctx.theta0_i, ctx.phi0_i)
    e1, e2 = fp._transverse_frame(u0)
    tr = tp.get(tuple(float(x) for x in ctx.track_id))
    if tr is None:
        return None
    tr = np.asarray(tr, float)
    al = np.asarray(ctx.alpha_mid, float)[:len(tr)]
    tr = tr[:len(al)]
    x0 = np.asarray(ctx.x0_fixed, float)
    L = float(ctx.total_len)
    B = np.sin(np.outer(al, np.pi * np.arange(1, K + 1)))
    d = tr - (x0[None, :] + al[:, None] * L * u0[None, :])
    bt1 = np.linalg.lstsq(B, d @ e1, rcond=None)[0]
    bt2 = np.linalg.lstsq(B, d @ e2, rcond=None)[0]
    return e1, e2, bt1, bt2


def is_drift_frame(e1, u0):
    return abs(float(e1[2])) < 1e-6 and float(np.linalg.norm(np.cross(fp._DRIFT_AXIS, u0))) > 1e-3


def reseed_batch(ck, b, f):
    """Call the PRODUCTION _seed_drift_from_hits on batch b, from a FRESH (zero) chain state,
    using this checkpoint's own contexts/true_positions and the fit's own kept target hits f.
    Returns (stub, ctxs); stub._chain_cache[b] holds the resulting (seeded or still-straight)
    coefficients, stub._straight_freeze[b] the set of track indices frozen at the straight line."""
    ctxs_raw = ck['chain_contexts'][b]
    ctxs = [types.SimpleNamespace(**c) for c in ctxs_raw]
    for c in ctxs:
        c.track_id = tuple(float(x) for x in c.track_id)
    Ks = [len(np.asarray(s['angles'])) for s in ck['chain_cache'][b]]
    tp = parse_true_positions(ck['true_positions'][b])
    stub = types.SimpleNamespace(
        _batch_chain_contexts={b: ctxs},
        _chain_cache={b: [{'angles': np.zeros(k, np.float32)} for k in Ks]},
        _chain_optimizer=optax.adam(1e-3),
        _batch_true_positions={b: tp},
    )
    fp.GradientDescentFitter._seed_drift_from_hits(
        stub, b, f['pixel_x'], f['pixel_y'], f['pixel_z'], f['event'], f['adcs'], f['pixel_id'])
    return stub, ctxs, tp


# ==========================================================================================
# Panel (a): worked example
# ==========================================================================================
def build_panel_a():
    ck = load_ckpt(f'{ARM}{WORKED_SAMPLE}')
    b = WORKED_BATCH
    f = target_npz(WORKED_SAMPLE, b)
    stub, ctxs, tp = reseed_batch(ck, b, f)
    straight_set = stub._straight_freeze[b]

    H = np.stack([f['pixel_x'], f['pixel_y'], f['pixel_z']], 1).astype(float)
    ev = f['event'].astype(float)
    q = f['adcs'].astype(float)
    pid = f['pixel_id']

    tid_by_idx = {ctx.track_id: i for i, ctx in enumerate(ctxs)}
    out = {}
    for tag, tid in (('curved', WORKED_CURVED_TID), ('straight', WORKED_STRAIGHT_TID)):
        ti = tid_by_idx[tid]
        ctx = ctxs[ti]
        u0 = fp._dir_from_angles(ctx.theta0_i, ctx.phi0_i)
        e1, e2 = fp._transverse_frame(u0)
        x0 = np.asarray(ctx.x0_fixed, float)
        L = float(ctx.total_len)
        K = len(np.asarray(ck['chain_cache'][b][ti]['angles'])) // 2

        # raw + per-pixel hits, exactly the _seed_drift_from_hits 3-D gate
        m = ev == float(ctx.track_id[0])
        P, w, I = H[m], q[m], pid[m]
        a = ((P - x0) @ u0) / L
        d = P - (x0[None, :] + a[:, None] * L * u0[None, :])
        sel = (a > 0.01) & (a < 0.99) & (np.linalg.norm(d - (d @ u0)[:, None] * u0[None, :], axis=1) < 1.0)
        a_hit, d_hit = a[sel], d[sel]
        up, inv = np.unique(I[sel], return_inverse=True)
        ws = np.bincount(inv, w[sel])
        Pm = np.stack([np.bincount(inv, w[sel] * P[sel, j]) / ws for j in range(3)], 1)
        am = ((Pm - x0) @ u0) / L
        dm = Pm - (x0[None, :] + am[:, None] * L * u0[None, :])

        # true trajectory: raw fine-segment positions (NOT basis-smoothed), sorted by arc length
        tr = np.asarray(tp[ctx.track_id], float)
        al = np.asarray(ctx.alpha_mid, float)[:len(tr)]
        tr = tr[:len(al)]
        order = np.argsort(al)
        al_s, tr_s = al[order], tr[order]
        nom_s = x0[None, :] + al_s[:, None] * L * u0[None, :]
        true_e2 = (tr_s - nom_s) @ e2
        true_e1 = (tr_s - nom_s) @ e1

        # hit-derived starting shape (fresh re-seed) and final fitted shape (end of the fit),
        # both evaluated on a smooth grid via the K-mode sine basis
        seed_ang = np.asarray(stub._chain_cache[b][ti]['angles'], float)
        fit_ang = np.asarray(ck['chain_cache'][b][ti]['angles'], float)
        Bg = np.sin(np.outer(AG, np.pi * np.arange(1, K + 1)))
        seed_e2_grid = Bg @ seed_ang[K:]
        fit_e2_grid = Bg @ fit_ang[K:]

        # true shape coefficients (for error numbers only)
        res = true_shape_coeffs(ctx, tp, K)
        _, _, bt1, bt2 = res
        err_straight_e2 = rms_um(K, bt2)
        err_hitseed_e2 = rms_um(K, seed_ang[K:] - bt2)
        err_fit_e2 = rms_um(K, fit_ang[K:] - bt2)

        # mean 3-D distance to truth (fit vs straight), matching event_display.py's metric
        nom_mid = x0[None, :] + al_s[:, None] * L * u0[None, :]
        pos_fit = nom_mid + np.outer(np.interp(al_s, AG, fit_e2_grid), e2) + \
            np.outer(np.interp(al_s, AG, Bg @ fit_ang[:K]), e1)
        mean3d_fit_um = float(np.linalg.norm(pos_fit - tr_s, axis=1).mean()) * 1e4
        mean3d_straight_um = float(np.linalg.norm(nom_mid - tr_s, axis=1).mean()) * 1e4

        out[tag] = dict(
            track_id=tid, L=L, K=K, frozen=bool(ti in straight_set),
            s_hit=a_hit * L, e2_hit=d_hit @ e2,
            s_pix=am * L, e2_pix=dm @ e2,
            s_true=al_s * L, e2_true=true_e2, e1_true=true_e1,
            s_grid=AG * L, e2_seed=seed_e2_grid, e2_fit=fit_e2_grid,
            err_straight_e2=err_straight_e2, err_hitseed_e2=err_hitseed_e2, err_fit_e2=err_fit_e2,
            mean3d_fit_um=mean3d_fit_um, mean3d_straight_um=mean3d_straight_um,
            n_hits_gated=int(sel.sum()), n_pix_gated=int(len(up)),
        )
    return out


# ==========================================================================================
# Panel (b): starting-shape quality over the recipe samples
# ==========================================================================================
def build_panel_b(samples=RECIPE_SAMPLES, max_batches=100):
    rows = []  # one dict per drift-frame track
    for s in samples:
        ck = load_ckpt(f'{ARM}{s}')
        for b in range(max_batches):
            try:
                f = target_npz(s, b)
            except FileNotFoundError:
                continue
            stub, ctxs, tp = reseed_batch(ck, b, f)
            for ti, ctx in enumerate(ctxs):
                u0 = fp._dir_from_angles(ctx.theta0_i, ctx.phi0_i)
                e1, e2 = fp._transverse_frame(u0)
                if not is_drift_frame(e1, u0):
                    continue
                seed_ang = np.asarray(stub._chain_cache[b][ti]['angles'], float)
                K = len(seed_ang) // 2
                res = true_shape_coeffs(ctx, tp, K)
                if res is None:
                    continue
                _, _, bt1, bt2 = res
                rows.append(dict(
                    sample=s, batch=b, frozen=int(ti in stub._straight_freeze[b]),
                    e1_straight=rms_um(K, bt1), e2_straight=rms_um(K, bt2),
                    e1_hit=rms_um(K, seed_ang[:K] - bt1), e2_hit=rms_um(K, seed_ang[K:] - bt2),
                    curv=max(rms_um(K, bt1), rms_um(K, bt2)),
                ))
    return rows


# ==========================================================================================
# Panel (c): final per-track drift-shape error, DSREP vs STRFRZ50 (vs HITSEED)
# ==========================================================================================
def final_e2_errors(run, samples, max_batches=100):
    E = []
    for s in samples:
        ck = load_ckpt(f'{run}{s}')
        for b, ctxs_raw in ck['chain_contexts'].items():
            if b not in ck['chain_cache']:
                continue
            tp = parse_true_positions(ck['true_positions'][b])
            for c_raw, st in zip(ctxs_raw, ck['chain_cache'][b]):
                ctx = types.SimpleNamespace(**c_raw)
                ctx.track_id = tuple(float(x) for x in ctx.track_id)
                u0 = fp._dir_from_angles(ctx.theta0_i, ctx.phi0_i)
                e1, e2 = fp._transverse_frame(u0)
                if not is_drift_frame(e1, u0):
                    continue
                ang = np.asarray(st['angles'], float)
                K = len(ang) // 2
                res = true_shape_coeffs(ctx, tp, K)
                if res is None:
                    continue
                _, _, bt1, bt2 = res
                E.append(rms_um(K, ang[K:] - bt2))
    return np.asarray(E)


def build_panel_c():
    # key 'STRFRZ50' = the recipe arm (ARM); the naive and hit-seeded comparison fits exist at K = 3 only
    if ARM != 'STRFRZ50':
        return dict(STRFRZ50=final_e2_errors(ARM, RECIPE_SAMPLES))
    return dict(
        DSREP=final_e2_errors('DSREP', NAIVE_SAMPLES),
        STRFRZ50=final_e2_errors('STRFRZ50', RECIPE_SAMPLES),
        HITSEED=final_e2_errors('HITSEED', HITSEED_SAMPLES),
    )


# ==========================================================================================
# Cache orchestration
# ==========================================================================================
def build_all():
    print('[1/3] panel (a): worked example (sample 8, batch 35) ...', flush=True)
    a = build_panel_a()
    for tag, d in a.items():
        print(f"    {tag} track {d['track_id']}: L={d['L']:.1f} cm, frozen={d['frozen']}, "
              f"straight-line err {d['err_straight_e2']:.0f} um, "
              f"hit-seed err {d['err_hitseed_e2']:.0f} um, fitted err {d['err_fit_e2']:.0f} um "
              f"| mean 3-D dist: fit {d['mean3d_fit_um']:.0f} um, straight {d['mean3d_straight_um']:.0f} um")

    print('[2/3] panel (b): starting-shape quality, samples', RECIPE_SAMPLES, '...', flush=True)
    b_rows = build_panel_b()
    print(f'    {len(b_rows)} drift-frame tracks')

    print('[3/3] panel (c): final shape error, DSREP vs STRFRZ50 vs HITSEED ...', flush=True)
    c = build_panel_c()
    for k, v in c.items():
        print(f'    {k}: {len(v)} tracks, median {np.median(v):.1f} um, mean {v.mean():.1f} um, '
              f'frac>500um {np.mean(v > 500) * 100:.2f}%')

    os.makedirs(CACHE_DIR, exist_ok=True)
    npz_payload = {}
    for tag, d in a.items():
        for k, v in d.items():
            if isinstance(v, (int, float, bool, str, tuple)):
                continue
            npz_payload[f'a_{tag}_{k}'] = np.asarray(v)
    b_keys = ['sample', 'batch', 'frozen', 'e1_straight', 'e2_straight', 'e1_hit', 'e2_hit', 'curv']
    for k in b_keys:
        npz_payload[f'b_{k}'] = np.asarray([r[k] for r in b_rows])
    for k, v in c.items():
        npz_payload[f'c_{k}'] = v
    np.savez_compressed(CACHE_NPZ, **npz_payload)

    numbers = {'a': {tag: {k: v for k, v in d.items() if isinstance(v, (int, float, bool, str))}
                     for tag, d in a.items()},
               'c_samples': {'DSREP': NAIVE_SAMPLES, 'STRFRZ50': RECIPE_SAMPLES, 'HITSEED': HITSEED_SAMPLES},
               'b_samples': RECIPE_SAMPLES}
    with open(CACHE_JSON, 'w') as jf:
        json.dump(numbers, jf, indent=1)
    return npz_payload, numbers, b_rows, c


def load_or_build(force=False):
    if not force and os.path.exists(CACHE_NPZ) and os.path.exists(CACHE_JSON):
        print(f'loading cache {CACHE_NPZ}')
        npz = dict(np.load(CACHE_NPZ, allow_pickle=True))
        with open(CACHE_JSON) as jf:
            numbers = json.load(jf)
        return npz, numbers
    npz, numbers, _, _ = build_all()
    return npz, numbers


# ==========================================================================================
# Figure
# ==========================================================================================
def _panel_label(ax, text):
    ax.text(0.04, 0.95, text, transform=ax.transAxes, fontsize=7, va='top', ha='left', zorder=10)


def plot_panel_a(fig, gs_cell, npz):
    ax = fig.add_subplot(gs_cell)
    d = {k: npz[f'a_curved_{k}'] for k in ('s_hit', 'e2_hit', 's_pix', 'e2_pix', 's_true', 'e2_true',
                                            's_grid', 'e2_seed', 'e2_fit')}
    L = float(npz['a_curved_s_grid'][-1])

    h_hits = ax.scatter(d['s_hit'], d['e2_hit'] * 10, s=3, c=C_HITS, alpha=0.18, lw=0,
                         label='single hits')
    h_pix = ax.scatter(d['s_pix'], d['e2_pix'] * 10, s=7, c=C_HITS, alpha=0.85, lw=0,
                        label='per-pixel mean')
    h_straight, = ax.plot([0, L], [0, 0], color=C_STRAIGHT, ls='--', lw=1.1, label='nominal axis (straight)')
    h_seed, = ax.plot(d['s_grid'], d['e2_seed'] * 10, color=C_HITSTART, ls='-.', lw=1.2,
                       label='hit-derived start')
    h_true, = ax.plot(d['s_true'], d['e2_true'] * 10, color=C_TRUE, lw=1.1, label='true')
    h_fit, = ax.plot(d['s_grid'], d['e2_fit'] * 10, color=C_FIT, lw=1.3, label='fitted')

    ax.set_xlabel('arc length (cm)')
    ax.set_ylabel(r'displacement along $\hat e_2$ (mm)')
    ax.set_xlim(0, L)
    ax.set_ylim(-3.0, 7.3)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.text(0.04, 0.95, '(a)', transform=ax.transAxes, fontsize=7, va='top', ha='left', zorder=10)
    ax.text(0.04, 0.80, f'event 2, track 4\n({L:.0f} cm)', transform=ax.transAxes, fontsize=6, va='top', ha='left',
            color='0.3')

    # ---- inset: a nearly-straight track from the same batch, classified straight (frozen) ----
    # placed bottom-right, a region the main curve (which peaks at s~35cm, y~4.3mm and only
    # approaches y=0 right at the two ends) never enters.
    axi = ax.inset_axes([0.60, 0.05, 0.37, 0.30])
    di = {k: npz[f'a_straight_{k}'] for k in ('s_hit', 'e2_hit', 's_pix', 'e2_pix', 's_true', 'e2_true', 's_grid')}
    Li = float(di['s_grid'][-1])
    axi.scatter(di['s_hit'], di['e2_hit'] * 1e4, s=2, c=C_HITS, alpha=0.15, lw=0)
    axi.scatter(di['s_pix'], di['e2_pix'] * 1e4, s=5, c=C_HITS, alpha=0.8, lw=0)
    axi.plot(di['s_true'], di['e2_true'] * 1e4, color=C_TRUE, lw=0.9)
    axi.axhline(0, color=C_STRAIGHT, ls='--', lw=1.0)
    axi.set_xlim(0, Li)
    axi.set_ylim(-2600, 2600)
    axi.set_ylabel(r'$\hat e_2$ ($\mu$m)', fontsize=5.5, labelpad=1)
    axi.set_xticklabels([])
    axi.tick_params(labelsize=5, length=2, pad=1)
    axi.text(0.97, 0.95, f'track 0 ({Li:.0f} cm)\nclassified straight', transform=axi.transAxes, fontsize=4.8,
             va='top', ha='right', linespacing=1.15)
    axi.grid(alpha=0.2, linewidth=0.3)
    for spine in axi.spines.values():
        spine.set_linewidth(0.4)

    return [h_true, h_fit, h_seed, h_straight, h_pix, h_hits]


def plot_panel_b(fig, gs_cell, npz):
    ax = fig.add_subplot(gs_cell)
    curv = npz['b_curv']
    classes = [('near-straight\ntracks', curv < 250),
               ('curved\ntracks', curv > 500)]
    width = 0.17
    offsets = {'e2_straight': -1.5, 'e2_hit': -0.5, 'e1_straight': 0.5, 'e1_hit': 1.5}
    colors = {'e2_straight': C_STRAIGHT, 'e2_hit': C_HITSTART, 'e1_straight': C_STRAIGHT, 'e1_hit': C_HITSTART}
    hatches = {'e2_straight': None, 'e2_hit': None, 'e1_straight': '////', 'e1_hit': '////'}
    x = np.arange(len(classes))
    for key, off in offsets.items():
        vals = [np.median(npz[f'b_{key}'][sel]) for _, sel in classes]
        ax.bar(x + off * width, vals, width, color=colors[key], hatch=hatches[key],
               edgecolor='k', linewidth=0.3, alpha=0.95 if hatches[key] is None else 0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in classes])
    ax.set_ylabel(r'median shape error vs. truth ($\mu$m)')
    ax.set_ylim(0, 1020)
    ax.grid(alpha=0.25, linewidth=0.5, axis='y')
    _panel_label(ax, '(b)')
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=C_STRAIGHT, edgecolor='k', linewidth=0.3, label='straight line'),
           Patch(facecolor=C_HITSTART, edgecolor='k', linewidth=0.3, label='hit-derived start'),
           Patch(facecolor='0.75', edgecolor='k', linewidth=0.3, hatch='////', label=r'$\hat e_1$ (pixel plane)'),
           Patch(facecolor='0.75', edgecolor='k', linewidth=0.3, label=r'$\hat e_2$ (drift)')]
    ax.legend(handles=leg, frameon=False, fontsize=6, loc='upper right', handlelength=1.4,
              labelspacing=0.3, borderaxespad=0.2)
    return x, classes


def plot_panel_c(fig, gs_cell, npz):
    ax = fig.add_subplot(gs_cell)
    specs = ([('DSREP', C_STRAIGHT, '--', 'naive')] if os.environ.get('INIT_NAIVE', '0') == '1' else []) + [   # 12 Oct: no straight-start fit
             ('HITSEED', C_HITSTART, '-.', 'no held tracks'),
             ('STRFRZ50', C_FIT, '-', 'procedure')]
    for key, col, ls, lab in specs:
        E = np.asarray(npz[f'c_{key}'])
        xs = np.sort(E)
        ccdf = 1.0 - np.arange(len(xs)) / len(xs)
        frac500 = 100.0 * np.mean(E > 500)
        ax.loglog(xs, ccdf, color=col, ls=ls, lw=1.3,
                  label=f'{lab}: {np.median(E):.0f}/{E.mean():.0f} $\\mu$m, {frac500:.1f}%$>$500$\\mu$m')
    ax.axvline(500, color='0.5', lw=0.6, ls=':')
    ax.annotate('500 $\\mu$m', xy=(500, 1.0), xycoords=('data', 'axes fraction'), xytext=(500, 1.015),
                textcoords=('data', 'axes fraction'), fontsize=5.5, color='0.4', ha='center', va='bottom',
                annotation_clip=False)
    ax.set_xlabel(r'final drift-shape error ($\mu$m)')
    ax.set_ylabel('fraction of tracks with larger error')
    ax.set_xlim(5, 2e4)
    ax.set_ylim(2e-4, 1.0)
    ax.grid(alpha=0.25, linewidth=0.5, which='major')
    _panel_label(ax, '(c)')
    ax.legend(frameon=False, fontsize=5.6, loc='lower left', handlelength=1.5, labelspacing=0.3,
              borderaxespad=0.25, title='median/mean error, % $>$500 $\\mu$m:', title_fontsize=5.3,
              alignment='left')


def build_figure(npz):
    # 12 Oct (Pierre): only the worked example (former panel a) is kept in the paper; INIT_PANELS=abc restores the
    # three-panel version
    if os.environ.get('INIT_PANELS', 'a') == 'a':
        fig = plt.figure(figsize=(3.4, 3.1))
        gs = fig.add_gridspec(1, 1, left=0.15, right=0.97, bottom=0.30, top=0.97)
        handles = plot_panel_a(fig, gs[0, 0], npz)
        for t in fig.axes[0].texts:
            if t.get_text() == '(a)':
                t.set_visible(False)
        fig.legend(handles, [h.get_label() for h in handles], loc='lower center', bbox_to_anchor=(0.55, 0.0), ncol=3,
                   frameon=False, fontsize=6.2, handlelength=1.6, columnspacing=1.0, handletextpad=0.4)
        os.makedirs(OUT_DIR, exist_ok=True)
        fig.savefig(f'{OUT_DIR}/initialisation.pdf', bbox_inches='tight')
        fig.savefig(f'{OUT_DIR}/initialisation.png', dpi=300, bbox_inches='tight')
        print(f'saved {OUT_DIR}/initialisation.pdf and .png (worked example only)')
        return
    fig = plt.figure(figsize=(7.0, 2.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.28, 1.0, 1.08], wspace=0.40, left=0.065, right=0.955,
                           bottom=0.185, top=0.96)
    handles = plot_panel_a(fig, gs[0, 0], npz)
    plot_panel_b(fig, gs[0, 1], npz)
    plot_panel_c(fig, gs[0, 2], npz)
    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=6, frameon=False,
               fontsize=6.6, handlelength=1.6, columnspacing=1.1, handletextpad=0.4)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(f'{OUT_DIR}/initialisation.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT_DIR}/initialisation.png', dpi=300, bbox_inches='tight')
    print(f'saved {OUT_DIR}/initialisation.pdf and .png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='ignore the cache, recompute from checkpoints')
    args = ap.parse_args()
    npz, numbers = load_or_build(force=args.force)
    build_figure(npz)
