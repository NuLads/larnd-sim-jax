"""Paper figure for the interpretability \\query{} note (intro of main.tex): "what is the
sensitivity brought by every event in the fit -> can probably be done in terms of Hessians by
looking at the curvature for each event. Can allow for nice plots."

Per-event Hessians are not available: PAPERJOBS/paper_hessian.py (hessian.json) computes the
5x5 Hessian of the production negative-log-likelihood w.r.t. log(Ab, eField, tran_diff,
long_diff, lifetime), at the true parameters/trajectories/energy losses, PER BATCH, and a batch
mixes 1-5 events (median 2) and 1-10 tracks (median 4). This script therefore works at the batch
level throughout and says so: it reconstructs each batch's true track segments exactly
(paper_common.load(200., 400, 1), the same call paper_hessian.py used -- verified below to give
the identical 321 batches and per-batch lengths, to the fourth significant figure, as the ones
stored in hessian.json), computes length-weighted geometric features of each batch's tracks, and
correlates them with each parameter's DIAGONAL information density H_jj/length_cm (the
curvature "with the other four parameters fixed", i.e. the quantity behind the "Others fixed"
column of Table tab:floor). That diagonal is well defined for every batch; the PROFILED
information 1/(H^-1)_jj that Table tab:floor's "Floor" column uses is not: at this batch size
(40-430 cm) the per-batch 5x5 Hessian is not even positive-definite for 24/321 batches, and only
24/321 (7.5%) have a condition number below 1e6 (checked explicitly below). Profiling is only
numerically stable once batches are aggregated into groups of several tens (checked by summing
H over each feature bin before inverting: condition numbers then sit at 1e5-1e7, same regime as
the production-sample floor computation) -- so the figure itself shows the diagonal, and the
text/caption separately reports what profiling an aggregate changes (it damps the lifetime trend
and uncovers an A_b vs. drift-distance trend invisible in the diagonal, from the A_b-lifetime
anti-correlation of Table tab:floor/Sec. subsec:uncertainty).

GEOMETRY CHECK (the task note explicitly flags this as something to verify, and it is NOT the
naive guess): after TracksDataset's swap_xz, the "z" field is the drift coordinate (range
+/-30.43 cm here, matching truth.drift_length=30.431 cm to 3 figures). But the ANODE planes sit
at |z| = drift_length (the two detector edges) and the shared CATHODE sits near z=0 (the middle,
tpc_borders cathode-side edge at |z|=0.159 cm) -- the OPPOSITE of the naive "anode at 0" guess.
Verified two independent ways: (1) truth.tpc_borders (printed below) gives z-borders
[0.159, 30.590] and [-30.590, -0.159] for the two TPCs, i.e. the far edge (|z|~30.6) is the
anode-side border; (2) directly from the simulator's own post-drift sigma_L, sigma_T (which can
only grow with actual drift distance): both FALL monotonically as |z| grows from 0 to 30.43 cm
(0.044->0.011 cm for sigma_L, 0.058->0.015 cm for sigma_T, binned over the real sample), so drift
distance DECREASES as |z| increases, i.e. drift_distance = drift_length - |z|. This script uses
that corrected quantity throughout; using |z| itself would flip every sign below.

Produces, for each batch: length_cm, n_events, n_tracks, mean dE/dx (length-weighted), mean drift
distance and its length-weighted spread (std and range of drift_length-|z|), and the
length-weighted mean track angle to the drift direction (0 deg = along the drift axis, 90 deg =
in the anode/cathode plane; from each track's straight-line endpoint-to-endpoint direction, the
same convention optimize/dataio.py's live_selection cos_theta cut uses).

CPU only; JAX_PLATFORMS is forced to 'cpu' before paper_common (which imports jax) is pulled in,
and only on the (one-time) cache-build path -- a cached run needs no jax at all. No SLURM. Only
writes NEW files: this script, its cache, and the two output figure files; no tracked file
(incl. main.tex) is edited.

Usage:
    python fig_information.py [--force]
Writes:
    /sdf/group/neutrino/pgranger/NuLads_draft/figures/information_per_batch.pdf
    /sdf/group/neutrino/pgranger/NuLads_draft/figures/information_per_batch.png
Caches per-batch features (and the raw H, g from hessian.json, for a self-contained, jax-free
re-plot) in:
    /sdf/scratch/neutrino/pgranger/review_2026_09/paperfig_cache/information_features.npz
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # must precede `import jax` anywhere (incl. via paper_common)
import sys
import json
import argparse

import numpy as np

PAPERJOBS = '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS'
HESSIAN_JSON = f'{PAPERJOBS}/hessian.json'
CACHE_DIR = '/sdf/scratch/neutrino/pgranger/review_2026_09/paperfig_cache'
CACHE_NPZ = f'{CACHE_DIR}/information_features.npz'
OUT_DIR = '/sdf/group/neutrino/pgranger/NuLads_draft/figures'

PARAMS = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
PARAM_LABEL = {'Ab': r'$A_b$', 'eField': r'$E$ field', 'tran_diff': r'$D_T$',
               'long_diff': r'$D_L$', 'lifetime': r'$\tau$'}
# Okabe-Ito palette (shared style guide); one colour per parameter, used consistently in every panel.
COLOR = {'lifetime': '#D55E00', 'long_diff': '#0072B2', 'tran_diff': '#009E73',
         'Ab': '#E69F00', 'eField': '#56B4E9'}


# ==========================================================================================
# Feature extraction (the expensive, jax-dependent path; only runs if the cache is missing)
# ==========================================================================================
def build_features():
    sys.path.insert(0, PAPERJOBS)
    import paper_common as pc  # noqa: E402  (chdirs to the repo on import; pulls in jax)

    print('loading hessian.json ...')
    d = json.load(open(HESSIAN_JSON))
    if d['params'] != PARAMS:
        raise RuntimeError(f'hessian.json params {d["params"]} != expected {PARAMS}')
    batches = d['batches']
    n = len(batches)

    print(f'reconstructing the {n} batches via paper_common.load({d["len"]}, 400, {d["data_seed"]}) ...')
    ds_sim, ds_tgt, fields = pc.load(d['len'], 400, d['data_seed'])
    if len(ds_tgt) != n:
        raise RuntimeError(f'batch-count mismatch: hessian.json has {n}, reload gives {len(ds_tgt)}')
    idx = {f: i for i, f in enumerate(fields)}

    def col(tr, name):
        return tr[:, idx[name]]

    truth, _response = pc.params()
    drift_length = float(truth.drift_length)
    print(f'drift_length = {drift_length:.3f} cm; tpc_borders =\n{np.asarray(truth.tpc_borders)}')

    length_cm = np.zeros(n)
    n_events = np.zeros(n, dtype=int)
    n_tracks = np.zeros(n, dtype=int)
    mean_dedx = np.zeros(n)
    mean_drift = np.zeros(n)     # length-weighted mean drift DISTANCE (drift_length - |z|), cm
    std_drift = np.zeros(n)      # length-weighted std of the same, within the batch
    range_drift = np.zeros(n)    # max-min of the same, within the batch
    mean_angle = np.zeros(n)     # length-weighted mean track angle to the drift axis, deg
    H_all = np.zeros((n, 5, 5))
    g_all = np.zeros((n, 5))

    for i in range(n):
        tr = np.asarray(ds_tgt[i])
        dx = col(tr, 'dx')
        real = dx > 0                      # padding rows carry dx=0 (and eventID=trackID=-1)
        trR, dxR = tr[real], dx[real]
        L = float(dxR.sum())
        b = batches[i]
        if abs(b['length_cm'] - L) > 1e-1:
            raise RuntimeError(f'batch {i}: length mismatch, hessian.json {b["length_cm"]} vs reload {L}')
        length_cm[i] = L

        ev, tid = col(trR, 'eventID'), col(trR, 'trackID')
        z, dedx = col(trR, 'z'), col(trR, 'dEdx')
        drift = np.clip(drift_length - np.abs(z), 0., None)

        mean_dedx[i] = float(np.sum(dedx * dxR) / L)
        mu_d = float(np.sum(drift * dxR) / L)
        mean_drift[i] = mu_d
        std_drift[i] = float(np.sqrt(np.sum(dxR * (drift - mu_d) ** 2) / L))
        range_drift[i] = float(drift.max() - drift.min())
        n_events[i] = len(np.unique(ev))

        # per-track angle to the drift axis, from straight-line endpoint-to-endpoint direction
        # (same convention as optimize/dataio.py's live_selection cos_theta cut). Rows of one
        # track are contiguous and in order within a batch (verified separately), so the first
        # row's *_start and the last row's *_end of each (eventID, trackID) group are its ends.
        key = ev * 100000 + tid
        uniq_keys = np.unique(key)
        n_tracks[i] = len(uniq_keys)
        order = np.argsort(key, kind='stable')
        key_s = key[order]
        starts = np.searchsorted(key_s, uniq_keys, side='left')
        ends = np.searchsorted(key_s, uniq_keys, side='right') - 1
        row_start, row_end = order[starts], order[ends]
        dxv = col(trR, 'x_end')[row_end] - col(trR, 'x_start')[row_start]
        dyv = col(trR, 'y_end')[row_end] - col(trR, 'y_start')[row_start]
        dzv = col(trR, 'z_end')[row_end] - col(trR, 'z_start')[row_start]
        straight = np.sqrt(dxv ** 2 + dyv ** 2 + dzv ** 2)
        cos_t = np.clip(np.abs(dzv) / np.maximum(straight, 1e-9), 0., 1.)
        theta = np.degrees(np.arccos(cos_t))
        track_len = np.array([dxR[key == k].sum() for k in uniq_keys])
        mean_angle[i] = float(np.sum(theta * track_len) / track_len.sum())

        H_all[i] = np.array(b['H'])
        g_all[i] = np.array(b['g'])

        if i % 50 == 0:
            print(f'  batch {i}/{n}: L={L:.1f} cm, {n_events[i]} events, {n_tracks[i]} tracks', flush=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(CACHE_NPZ, params=np.array(PARAMS), drift_length_cm=np.array(drift_length),
              length_cm=length_cm, n_events=n_events, n_tracks=n_tracks, mean_dedx=mean_dedx,
              mean_drift=mean_drift, std_drift=std_drift, range_drift=range_drift,
              mean_angle=mean_angle, H=H_all, g=g_all)
    print(f'cached features for {n} batches -> {CACHE_NPZ}')
    return dict(np.load(CACHE_NPZ, allow_pickle=True))


def load_or_build(force=False):
    if not force and os.path.exists(CACHE_NPZ):
        print(f'loading cache {CACHE_NPZ}')
        return dict(np.load(CACHE_NPZ, allow_pickle=True))
    return build_features()


# ==========================================================================================
# Derived statistics
# ==========================================================================================
def binned_aggregate(feature, Hjj, length_cm, nbins=8, n_boot=4000, seed=0):
    """Equal-population bins of `feature`; each bin's aggregate info density is the SUM of Hjj
    over the bin divided by the SUM of length (additive, so individual-batch sign/noise issues
    average out), with a bootstrap (resample batches within the bin) 16-84 percentile band."""
    rng = np.random.default_rng(seed)
    order = np.argsort(feature)
    bins = np.array_split(order, nbins)
    xs, ys, lo, hi = [], [], [], []
    for idxb in bins:
        m = len(idxb)
        xs.append(float(np.average(feature[idxb], weights=length_cm[idxb])))
        ys.append(float(Hjj[idxb].sum() / length_cm[idxb].sum()))
        samp = rng.integers(0, m, size=(n_boot, m))
        bi = idxb[samp]
        boots = Hjj[bi].sum(axis=1) / length_cm[bi].sum(axis=1)
        lo.append(np.percentile(boots, 16))
        hi.append(np.percentile(boots, 84))
    return np.array(xs), np.array(ys), np.array(lo), np.array(hi)


def lorenz_curve(Hjj, length_cm):
    """Cumulative information fraction vs. cumulative length fraction, batches sorted by
    decreasing own density. concentration = 2*int(Hc dLc) - 1 in [0, 1); 0 = info proportional
    to length (uniform density), ->1 = info concentrated in a vanishing fraction of length."""
    order = np.argsort(-Hjj / length_cm)
    Lc = np.concatenate([[0.], np.cumsum(length_cm[order]) / length_cm.sum()])
    Hc = np.concatenate([[0.], np.cumsum(Hjj[order]) / Hjj.sum()])
    concentration = 2 * np.trapezoid(Hc, Lc) - 1
    return Lc, Hc, concentration


def profiled_bin_check(feature, Hsym, length_cm, nbins=8):
    """Profiled (all-5-free) info density of each bin, from the bin-SUMMED 5x5 Hessian --
    stable at this aggregation (unlike single-batch), used only for the printed/caption numbers,
    not for the figure itself. Returns relative-to-global arrays, shape (nbins, 5), and the
    bins' condition numbers."""
    order = np.argsort(feature)
    bins = np.array_split(order, nbins)
    glob_diag = np.einsum('bjj->bj', Hsym).sum(0) / length_cm.sum()
    out, conds = [], []
    for idxb in bins:
        Hb = Hsym[idxb].sum(0)
        Lb = length_cm[idxb].sum()
        eig = np.linalg.eigvalsh(Hb)
        conds.append(eig[-1] / eig[0] if eig[0] > 0 else np.inf)
        prof = (1.0 / np.diag(np.linalg.inv(Hb))) / Lb
        out.append(prof / glob_diag)
    return np.array(out), np.array(conds)


# ==========================================================================================
# Figure
# ==========================================================================================
def make_figure(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats

    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    })

    length_cm = data['length_cm']
    mean_drift, mean_angle, mean_dedx = data['mean_drift'], data['mean_angle'], data['mean_dedx']
    H = data['H']
    Hsym = 0.5 * (H + H.transpose(0, 2, 1))
    diag = np.einsum('bjj->bj', Hsym)                     # (n, 5), H_jj per batch
    glob = diag.sum(0) / length_cm.sum()                  # global diagonal info density per param
    rel = diag / length_cm[:, None] / glob[None, :]        # relative info density, length-wtd mean = 1
    n = len(length_cm)

    eig0 = np.linalg.eigvalsh(Hsym)[:, 0]
    cond = np.where(eig0 > 0, np.linalg.eigvalsh(Hsym)[:, -1] / np.where(eig0 > 0, eig0, 1), np.inf)
    n_nonpsd = int(np.sum(eig0 <= 0))
    n_stable = int(np.sum((eig0 > 0) & (cond <= 1e6)))

    def corr(fv, j):
        y = rel[:, j]
        r, _ = stats.pearsonr(fv, y)
        rho, _ = stats.spearmanr(fv, y)
        return r, rho

    print('\n=== per-batch summary ===')
    print(f'{n} batches, {length_cm.sum()/100:.1f} m total; events/batch {int(data["n_events"].min())}'
          f'-{int(data["n_events"].max())} (median {np.median(data["n_events"]):.0f}); '
          f'tracks/batch {int(data["n_tracks"].min())}-{int(data["n_tracks"].max())} '
          f'(mean {data["n_tracks"].mean():.2f})')
    print(f'mean dE/dx range [{mean_dedx.min():.3f}, {mean_dedx.max():.3f}] MeV/cm (1-99 pct '
          f'[{np.percentile(mean_dedx,1):.3f}, {np.percentile(mean_dedx,99):.3f}])')
    print(f'batches with non-PSD H: {n_nonpsd}/{n};  batches with cond(H) <= 1e6: {n_stable}/{n}'
          f'  (median cond among PSD batches: {np.nanmedian(np.where(eig0>0, cond, np.nan)):.2e})')
    print()
    print(f'{"param":10s} {"glob info/cm":>13s} {"rel range":>18s}  {"r,rho vs drift":>16s}  {"r,rho vs angle":>16s}  {"r,rho vs dEdx":>14s}')
    for j, p in enumerate(PARAMS):
        r_d, rho_d = corr(mean_drift, j)
        r_a, rho_a = corr(mean_angle, j)
        r_e, rho_e = corr(mean_dedx, j)
        print(f'{p:10s} {glob[j]:13.4e} [{rel[:,j].min():6.3f},{rel[:,j].max():6.3f}]  '
              f'{r_d:+.2f},{rho_d:+.2f}         {r_a:+.2f},{rho_a:+.2f}         {r_e:+.2f},{rho_e:+.2f}')

    # ---- Lorenz / concentration numbers ----
    print('\n=== concentration of information across batches (Lorenz curve, fig panel d) ===')
    lorenz = {}
    for j, p in enumerate(PARAMS):
        Lc, Hc, c = lorenz_curve(diag[:, j], length_cm)
        f10, f20 = np.interp([0.1, 0.2], Lc, Hc)
        lorenz[p] = (Lc, Hc, c, f10, f20)
        print(f'{p:10s} concentration={c:.3f}  top10%-of-length carries {100*f10:4.1f}% of info, '
              f'top20%% carries {100*f20:4.1f}%')

    # ---- profiled-info aggregate check (text/caption numbers only, not plotted) ----
    print('\n=== profiled (all-5-free) info density, aggregated in drift-distance bins (relative) ===')
    prof_drift, cond_drift = profiled_bin_check(mean_drift, Hsym, length_cm)
    print('bin condition numbers:', ' '.join(f'{c:.1e}' for c in cond_drift))
    for j, p in enumerate(PARAMS):
        print(f'{p:10s} profiled(rel) by drift-distance bin: ' + ' '.join(f'{v:.2f}' for v in prof_drift[:, j]))

    # ---- binned trends for the three scatter panels (normalised by the SAME global per-param
    # density as the `rel` scatter, so the trend line overlays the cloud it summarises instead
    # of sitting offset by the parameter's absolute scale -- critical for eField, whose global
    # density is ~1e6 vs ~O(1) for the others) ----
    def binned_relative(fv, p):
        j = PARAMS.index(p)
        xs, ys, lo, hi = binned_aggregate(fv, diag[:, j], length_cm)
        return xs, ys / glob[j], lo / glob[j], hi / glob[j]

    bins_drift_a = {p: binned_relative(mean_drift, p) for p in ('lifetime', 'long_diff')}
    bins_angle_b = {p: binned_relative(mean_angle, p) for p in ('tran_diff', 'Ab')}
    bins_drift_c = {p: binned_relative(mean_drift, p) for p in ('eField',)}
    n_neg_efield = int(np.sum(rel[:, PARAMS.index('eField')] < 0))
    print(f'\neField: {n_neg_efield}/{n} batches have formally negative diagonal curvature (excluded from '
          f'panel c raw scatter; included in the binned/bootstrap aggregate, which is positive in every bin).')

    # ------------------------------------------------------------------------------------
    # The drift field is left out of the figure: its per-batch diagonal curvature is negative in a few batches and a
    # single batch carries ~28% of its total (see the printout below), so it is not interpretable batch by batch.
    jE = PARAMS.index('eField'); dE = diag[:, jE]; iE = int(np.argmax(dE))
    print(f'\neField omitted from the figure: largest single batch (#{iE}, {length_cm[iE]:.0f} cm) carries '
          f'{100*dE[iE]/dE.sum():.1f}% of the total diagonal curvature; {int(np.sum(dE < 0))} batches negative.')
    fig = plt.figure(figsize=(7.0, 2.45))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.075, right=0.985, top=0.95, bottom=0.18)

    def panel_label(ax, text):
        ax.text(0.04, 0.95, text, transform=ax.transAxes, fontsize=7, va='top', ha='left', zorder=10)

    def scatter_and_bins(ax, fv, j_param, bins, log=True):
        p = j_param
        y = rel[:, PARAMS.index(p)]
        if log:
            pos = y > 0
            ax.scatter(fv[pos], y[pos], s=5, color=COLOR[p], alpha=0.18, lw=0, zorder=2)
        else:
            ax.scatter(fv, y, s=5, color=COLOR[p], alpha=0.18, lw=0, zorder=2)
        xs, ys, lo, hi = bins[p]
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt='o-', color=COLOR[p], ms=3.5, lw=1.3,
                    capsize=2, elinewidth=0.8, zorder=5, label=PARAM_LABEL[p])

    # --- (a) lifetime & long_diff vs mean drift distance ---
    ax = fig.add_subplot(gs[0, 0])
    for p in ('lifetime', 'long_diff'):
        scatter_and_bins(ax, mean_drift, p, bins_drift_a)
    ax.set_yscale('log')
    ax.set_xlabel('batch mean drift distance (cm)')
    ax.set_ylabel('relative information density\n(batch mean = 1)')
    ax.grid(alpha=0.25, linewidth=0.5)
    panel_label(ax, '(a)')
    r_tau, rho_tau = corr(mean_drift, PARAMS.index('lifetime'))
    r_dl, rho_dl = corr(mean_drift, PARAMS.index('long_diff'))
    ax.text(0.97, 0.06, rf'$\tau$: $\rho$={rho_tau:+.2f}' + '\n' + rf'$D_L$: $\rho$={rho_dl:+.2f}',
            transform=ax.transAxes, fontsize=6.3, va='bottom', ha='right', linespacing=1.3)
    ax.legend(frameon=False, loc='lower left', handlelength=1.3, labelspacing=0.3, borderaxespad=0.2)

    # --- (b) tran_diff & Ab vs mean track angle to the drift direction ---
    ax = fig.add_subplot(gs[0, 1])
    for p in ('tran_diff', 'Ab'):
        scatter_and_bins(ax, mean_angle, p, bins_angle_b)
    ax.set_yscale('log')
    ax.set_xlabel(r'batch mean track angle to drift axis (deg)')
    ax.grid(alpha=0.25, linewidth=0.5)
    panel_label(ax, '(b)')
    r_td, rho_td = corr(mean_angle, PARAMS.index('tran_diff'))
    r_ab, rho_ab = corr(mean_angle, PARAMS.index('Ab'))
    ax.text(0.97, 0.94, rf'$D_T$: $\rho$={rho_td:+.2f}' + '\n' + rf'$A_b$: $\rho$={rho_ab:+.2f}',
            transform=ax.transAxes, fontsize=6.3, va='top', ha='right', linespacing=1.3)
    ax.legend(frameon=False, loc='lower left', handlelength=1.3, labelspacing=0.3, borderaxespad=0.2)

    # --- (c) cumulative share of information vs. cumulative share of length (Lorenz curves) ---
    ax = fig.add_subplot(gs[0, 2])
    ax.plot([0, 1], [0, 1], color='0.5', ls='--', lw=0.9, zorder=1, label='uniform density')
    for p in [q for q in PARAMS if q != 'eField']:
        Lc, Hc, c, f10, f20 = lorenz[p]
        ax.plot(Lc, Hc, color=COLOR[p], lw=1.3, label=f'{PARAM_LABEL[p]} (C={c:.2f})', zorder=3)
    ax.set_xlabel('cumulative fraction of track length')
    ax.set_ylabel('cumulative fraction of information')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, linewidth=0.5)
    panel_label(ax, '(c)')
    ax.legend(frameon=False, loc='lower right', handlelength=1.4, labelspacing=0.25, borderaxespad=0.2,
              fontsize=6.3)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(f'{OUT_DIR}/information_per_batch.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT_DIR}/information_per_batch.png', dpi=300, bbox_inches='tight')
    print(f'\nsaved {OUT_DIR}/information_per_batch.pdf and .png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='ignore the cache, recompute from paper_common')
    args = ap.parse_args()
    data = load_or_build(force=args.force)
    make_figure(data)
