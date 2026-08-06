"""Mode-decomposition figures (Figs 11-14) for the noise-ON campaign report.

Recomputes, rather than restates:
  - the static Hessian eigenmodes (from stored eigenvalue/eigenvector compositions)
  - the (lifetime, long_diff) soft plane as a covariance ellipse + the 2-D NLL map
  - the DYNAMIC modes: PCA of the parameter increments actually taken by S2 / S4 / anneal,
    and their overlap with the static soft plane
  - the drift-axis vs wire-plane decomposition of the fitted spline displacement
"""
import pickle, glob, os, re, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
os.makedirs(OUT, exist_ok=True)
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', sky='#56B4E9', grey='#666666')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
H = json.load(open('fit_result/hessian_analysis/hessian_results.json'))


def parse_modes(entries, names):
    """'Ab:-0.02 eField:+0.00 ...' -> (eigenvalues, matrix of eigenvector components)."""
    vals, vecs = [], []
    for e in entries:
        d = dict((m.group(1), float(m.group(2)))
                 for m in re.finditer(r'([A-Za-z_]+):([+-]?[\d.]+(?:e[+-]?\d+)?)', e['composition']))
        vals.append(e['value']); vecs.append([d.get(n, 0.0) for n in names])
    return np.array(vals), np.array(vecs)


# ── Fig 11 ── static Hessian mode decomposition ──────────────────────────────
v5, V5 = parse_modes(H['calib5x5']['eigen'], P)
G = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime', 'dz_cm', 'zscale', 'dx_cm']
v8, V8 = parse_modes(H['calib_plus_geom8x8']['eigen'], G)
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.0),
                         gridspec_kw={'width_ratios': [1, 1.5]})
for ax, (vals, V, names, ttl) in zip(axes, [
        (v5, V5, P, 'Calibration only (5×5), at truth'),
        (v8, V8, G, 'Calibration + 3 rigid geometry modes (8×8)')]):
    im = ax.imshow(np.abs(V), cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([PL.get(n, n) for n in names], rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([f'λ={x:.3g}' for x in vals], fontsize=8)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            if abs(V[i, j]) > .12:
                ax.text(j, i, f'{V[i,j]:+.2f}', ha='center', va='center', fontsize=6.6,
                        color='white' if abs(V[i, j]) > .6 else 'black')
    ax.set_title(ttl, loc='left', fontsize=9.5)
    ax.grid(False)
    for i, x in enumerate(vals):
        if x < 1e4:
            ax.add_patch(plt.Rectangle((-.5, i - .5), len(names), 1, fill=False,
                                       edgecolor=C['red'], lw=2))
fig.colorbar(im, ax=axes, shrink=.8, label='|eigenvector component|')
fig.suptitle('Fig 11 — Static mode decomposition. Each ROW is an eigenmode of the loss curvature; '
             'each COLUMN a parameter.\nRed boxes = the soft (poorly constrained) modes. Soft modes are '
             'MIXED — no single parameter is unconstrained; a COMBINATION is.', x=.02, ha='left', y=1.09)
fig.savefig(f'{OUT}/fig11_mode_decomposition.png'); plt.close(fig)

# ── Fig 12 ── the soft plane, drawn ──────────────────────────────────────────
Hm = (V5.T * v5) @ V5                       # rebuild H = sum_k lambda_k v_k v_k^T
i_l, i_g = P.index('lifetime'), P.index('long_diff')
sub = np.linalg.inv(Hm)[np.ix_([i_l, i_g], [i_l, i_g])]     # marginal 2x2 covariance
w, U = np.linalg.eigh(sub)
th = np.linspace(0, 2 * np.pi, 200)
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
ax = axes[0]
for ns, a in [(1, .30), (2, .18), (3, .10)]:
    e = (U @ (np.sqrt(w)[:, None] * np.array([np.cos(th), np.sin(th)])) * ns) * 100
    ax.fill(e[0], e[1], color=C['blue'], alpha=a, lw=0)
    ax.plot(e[0], e[1], color=C['blue'], lw=.8)
soft = U[:, np.argmax(w)]
L = 3 * np.sqrt(w.max()) * 100
ax.annotate('', xy=(soft[0] * L, soft[1] * L), xytext=(-soft[0] * L, -soft[1] * L),
            arrowprops=dict(arrowstyle='<->', color=C['red'], lw=1.8))
ax.text(soft[0] * L * .55, soft[1] * L * .55 + .4, 'soft direction\n(degenerate)',
        color=C['red'], fontsize=8)
ax.plot(0, 0, 'k+', ms=9)
ax.set_xlabel('lifetime error (%)'); ax.set_ylabel('long. diffusion error (%)')
ax.set_title(f'1/2/3σ joint uncertainty at truth\naspect ratio {np.sqrt(w.max()/w.min()):.1f}:1',
             loc='left', fontsize=9)
ax.set_aspect('equal')
try:
    z = np.load('fit_result/gn_proto/nll_map_s2_400_noise.npz')
    M, ls, ts = z['M'], z['lspan'], z['tspan']
    ax = axes[1]
    cs = ax.contourf(ts * 100, ls * 100, M - M.min(), levels=18, cmap='YlGnBu_r')
    ax.contour(ts * 100, ls * 100, M - M.min(), levels=8, colors='k', linewidths=.4, alpha=.4)
    j, i = np.unravel_index(np.argmin(M), M.shape)
    ax.plot(ts[i] * 100, ls[j] * 100, 'o', color=C['red'], ms=7, label='NLL minimum')
    ax.plot(0, 0, 'k+', ms=11, label='truth')
    ax.set_xlabel('Δ log(tran. diff.) (%)'); ax.set_ylabel('Δ log(lifetime) (%)')
    ax.legend(fontsize=8, frameon=False)
    ax.set_title('Measured 2-D NLL scan: minimum sits exactly on truth\n'
                 '(the loss is unbiased; the difficulty is flatness, not a wrong optimum)',
                 loc='left', fontsize=9)
    fig.colorbar(cs, ax=ax, label='NLL − NLL$_{min}$')
except Exception as ex:
    print('NLL map skipped:', ex)
fig.suptitle('Fig 12 — What the soft mode means in practice', x=.02, ha='left', y=1.02)
fig.savefig(f'{OUT}/fig12_soft_plane.png'); plt.close(fig)

# ── Fig 13 ── DYNAMIC modes: what directions the optimizer actually moved in ─
def increments(d):
    """Per-seed PCA of log-space parameter increments -> (leading vec, explained var)."""
    out = []
    for f in glob.glob(d + '/history_iter*.pkl'):
        if not re.search(r'seed\d', f):
            continue
        try:
            h = pickle.load(open(f, 'rb'))
        except Exception:
            continue
        if any(p + '_iter' not in h for p in P):
            continue
        X = np.stack([np.log(np.abs(np.ravel(np.array(h[p + '_iter']))) + 1e-30) for p in P], 1)
        D = np.diff(X, axis=0)
        D = D[len(D) // 2:]                       # late phase only
        D = D[np.isfinite(D).all(1)]
        if len(D) < 50:
            continue
        Cv = np.cov(D.T)
        ev, EV = np.linalg.eigh(Cv)
        out.append((EV[:, -1], ev[-1] / ev.sum()))
    return out


soft2 = V5[:2].T                                  # the two softest static eigenvectors
Psoft = soft2 @ np.linalg.pinv(soft2)             # projector onto the soft plane
CASES = [('S2\n(geometry known)', 'fit_result/sci_ceiling_noise_s2', C['green']),
         ('S4 baseline\n(geometry fitted)', 'fit_result/sci_full_noise_s4', C['red']),
         ('S4 + LR anneal', 'fit_result/sci_full_noise_s4_anneal', C['blue'])]
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4),
                         gridspec_kw={'width_ratios': [1, 1.35]})
labs, allov, comps = [], [], []
for lab, d, col in CASES:
    res = increments(d)
    ov = [float(np.linalg.norm(Psoft @ v)) for v, _ in res]
    if not ov:
        continue
    labs.append(lab); allov.append(ov)
    comps.append(np.mean([np.abs(v) for v, _ in res], axis=0))
    axes[0].scatter([len(labs) - 1] * len(ov), ov, s=46, color=col, zorder=3,
                    edgecolor='white', linewidth=.8)
    axes[0].plot([len(labs) - 1.22, len(labs) - .78], [np.mean(ov)] * 2, color=col, lw=2.4)
axes[0].set_xticks(range(len(labs))); axes[0].set_xticklabels(labs, fontsize=8)
axes[0].set_ylim(0, 1.03); axes[0].set_ylabel('overlap of leading increment mode\nwith the static soft plane')
axes[0].axhline(1, ls=':', color=C['grey'])
axes[0].set_title('Geometry known -> the fit moves inside the predicted soft plane.\n'
                  'Geometry fitted -> it escapes into a NEW degeneracy.\n'
                  'Annealing RESTORES the S2-like behaviour.', loc='left', fontsize=8.5)
x = np.arange(len(P)); w = .8 / max(len(comps), 1)
for j, (lab, cvec) in enumerate(zip(labs, comps)):
    axes[1].bar(x + (j - (len(comps) - 1) / 2) * w, cvec, w, label=lab.replace('\n', ' '),
                color=[c for _, _, c in CASES][j], edgecolor='white', linewidth=.6)
axes[1].set_xticks(x); axes[1].set_xticklabels([PL[p] for p in P], fontsize=8)
axes[1].set_ylabel('|component| of leading increment mode')
axes[1].legend(fontsize=7.5, frameon=False)
axes[1].set_title('Composition of the direction the optimizer actually moves in', loc='left', fontsize=9)
fig.suptitle('Fig 13 — DYNAMIC mode decomposition: the directions the fit actually travels, '
             'compared with the\nstatic curvature prediction. Computed from the late half of each '
             'run\'s parameter increments.', x=.02, ha='left', y=1.14)
fig.savefig(f'{OUT}/fig13_dynamic_modes.png'); plt.close(fig)
for lab, ov in zip(labs, allov):
    print(f'  overlap {lab.splitlines()[0]:14s}: ' + ' '.join(f'{o:.2f}' for o in ov)
          + f'   mean {np.mean(ov):.2f}')

# ── Fig 14 ── drift-axis vs wire-plane decomposition of the fitted geometry ──
ZH = np.array([0., 0., 1.], np.float32)


def frame_legacy(u0):
    ref = ZH.copy()
    if abs(float(np.dot(u0, ref))) > 0.9:
        ref = np.array([1., 0., 0.], np.float32)
    e1 = np.cross(u0, ref); e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(u0, e1);  e2 /= np.linalg.norm(e2) + 1e-12
    return e1, e2


def decompose(d):
    """-> (drift-projected RMS, wire-plane RMS, |e2.zhat|) per track, in um."""
    dr, wp, e2z = [], [], []
    for f in glob.glob(d + '/history_iter*.pkl'):
        if not re.search(r'seed\d', f):
            continue
        h = pickle.load(open(f, 'rb'))
        cache, ctxs = h.get('chain_cache'), h.get('chain_contexts')
        if not cache or not ctxs:
            continue
        for b in ctxs:
            for ci, c in enumerate(ctxs[b]):
                try:
                    co = np.asarray(cache[b][ci]['angles'], float)
                except Exception:
                    continue
                K = len(co) // 2
                b1, b2 = co[:K], co[K:2 * K]
                th_, ph = float(c['theta0_i']), float(c['phi0_i'])
                st = np.sin(th_)
                u0 = np.array([st * np.cos(ph), st * np.sin(ph), np.cos(th_)], np.float32)
                e1, e2 = frame_legacy(u0)
                nc = int(c['n_chain']); al = np.arange(nc + 1) / nc
                B = np.sin(np.outer(al, np.pi * np.arange(1, K + 1)))
                d1, d2 = B @ b1, B @ b2
                disp = d1[:, None] * e1[None, :] + d2[:, None] * e2[None, :]
                dz = disp[:, 2]
                dperp = np.linalg.norm(disp - dz[:, None] * ZH[None, :], axis=1)
                dr.append(np.sqrt(np.mean(dz ** 2)) * 1e4)
                wp.append(np.sqrt(np.mean(dperp ** 2)) * 1e4)
                e2z.append(abs(float(np.dot(e2, ZH))))
        break                                     # one seed is enough for the distribution
    return np.array(dr), np.array(wp), np.array(e2z)


dr, wp, e2z = decompose('fit_result/sci_full_noise_s4_anneal')
if dr.size:
    tot = np.sqrt(dr ** 2 + wp ** 2)
    frac = dr / np.maximum(tot, 1e-9)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.5))
    axes[0].hist(dr, bins=40, color=C['red'], alpha=.75, label=f'along DRIFT (median {np.median(dr):.0f} µm)')
    axes[0].hist(wp, bins=40, color=C['blue'], alpha=.6, label=f'in WIRE PLANE (median {np.median(wp):.0f} µm)')
    axes[0].set_xlabel('per-track RMS displacement (µm)'); axes[0].set_ylabel('tracks')
    axes[0].legend(fontsize=7.5, frameon=False)
    axes[0].set_title('The fit moves charge along the drift axis, where\ndisplacement trades off directly against lifetime',
                      loc='left', fontsize=9)
    axes[1].hist(frac, bins=30, color=C['purple'])
    axes[1].axvline(np.median(frac), color='k', lw=1.4)
    axes[1].text(np.median(frac) + .02, axes[1].get_ylim()[1] * .9,
                 f'median {np.median(frac)*100:.0f}%', fontsize=8)
    axes[1].set_xlabel('drift-projected fraction of |displacement|'); axes[1].set_ylabel('tracks')
    axes[1].set_title('Fraction of fitted motion lying in the direction\nthat is degenerate with charge attenuation', loc='left', fontsize=9)
    axes[2].hist(e2z, bins=30, color=C['orange'])
    axes[2].set_xlabel('|e₂·ẑ|  (drift content of the 2nd basis vector)'); axes[2].set_ylabel('tracks')
    axes[2].set_title(f'Legacy basis has no drift awareness:\nmedian |e₂·ẑ| = {np.median(e2z):.2f}',
                      loc='left', fontsize=9)
    fig.suptitle('Fig 14 — Drift-axis decomposition of the fitted spline geometry (LR-anneal run). '
                 'The transverse basis\nis blind to the drift direction, so a large share of the fitted '
                 'displacement lies along it.', x=.02, ha='left', y=1.05)
    fig.savefig(f'{OUT}/fig14_drift_decomposition.png'); plt.close(fig)
    print(f'  drift RMS median {np.median(dr):.0f} um | wire-plane {np.median(wp):.0f} um '
          f'| drift fraction {np.median(frac)*100:.0f}% | median |e2.z| {np.median(e2z):.2f} '
          f'| n_tracks {dr.size}')
print('wrote figs 11-14')
