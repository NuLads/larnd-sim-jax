"""a10: the fit's path through the (A_b, lifetime) plane, then the end points against the loss valley computed at TRUE inputs."""
from matplotlib.lines import Line2D
from traj_helpers import *

args = parse_args()
DUR = 48.0
KNOTS = [(0, 0), (4, 0), (13, 1000), (14, 1000), (19.5, 5000), (25, 14000), (25.6, 14000), (30, 20000), (DUR, 20000)]
it_at = time_map(KNOTS)
T_Z1 = (14.5, 19.5)        # wide -> mid view (the cloud is resolved while it settles)
T_Z2 = (31.0, 33.5)        # mid -> final region
T_END = (30.2, 31.2)       # moving dots -> end points (last-1000 means)
T_SURF = 34.0; T_LINE = 37.5; T_MEAN = 41.0

AB, TA = errs('Ab'), errs('lifetime')
RUN = 0                                  # sample 1: the lowest sample number, not chosen for its behaviour
assert SAMPLE[EXP[RUN]] == 1
EA, ET = endpoint('Ab'), endpoint('lifetime')
sa, st = ensemble(EA), ensemble(ET)
SL, IC = np.polyfit(EA, ET, 1); R = np.corrcoef(EA, ET)[0, 1]
_rng = np.random.default_rng(0); _b = [np.polyfit(EA[k], ET[k], 1)[0] for k in _rng.integers(0, 36, (5000, 36))]
CI = np.percentile(_b, [2.5, 97.5])
# r across the 36 fits, each fit averaged over one full 100-batch pass
_ca = np.cumsum(AB, 1, dtype=float); _ct = np.cumsum(TA, 1, dtype=float)
RIT = np.arange(100, 20001, 20)
RV = np.array([np.corrcoef(_ca[:, i] - _ca[:, i - 100], _ct[:, i] - _ct[:, i - 100])[0, 1] for i in RIT])
# real loss surface: production loss, noise ON, sum over the 100 batches, mean over the 10 sim seeds
_Q = [dict(np.load(f)) for f in sorted(glob.glob('/sdf/scratch/neutrino/pgranger/review_2026_09/E_truthdiag/out/ab2dfine_*.npz'))]
GA, GT = 100 * (_Q[0]['ab_grid'] - 1), 100 * (_Q[0]['lt_grid'] - 1)
Z = np.concatenate([q['nll2_on'] for q in _Q], 0).sum(0).mean(-1); Z = Z - Z.min()
assert len(np.concatenate([q['batch'] for q in _Q])) == 100


def _quad(Z):
    X, Y = np.meshgrid(GA, GT, indexing='ij')
    M = np.vstack([np.ones(X.size), X.ravel(), Y.ravel(), X.ravel() ** 2, X.ravel() * Y.ravel(), Y.ravel() ** 2]).T
    c = np.linalg.lstsq(M, Z.ravel(), rcond=None)[0]
    H = np.array([[2 * c[3], c[4]], [c[4], 2 * c[5]]]); x = np.linalg.solve(H, -c[1:3])
    return x, -c[4] / (2 * c[5])


# same 2-D quadratic as analyze_2dfine.py, on the seed-mean surface; the valley slope agrees with its -20.3 (per-seed mean)
MIN0, VAL = _quad(Z)
VALLEY = -20.3
_o = st['mean'] - (MIN0[1] + VALLEY * (sa['mean'] - MIN0[0]))
print(f'endpoints: r {R:.3f} OLS {SL:.2f} CI {CI}, centroid {sa["mean"]:.3f} {st["mean"]:.2f}; surface min {MIN0}, valley fit {VAL:.2f}; centroid minus valley line {_o:+.2f}')
print('r(it): ' + ' '.join(f'{i}:{np.interp(i, RIT, RV):+.2f}' for i in (500, 1000, 2000, 5000, 10000, 14000, 20000)))
_final = f'End points share that direction: slope {SL:.1f} (95% CI {CI[0]:.1f} to {CI[1]:.1f}), r = {R:.2f}, as a broad band'.replace('-', '−')
CAPS = [(0, 4.1, 'Collected charge scales with $A_b$ and with $e^{-t/\\tau}$: more of one can pay for less of the other'),
        (4.1, 14.2, 'Line: one fit, sample 1. Dots: all 36. Early on, each fit swings hard in both'),
        (14.2, 25.8, 'The cloud lies down along one direction as the easy errors disappear'),
        (25.8, 30.8, 'The last re-heat shakes the cloud, then it settles shorter along that direction'),
        (30.8, 34.1, 'Each dot is now one sample: mean of its last 1000 iterations'),
        (34.1, 37.4, 'Loss computed at TRUE inputs, near truth only: a long, narrow valley'),
        (37.4, 40.9, 'Valley direction from that loss: $d\\tau/\\tau = -20.3\\,dA_b/A_b$'),
        (40.9, 99, _final)]
WIDE = (-5.6, 4.1, -50, 82)
MID = (-1.0, 0.75, -14, 20)
FINAL = (-0.52, 0.47, -9.5, 13.5)
AX = (150, 110, 1710, 800)


def _geo(A, B, u):
    """Zoom from view A to view B with a constant apparent speed: spans interpolate geometrically."""
    xa, ya, xb, yb = (A[0] + A[1]) / 2, (A[2] + A[3]) / 2, (B[0] + B[1]) / 2, (B[2] + B[3]) / 2
    sxa, sya, sxb, syb = A[1] - A[0], A[3] - A[2], B[1] - B[0], B[3] - B[2]
    gx, gy = (sxb / sxa) ** u, (syb / sya) ** u
    fx, fy = (1 - gx) / (1 - sxb / sxa), (1 - gy) / (1 - syb / sya)
    xc, yc, sx, sy = lerp(xa, xb, fx), lerp(ya, yb, fy), sxa * gx, sya * gy
    return xc - sx / 2, xc + sx / 2, yc - sy / 2, yc + sy / 2


def view(t):
    return _geo(WIDE, MID, ease(seg(t, *T_Z1))) if t < T_Z2[0] else _geo(MID, FINAL, ease(seg(t, *T_Z2)))


def setup():
    plt.rcParams.update({'path.simplify': True, 'path.simplify_threshold': 0.2})
    fig = new_figure()
    _, sub = title(fig, 'Recombination and lifetime share the charge scale')
    footer(fig, 'Simulated data at known truth, endpoints smeared by 1 mm. Line and moving dots are raw per-iteration values.', right='EXP<s>C75H; surface: E_truthdiag/out/ab2dfine_*.npz')
    ax = fig.add_axes(box_px(*AX)); style_axes(ax, grid=None); ax.set_xticks([]); ax.set_yticks([])
    s = dict(fig=fig, sub=sub, ax=ax)
    s['tx'] = SoftTicks(ax, 'x', AX[2], 120, fmt_pct, zero_grid=False)
    s['ty'] = SoftTicks(ax, 'y', AX[3], 76, fmt_pct, zero_grid=False)
    fig.text((AX[0] + AX[2]) / W, 58 / H, f'{PARAM_LABEL["Ab"]} ({PARAM_NAME["Ab"]}), from truth', ha='right', fontsize=px(25), color=C.ink2)
    fig.text(60 / W, (AX[1] + AX[3] + 12) / H, f'{PARAM_LABEL["lifetime"]} ({PARAM_NAME["lifetime"]}), from truth', fontsize=px(25), color=C.ink2)
    s['lab'] = fig.text(1860 / W, 1012 / H, '', ha='right', va='baseline', fontsize=px(36), color=C.ink, fontweight='semibold')
    ax.axhline(0, color=C.truth, lw=px(2.2), ls=(0, (4, 3)), zorder=3); ax.axvline(0, color=C.truth, lw=px(2.2), ls=(0, (4, 3)), zorder=3)
    # loss surface patch, only ever visible in the final view
    lv = [0, 2, 5, 10, 20, 40, 80, 140]
    s['cf'] = ax.contourf(GA, GT, Z.T, levels=lv, cmap=LinearSegmentedColormap.from_list('v', [C.ink3, C.panel]), zorder=1, extend='neither')
    s['cl'] = ax.contour(GA, GT, Z.T, levels=lv[1:-1], colors=[C.bg], linewidths=px(1.2), zorder=1.5)
    s['box'] = ax.add_patch(plt.Rectangle((GA[0], GT[0]), GA[-1] - GA[0], GT[-1] - GT[0], fill=False, ec=C.ink2, lw=px(2), zorder=2))
    s['tag'] = ax.annotate('loss at TRUE inputs, brighter = lower\n7 × 8 grid, 100 batches', xy=(GA[0], GT[0]), xycoords='data', xytext=(-px(14), px(4)), textcoords='offset points', ha='right', va='bottom', fontsize=px(24), color=C.ink2, linespacing=1.15)
    xx = np.array(FINAL[:2]) * 1.0
    s['val'], = ax.plot(xx, MIN0[1] + VALLEY * (xx - MIN0[0]), color=C.ink, lw=px(2.6), ls=(0, (9, 5)), zorder=4)
    # run path, ensemble, end points, start, truth
    s['path'], = ax.plot([], [], color=C.model, lw=px(2.6), zorder=5, solid_joinstyle='round')
    s['cloud'], = ax.plot([], [], 'o', color=C.model, ms=px(12.5), mec=C.bg, mew=px(0.9), alpha=0.85, zorder=6, ls='none')
    s['headh'], = ax.plot([], [], 'o', color=C.model, ms=px(21), mec=C.ink, mew=px(2.5), zorder=8, ls='none')
    s['ends'], = ax.plot(EA, ET, 'o', color=C.model, ms=px(14), mec=C.bg, mew=px(1), zorder=6.5, ls='none')
    s['start'], = ax.plot([START['Ab']], [START['lifetime']], 'o', color=C.start, ms=px(20), zorder=7, ls='none')
    s['stag'] = ax.annotate('start', xy=(START['Ab'], START['lifetime']), xytext=(px(4), -px(26)), textcoords='offset points', ha='center', va='top', fontsize=px(24), color=C.ink2)
    s['ttag'] = ax.annotate('truth', xy=(0, 1), xycoords=ax.get_xaxis_transform(), xytext=(px(10), -px(8)), textcoords='offset points', ha='left', va='top', fontsize=px(24), color=C.ink2)
    s['mean'] = ax.errorbar([sa['mean']], [st['mean']], xerr=[sa['se']], yerr=[st['se']], fmt='s', color=C.ink, ms=px(15), mec=C.bg, mew=px(1.2), elinewidth=px(3), capsize=px(8), capthick=px(3), zorder=9)
    # legend block, top right: one sample, mean, valley
    s['leg'] = []
    for yf, kw, name in ((0.965, dict(marker='o', color=C.model, ms=px(14), mec=C.bg, ls='none'), 'one sample'),
                         (0.915, dict(marker='s', color=C.ink, ms=px(15), mec=C.bg, ls='none'), 'mean of 36 ± SE'),
                         (0.865, dict(color=C.ink, lw=px(2.6), ls=(0, (9, 5))), 'valley direction, extended past the box')):
        h = Line2D([0.66, 0.66] if 'marker' in kw else [0.645, 0.675], [yf, yf], transform=ax.transAxes, zorder=10, **kw); ax.add_line(h)
        tx = ax.text(0.69, yf, name, transform=ax.transAxes, ha='left', va='center', fontsize=px(24), color=C.ink2, zorder=10)
        s['leg'].append((h, tx))
    s['rt'] = ax.annotate('', xy=(0.985, 0.965), xycoords='axes fraction', ha='right', va='top', fontsize=px(28), color=C.ink2)
    return s


def _alpha_cs(cs, a):
    if hasattr(cs, 'collections'):
        for c in cs.collections:
            c.set_alpha(a)
    else:
        cs.set_alpha(a)


def draw(s, i, t):
    it = it_at(t); k = int(it)
    x0, x1, y0, y1 = view(t); ax = s['ax']
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    s['tx'].update(x0, x1); s['ty'].update(y0, y1)
    s['headh'].set_data([AB[RUN, k]], [TA[RUN, k]]); s['headh'].set_visible(k > 0)     # at iteration 0 every fit sits on the grey start marker
    s['cloud'].set_data(AB[:, k], TA[:, k]); s['cloud'].set_visible(k > 0)
    ae = fade(t, T_END[0], 1e9, T_END[1] - T_END[0])
    s['cloud'].set_alpha(0.85 * (1 - ae)); s['ends'].set_alpha(ae); s['ends'].set_visible(ae > 0)
    s['headh'].set_alpha(1 - ae)
    # the single path keeps all its iterations in the wide view, then only its most recent ones, and leaves before the end points
    tail = int(lerp(20001, 250, ease(seg(t, T_Z1[0], T_Z1[0] + 2.0))))
    j0 = max(0, k - tail)
    s['path'].set_data(AB[RUN, j0:k + 1], TA[RUN, j0:k + 1])
    s['path'].set_alpha(1 - ae)
    w = 1 - ease(seg(t, *T_Z1))
    s['stag'].set_alpha(w); s['start'].set_alpha(w)
    a = fade(t, T_SURF, 1e9, 1.0)
    _alpha_cs(s['cf'], a); _alpha_cs(s['cl'], a); s['box'].set_alpha(a); s['tag'].set_alpha(a)
    for c in (s['cf'], s['cl']):
        c.set_visible(a > 0)
    s['box'].set_visible(a > 0); s['tag'].set_visible(a > 0)
    a = fade(t, T_LINE, 1e9, 0.9)
    s['val'].set_alpha(a); s['val'].set_visible(a > 0)
    for (h, tx), q in zip(s['leg'], [fade(t, T_END[0], 1e9, 0.9), fade(t, T_MEAN, 1e9, 0.9), a]):
        h.set_alpha(q); tx.set_alpha(q)
    a = fade(t, T_MEAN, 1e9, 0.9)
    for art in [s['mean'][0]] + list(s['mean'][1]) + list(s['mean'][2]):
        art.set_alpha(a); art.set_visible(a > 0)
    rv = float(np.interp(it, RIT, RV)); ar = fade(t, 14.4, T_END[0] - 0.6, 0.6)
    s['rt'].set_text(f'correlation across the 36 fits: r = {rv:+.2f}'.replace('-', '−') if ar > 0 else ''); s['rt'].set_alpha(ar)
    s['lab'].set_text(f'iteration {k:,}')
    cap, c = caption_at(t, CAPS)
    s['sub'].set_text(cap); s['sub'].set_alpha(c)


render(args, 'a10_parameter_path', DUR, setup, draw,
       beats=[(0, 'charge is shared'), (13.6, 'iteration 1000: one path, 36 dots'), (25.3, 'iteration 14000: the cloud has tilted'),
              (33.6, 'end points, zoomed'), (37.0, 'loss at true inputs'), (40.5, 'valley direction'), (44.0, 'end points against the valley')])
