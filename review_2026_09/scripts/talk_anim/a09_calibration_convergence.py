"""a09: the 36 realistic-end production fits draw themselves over 20000 iterations, as % from truth per parameter, plus the loss."""
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D
import common
from traj_helpers import *

args = parse_args(lambda ap: ap.add_argument('--crf', type=int, default=None, help='x264 CRF of the --final encode (common.py default 17)'))
if args.crf is not None:
    _enc0 = common._enc_cmd
    common._enc_cmd = lambda w, h, fps, out, crf: _enc0(w, h, fps, out, args.crf)
ST = [(0, 7500, 'main fit: coarse geometry'), (7500, 10000, 'all modes'), (10000, 14000, 'freeze released'), (14000, 20000, 'fast continuation')]
KNOTS = [(0, 0), (3, 0), (12, 1000), (13.2, 1000), (21, 7500), (22.2, 7500), (26.2, 10000), (27.4, 10000), (31.6, 14000), (32.8, 14000), (38, 20000), (42.5, 20000)]
DUR = 42.5
it_at = time_map(KNOTS)
CAPS = [(0, 3.1, 'All 36 samples start from the same wrong guess, and share one truth'),
        (3.1, 12.6, 'First 1000 iterations: E and $A_b$ arrive early, lifetime swings hard'),
        (12.6, 21.8, 'Until 7500: coarse geometry, only the lower spline modes are free'),
        (21.8, 27.0, 'From 7500: all spline modes free; calibration learning rate nearly gone'),
        (27.0, 32.4, '10000: learning rate re-heated, drift freeze of straight tracks released'),
        (32.4, 38.4, '14000: larger re-heat, the $D_T$ and $D_L$ spreads tighten'),
        (38.4, 99, 'Axes zoomed in: each panel now spans only the final spread')]

RAW = {p: errs(p) for p in PARAMS}
E = {p: widening_mean(RAW[p]) for p in PARAMS}        # lines: raw to 1000, then a trailing mean widening to 100 iterations by 3000
MED = {p: np.median(E[p], 0) for p in PARAMS}
PY = {p: Pyramid(E[p]) for p in PARAMS}


def _binmean(y, levels=8):
    """Median line per level: bin averages, not min/max. Bold min/max decimation of a jittery median draws a solid block once zoomed."""
    out = []
    for k in range(levels):
        b = 2 ** k; n = len(y) // b
        out.append((np.arange(n) * b + (b - 1) / 2.0, y[:n * b].reshape(n, b).mean(1)))
    return out


PM = {p: _binmean(MED[p]) for p in PARAMS}
BAND = {p: EnvPyramid(RAW[p].min(0), RAW[p].max(0)) for p in PARAMS}   # raw per-iteration min and max over the 36 fits
LC, LX = loss_cycles(); LC = LC / 1e3; LMED = np.median(LC, 0)
IT_LR, LR = calib_lr()
PANEL_W, PANEL_H = 490, 316
COL = [120, 740, 1360]; ROW = [482, 114]


def _lims(smin, smax, its, cyc=1):
    tg = np.arange(0, DUR + 1 / 60, 1 / 60)
    I = np.array([it_at(t) for t in tg]) / cyc
    w = np.clip((I * 0.9 - 40 / cyc).astype(int), 0, len(smin) - 1)
    lo, hi = np.minimum(smin[w], 0), np.maximum(smax[w], 0)
    sp = hi - lo; lo = lo - 0.1 * sp; hi = hi + 0.1 * sp
    return tg, lo, hi


def _smooth(lo, hi, sig=48):
    L = -np.exp(gaussian_filter1d(np.log(np.maximum(-lo, 1e-4)), sig, mode='nearest'))
    U = np.exp(gaussian_filter1d(np.log(np.maximum(hi, 1e-4)), sig, mode='nearest'))
    return L, U


LIM = {}
for p in PARAMS:
    smin = np.minimum.accumulate(E[p].min(0)[::-1])[::-1]; smax = np.maximum.accumulate(E[p].max(0)[::-1])[::-1]
    tg, lo, hi = _lims(smin, smax, None)
    LIM[p] = _smooth(lo, hi)
_mn = np.minimum.accumulate(LC.min(0)[::-1])[::-1]; _mx = np.maximum.accumulate(LC.max(0)[::-1])[::-1]
_I = np.array([it_at(t) for t in tg])
_w = np.clip((_I * 0.8 / 100).astype(int), 0, 199)
_l, _u = _mn[_w], _mx[_w]; _sp = _u - _l
LIM['loss'] = (gaussian_filter1d(_l - 0.1 * _sp, 48, mode='nearest'), gaussian_filter1d(_u + 0.1 * _sp, 48, mode='nearest'))
# Once a panel zooms in, the dense early lines are clipped into a solid block. A veil in the panel colour, above the lines and band
# but below the median, truth and ticks, dims them to a quiet texture. Strength follows that panel's zoom relative to its wide view
# at iteration 1000 (the t = 12 s knot), so the first section is untouched; it tapers out by iteration 5000 and stays behind the head.
VEIL_MAX = 0.88
SPAN0 = {p: float(np.interp(12.0, tg, LIM[p][1]) - np.interp(12.0, tg, LIM[p][0])) for p in PARAMS}


def setup():
    plt.rcParams.update({'path.simplify': True, 'path.simplify_threshold': 0.2})
    fig = new_figure()
    ta, sub = title(fig, 'Fast first, slow to settle: 36 fits from one wrong start')
    footer(fig, 'Simulated data, known truth, ends smeared 1 mm. Lines: raw to iteration 1000, then running mean up to 100. Once axes zoom, early lines dim, median smooths. Band: raw min to max.',
           right='EXP<s>C75H, s = 1\u201312, 17\u201340')
    s = dict(fig=fig, sub=sub, ax={}, tk={}, tx={}, ln={}, md={})
    # stage strip with the real calibration learning rate
    lr = fig.add_axes(box_px(COL[0], 846, COL[2] + PANEL_W - COL[0], 56))
    for sp in lr.spines.values():
        sp.set_visible(False)
    lr.set_xlim(0, 20000); lr.set_ylim(-5.45, -0.85); lr.set_xticks([]); lr.set_yticks([])
    y = np.log10(np.maximum(LR, 1e-9))
    for k, (a, b, name) in enumerate(ST):
        lr.axvspan(a, b, color=C.panel if k % 2 == 0 else C.bg, lw=0, zorder=0)
        lr.annotate(name, xy=((a + b) / 2, 1), xycoords=lr.get_xaxis_transform(), xytext=(0, px(6)), textcoords='offset points', ha='center', va='bottom', fontsize=px(23), color=C.ink2)
        if a:
            lr.axvline(a, color=C.spine, lw=px(1.5))
    lr.plot(IT_LR[1:], y[1:], color=C.ink3, lw=px(1.8), alpha=0.75, zorder=2)
    s['lrd'], = lr.plot([], [], color=C.ink, lw=px(2.6), zorder=3)
    s['head'] = lr.axvline(0, color=C.ink, lw=px(2.5), zorder=4)
    lr.text(600, -4.42, 'calibration learning rate (log scale)', ha='left', va='center', fontsize=px(22), color=C.ink3)
    fig.text((COL[0] - 8) / W, 893 / H, '0.1', ha='right', va='center', fontsize=px(22), color=C.ink3)
    fig.text((COL[0] - 8) / W, 852 / H, r'$10^{-5}$', ha='right', va='center', fontsize=px(22), color=C.ink3)
    s['lab'] = fig.text(1860 / W, 1012 / H, '', ha='right', va='baseline', fontsize=px(36), color=C.ink, fontweight='semibold')
    # legend (right of the caption line)
    lx = [1468, 1610, 1760]
    for x, c, ls, lw, a, name in [(lx[0], C.truth, (0, (4, 3)), 2.5, 1, 'truth'), (lx[1], C.ink2, '-', 4.5, 1, 'median'), (lx[2], C.ink2, '-', 1.3, 0.9, '36 fits')]:
        fig.add_artist(Line2D([x / W, (x + 34) / W], [975 / H] * 2, color=c, ls=ls, lw=px(lw), alpha=a))
        fig.text((x + 42) / W, 975 / H, name, va='center', fontsize=px(23), color=C.ink2)
    keys = PARAMS + ['loss']
    for k, p in enumerate(keys):
        cx, cy = COL[k % 3], ROW[k // 3]
        ax = fig.add_axes(box_px(cx, cy, PANEL_W, PANEL_H)); style_axes(ax, grid=None); ax.set_xticks([]); ax.set_yticks([])
        s['ax'][p] = ax
        if p == 'loss':
            s['tk'][p] = SoftTicks(ax, 'y', PANEL_H, 58, lambda v: f'{v:g}k')
            col = C.ink2
            ln = [ax.plot([], [], color=col, lw=px(1.3), alpha=0.32, zorder=2)[0] for _ in range(36)]
            fig.text(cx / W, (cy + PANEL_H + 12) / H, 'Loss per batch', fontsize=px(26), color=C.ink, va='baseline')
            fig.text((cx + PANEL_W) / W, (cy + PANEL_H + 12) / H, '100-batch means', fontsize=px(22), color=C.ink3, va='baseline', ha='right')
        else:
            s['tk'][p] = SoftTicks(ax, 'y', PANEL_H, 58, fmt_pct, zero_grid=False)
            col = C[p]
            ax.axhline(0, color=C.truth, lw=px(2.2), ls=(0, (4, 3)), zorder=5)
            ln = [ax.plot([], [], color=col, lw=px(1.3), alpha=0.5, zorder=2)[0] for _ in range(36)]
            s.setdefault('band', {})[p] = ax.fill([], [], color=col, alpha=0.13, lw=0, zorder=1.2)[0]
            fig.text(cx / W, (cy + PANEL_H + 12) / H, f'{PARAM_LABEL[p]}  {PARAM_NAME[p]}', fontsize=px(26), color=C.ink, va='baseline')
            fig.text((cx + PANEL_W) / W, (cy + PANEL_H + 12) / H, 'start ' + fmt_pct(round(START[p], 1)), fontsize=px(22), color=C.ink3, va='baseline', ha='right')
            s.setdefault('sd', {})[p] = ax.plot([0], [START[p]], 'o', color=C.start, ms=px(13), zorder=7, clip_on=False)[0]
            u = np.linspace(0, 1, 256); g = np.ones((1, 256, 4)); g[..., :3] = to_rgb(C.panel); g[0, :, 3] = 1 - ease(clamp((u - 0.5) / 0.5))
            s.setdefault('veil', {})[p] = ax.imshow(g, extent=(0, 5000, 0, 1), aspect='auto', origin='lower', interpolation='bilinear', zorder=3)
        md, = ax.plot([], [], color=col, lw=px(4.5), zorder=4, path_effects=None)
        halo, = ax.plot([], [], color=C.bg, lw=px(7), zorder=3.5, alpha=0.85)
        for j, (a, b, _) in enumerate(ST):
            if j % 2 == 0:
                ax.axvspan(a, b, color=C.panel, lw=0, zorder=0.2)
        for v in (7500, 10000, 14000):
            ax.axvline(v, color=C.spine, lw=px(1.5), zorder=1)
        s['ln'][p] = ln; s['md'][p] = (md, halo)
        s['tk'][p + 'x'] = SoftTicks(ax, 'x', PANEL_W, 112, lambda v: f'{int(v):,}', labels=(k // 3 == 1), zero_grid=False)
    fig.text(1860 / W, 54 / H, 'iteration', ha='right', va='baseline', fontsize=px(24), color=C.ink2)
    return s


def draw(s, i, t):
    it = it_at(t)
    xm = max(700.0, it * (1 + 0.14 * (1 - it / 20000)))
    xm = min(xm, 20000.0)
    bins = xm / (PANEL_W * 2) * 1.6
    for p in PARAMS + ['loss']:
        ax = s['ax'][p]
        lo, hi = np.interp(t, tg, LIM[p][0]), np.interp(t, tg, LIM[p][1])
        ax.set_xlim(0, xm); ax.set_ylim(lo, hi)
        s['tk'][p].update(lo, hi); s['tk'][p + 'x'].update(0, xm)
        if p != 'loss':       # unclipped so it is not cut by the axis at x = 0, so it must leave as soon as the zoom passes the start value
            q = min(START[p] - lo, hi - START[p]) / (hi - lo)
            s['sd'][p].set_visible(q > 0); s['sd'][p].set_alpha(float(clamp(q / 0.03)))
            m = float(ease(seg(np.log(SPAN0[p] / (hi - lo)), np.log(1 / 0.8), np.log(1 / 0.35)))) * float(ease(seg(it, 1500, 3200)))   # 0 = wide, 1 = fully zoomed
            v = VEIL_MAX * m
            vl = s['veil'][p]; vl.set_visible(v > 0.003)
            if v > 0.003:
                vl.set_extent((0, min(5000.0, max(1500.0, it - 150)), lo, hi)); vl.set_alpha(v)
            ax.set_xlim(0, xm); ax.set_ylim(lo, hi)       # set_extent re-autoscales: restore the frame's limits
        md, halo = s['md'][p]
        if p == 'loss':
            n = int(it // 100)
            for r, l in enumerate(s['ln'][p]):
                l.set_data(LX[:n], LC[r, :n])
            md.set_data(LX[:n], LMED[:n]); halo.set_data(LX[:n], LMED[:n])
        else:
            s['band'][p].set_xy(BAND[p].poly(bins, it))
            X, Y = PY[p].level(bins); n = int(np.searchsorted(X, it, side='right'))
            for r, l in enumerate(s['ln'][p]):
                l.set_data(X[:n], Y[r, :n])
            X, Y = PM[p][int(np.clip(np.floor(np.log2(max(bins * (1 + 5 * m), 1.0))), 0, len(PM[p]) - 1))]
            n = int(np.searchsorted(X, it, side='right'))
            X, Y = np.r_[X[:n], it], np.r_[Y[:n], MED[p][int(it)]]        # the line always reaches the current iteration
            md.set_data(X, Y); halo.set_data(X, Y)
    n = int(it) + 1
    s['lrd'].set_data(IT_LR[1:max(n, 2)], np.log10(np.maximum(LR, 1e-9))[1:max(n, 2)])
    s['head'].set_xdata([it, it])
    s['lab'].set_text(f'iteration {int(it):,}')
    cap, a = caption_at(t, CAPS)
    s['sub'].set_text(cap); s['sub'].set_alpha(a)


render(args, 'a09_calibration_convergence', DUR, setup, draw,
       beats=[(0, 'same wrong start, 36 samples'), (12.2, 'iteration 1000: first fast phase'), (21.2, 'iteration 7500: all spline modes free next'),
              (26.4, 'iteration 10000: main fit done'), (31.8, 'iteration 14000: freeze-release stage done'), (39.0, 'iteration 20000: final spread')])
