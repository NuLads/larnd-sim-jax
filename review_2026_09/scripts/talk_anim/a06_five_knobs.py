"""a06: five knobs. Each calibration parameter is swept alone on the real event (true segments, other four at truth), with the
production simulation and the production point-process loss. Left: what the parameter physically does, data fixed, model moving.
Right: the loss of this event against that parameter. Inputs come from a06_sweep.py via a06_obs.py.
Only the dots are evaluations. The marker stops at each dot that has a jax.grad value, and the tangent appears only there, anchored on the dot.
Between dots the model curve (linear) and the loss curve (a06_curve: Hermite through values and gradients) are interpolated for display, and the
footer says so. Because the loss curve uses the gradients, a tangent on it proves nothing: the finite-difference numbers are the independent test."""
from common import *
from a06_curve import Curve
from matplotlib.patches import Rectangle

args = parse_args()
O = dict(np.load(f'{DATA}/a06_obs.npz', allow_pickle=True))   # eager: render workers fork, and a lazy npz handle cannot be shared
SEQ = ['Ab', 'lifetime', 'eField', 'tran_diff', 'long_diff']
LOGAX = dict(Ab=False, lifetime=True, eField=False, tran_diff=True, long_diff=True)
T0, TAIL = 2.6, 6.4
TRV, DWL, LEG, HOLD0, HOLD1 = 0.40, 0.45, 1.0, 1.0, 0.3        # s: travel between dots, dwell at a dot, long legs, truth holds
FOOT2 = ('Dots are the evaluated points. Curves and motion between dots are interpolated through the values and gradients, for display only. '
         'The gradient is tested separately by finite differences.')


def w_of(p, v):
    return np.log(v) if LOGAX[p] else np.asarray(v, float)


def v_of(p, w):
    return np.exp(w) if LOGAX[p] else w


def schedule(p):
    """Stops (start, end, node) of a parameter segment: truth, down to the low end, up through every point that has a gradient, back to truth."""
    n = len(O[p + '_vals']); i0 = int(np.argmin(np.abs(O[p + '_vals'] / float(O[p + '_true']) - 1)))
    stops = [(-1.0, HOLD0, i0), (HOLD0 + LEG, HOLD0 + LEG + DWL, 0)]
    t = stops[-1][1]
    for k in range(1, n):
        stops.append((t + TRV, t + TRV + DWL, k)); t = stops[-1][1]
    stops.append((t + LEG, t + LEG + HOLD1, i0))
    return stops


STOPS = {p: schedule(p) for p in SEQ}
DPS = [STOPS[p][-1][1] for p in SEQ]
STARTS = T0 + np.r_[0, np.cumsum(DPS)]
TE = float(STARTS[-1])
DUR = TE + TAIL
BEATS = [(0.0, 'data fixed, model moves')] + [(float(STARTS[k]) + 0.6, PARAM_NAME[p]) for k, p in enumerate(SEQ)] + [(TE + 0.9, 'gradient predicts the response')]


def sweep_at(p, u, wg):
    """Marker position, and the tangent's node and opacity. The tangent is anchored on the dot and only exists around a stop."""
    st = STOPS[p]
    for j, (s, e, k) in enumerate(st):
        if s <= u <= e:
            w = wg[k]; break
        if j + 1 < len(st) and e < u < st[j + 1][0]:
            w = lerp(wg[k], wg[st[j + 1][2]], ease(seg(u, e, st[j + 1][0]))); break
    else:
        w = wg[st[-1][2]]
    best, node = 0.0, st[0][2]
    for j, (s, e, k) in enumerate(st):
        a = float(ease(seg(u, s - 0.12, s + 0.04))) * (1.0 if j == len(st) - 1 else 1 - float(ease(seg(u, e - 0.04, e + 0.12))))
        if a > best: best, node = a, k
    return float(w), node, best


def X_of(p, w):
    """Display coordinate of the loss panels: ln(value/truth) for the log-swept parameters, % change otherwise. Linear in w, so a tangent stays a tangent."""
    tv = float(O[p + '_true'])
    return w - np.log(tv) if LOGAX[p] else 100 * (w / tv - 1)


def x_axis(ax, p, small=False):
    if LOGAX[p]:
        lo, hi = float(O[p + '_lo']) / float(O[p + '_true']), float(O[p + '_hi']) / float(O[p + '_true'])
        fs = [f for f in (0.2, 0.25, 0.3, 0.5, 1, 2, 2.5, 4, 5) if lo * 0.999 <= f <= hi * 1.001]
        if len(fs) > 3: fs = [fs[0], 1, fs[-1]] if small else [f for f in fs if f in (fs[0], fs[-1], 0.5, 1, 2)]
        ax.set_xticks([np.log(f) for f in fs]); ax.set_xticklabels([('×' + (f'{f:g}')) for f in fs])
        ax.set_xlabel('value relative to truth', fontsize=px(22) if small else None)
    else:
        ax.set_xlabel('change from truth (%)', fontsize=px(22) if small else None)


def curve_at(wg, Y, w):
    i = int(np.clip(np.searchsorted(wg, w) - 1, 0, len(wg) - 2)); s = float(clamp((w - wg[i]) / (wg[i + 1] - wg[i])))
    return Y[i] * (1 - s) + Y[i + 1] * s


def dots(ax, p, d, ms):
    """Evaluated points: filled where a gradient exists too, open where only the value was evaluated."""
    ax.plot(X_of(p, d['wg']), d['L'] - d['L0'], 'o', ms=px(ms), color=C[p], mew=0, zorder=3)
    v = ~d['c'].has
    if v.any(): ax.plot(X_of(p, d['wn'][v]), d['Ln'][v] - d['L0'], 'o', ms=px(ms), mfc=C.bg, mec=C[p], mew=px(2), zorder=3)


def setup():
    fig = new_figure()
    tt, sub = title(fig, 'Turn one knob: the event responds smoothly', 'Data stay fixed. The model moves.')
    footer(fig, 'Simulated data: noisy simulation at known truth. One event, true track segments, four parameters held at truth.', right='production simulation and loss, sample 8 batch 35 event 2')
    fig.text(60 / W, 56 / H, FOOT2, fontsize=px(22), color=C.ink2, va='baseline')
    S = dict(fig=fig, tt=tt, sub=sub, P={})
    for p in SEQ:
        d = {}
        a = fig.add_axes(box_px(130, 150, 960, 730)); style_axes(a)
        b = fig.add_axes(box_px(1230, 150, 630, 730)); style_axes(b)
        a.set_xlabel(str(O[p + '_xlabel'])); a.set_ylabel(str(O[p + '_ylabel'])); a.set_xlim(*O[p + '_xlim']); a.set_ylim(*O[p + '_ylim'])
        d['data'], = a.plot(O[p + '_xd'], O[p + '_yd'], 'o', ms=px(float(O[p + '_ms'])), color=C.data, mew=0, zorder=3)
        d['model'], = a.plot(O[p + '_xm'], O[p + '_ym'][0], '-', color=C.model, lw=px(4), zorder=2)
        lx, ly = tuple(O[p + '_ldpos'])
        a.plot([lx], [ly], 'o', ms=px(15), color=C.data, mew=0, transform=a.transAxes, clip_on=False); a.text(lx + 0.022, ly, 'data', color=C.ink, fontsize=px(26), transform=a.transAxes, va='center')
        a.plot([lx + 0.12, lx + 0.17], [ly, ly], '-', color=C.model, lw=px(4), transform=a.transAxes, clip_on=False); a.text(lx + 0.182, ly, 'model', color=C.ink, fontsize=px(26), transform=a.transAxes, va='center')
        wg = w_of(p, O[p + '_vals']); L, G = O[p + '_nll'], O[p + '_grad']
        dW = G * (O[p + '_vals'] if LOGAX[p] else 1.0)           # dL/dw from the real jax.grad values
        wn, Ln = w_of(p, O[p + '_xv']), O[p + '_xL']; cv = Curve(wn, Ln, wg, dW)
        wf = np.linspace(wg[0], wg[-1], 480); Lf = cv(wf); L0 = float(O[p + '_L0'])
        d.update(wg=wg, L=L, dW=dW, L0=L0, wn=wn, Ln=Ln, xr=X_of(p, wf), Lf=Lf - L0, c=cv)
        b.plot(d['xr'], Lf - L0, color=C[p], lw=px(4)); dots(b, p, d, 12)
        b.axvline(0, color=C.truth, ls=(0, (4, 4)), lw=px(2)); x_axis(b, p); b.set_ylabel('event loss, relative to truth')
        b.set_xlim(d['xr'][0], d['xr'][-1]); yl = np.max(L - L0); b.set_ylim(min(-0.08 * yl, Lf.min() - L0 - 0.04 * yl), 1.12 * yl)
        d['tan'], = b.plot([], [], '-', color=C.ink, lw=px(3), zorder=4); d['dot'], = b.plot([], [], 'o', ms=px(17), color=C[p], mec=C.ink, mew=px(2.5), zorder=5, clip_on=False)
        d['truth'] = b.text(0, 1.0, ' truth', transform=b.get_xaxis_transform(), color=C.ink2, fontsize=px(24), va='top')
        d['read'] = fig.text(1230 / W, 905 / H, '', color=C.ink, fontsize=px(28))
        d['cap'] = fig.text(130 / W, 905 / H, str(O[p + '_cap']), color=C.ink, fontsize=px(28))
        d.update(a=a, b=b); S['P'][p] = d
    # closing frame: all five loss curves, each with its tangent at the evaluated point that was checked by finite differences
    S['end'] = []
    bw = 320; gap = 45; x0 = 80
    for k, p in enumerate(SEQ):
        d = S['P'][p]; e = fig.add_axes(box_px(x0 + k * (bw + gap), 330, bw, 480)); style_axes(e, grid=None)
        e.plot(d['xr'], d['Lf'], color=C[p], lw=px(4)); dots(e, p, d, 10); e.axvline(0, color=C.truth, ls=(0, (4, 4)), lw=px(2))
        e.set_xlim(d['xr'][0], d['xr'][-1]); yl = np.max(d['L'] - d['L0']); e.set_ylim(min(-0.08 * yl, d['Lf'].min() - 0.04 * yl), 1.12 * yl); x_axis(e, p, small=True); e.tick_params(labelsize=px(22)); e.set_yticks([0, yl]); e.set_yticklabels(['0', f'{yl:,.0f}'.replace(',', ' ')])
        n = int(np.argmin(np.abs(O[p + '_vals'] / float(O[p + '_true']) - float(O[p + '_endc'])))); wc = d['wg'][n]; yc = d['L'][n] - d['L0']; g = d['dW'][n]
        dw = 0.22 * (d['wg'][-1] - d['wg'][0]); ws = np.array([wc - dw, wc + dw]); xs = X_of(p, ws)
        e.plot(xs, yc + g * (ws - wc), color=C.ink, lw=px(3), clip_on=True); e.plot([X_of(p, wc)], [yc], 'o', ms=px(15), color=C[p], mec=C.ink, mew=px(2.5))
        h = fig.text((x0 + k * (bw + gap)) / W, 830 / H, f"{PARAM_LABEL[p]}  {PARAM_NAME[p]}", color=C.ink, fontsize=px(26))
        S['end'] += [(e, h)]
    S['veil'] = fig.add_artist(Rectangle((0, 80 / H), 1, 860 / H, transform=fig.transFigure, color=C.bg, lw=0, zorder=50, alpha=0.0))
    S['endcap'] = fig.text(960 / W, 190 / H, str(O['end_caption']), color=C.ink, fontsize=px(34), ha='center')
    S['endsub'] = fig.text(960 / W, 140 / H, str(O['end_sub']), color=C.ink2, fontsize=px(25), ha='center')
    return S


def show(d, on):
    for k in ('a', 'b', 'read', 'cap'):
        d[k].set_visible(on)


def draw(s, i, t):
    k = int(np.searchsorted(STARTS, t, side='right') - 1) if T0 <= t < TE else (-1 if t < T0 else len(SEQ))
    s['veil'].set_alpha(float(ease(clamp(1 - min(abs(t - b) for b in STARTS[1:]) / 0.3))))   # panel changes happen behind a brief dip to the background
    active = SEQ[0] if t < T0 else (SEQ[k] if k < len(SEQ) else None)
    for p, d in s['P'].items():
        show(d, p == active)
    endon = active is None
    for e, h in s['end']:
        e.set_visible(endon); h.set_visible(endon)
    s['endcap'].set_visible(endon); s['endsub'].set_visible(endon)
    if endon:
        a = float(ease(seg(t, TE, TE + 0.7)))
        for e, h in s['end']:
            for art in list(e.lines) + [e.xaxis.label] + list(e.spines.values()) + e.get_xticklabels(): art.set_alpha(a)
            h.set_alpha(a)
        s['endcap'].set_alpha(float(ease(seg(t, TE + 0.7, TE + 1.3)))); s['endsub'].set_alpha(float(ease(seg(t, TE + 1.2, TE + 1.8))))
        s['tt'].set_text('The gradient predicts the response'); s['sub'].set_text('Autodiff through the full simulation, checked against finite differences')
        return
    p = active; d = s['P'][p]; u = 0.0 if t < T0 else t - float(STARTS[k])
    s['tt'].set_text('Turn one knob: the event responds smoothly')
    s['sub'].set_text('Data stay fixed. The model moves.' if t < T0 else f'{PARAM_LABEL[p]}  {PARAM_NAME[p]}')
    w, node, a = sweep_at(p, u, d['wg']); v = float(v_of(p, w)); tv = float(O[p + '_true'])
    d['model'].set_ydata(curve_at(d['wg'], O[p + '_ym'], w))
    d['dot'].set_data([X_of(p, w)], [float(d['c'](w)[0]) - d['L0']])
    wc, yc, g = d['wg'][node], d['L'][node] - d['L0'], d['dW'][node]             # tangent: real value and real jax.grad at the evaluated point
    dw = 0.16 * (d['wg'][-1] - d['wg'][0]); ws = np.array([wc - dw, wc + dw])
    d['tan'].set_data(X_of(p, ws), yc + g * (ws - wc)); d['tan'].set_alpha(a); d['tan'].set_visible(a > 0.003)
    pc = round(100 * (v / tv - 1), 1)
    rel = f'×{v / tv:.2f}' if LOGAX[p] else ('0.0%' if pc == 0 else f'{pc:+.1f}%'.replace('-', '−'))
    d['read'].set_text(f"{PARAM_LABEL[p]} = {str(O[p + '_fmt']) % (v * float(O[p + '_scale']))} {str(O[p + '_unit'])}   ({rel})"); d['read'].set_alpha(float(lerp(0.55, 1.0, a)))


if __name__ == '__main__':
    render(args, 'a06_five_knobs', duration=DUR, setup=setup, draw=draw, beats=BEATS)
