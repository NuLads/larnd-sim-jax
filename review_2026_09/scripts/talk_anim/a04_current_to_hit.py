"""a04: how a self-triggering pixel turns current into a hit. One real pixel of a real muon batch, swept in time, through the close double hit.
Current and charge: production simulation (fig3 cache). Noise: throw 0 of the 10,000 cached throws, recovered by a04_replay.npz (bitwise replay).
Nothing is schematic except the cursor and the moving dot that carries the sample down to the hit list."""
from common import *
from fee_pair import *
from matplotlib.patches import Polygon, Rectangle
args = parse_args()

M = 0                                    # first cached throw: no selection
q, q0, HITS = throw(M); check(M)
h0, h1 = HITS[0], HITS[1]
NAME = 'a04_current_to_hit'
DUR = 42.0
BOXA = (130, 712, 1730, 200); BOXB = (130, 322, 1730, 362); BOXC = (130, 122, 1730, 168)
YA = (-4.5, 17); YB = (-2.2, 14.5); YC = (73, 97)
XL = (T0 * DT, T1 * DT)
c0, c1 = h0['cross'], h1['cross']; e0, e1 = h0['end'], h1['end']; r0, r1 = h0['reset'], h1['reset']
# cursor timeline: (t start, t end, tick start, tick end)
TL = [(2.2, 10.5, T0, c0), (13.0, 16.2, c0, e0), (19.2, 19.5, e0, r0), (22.0, 24.6, r0, c1), (26.6, 28.8, c1, e1), (31.6, 31.9, e1, r1), (33.2, 35.6, r1, T1)]
Tcross0 = 10.5; Tsample0 = 16.2; Treset0 = 19.5; Tcross1 = 24.6; Tsample1 = 28.8; Treset1 = 31.9; Tend = 35.6
CAPS = [(0.0, 2.2, 'One pixel as a muon track passes'),
        (2.4, 6.0, 'Drifting charge induces a current on the pixel'),
        (6.2, 10.4, 'The chip integrates it: accumulated charge climbs'),
        (10.6, 12.9, 'Threshold crossed: this tick becomes the hit time'),
        (13.1, 16.1, 'The chip holds for 1.8 µs while charge keeps arriving'),
        (16.3, 19.1, 'End of hold: the charge is digitised, with its own noise'),
        (19.6, 21.9, 'Reset: the integrator restarts from a new random baseline'),
        (22.1, 26.5, 'Current is still flowing, so the pixel fires again'),
        (26.7, 31.4, 'Same steps: hold, digitise'),
        (32.0, 35.4, 'Reset again: nothing left to cross threshold'),
        (35.7, 99, 'A hit is two numbers: a time and a charge')]


def cursor(t):
    c = T0
    for a, b, u, v in TL:
        if t >= b:
            c = v
        elif t >= a:
            return lerp(u, v, float(ease(seg(t, a, b)))) if b - a > 0.5 else lerp(u, v, seg(t, a, b))
    return c


def to_px(box, xl, yl, x, y):
    return box[0] + (x - xl[0]) / (xl[1] - xl[0]) * box[2], box[1] + (y - yl[0]) / (yl[1] - yl[0]) * box[3]


def trace_upto(c):
    """Chip state up to fractional tick c, with the reset drops drawn as vertical jumps."""
    n = int(np.floor(c)); ii = np.arange(T0, n + 1)
    x = ii.astype(float); y = q[ii] / 1e3; y0 = q0[ii] / 1e3
    if c > n and n + 1 < NT and (n + 1) not in (r0, r1):
        f = c - n
        x = np.append(x, c); y = np.append(y, lerp(q[n], q[n + 1], f) / 1e3); y0 = np.append(y0, lerp(q0[n], q0[n + 1], f) / 1e3)
    return x * DT, y, y0


def setup():
    fig = new_figure()
    title(fig, 'A self-triggering pixel turns current into hits')
    sub = fig.text(60 / W, 958 / H, '', fontsize=px(30), color=C.ink2, va='baseline')
    footer(fig, 'Simulation: one real pixel of a muon batch, production electronics. Noise: throw 0 of 10,000. Same pixel fires again 125 µs later (other track, not shown).',
           right='fig3_probabilistic_method cache')
    A = fig.add_axes(box_px(*BOXA)); B = fig.add_axes(box_px(*BOXB)); Cc = fig.add_axes(box_px(*BOXC))
    for ax, yl in ((A, YA), (B, YB), (Cc, YC)):
        style_axes(ax, 'y'); ax.set_xlim(*XL); ax.set_ylim(*yl)
    A.tick_params(labelbottom=False); B.tick_params(labelbottom=False)
    A.set_yticks([0, 8, 16]); B.set_yticks([0, 5, 10]); Cc.set_yticks([76, 86, 96])
    A.set_ylabel('current\n(ke⁻/µs)'); B.set_ylabel('charge on integrator (ke⁻)'); Cc.set_ylabel('ADC'); Cc.set_xlabel('time (µs)')
    Cc.set_xticks(np.arange(24, 36, 2)); A.set_xticks(np.arange(24, 36, 2)); B.set_xticks(np.arange(24, 36, 2))
    fig.align_ylabels([A, B, Cc])
    A.axhline(0, color=C.spine, lw=px(1.5), zorder=1)
    # current: area since the last reset is what the integrator currently holds
    tt = np.arange(T0, T1 + 1)
    pold = Polygon(np.zeros((3, 2)), closed=True, fc=C.ink3, ec='none', alpha=0.28, zorder=2); A.add_patch(pold)
    pnow = Polygon(np.zeros((3, 2)), closed=True, fc=C.ink2, ec='none', alpha=0.75, zorder=3); A.add_patch(pnow)
    lI, = A.plot([], [], color=C.ink, lw=px(3), zorder=4)
    # threshold + hold + charge
    B.axhline(VTH / 1e3, color=C.ink2, lw=px(2.5), zorder=2)
    thr = B.text(XL[0] + 0.12, VTH / 1e3 + 0.35, 'threshold 5 ke⁻', fontsize=px(24), color=C.ink2, va='bottom', ha='left')
    B.axhline(0, color=C.spine, lw=px(1.5), zorder=1)
    span = [Rectangle((0, YB[0]), 0, YB[1] - YB[0], fc=C.ink3, ec='none', alpha=0.20, zorder=0) for _ in range(2)]
    for r_ in span:
        B.add_patch(r_)
    hold_txt = [B.text(0, YB[0] + 0.3, 'hold 1.8 µs', fontsize=px(24), color=C.ink2, va='bottom', ha='center', alpha=0) for _ in range(2)]
    lq0, = B.plot([], [], color=C.truth, lw=px(2.5), ls=(0, (3, 2.2)), zorder=3)
    lq, = B.plot([], [], color=C.data, lw=px(4.5), zorder=4)
    head, = B.plot([], [], 'o', color=C.data, ms=px(13), zorder=6)
    ring = [B.plot([], [], 'o', mfc='none', mec=C.ink, mew=px(3), ms=px(24), zorder=7)[0] for _ in range(2)]
    samp = [B.plot([], [], 'o', color=C.data, mec=C.ink, mew=px(2.5), ms=px(19), zorder=7)[0] for _ in range(2)]
    pre = [B.plot([], [], 'o', mfc='none', mec=C.data, mew=px(2.5), ms=px(15), zorder=6)[0] for _ in range(2)]
    stxt = [B.text(0, 0, '', fontsize=px(24), color=C.ink, va='center', ha='left', alpha=0, zorder=8) for _ in range(2)]
    rtxt = [B.text(0, 0, '', fontsize=px(24), color=C.ink, va='center', ha='left', alpha=0, zorder=8) for _ in range(2)]
    # small key, upper left of the charge panel: stays clear of the curve
    k1, = B.plot([XL[0] + 0.12, XL[0] + 0.6], [13.15, 13.15], color=C.data, lw=px(4.5))
    k2, = B.plot([XL[0] + 0.12, XL[0] + 0.6], [11.75, 11.75], color=C.truth, lw=px(2.5), ls=(0, (3, 2.2)))
    keyt = B.text(XL[0] + 0.72, 13.15, '', fontsize=px(24), color=C.ink, va='center')
    B.text(XL[0] + 0.72, 11.75, 'same charge, no noise', fontsize=px(24), color=C.ink, va='center')
    # hits
    Cc.axhline(PED, color=C.spine, lw=px(1.5), zorder=1)
    Cc.text(XL[0] + 0.12, PED + 1.1, 'pedestal, no charge', fontsize=px(22), color=C.ink3, va='bottom')
    stem = [Cc.plot([], [], color=C.data, lw=px(4), zorder=3)[0] for _ in range(2)]
    mark = [Cc.plot([], [], 'o', color=C.data, mec=C.ink, mew=px(2.5), ms=px(22), zorder=4)[0] for _ in range(2)]
    htxt = [Cc.text(0, 0, '', fontsize=px(26), color=C.ink, va='center', ha='left', alpha=0, zorder=5) for _ in range(2)]
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(0, H); ov.axis('off'); ov.patch.set_alpha(0)
    vcur, = ov.plot([], [], color=C.ink2, lw=px(2), zorder=5)
    fly, = ov.plot([], [], 'o', color=C.data, mec=C.ink, mew=px(2.5), ms=px(19), zorder=9)
    drop = [ov.plot([], [], color=C.ink2, lw=px(2), ls=(0, (1.2, 2)), zorder=4)[0] for _ in range(2)]
    veil = Rectangle((BOXA[0] - 100, BOXB[1] - 6), BOXA[2] + 110, BOXA[1] + BOXA[3] - BOXB[1] + 12, fc=C.bg, ec='none', alpha=0, zorder=3); ov.add_patch(veil)
    return dict(fig=fig, sub=sub, A=A, B=B, Cc=Cc, pold=pold, pnow=pnow, lI=lI, span=span, hold_txt=hold_txt, lq=lq, lq0=lq0, head=head, ring=ring, samp=samp, pre=pre,
                stxt=stxt, rtxt=rtxt, stem=stem, mark=mark, keyt=keyt, htxt=htxt, drop=drop, vcur=vcur, fly=fly, ov=ov, veil=veil)


def area_poly(a, b):
    """Polygon of the current between fractional ticks a<b."""
    if b <= a:
        return np.zeros((3, 2))
    ii = np.arange(int(np.ceil(a)), int(np.floor(b)) + 1)
    x = np.r_[a, ii, b]; y = np.interp(x, np.arange(NT), I_E / 1e3)
    return np.c_[np.r_[x[0], x, x[-1]] * DT, np.r_[0, y, 0]]


def draw(s, i, t):
    c = cursor(t)
    # caption
    txt, a_ = '', 0.0
    for a, b, tx in CAPS:
        if a <= t < b + 0.25:
            txt, a_ = tx, fade(t, a, b, 0.25)
            if a == 0.0:
                a_ = 1 - float(ease(seg(t, b, b + 0.25)))
    s['sub'].set_text(txt); s['sub'].set_alpha(a_)
    # current
    n = np.arange(T0, int(np.floor(c)) + 1)
    x = np.r_[n, c]
    s['lI'].set_data(x * DT, np.interp(x, np.arange(NT), I_E / 1e3))
    last = T0 if c < r0 else (r0 if c < r1 else r1)
    s['pnow'].set_xy(area_poly(last, c)); s['pold'].set_xy(area_poly(T0, min(last, c)))
    # charge
    xq, yq, yq0 = trace_upto(c)
    s['lq'].set_data(xq, yq); s['lq0'].set_data(xq, yq0)
    base = h0['eps_r'] if c < r0 else (h0['base_next'] if c < r1 else h1['base_next'])
    s['keyt'].set_text(f'with noise: baseline ε$_r$ = {base / 1e3:+.2f} ke⁻'.replace('-', '−'))
    s['head'].set_data([xq[-1]], [yq[-1]])
    # hold windows, crossing rings, samples
    for k, (h, tc, ts) in enumerate(((h0, Tcross0, Tsample0), (h1, Tcross1, Tsample1))):
        on = c >= h['cross'] - 1e-6
        s['ring'][k].set_data(([h['cross'] * DT], [VTH / 1e3]) if t >= tc else ([], []))
        xa = (h['tick'] + 1) * DT
        wd = max(min(c, h['end']) * DT - xa, 0) if on else 0
        s['span'][k].set_x(xa); s['span'][k].set_width(wd)
        s['hold_txt'][k].set_position((xa + HOLD * DT / 2, YB[0] + 0.3)); s['hold_txt'][k].set_alpha(float(ease(seg(c, h['tick'] + 1 + HOLD / 2 + 4, h['tick'] + 1 + HOLD / 2 + 7))))   # only once the band has passed the label
        if t >= ts:
            s['samp'][k].set_data([h['end'] * DT], [h['q_meas'] / 1e3]); s['pre'][k].set_data([h['end'] * DT], [h['q_pre'] / 1e3])
        else:
            s['samp'][k].set_data([], []); s['pre'][k].set_data([], [])
        ua = fade(t, ts + 0.2, 1e9, 0.3)
        lab = f'{h["q_pre"] / 1e3:.2f} {"−" if h["eps_u"] < 0 else "+"} {abs(h["eps_u"]) / 1e3:.2f} (ε$_u$) = {h["q_meas"] / 1e3:.2f} ke⁻'
        s['stxt'][k].set_text(lab); s['stxt'][k].set_alpha(fade(t, ts + 0.2, Tend - 0.6, 0.3))
        # text to the right of the sample for hit 1 would sit on the rising curve: put it on the free side
        if k == 0:
            s['stxt'][k].set_position((h['end'] * DT - 0.16, h['q_meas'] / 1e3 + 1.7)); s['stxt'][k].set_ha('right')
        else:
            s['stxt'][k].set_position((h['end'] * DT + 0.16, h['q_meas'] / 1e3 + 1.6)); s['stxt'][k].set_ha('left')
        # reset label
        tr = (Treset0, Treset1)[k]
        s['rtxt'][k].set_text(f'new baseline ε$_r$ = {h["base_next"] / 1e3:+.2f} ke⁻'.replace('-', '−'))
        s['rtxt'][k].set_position(((h['reset'] * DT + 0.16), h['base_next'] / 1e3 + (0.0 if k == 0 else 1.0)))
        s['rtxt'][k].set_alpha(fade(t, tr + 0.05, tr + 2.3, 0.3))
    # the sample flies to the hit list; drop line marks the crossing tick
    FLY = 1.5
    fx = fy = None
    for k, (h, tc, ts) in enumerate(((h0, Tcross0, Tsample0), (h1, Tcross1, Tsample1))):
        xs = h['tick'] * DT
        p = seg(t, ts + 0.5, ts + 0.5 + FLY)
        if t >= tc:
            g = ease(seg(t, tc, tc + 0.6))
            # vertical guide from the threshold crossing to the hit axis
            xa_, ya = to_px(BOXB, XL, YB, xs, VTH / 1e3); _, yb = to_px(BOXC, XL, YC, xs, PED)
            if t >= ts + 0.5 + FLY:
                yb = to_px(BOXC, XL, YC, xs, h['adc'])[1] + 12
            s['drop'][k].set_data([xa_, xa_], [ya - 13, lerp(ya - 13, yb, g)])
        else:
            s['drop'][k].set_data([], [])
        if 0 < p < 1:
            pa = to_px(BOXB, XL, YB, h['end'] * DT, h['q_meas'] / 1e3); pb = to_px(BOXC, XL, YC, xs, h['adc'])
            e_ = float(ease(p)); fx, fy = lerp(pa[0], pb[0], e_), lerp(pa[1], pb[1], e_)
        if p >= 1:
            s['stem'][k].set_data([xs, xs], [PED, h['adc']]); s['mark'][k].set_data([xs], [h['adc']])
        else:
            s['stem'][k].set_data([], []); s['mark'][k].set_data([], [])
        s['htxt'][k].set_text(f'hit {k + 1}:  tick {h["tick"]} = {h["tick"] * DT:.1f} µs\nADC {h["adc"]:.1f} = {h["q_meas"] / 1e3:.2f} ke⁻')
        s['htxt'][k].set_position((xs + 0.15, 88.5)); s['htxt'][k].set_alpha(fade(t, ts + 0.5 + FLY, 1e9, 0.3))
    s['fly'].set_data(([fx], [fy]) if fx is not None else ([], []))
    xc = BOXA[0] + (c * DT - XL[0]) / (XL[1] - XL[0]) * BOXA[2]
    al = 1 - float(ease(seg(t, Tend, Tend + 0.8)))
    s['vcur'].set_data([xc, xc], [BOXC[1], BOXA[1] + BOXA[3]]); s['vcur'].set_alpha(al)
    s['head'].set_alpha(al)
    s['veil'].set_alpha(0.55 * float(ease(seg(t, Tend + 0.4, Tend + 1.6))))


BEATS = [(0, 'The pixel and its current'), (6.2, 'Charge accumulates on the integrator'), (Tcross0, 'Threshold crossed: hit time'), (Tsample0, 'End of hold: digitise'),
         (Treset0, 'Reset, new baseline noise'), (Tcross1, 'Second crossing'), (Tsample1, 'Second sample'), (Treset1, 'Second reset'), (Tend, 'A hit is a time and a charge')]
render(args, NAME, DUR, setup, draw, beats=BEATS)
