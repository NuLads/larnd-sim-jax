"""a05: why the electronics can be differentiated. Starts on the last frame of a04, replays the same waveform with the real cached throws, builds the hit-time
histogram from all 10,000 of them, lays the analytic P(t) (log_prob_pixel, production beam search, K=100) on it with NO rescaling, then shows one throw (staircase in a
waveform scale) against the model (smooth). Histogram: counts/10,000 per tick. Model: P(hit n in tick i), unconditional, so sum_i P = lambda_n, not 1.
Both index the tick interval [i, i+1] in which the crossing happens (fee_jax: idx_t and g_i). Bin i is drawn on [i, i+1]. Sentinel tick 1998 = empty slot, excluded.
Agreement numbers on screen: a05_agreement.py / a05_agreement.json. Scale scan: a04_build_replay.py (production functions on the waveform times 1+s, same noise key)."""
from common import *
from fee_pair import *
from matplotlib.patches import Polygon, Rectangle
import json, os
args = parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
AG = json.load(open(f'{HERE}/a05_agreement.json'))
R = np.load(f'{HERE}/a04_replay.npz')
NAME = 'a05_random_to_differentiable'
XL = (T0 * DT, T1 * DT); YA = (-4.5, 17); YB = (-2.2, 14.5); YB1 = (-3.2, 20.0); YC0 = (73, 97)
BOXA = (130, 712, 1730, 200); BOXB0 = (130, 322, 1730, 362); BOXC0 = (130, 122, 1730, 168)
BOXB1 = (130, 612, 1730, 300); BOXC1 = (130, 122, 1730, 450)

# ---- timeline -----------------------------------------------------------------------------------------------------------
T_FADEA = (2.5, 3.1); T_MORPH = (3.0, 4.3); T_START = 4.5
KG = 0.85                                # exponential growth of N in the fast stage, per second
NSLOW = 3                                # throws 1..NSLOW get the full slow sweep
TM = np.zeros(NTHROW + 1); DM = np.zeros(NTHROW)
t_ = T_START
for m in range(1, NTHROW):
    d_ = 1.9 if m <= NSLOW else max(1.9 * 0.72 ** (m - NSLOW), 1.0 / (KG * m))
    TM[m] = t_; DM[m] = d_; t_ += d_
TM[NTHROW] = t_; TM[0] = 0.0
T_FULL = float(t_)                       # all throws in
T_REVEAL = T_FULL + 3.0; T_REVEAL_END = T_REVEAL + 3.2
T_AGREE = T_REVEAL_END + 0.4
T_FIT = T_AGREE + 6.0
T_SWEEP = (T_FIT + 1.6, T_FIT + 8.2)
T_LAST = T_SWEEP[1] + 0.2
DUR = T_LAST + 3.6
SLOW = 0.30                              # a throw is swept individually while its duration exceeds this
FOOT0 = 'Simulation: same pixel as before, production electronics. 10,000 real throws; P(t) from the production beam search (K = 100).'
FOOT1 = 'Scale test: production code re-run on the same waveform × (1 + s), same noise key for the throw. A uniform scale, not a physics parameter.'
CAPS = [(0.0, 4.2, 'Replay the same pixel: same current, new noise'),
        (T_START + 0.1, TM[3] - 0.1, 'Each replay draws a new baseline, so the crossing moves'),
        (TM[3], TM[5] - 0.1, 'Same waveform, different tick'),
        (TM[5], T_FULL + 0.1, 'Keep replaying: the hit times build a distribution'),
        (T_FULL + 0.3, T_REVEAL - 0.1, '10,000 real throws of the production electronics'),
        (T_REVEAL + 0.1, T_AGREE - 0.1, 'The model computes the same distribution directly: one pass, no random numbers'),
        (T_AGREE + 0.1, T_FIT - 0.1, 'Same axis, no rescaling. The second hit sums over the possible resets of the first'),
        (T_FIT + 0.1, T_SWEEP[1] - 0.1, 'Scale the waveform: the throw jumps tick by tick, the probability glides'),
        (T_LAST, 999, 'The fit uses this smooth probability, not one random outcome')]

# ---- data ---------------------------------------------------------------------------------------------------------------
TT = np.arange(T0, T1 + 1)
TRC = np.empty((NTHROW, TT.size)); HITS = {}
for m in range(NTHROW):
    q_, q0_, h_ = throw(m)
    TRC[m] = q_[T0:T1 + 1] / 1e3
    if m < 400:
        HITS[m] = h_
for m in range(0, NTHROW, 97):
    check(m)
EDGES = np.arange(T0, T1 + 1) * DT
_ok = [(TICKS[:, n] != SENT) & (TICKS[:, n] >= T0) & (TICKS[:, n] < T1) for n in range(2)]


def hist_upto(n, N):
    t = TICKS[:N, n]
    t = t[_ok[n][:N]]
    return np.bincount(t - T0, minlength=T1 - T0).astype(float)


# y scale of the histogram per 30-fps time step: smooth envelope of the tallest bin
_grid = np.arange(0, DUR + 1 / 30, 1 / 30)
_Nf = np.array([int(np.clip(np.searchsorted(TM[:NTHROW], t, 'right'), 1, NTHROW)) for t in _grid])
_cnt = np.array([max(hist_upto(0, n).max(), hist_upto(1, n).max()) for n in _Nf], float)
_env = np.maximum(np.maximum.accumulate(_cnt) * 1.28, 7.0)
_w = np.hanning(61); _w /= _w.sum()
_pad = np.r_[np.full(30, _env[0]), _env, np.full(30, _env[-1])]
YMAX = np.convolve(_pad, _w, 'same')[30:-30]
YMAX = np.maximum(YMAX, _env / 1.28 * 1.06)
YFINAL = float(max(hist_upto(0, NTHROW).max(), hist_upto(1, NTHROW).max()) * 1.28)
YMAX[_grid >= T_FULL] = np.minimum(YMAX[_grid >= T_FULL], YFINAL); YMAX[_grid >= T_FULL + 1.0] = YFINAL


def ymax(t):
    return float(YMAX[min(int(round(t * 30)), YMAX.size - 1)])


# charge cloud of all throws so far, per 30-fps step: cumulative, so it never flickers
NY = 230; YL = (-3.5, 20.5); DY = (YL[1] - YL[0]) / NY
_cell = np.clip(((TRC - YL[0]) / DY).astype(int), 0, NY - 1)
_ids = np.tile(np.arange(TT.size), (NTHROW, 1)) * NY + _cell
_fast = np.where(_grid >= TM[4])[0]
_Ns = _Nf[_fast]; _uniq, _inv = np.unique(_Ns, return_inverse=True)
CLOUD = np.empty((_uniq.size, TT.size, NY), np.uint16)
_acc = np.zeros(TT.size * NY, np.int64); _prev = 0
for k, n in enumerate(_uniq):
    _acc += np.bincount(_ids[_prev:n].ravel(), minlength=TT.size * NY); _prev = n
    CLOUD[k] = _acc.reshape(TT.size, NY)


def cloud_at(t):
    i = int(round(t * 30))
    if i < _fast[0]:
        return None, 0
    j = min(i - _fast[0], _inv.size - 1)
    return CLOUD[_inv[j]], int(_Ns[j])


# scale scan
SC = np.asarray(R['scales']); LP = np.asarray(R['scan_logp'], float); T00 = int(R['scan_tick0'])
PS = np.exp(np.clip(LP, -40, 0))          # (scales, 3, ticks from T00)
_tt = np.arange(PS.shape[2]) + T00
ET0 = (PS[:, 0] * _tt).sum(1) / PS[:, 0].sum(1); TK0 = np.asarray(R['scan_ticks'])[:, 0, 0]
P0 = PROB[0]; P1 = PROB[1]


def scale_at(t):
    a, b = T_SWEEP
    if t <= a or t >= b:
        return 0.0
    return float(0.04 * np.sin(2 * np.pi * ease((t - a) / (b - a))))


def to_px(box, yl, x, y):
    return box[0] + (x - XL[0]) / (XL[1] - XL[0]) * box[2], box[1] + (y - yl[0]) / (yl[1] - yl[0]) * box[3]


def area_poly(a, b):
    ii = np.arange(int(np.ceil(a)), int(np.floor(b)) + 1)
    x = np.r_[a, ii, b]; y = np.interp(x, np.arange(NT), I_E / 1e3)
    return np.c_[np.r_[x[0], x, x[-1]] * DT, np.r_[0, y, 0]]


def setup():
    fig = new_figure()
    title(fig, 'One random hit time has no gradient. Its probability does.')
    sub = fig.text(60 / W, 958 / H, '', fontsize=px(30), color=C.ink2, va='baseline')
    ftxt, _ = footer(fig, FOOT0, right='fig3_probabilistic_method cache')
    A = fig.add_axes(box_px(*BOXA)); B = fig.add_axes(box_px(*BOXB0)); Cc = fig.add_axes(box_px(*BOXC0))
    style_axes(A, 'y'); style_axes(B, 'y'); style_axes(Cc, 'y')
    for ax, yl in ((A, YA), (B, YB), (Cc, YC0)):
        ax.set_xlim(*XL); ax.set_ylim(*yl); ax.set_xticks(np.arange(24, 36, 2))
    A.tick_params(labelbottom=False); B.tick_params(labelbottom=False)
    A.set_yticks([0, 8, 16]); B.set_yticks([0, 5, 10, 15, 20]); Cc.set_yticks([76, 86, 96])
    A.set_ylabel('current\n(ke⁻/µs)'); B.set_ylabel('charge on integrator (ke⁻)'); Cc.set_ylabel('ADC'); Cc.set_xlabel('time (µs)')
    fig.align_ylabels([A, B, Cc])
    # opening state: the end of a04
    A.axhline(0, color=C.spine, lw=px(1.5), zorder=1)
    A.add_patch(Polygon(area_poly(T0, T1), closed=True, fc=C.ink3, ec='none', alpha=0.28, zorder=2))
    A.plot(tomu(np.arange(T0, T1 + 1)), I_E[T0:T1 + 1] / 1e3, color=C.ink, lw=px(3), zorder=4)
    B.axhline(VTH / 1e3, color=C.ink2, lw=px(2.5), zorder=2); B.axhline(0, color=C.spine, lw=px(1.5), zorder=1)
    B.text(XL[0] + 0.12, VTH / 1e3 + 0.35, 'threshold 5 ke⁻', fontsize=px(24), color=C.ink2, va='bottom', ha='left')
    q_, q0_, h_ = throw(0)
    open_art = []                                  # fade out together
    for h in h_[:2]:
        r = Rectangle(((h['tick'] + 1) * DT, YB[0]), HOLD * DT, YB[1] - YB[0], fc=C.ink3, ec='none', alpha=0.20, zorder=0); B.add_patch(r); open_art.append((r, 0.20))
        tx = B.text((h['tick'] + 1 + HOLD / 2) * DT, YB[0] + 0.3, 'hold 1.8 µs', fontsize=px(24), color=C.ink2, va='bottom', ha='center'); open_art.append((tx, 1))
        for a_ in (B.plot([h['cross'] * DT], [VTH / 1e3], 'o', mfc='none', mec=C.ink, mew=px(3), ms=px(24), zorder=7)[0],
                   B.plot([h['end'] * DT], [h['q_meas'] / 1e3], 'o', color=C.data, mec=C.ink, mew=px(2.5), ms=px(19), zorder=7)[0],
                   B.plot([h['end'] * DT], [h['q_pre'] / 1e3], 'o', mfc='none', mec=C.data, mew=px(2.5), ms=px(15), zorder=6)[0]):
            open_art.append((a_, 1))
    l0, = B.plot(TT * DT, q0_[T0:T1 + 1] / 1e3, color=C.truth, lw=px(2.5), ls=(0, (3, 2.2)), zorder=3); open_art.append((l0, 1))
    k1, = B.plot([XL[0] + 0.12, XL[0] + 0.6], [13.15, 13.15], color=C.data, lw=px(4.5)); k2, = B.plot([XL[0] + 0.12, XL[0] + 0.6], [11.75, 11.75], color=C.truth, lw=px(2.5), ls=(0, (3, 2.2)))
    t1 = B.text(XL[0] + 0.72, 13.15, f'with noise: baseline ε$_r$ = {h_[1]["base_next"] / 1e3:+.2f} ke⁻'.replace('-', '−'), fontsize=px(24), color=C.ink, va='center')
    t2 = B.text(XL[0] + 0.72, 11.75, 'same charge, no noise', fontsize=px(24), color=C.ink, va='center')
    open_art += [(k1, 1), (k2, 1), (t1, 1), (t2, 1)]
    lthrow, = B.plot(TT * DT, TRC[0], color=C.data, lw=px(4.5), zorder=5)
    ghosts = [B.plot([], [], color=C.data, lw=px(2.2), zorder=3)[0] for _ in range(8)]
    ring, = B.plot([], [], 'o', mfc='none', mec=C.ink, mew=px(3), ms=px(24), zorder=7)
    ring2, = B.plot([], [], 'o', mfc='none', mec=C.ink, mew=px(3), ms=px(24), zorder=7)
    rgba = np.zeros((NY, TT.size, 4)); rgba[..., :3] = to_rgb(C.data)
    im = B.imshow(rgba, extent=(T0 * DT - DT / 2, T1 * DT + DT / 2, YL[0], YL[1]), origin='lower', aspect='auto', interpolation='bilinear', zorder=2)
    B.set_xlim(*XL); B.set_ylim(*YB)
    # hits of throw 0 as in a04
    pl = Cc.axhline(PED, color=C.spine, lw=px(1.5), zorder=1)
    ptxt = Cc.text(XL[0] + 0.12, PED + 1.1, 'pedestal, no charge', fontsize=px(22), color=C.ink3, va='bottom')
    adc_art = [(ptxt, 1), (pl, 1)]
    for k, h in enumerate(h_[:2]):
        xs = h['tick'] * DT
        adc_art.append((Cc.plot([xs, xs], [PED, h['adc']], color=C.data, lw=px(4), zorder=3)[0], 1))
        adc_art.append((Cc.plot([xs], [h['adc']], 'o', color=C.data, mec=C.ink, mew=px(2.5), ms=px(22), zorder=4)[0], 1))
        adc_art.append((Cc.text(xs + 0.15, 88.5, f'hit {k + 1}:  tick {h["tick"]} = {h["tick"] * DT:.1f} µs\nADC {h["adc"]:.1f} = {h["q_meas"] / 1e3:.2f} ke⁻', fontsize=px(26), color=C.ink, va='center', ha='left', zorder=5), 1))
    # histogram lives on its own axes on top of the same box
    Hh = fig.add_axes(box_px(*BOXC0)); style_axes(Hh, None); Hh.set_xlim(*XL); Hh.patch.set_alpha(0)
    Hh.set_xticks([]); Hh.set_yticks([]); Hh.spines['bottom'].set_visible(False); Hh.spines['left'].set_visible(False)
    st0 = Hh.stairs(np.zeros(T1 - T0), EDGES, fill=True, color=C.data, alpha=0.9, zorder=3)
    st1 = Hh.stairs(np.zeros(T1 - T0), EDGES, fill=True, color=C.data, alpha=0.9, zorder=3)
    grid = [Hh.plot((XL[0], 31.6), [v, v], color=C.grid, lw=px(1.2), zorder=1, alpha=0)[0] for v in (1000, 2000)]
    lm0, = Hh.plot([], [], color=C.model, lw=px(4.5), zorder=5); lm1, = Hh.plot([], [], color=C.model, lw=px(4.5), zorder=5)
    lab0 = Hh.text(279.5 * DT, 0, 'first hit\nslot n = 0', fontsize=px(26), color=C.ink, ha='center', va='bottom', alpha=0, linespacing=1.05, zorder=6)
    lab1 = Hh.text(302.5 * DT, 0, 'second hit\nslot n = 1', fontsize=px(26), color=C.ink, ha='right', va='center', alpha=0, linespacing=1.05, zorder=6)
    lmod = Hh.text(XL[0] + 0.14, 0, 'model P(t)', fontsize=px(34), fontweight='semibold', color=C.ink, va='top', alpha=0, zorder=6)
    sw0, = Hh.plot([], [], color=C.data, lw=px(12), solid_capstyle='butt', zorder=6, alpha=0); sw1, = Hh.plot([], [], color=C.model, lw=px(4.5), zorder=6, alpha=0)
    count = Hh.text(XL[0] + 0.14, 0, '', fontsize=px(34), fontweight='semibold', color=C.ink, va='top', alpha=0, zorder=6)
    # inset: hit time against waveform scale (added before the overlay so the overlay veil can fade it)
    In = fig.add_axes(box_px(1330, 262, 510, 276)); style_axes(In, 'y'); In.patch.set_alpha(0)
    In.set_xlim(-4.4, 4.4); In.set_ylim(275.2, 286.4); In.set_xticks([-4, 0, 4]); In.set_xticklabels(['−4', '0', '+4']); In.set_yticks([276, 280, 284])
    In.set_xlabel('waveform scale (%)'); In.set_ylabel('first-hit tick')
    In.step(SC * 100, TK0, where='mid', color=C.data, lw=px(4)); In.plot(SC * 100, ET0, color=C.model, lw=px(4.5))
    dot0, = In.plot([], [], 'o', color=C.data, mec=C.ink, mew=px(2.5), ms=px(17), zorder=6); dot1, = In.plot([], [], 'o', color=C.model, mec=C.ink, mew=px(2.5), ms=px(17), zorder=6)
    In.text(4.3, 285.7, 'one throw, same noise: flat, then a jump', fontsize=px(24), color=C.ink, ha='right', va='center')
    In.text(-4.3, 275.9, 'mean of P(t): smooth', fontsize=px(24), color=C.ink, ha='left', va='center')
    In.set_visible(False)
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(0, H); ov.axis('off'); ov.patch.set_alpha(0)
    veil = Rectangle((BOXA[0] - 100, BOXB0[1] - 6), BOXA[2] + 110, BOXA[1] + BOXA[3] - BOXB0[1] + 12, fc=C.bg, ec='none', alpha=0, zorder=3); ov.add_patch(veil)
    veilA = Rectangle((BOXA[0] - 120, BOXA[1] - 14), BOXA[2] + 130, BOXA[3] + 18, fc=C.bg, ec='none', alpha=0, zorder=3); ov.add_patch(veilA)
    veilI = Rectangle((1240, 190), 610, 365, fc=C.bg, ec='none', alpha=0, zorder=3, visible=False); ov.add_patch(veilI)
    fall = [ov.add_patch(Rectangle((0, 0), 15, 22, fc=C.data, ec='none', zorder=8, visible=False)) for _ in range(3)]
    cursor, = ov.plot([], [], color=C.ink2, lw=px(2), zorder=5)
    yl_hits = ov.text(77, 0, '', rotation=90, fontsize=px(26), color=C.ink2, ha='center', va='center')
    yt = [ov.text(BOXC1[0] - 14, 0, s_, fontsize=px(22), color=C.ink3, ha='right', va='center', alpha=0) for s_ in ('0', '0.1', '0.2')]
    ytk = [ov.plot([], [], color=C.ink3, lw=px(1.5))[0] for _ in range(3)]
    # agreement card
    a0, a1 = AG['n0_window'], AG['n1_window']
    rows = [('hit probability', f'{a0["lam_model"]:.4f} | {a0["lam_throws"]:.4f}', f'{a1["lam_model"]:.4f} | {a1["lam_throws"]:.4f}'),
            ('mean tick', f'{a0["mean_model"]:.2f} | {a0["mean_throws"]:.2f}', f'{a1["mean_model"]:.2f} | {a1["mean_throws"]:.2f}'),
            ('χ² / bins', f'{a0["chi2"]:.0f} / {a0["nbins"]}', f'{a1["chi2"]:.0f} / {a1["nbins"]}')]
    card = []
    X0, X1, X2 = 1290, 1520, 1740; Y0 = 526; LH = 38
    card.append(ov.text(X1, Y0, 'first hit', fontsize=px(24), color=C.ink2, ha='center', va='center', alpha=0))
    card.append(ov.text(X2, Y0, 'second hit', fontsize=px(24), color=C.ink2, ha='center', va='center', alpha=0))
    for i, r_ in enumerate(rows):
        y = Y0 - LH * (i + 1)
        card.append(ov.text(X0, y, r_[0], fontsize=px(24), color=C.ink2, ha='left', va='center', alpha=0))
        card.append(ov.text(X1, y, r_[1], fontsize=px(24), color=C.ink, ha='center', va='center', alpha=0))
        card.append(ov.text(X2, y, r_[2], fontsize=px(24), color=C.ink, ha='center', va='center', alpha=0))
    card.append(ov.text(X0, Y0 - LH * 4 - 2, 'model | 10,000 throws, this strike', fontsize=px(22), color=C.ink3, ha='left', va='center', alpha=0))
    return dict(fig=fig, sub=sub, ftxt=ftxt, A=A, B=B, Cc=Cc, Hh=Hh, open_art=open_art, adc_art=adc_art, lthrow=lthrow, ghosts=ghosts, ring=ring, ring2=ring2, im=im, rgba=rgba, st0=st0, st1=st1, grid=grid,
                lm0=lm0, lm1=lm1, lab0=lab0, lab1=lab1, lmod=lmod, sw0=sw0, sw1=sw1, count=count, veil=veil, veilA=veilA, veilI=veilI, fall=fall, cursor=cursor, yl_hits=yl_hits, yt=yt, ytk=ytk, card=card, In=In, dot0=dot0,
                dot1=dot1, ov=ov)


def draw(s, i, t):
    # caption
    txt, a_ = '', 0.0
    for a, b, tx in CAPS:
        if a <= t < b + 0.25:
            txt, a_ = tx, (fade(t, a, b, 0.25) if a > 0 else 1 - float(ease(seg(t, b, b + 0.25))))
    s['sub'].set_text(txt); s['sub'].set_alpha(a_); s['ftxt'].set_text(FOOT1 if t >= T_FIT else FOOT0)
    s['veil'].set_alpha(0.55 * (1 - float(ease(seg(t, 0.3, 1.3)))))
    # current panel leaves first, then the two panels morph
    aa = float(ease(seg(t, *T_FADEA)))
    s['veilA'].set_alpha(aa); s['A'].set_visible(aa < 0.997); s['veilA'].set_visible(0.003 < aa < 0.997)
    u = float(ease(seg(t, *T_MORPH)))
    boxB = tuple(lerp(np.array(BOXB0, float), np.array(BOXB1, float), u)); boxC = tuple(lerp(np.array(BOXC0, float), np.array(BOXC1, float), u))
    s['B'].set_position(box_px(*boxB)); s['Cc'].set_position(box_px(*boxC)); s['Hh'].set_position(box_px(*boxC)); s['B'].set_ylim(lerp(YB[0], YB1[0], u), lerp(YB[1], YB1[1], u))
    oa = 1 - float(ease(seg(t, T_FADEA[0], T_FADEA[1])))
    for a_, base in s['open_art'] + s['adc_art']:
        a_.set_visible(oa > 0.003)
        a_.set_alpha(oa * base)
    if oa <= 0.003:
        s['Cc'].set_yticks([]); s['Cc'].set_ylabel('')
    else:
        s['Cc'].set_yticks([76, 86, 96]); s['Cc'].set_ylabel('ADC'); s['Cc'].yaxis.label.set_alpha(oa)
        for l in s['Cc'].get_yticklabels():
            l.set_alpha(oa)
    # throws
    mcur = int(np.clip(np.searchsorted(TM[:NTHROW], t, 'right') - 1, 0, NTHROW - 1))
    if t < T_START:
        mcur = 0
    Dm = DM[mcur] if mcur > 0 else 0
    slow = 0 < mcur and Dm >= SLOW
    N = mcur + 1
    c = T1
    falling = []
    ym = ymax(t)
    h0 = hist_upto(0, mcur if slow else N); h1 = hist_upto(1, mcur if slow else N)
    if slow:
        sw = 0.72 * Dm
        c = T0 + (T1 - T0) * seg(t, TM[mcur], TM[mcur] + sw)
        for h in HITS[mcur][:2]:
            if h['failed'] or h['tick'] >= T1:
                continue
            tc = TM[mcur] + sw * (h['cross'] - T0) / (T1 - T0)
            land = tc + 0.34
            if t >= land:
                (h0 if h['n'] == 0 else h1)[h['tick'] - T0] += 1
            elif t >= tc:
                falling.append((h, seg(t, tc, land)))
    # blocks of throw 0 grow in at the end of the morph
    gr = float(ease(seg(t, T_MORPH[1] - 0.6, T_MORPH[1])))
    if t < T_START:
        h0 = h0 * gr; h1 = h1 * gr
    # charge panel
    cloud, ncl = cloud_at(t)
    ln = np.log10(max(N, 1))
    ac = float(ease(seg(ln, np.log10(6), np.log10(24))))           # cumulative cloud of all throws so far
    ath = 1 - float(ease(seg(ln, np.log10(7), np.log10(16))))      # single throw + ghosts
    if ac > 0.002 and cloud is not None:
        fr = cloud.T.astype(float) / max(ncl, 1)
        s['rgba'][..., 3] = np.clip(fr / 0.05, 0, 1) ** 0.62 * ac * float(1 - 0.5 * ease(seg(t, T_FIT, T_FIT + 1.2)))
        s['im'].set_data(s['rgba']); s['im'].set_visible(True)
    else:
        s['im'].set_visible(False)
    n_ = int(np.floor(c)); idx = np.arange(T0, min(n_, T1) + 1)
    s['lthrow'].set_data(idx * DT, TRC[mcur][idx - T0]); s['lthrow'].set_alpha(ath); s['lthrow'].set_visible(ath > 0.003)
    for k, g in enumerate(s['ghosts']):
        m = mcur - 1 - k
        if m >= 0 and ath > 0.003 and t >= T_START:
            g.set_data(TT * DT, TRC[m]); g.set_alpha(ath * 0.34 * (1 - k / 8)); g.set_visible(True)
        else:
            g.set_visible(False)
    # hit rings on the current throw
    rr = [(), ()]
    if slow:
        for k, h in enumerate(HITS[mcur][:2]):
            if c >= h['cross']:
                rr[k] = ([h['cross'] * DT], [VTH / 1e3])
    s['ring'].set_data(*(rr[0] if rr[0] else ([], []))); s['ring2'].set_data(*(rr[1] if rr[1] else ([], [])))
    xc = boxB[0] + (c * DT - XL[0]) / (XL[1] - XL[0]) * boxB[2]
    if slow and c < T1:
        s['cursor'].set_data([xc, xc], [boxB[1], boxB[1] + boxB[3]])
    else:
        s['cursor'].set_data([], [])
    # histogram
    Hh = s['Hh']; Hh.set_ylim(0, ym)
    fa = float(1 - 0.70 * ease(seg(t, T_FIT, T_FIT + 1.2)))
    s['st0'].set_data(h0, EDGES); s['st1'].set_data(h1, EDGES); s['st0'].set_alpha(0.9 * fa); s['st1'].set_alpha(0.9 * fa)
    for k, r_ in enumerate(s['fall']):
        if k < len(falling):
            h, p = falling[k]
            x = (h['tick'] + 0.5) * DT
            xp, ya = to_px(BOXB1, YB1, x, VTH / 1e3)
            stack = (h0 if h['n'] == 0 else h1)[h['tick'] - T0]
            _, yb = to_px(BOXC1, (0, ym), x, stack)
            r_.set_visible(True); r_.set_xy((xp - 7.5, lerp(ya, yb, float(p ** 2)))); r_.set_height(450 / ym * lerp(0.4, 1, p))
        else:
            r_.set_visible(False)
    # labels
    Lb = fade(t, TM[40], 1e9, 0.5)
    s['lab0'].set_position((279.5 * DT, ym * 0.31)); s['lab0'].set_alpha(Lb)
    s['lab1'].set_position((302.5 * DT, ym * 0.60)); s['lab1'].set_alpha(Lb)
    s['count'].set_position((XL[0] + 0.14, ym * 0.965)); s['count'].set_text(f'{N:,} throw{"s" if N > 1 else ""}'); s['count'].set_alpha(fade(t, T_MORPH[1] - 0.3, 1e9, 0.5))
    # model curve: drawn left to right, then follows the scale. Each slot drawn around its own peak so the flat zeros do not cross.
    r = seg(t, T_REVEAL, T_REVEAL_END)
    sc = scale_at(t)
    if r > 0:
        js = int(np.clip(round((sc - SC[0]) / (SC[1] - SC[0])), 0, SC.size - 1))
        xr = lerp(T0, T1, float(ease(r)))
        pp0, pp1 = (P0[T0:T1], P1[T0:T1]) if sc == 0.0 else (PS[js, 0, T0 - T00:T1 - T00], PS[js, 1, T0 - T00:T1 - T00])
        xc_ = np.arange(T0, T1) + 0.5
        m_ = xc_ <= xr
        m0 = m_ & (xc_ < 298); m1 = m_ & (xc_ > 293)
        s['lm0'].set_data(xc_[m0] * DT, pp0[m0] * NTHROW); s['lm1'].set_data(xc_[m1] * DT, pp1[m1] * NTHROW)
    else:
        s['lm0'].set_data([], []); s['lm1'].set_data([], [])
    la = fade(t, T_REVEAL, 1e9, 0.5)
    s['lmod'].set_position((XL[0] + 0.14, ym * 0.845)); s['lmod'].set_alpha(la)
    s['sw0'].set_data([XL[0] + 1.62, XL[0] + 2.02], [ym * 0.925, ym * 0.925]); s['sw0'].set_alpha(la * fa)
    s['sw1'].set_data([XL[0] + 1.62, XL[0] + 2.02], [ym * 0.805, ym * 0.805]); s['sw1'].set_alpha(la)
    # y axis on the final scale
    ya = fade(t, T_REVEAL - 1.2, 1e9, 0.6)
    for g in s['grid']:
        g.set_alpha(ya)
    for k, (v, tx) in enumerate(zip((0, 1000, 2000), s['yt'])):
        y = to_px(BOXC1, (0, YFINAL), 0, v)[1]
        tx.set_position((BOXC1[0] - 14, y)); tx.set_alpha(ya)
        s['ytk'][k].set_data([BOXC1[0] - 8, BOXC1[0]], [y, y]); s['ytk'][k].set_alpha(ya)
    s['yl_hits'].set_position((77, boxC[1] + boxC[3] / 2))
    s['yl_hits'].set_text('probability per tick' if t >= T_REVEAL - 1.2 else 'hits per tick')
    s['yl_hits'].set_alpha(ya if t >= T_REVEAL - 1.2 else float(ease(seg(t, T_MORPH[1] - 0.6, T_MORPH[1]))) * (1 - float(ease(seg(t, T_REVEAL - 1.7, T_REVEAL - 1.3)))))
    # agreement card, then inset
    ca = fade(t, T_AGREE, T_FIT - 0.2, 0.5)
    for x in s['card']:
        x.set_alpha(ca); x.set_visible(ca > 0.003)
    ia = fade(t, T_FIT + 0.3, 1e9, 0.6)
    In = s['In']; In.set_visible(ia > 0.003)
    s['veilI'].set_visible(0.003 < ia < 0.997); s['veilI'].set_alpha(1 - ia)
    js = int(np.clip(round((sc - SC[0]) / (SC[1] - SC[0])), 0, SC.size - 1))
    s['dot0'].set_data([sc * 100], [TK0[js]]); s['dot1'].set_data([sc * 100], [ET0[js]])


BEATS = [(0, 'End of a04: one throw, two hits'), (T_START + 1.75, 'Replay: new baseline, new tick'), (TM[5], 'Speeding up'), (T_FULL, '10,000 throws'), (T_REVEAL, 'Analytic P(t) laid on the histogram'),
         (T_AGREE, 'Measured agreement'), (T_FIT, 'Scale the waveform'), (T_LAST, 'The fit uses the smooth probability')]
print(f'duration {DUR:.1f} s, all throws in at {T_FULL:.1f} s, reveal {T_REVEAL:.1f}, agree {T_AGREE:.1f}, fit {T_FIT:.1f}, last {T_LAST:.1f}', flush=True)
render(args, NAME, DUR, setup, draw, beats=BEATS)
