"""a07: how the fit scores the model against data. One real pixel of the main track at two real saved states (start = straight-line
tracks at the initial calibration, iteration 20000), then every pixel, then one number and its gradient. No interpolation between states.
Inputs: a07_prep.py (per-hit and per-pixel terms from the production prediction; pixel sum reproduces the production loss to <0.01).
The time/charge split of a hit is log(sum w P) and the remainder of log I. For a hit on the floor only the floor is meaningful."""
from common import *

args = parse_args()
c = dict(np.load(f'{DATA}/a07_pixel.npz')); s_ = dict(np.load(f'{DATA}/a07_score.npz'))   # eager: render workers fork, and a lazy npz handle cannot be shared
TW = (228, 322)
T_ = dict(a=0.0, b=4.0, score=11.0, start=19.0, strip=27.0, grad=35.0, end=42.5)
BEATS = [(1.0, 'one pixel of the track'), (6.5, 'model and data on this pixel'), (13.0, 'scoring the hits'), (21.0, 'start against iteration 20000'),
         (29.5, 'every pixel adds up'), (37.5, 'one number, one gradient')]
P_ORDER = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']   # order of the saved gradient arrays
m = s_['track'] == 4
HITT, HITQ = c['ticks'], c['q']
LT = float(np.load(f'{DATA}/a06_sweep/main_Ab_0.83488135.npz')['nll'])            # production loss at true segments and true parameters


def th(x):
    return f'{float(x):,.0f}'.replace(',', '\u2009')


def fmt(x, n=2):
    return f'{x:.{n}f}'.replace('-', '−')


def grp(fig, items):
    return [fig.text(x / W, y / H, tx, color=col, fontsize=px(sz), ha=ha, va='baseline') for x, y, tx, col, sz, ha in items]


def setup():
    fig = new_figure()
    tt, sub = title(fig, '', '')
    footer(fig, 'Simulated data: noisy simulation at known truth. Sample 8, batch 35, event 2.', right='run KN10N8C75H, production loss. Start = straight tracks, initial calibration')
    S = dict(fig=fig, tt=tt, sub=sub)
    # scene A: the data on the main track, real aspect
    a = fig.add_axes(box_px(100, 330, 1720, 400)); style_axes(a, grid=None); a.set_xlim(-32, 32); a.set_aspect('equal', adjustable='datalim'); a.set_xlabel('x (cm)'); a.set_ylabel('y (cm)')
    q = np.zeros(len(s_['allp'])); pid = np.load(f'{DATA}/a06_hero_small.npz')['data_pixel_id']; dq = np.load(f'{DATA}/a06_hero_small.npz')['data_q']
    np.add.at(q, np.searchsorted(s_['allp'], pid), dq); k = m & (q > 0)
    a.scatter(s_['X'][k], s_['Y'][k], c=q[k], cmap=cmap_to(C.data), s=px(13) ** 2, marker='s', linewidths=0)
    a.plot([float(c['x'])], [float(c['y'])], 's', ms=px(30), mfc='none', mec=C.ink, mew=px(3))
    S['a'] = a
    # scene B: the pixel
    b = fig.add_axes(box_px(130, 480, 1000, 400)); style_axes(b); d = fig.add_axes(box_px(130, 150, 1000, 280)); style_axes(d)
    tick = np.arange(c['iter20000_P'].shape[1])
    LF, LS = c['iter20000_P'].sum(0), c['start_P'].sum(0)
    b.set_xlim(*TW); b.set_ylim(0, 0.175); b.set_ylabel('P(hit at tick t)'); b.set_xticklabels([])
    d.set_xlim(*TW); d.set_ylim(0, 24); d.set_xlabel('time (ticks of 0.1 µs)'); d.set_ylabel('expected charge (ke)')
    S['pf'], = b.plot(tick, LF, color=C.model, lw=px(4)); S['ps'], = b.plot(tick, LS, color=C.start, lw=px(4))
    S['fill'] = b.fill_between(tick, 0, LF, color=C.model, alpha=0.18, lw=0)
    S['hitl'] = [b.plot([t, t], [0, 0.175], color=C.data, lw=px(2.5), ls=(0, (2, 3)))[0] for t in HITT]
    S['hitm'], = b.plot(HITT, np.full(2, 0.004), 'D', ms=px(15), color=C.data, mew=0, clip_on=False, zorder=5)
    S['qf'] = []; S['qs'] = []
    for st, key, col in (('iter20000', 'qf', C.model), ('start', 'qs', C.start)):
        P, Q = c[st + '_P'], c[st + '_Q']
        for n in range(3):
            ok = P[n] > 2e-4
            if ok.sum() > 3:
                S[key].append(d.plot(tick[ok], Q[n][ok], color=col, lw=px(4))[0])
    S['qm'], = d.plot(HITT, HITQ, 'D', ms=px(15), color=C.data, mew=0, zorder=5)
    S['tb'] = [b.text(273, 0.083, 'iteration 20000', color=C.ink, fontsize=px(26), ha='right'), b.text(258, 0.040, 'start', color=C.ink, fontsize=px(26), ha='right'),
               b.text(HITT[1] + 2, 0.163, 'data hits', color=C.ink, fontsize=px(26))]
    S['b'] = b; S['d'] = d
    # right column: the formula, then the scorecards
    X = 1215
    S['formula'] = grp(fig, [(X, 872, r'$L\ =\ -\sum_{\mathrm{hits}}\ \log I_i$', C.ink, 32, 'left'), (X + 340, 872, 'hits seen', C.ink2, 24, 'left'),
                             (X + 42, 802, r'$+\ N_{\mathrm{miss}}\times 18.4$', C.ink, 32, 'left'), (X + 340, 802, 'on pixels not simulated', C.ink2, 24, 'left'),
                             (X + 42, 748, r'$+\ \sum_{j}\sum_{n,t} P(t\,|\,j,n)$', C.ink, 32, 'left'), (X + 340, 748, 'hits expected', C.ink2, 24, 'left'),
                             (X, 690, r'$I_i$: time and charge probability, $\pm$2 ticks, floor $10^{-8}$', C.ink2, 24, 'left')])
    def card(st, y, head):
        lT, lI = c[st + '_logT'], c[st + '_logI']; flo = lI <= -18.42
        rows = [(X, y, head, C.ink, 27, 'left'), (X + 300, y, 'time', C.ink2, 24, 'right'), (X + 440, y, 'charge', C.ink2, 24, 'right'), (X + 620, y, 'log I', C.ink2, 24, 'right')]
        for i in range(2):
            yy = y - 40 * (i + 1)
            rows += [(X, yy, f'hit at {HITT[i]}, {HITQ[i]:.1f} ke', C.ink, 25, 'left'), (X + 300, yy, fmt(lT[i]), C.ink, 25, 'right'),
                     (X + 440, yy, 'floor' if flo[i] else fmt(lI[i] - lT[i]), C.ink, 25, 'right'), (X + 620, yy, fmt(lI[i]), C.ink, 25, 'right')]
        rows += [(X, y - 120, 'hits expected (data: 2)', C.ink, 25, 'left'), (X + 620, y - 120, fmt(float(c[st + '_exp'])), C.ink, 25, 'right'),
                 (X, y - 164, 'pixel loss', C.ink, 28, 'left'), (X + 620, y - 164, fmt(float(c[st + '_L'])), C.ink, 28, 'right')]
        return grp(fig, rows)
    S['cf'] = card('iter20000', 622, 'Iteration 20000'); S['cs'] = card('start', 382, 'Start')
    # scene C: every pixel of the track, both states
    vm = float(np.percentile(s_['L0'][m], 98)); S['strip'] = []
    for (st, key, y0, lab) in (('start', 'L0', 560, 'Start'), ('iter20000', 'L1', 300, 'Iteration 20000')):
        e = fig.add_axes(box_px(100, y0, 1450, 200)); style_axes(e, grid=None, spines=()); e.set_xticks([]); e.set_yticks([]); e.set_xlim(-32, 32); e.set_ylim(27.8, 34.7)
        e.scatter(s_['X'][m], s_['Y'][m], c=s_[key][m], cmap=cmap_to(C.ink), vmin=0, vmax=vm, s=px(13) ** 2, marker='s', linewidths=0)
        S['strip'].append(e)
    S['st'] = grp(fig, [(100, 770, 'Start', C.ink, 28, 'left'), (100, 510, 'Iteration 20000', C.ink, 28, 'left'),
                        (1590, 680, 'this track', C.ink2, 24, 'left'), (1590, 630, f"{s_['L0'][m].sum():,.0f}".replace(',', ' '), C.ink, 44, 'left'),
                        (1590, 420, 'this track', C.ink2, 24, 'left'), (1590, 370, f"{s_['L1'][m].sum():,.0f}".replace(',', ' '), C.ink, 44, 'left'),
                        (100, 215, 'Whole event: L = ' + th(s_['nll0']) + ' at start, ' + th(s_['nll1']) + ' at iteration 20000, ' + th(LT) + ' with true inputs', C.ink, 34, 'left'),
                        (100, 170, f"Brighter pixel = larger contribution, 0 to {vm:.0f}. {int((~s_['match0']).sum())} hits at start are on pixels the model never simulated.", C.ink2, 24, 'left')])
    # scene D: one number, one gradient
    g0, v0 = c['start_grad'], c['start_val']; S['gr'] = []
    S['L'] = grp(fig, [(150, 560, 'one number', C.ink2, 28, 'left'), (150, 480, f"L = {float(s_['nll0']):,.0f}".replace(',', ' '), C.ink, 60, 'left')])
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.patch.set_alpha(0); S['ax'] = ax; S['arr'] = []
    for k, p in enumerate(PARAMS):
        i = P_ORDER.index(p); y = 800 - k * 118; dln = -float(g0[i] * v0[i]); sgn = np.sign(dln)
        S['arr'].append(ax.annotate('', xy=(1090, y + 12 + 38 * sgn), xytext=(1090, y + 12 - 38 * sgn), arrowprops=dict(arrowstyle='-|>,head_width=0.5,head_length=0.8', color=C[p], lw=px(5))))
        S['gr'] += grp(fig, [(640, y, PARAM_LABEL[p], C.ink, 36, 'left'), (720, y, PARAM_NAME[p], C.ink, 28, 'left'), (1160, y, fmt(dln, 0) if abs(dln) >= 100 else fmt(dln, 1), C.ink, 32, 'left')])
    S['gr'] += grp(fig, [(1160, 880, r'$-\partial L/\partial \ln\theta$', C.ink2, 26, 'left'),
                         (640, 880, 'jax.grad, this event at start', C.ink2, 26, 'left')])
    nf = int(s_['floored0']) + int((~s_['match0']).sum())
    S['gcap'] = grp(fig, [(960, 190, 'The gradient of that one number tells every parameter which way to move', C.ink, 34, 'center'),
                          (960, 140, f'One event at start: {nf} of 638 hits are on the floor, and 3 of 5 arrows point away from truth. Each fit step sums a batch of tracks.', C.ink2, 25, 'center')])
    return S


def vis(items, a):
    for it in items:
        it.set_visible(a > 0.003); it.set_alpha(float(a))


def axvis(ax, a):
    ax.set_visible(a > 0.003)
    for ch in ax.get_children():
        try: ch.set_alpha(float(a))
        except Exception: pass


def draw(s, i, t):
    A, B, SC, ST, CC, G, E = T_['a'], T_['b'], T_['score'], T_['start'], T_['strip'], T_['grad'], T_['end']
    a_map = fade(t, -1, B - 0.5, 0.5); a_pix = fade(t, B, CC - 0.5, 0.5); a_str = fade(t, CC, G - 0.5, 0.5); a_g = fade(t, G, 1e9, 0.5)
    titles = [(G, 'One number, one gradient', ''), (CC, 'Every pixel adds to one number', ''),
              (ST, 'Same pixel: better prediction, better score', 'Start predicts one hit here. The data have two.'),
              (SC, 'Each hit is scored in time and charge', 'Higher log I is better. Every hit the model expects also adds to the loss.'),
              (B, "The model's prediction for one pixel", 'Blue: probability of a hit at each tick. Orange: the hits in the data.'),
              (0, 'Every hit gets a score', f"One pixel of the main track, picked at the median improvement of its {int(c['ncand'])} two-hit pixels")]
    for t0, a, b in titles:
        if t >= t0:
            s['tt'].set_text(a); s['sub'].set_text(b); break
    axvis(s['a'], a_map)
    for ax in (s['b'], s['d']):
        ax.set_visible(a_pix > 0.003)
        for art in [ax.xaxis.label, ax.yaxis.label] + list(ax.spines.values()) + ax.get_xticklabels() + ax.get_yticklabels() + ax.get_xgridlines() + ax.get_ygridlines() + ax.get_xticklines() + ax.get_yticklines():
            art.set_alpha(a_pix)
    a_st = float(ease(seg(t, ST + 0.3, ST + 1.1)))
    for k, h in (('pf', 1), ('qm', 1), ('hitm', 1)): s[k].set_alpha(a_pix)
    s['ps'].set_alpha(a_st * a_pix); s['fill'].set_alpha(0.18 * a_pix)
    for l in s['hitl']: l.set_alpha(a_pix)
    for l in s['qf']: l.set_alpha(a_pix)
    for l in s['qs']: l.set_alpha(a_st * a_pix)
    s['tb'][0].set_alpha(a_pix); s['tb'][1].set_alpha(a_st * a_pix); s['tb'][2].set_alpha(a_pix)
    a_sc = fade(t, SC, CC - 0.5, 0.6) * (a_pix > 0)
    vis(s['formula'], a_sc); vis(s['cf'], a_sc); vis(s['cs'], a_st * a_sc)
    for e in s['strip']: e.set_visible(a_str > 0.003); [ch.set_alpha(a_str) for ch in e.collections]
    vis(s['st'], a_str)
    vis(s['L'], a_g); vis(s['gr'], a_g); s['ax'].set_visible(a_g > 0.003)
    for ar in s['arr']: ar.set_alpha(a_g); ar.arrow_patch.set_alpha(a_g)
    vis(s['gcap'], fade(t, G + 1.5, 1e9, 0.6))


if __name__ == '__main__':
    render(args, 'a07_scoring_a_hit', duration=T_['end'], setup=setup, draw=draw, beats=BEATS)
