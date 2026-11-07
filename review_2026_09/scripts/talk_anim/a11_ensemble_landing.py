"""a11: the 36 realistic-end samples land one at a time, ordered by sample number; then the mean +- SE and the spread."""
from traj_helpers import *

args = parse_args()
ORDER = np.argsort(SAMPLE[EXP]); SMP = SAMPLE[EXP][ORDER]
assert np.all(np.diff(SMP) > 0)
V = {p: endpoint(p)[ORDER] for p in PARAMS}
ST = {p: ensemble(V[p]) for p in PARAMS}
for p in PARAMS:
    print(f'{p:10s} mean {ST[p]["mean"]:+.3f} se {ST[p]["se"]:.3f} sd {ST[p]["sd"]:.3f} min {ST[p]["lo"]:+.2f} max {ST[p]["hi"]:+.2f} mean/se {ST[p]["mean"]/ST[p]["se"]:+.2f}')
HALF = dict(Ab=0.4, eField=0.3, tran_diff=5.0, long_diff=20.0, lifetime=12.0)   # symmetric, so truth stays one vertical line
DEC = dict(Ab=3, eField=3, tran_diff=2, long_diff=2, lifetime=2)   # E needs 3 decimals: its SE rounds to 0.01 otherwise, misstating its size in SE
ROWH = 168; R0 = 762                     # baseline of the top row
Y = {p: R0 - i * ROWH for i, p in enumerate(PARAMS)}
XC, XW = 860, 500                        # x of truth, half width
NUMX = (1450, 1668, 1860)
D = 14.5                                 # dot diameter, design px
T0, DT0, DT1 = 3.2, 0.72, 0.30
TL = np.cumsum([0] + [DT0 if k < 4 else DT1 for k in range(35)]) + T0
TLAND = TL[-1] + 0.45
T_MEAN, T_SD, T_BIAS = TLAND + 2.0, TLAND + 6.0, TLAND + 10.0
DUR = T_BIAS + 5.5
DROP = 0.42


def xpos(p, v):
    return XC + XW * np.asarray(v) / HALF[p]


def swarm(xs):
    """Exact x kept; each dot rises to the lowest height clear of the dots already there, in landing order."""
    ys = []
    for i, x in enumerate(xs):
        cand = [0.0]
        for j in range(i):
            dx = abs(x - xs[j])
            if dx < D:
                cand.append(ys[j] + np.sqrt(D ** 2 - dx ** 2) + 0.01)
        for c in sorted(cand):
            if all((x - xs[j]) ** 2 + (c - ys[j]) ** 2 >= D ** 2 - 1e-6 for j in range(i)):
                ys.append(c); break
    return np.array(ys)


XP = {p: xpos(p, V[p]) for p in PARAMS}
SW = {p: swarm(XP[p]) for p in PARAMS}
print('max stack height px', {p: round(float(SW[p].max()), 1) for p in PARAMS})
NSE = np.array([abs(ST[p]['mean']) / ST[p]['se'] for p in PARAMS if p != 'Ab'])
CAPS = [(0, T0 - 0.2, 'One fit per sample, same recipe. Ordered by sample number'),
        (T0 - 0.2, TLAND + 1.6, 'Each dot: one sample, mean of its last 1000 iterations'),
        (TLAND + 1.6, T_SD - 0.2, 'Mean ± standard error of the 36 samples'),
        (T_SD - 0.2, T_BIAS - 0.2, 'Spread: the standard deviation, what one sample can expect'),
        (T_BIAS - 0.2, 999, f'Real biases: E, $D_T$, $D_L$, $\\tau$ sit {NSE.min():.0f} to {NSE.max():.0f} standard errors from truth; $A_b$ does not')]
fm_ = lambda v, d=2: f'{v:+.{d}f}'.replace('-', '−')


def setup():
    fig = new_figure()
    _, sub = title(fig, 'Where 36 independent fits land: near truth, with real biases')
    footer(fig, 'Simulated data at known truth, endpoints smeared by 1 mm. Samples 1–12 and 17–40, final error at iteration 20000.', right='EXP<s>C75H; last-1000 mean / target − 1')
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off'); ax.patch.set_alpha(0)
    s = dict(fig=fig, sub=sub, ax=ax, dots={}, new={}, band={}, mid={}, sdb={}, num={})
    yt = Y[PARAMS[0]] + 132
    ax.text(XC, yt + 6, 'truth', ha='center', va='bottom', fontsize=px(24), color=C.ink2)
    for x, name, ha in ((NUMX[0], 'start', 'right'), (NUMX[1], 'mean ± SE', 'right'), (NUMX[2], 'spread (SD)', 'right')):
        s['num'][name] = ax.text(x, yt + 6, name, ha=ha, va='bottom', fontsize=px(24), color=C.ink2)
    for p in PARAMS:
        y = Y[p]
        ax.plot([XC - XW, XC + XW], [y, y], color=C.spine, lw=px(1.8), zorder=1)
        ax.plot([XC, XC], [y, y + 128], color=C.truth, lw=px(2.4), ls=(0, (4, 3)), zorder=3)
        h = HALF[p]; step = dict(Ab=0.2, eField=0.1, tran_diff=2.5, long_diff=10.0, lifetime=4.0)[p]
        for k in range(-int(round(h / step)), int(round(h / step)) + 1):
            v = k * step; xt = xpos(p, v)
            ax.plot([xt, xt], [y, y - 7], color=C.spine, lw=px(1.5), zorder=1)
            ax.text(xt, y - 11, fmt_pct(v), ha='center', va='top', fontsize=px(22), color=C.ink3)
        ax.text(60, y + 62, PARAM_LABEL[p], ha='left', va='center', fontsize=px(34), color=C.ink)
        ax.text(60, y + 22, PARAM_NAME[p], ha='left', va='center', fontsize=px(23), color=C.ink2)
        ax.text(NUMX[0], y + 40, fmt_pct(round(START[p], 1)), ha='right', va='center', fontsize=px(26), color=C.ink3)
        s['band'][p] = ax.add_patch(plt.Rectangle((0, y + 3), 1, 118, fc=C.ink2, ec='none', alpha=0.0, zorder=2.5))
        s['sdb'][p], = ax.plot([], [], color=C.ink, lw=px(3), zorder=6, solid_capstyle='butt')
        s['mid'][p], = ax.plot([], [], color=C.ink, lw=px(4), zorder=7, solid_capstyle='butt')
        s['dots'][p] = ax.scatter(np.full(36, -99.), np.zeros(36), s=px(D - 1.2) ** 2, color=C[p], lw=0, zorder=4)
        s['new'][p], = ax.plot([], [], 'o', ms=px(D + 3), mfc=C[p], mec=C.ink, mew=px(2.2), zorder=5, ls='none')
        s['num'][p + 'm'] = ax.text(NUMX[1], y + 40, f'{fm_(ST[p]["mean"], DEC[p])} ± {ST[p]["se"]:.{DEC[p]}f}%', ha='right', va='center', fontsize=px(28), color=C.ink, fontweight='semibold')
        s['num'][p + 's'] = ax.text(NUMX[2], y + 40, f'{ST[p]["sd"]:.{DEC[p]}f}%', ha='right', va='center', fontsize=px(28), color=C.ink)
    s['ct'] = fig.text(1860 / W, 1012 / H, '', ha='right', va='baseline', fontsize=px(36), color=C.ink, fontweight='semibold')
    return s


def draw(s, i, t):
    n = int(np.searchsorted(TL, t + 1e-6))     # samples released so far
    ydrop = np.zeros(36); al = np.zeros(36)
    u = np.clip((t - (TL - DROP)) / DROP, 0, 1)
    for p in PARAMS:
        y = Y[p] + 8 + SW[p] + D / 2 - 7
        top = Y[p] + 124
        yy = top + (y - top) * u ** 2
        ao = np.clip(u / 0.25, 0, 1)
        rgba = np.tile(to_rgba(C[p]), (36, 1)); rgba[:, 3] = ao
        d = s['dots'][p]; d.set_offsets(np.c_[XP[p], yy]); d.set_facecolors(rgba)
        last = n - 1
        if 0 <= last < 36 and t < TLAND:
            s['new'][p].set_data([XP[p][last]], [yy[last]])
        else:
            s['new'][p].set_data([], [])
        a = fade(t, T_MEAN, 1e9, 0.8); m, e = xpos(p, ST[p]['mean']), XW * ST[p]['se'] / HALF[p]
        b = s['band'][p]; b.set_x(m - e); b.set_width(2 * e); b.set_alpha(0.38 * a)
        s['mid'][p].set_data([m, m], [Y[p] + 3, Y[p] + 121]); s['mid'][p].set_alpha(a)
        s['num'][p + 'm'].set_alpha(a)
        a = fade(t, T_SD, 1e9, 0.8); sd = XW * ST[p]['sd'] / HALF[p]
        yb = Y[p] + 113
        s['sdb'][p].set_data([m - sd, m - sd, m + sd, m + sd], [yb - 10, yb, yb, yb - 10]); s['sdb'][p].set_alpha(a)
        s['num'][p + 's'].set_alpha(a)
    s['num']['mean ± SE'].set_alpha(fade(t, T_MEAN, 1e9, 0.8)); s['num']['spread (SD)'].set_alpha(fade(t, T_SD, 1e9, 0.8))
    c = int(np.sum(TL <= t))
    s['ct'].set_text(f'{c} of 36' if c < 36 else '36 samples')
    if 0 < c < 36:
        s['ct'].set_text(f'sample {SMP[c - 1]}   ({c} of 36)')
    if c == 0:
        s['ct'].set_text('')
    cap, a = caption_at(t, CAPS)
    s['sub'].set_text(cap); s['sub'].set_alpha(a)


render(args, 'a11_ensemble_landing', DUR, setup, draw,
       beats=[(0, 'five parameters, start values'), (float(TLAND + 0.6), 'all 36 samples landed'), (float(T_MEAN + 1.5), 'mean and standard error'),
              (float(T_SD + 1.5), 'spread'), (float(T_BIAS + 1.5), 'biases stated')])
