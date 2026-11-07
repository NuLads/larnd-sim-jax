"""a01: orientation in 3-D. The real Module-0 volume, the three true muon tracks of the example event, ionisation charge sampled from the real
segments (weighted by the production electron counts) drifting to the anode at the truth drift velocity, the real target hits, and the hits
reassembled in 3-D from (pixel x, pixel y, tick). Own perspective projection onto a 2-D axes (full draw-order control).
The view is rotated so that the anode is the floor: the real drift axis is horizontal. Data: data/a01_03_sim.npz (a0x_extract.py)."""
from common import *
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

args = parse_args()
Z = np.load(f'{DATA}/a01_03_sim.npz')
seg = Z['seg']; X, Y, ZZ, TID = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 12]
nq, tdrift = Z['nq'], Z['tdrift']
tb = Z['tpc_borders']; ZA = float(tb[0, 2, 0]); ZC = float(tb[0, 2, 1]); XL, XH = tb[0, 0]; YL, YH = tb[0, 1]
vd = float(Z['vd']); TS = float(Z['t_sampling'])
hx, hy, hz, ht, hq = [Z[k].astype(float) for k in ('hit_x', 'hit_y', 'hit_ztruth', 'hit_t', 'hit_q')]
ht_us = ht * TS
FULLD = float(Z['drift_length']) / vd
NCH = 1800
rng = np.random.default_rng(11)
iS = rng.choice(len(nq), NCH, p=nq / nq.sum())
PERDOT = nq.sum() / NCH
TOTAL = nq.sum()
print('electrons released', TOTAL, 'per dot', PERDOT, 'v mm/us', vd * 10, 'full drift us', FULLD, 'max tdrift', tdrift.max(), 'last hit us', ht_us.max())

# beats (seconds)
T_VOL, T_TRK, T_CH, T_DR, T_HOLD, T_UP, T_END, DUR = 0.0, 4.5, 8.5, 12.0, 22.5, 26.0, 32.0, 36.0
CLOCK_MAX = 110.0                                                       # us shown during the drift beat

def cam(t):
    z = ease(seg_(t, 7.0, 11.5))
    r = ease(seg_(t, T_UP - 0.5, T_END))
    return dict(yaw=np.radians(lerp(lerp(-50, -52, z), -30, r)), tilt=np.radians(lerp(lerp(22, 20, z), 24, r)),
                k=lerp(7.9, lerp(11.6, 12.4, r), z), u0=lerp(960, lerp(945, 960, r), z), v0=lerp(330, lerp(400, 400, r), z), f=620.0)

seg_ = lambda t, a, b: float(clamp((t - a) / (b - a)))

def proj(x, y, z, c):
    """World (code coordinates, cm) -> design px. Long axis y across the screen, x into the screen, height above the event's anode = ZA - z."""
    A, B, H = np.asarray(y, float), np.asarray(x, float), ZA - np.asarray(z, float)
    a = A * np.cos(c['yaw']) + B * np.sin(c['yaw']); b = -A * np.sin(c['yaw']) + B * np.cos(c['yaw'])
    v = H * np.cos(c['tilt']) + b * np.sin(c['tilt']); d = b * np.cos(c['tilt']) - H * np.sin(c['tilt'])
    s = c['f'] / (c['f'] + d)
    return c['u0'] + c['k'] * s * a, c['v0'] + c['k'] * s * v, s

def rect(zv, c, x0=XL, x1=XH, y0=YL, y1=YH):
    u, v, _ = proj([x0, x0, x1, x1], [y0, y1, y1, y0], [zv] * 4, c)
    return np.c_[u, v]

def setup():
    fig = new_figure()
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    a, b = title(fig, 'A muon leaves charge; the pixels turn it back into 3-D')
    footer(fig, 'Simulation: true muon segments and noisy target hits at known truth. View rotated: the anode is drawn as the floor', 'Example event: sample 8, batch 35, event 2')
    s = dict(fig=fig, ax=ax, sub=b)
    s['floor'] = ax.add_patch(Polygon(np.zeros((4, 2)), closed=True, fc=C.spine, ec=C.ink2, lw=px(3.5), zorder=1))
    s['cath'] = ax.add_patch(Polygon(np.zeros((4, 2)), closed=True, fc=C.spine, ec=C.ink3, lw=px(2.5), zorder=2))
    s['tiles'] = LineCollection([], colors=C.ink3, linewidths=px(2), zorder=1.5); ax.add_collection(s['tiles'])
    s['box'] = LineCollection([], colors=C.ink3, linewidths=px(2.5), zorder=2.2); ax.add_collection(s['box'])
    s['trk'] = [ax.plot([], [], color=C.truth, lw=px(4.5), ls=(0, (4, 2.5)), zorder=4, dash_capstyle='butt')[0] for _ in range(3)]
    s['ch'] = ax.scatter([], [], s=1, color=C.ink2, linewidths=0, zorder=5)
    s['glow'] = ax.scatter([], [], s=1, color=C.data, linewidths=0, zorder=6)
    s['hit'] = ax.scatter([], [], s=1, linewidths=0, zorder=7)
    s['arrow'], = ax.plot([], [], color=C.ink3, lw=px(3), zorder=3, solid_capstyle='butt')
    s['head'] = ax.add_patch(Polygon(np.zeros((3, 2)), closed=True, fc=C.ink3, ec='none', zorder=3))
    lk = dict(fontsize=px(24), color=C.ink2, zorder=9)
    s['lab_an'] = ax.text(0, 0, 'anode: pixel plane', ha='left', va='top', **lk)
    s['lab_ca'] = ax.text(0, 0, 'cathode', ha='left', va='center', **lk)
    s['lab_an2'] = ax.text(0, 0, 'second anode', ha='left', va='bottom', **lk)
    s['lab_dr'] = ax.text(0, 0, 'drift', ha='right', va='center', **lk)
    s['clock'] = fig.text(1860 / W, 1006 / H, '', fontsize=px(52), fontweight='semibold', color=C.ink, ha='right', va='baseline')
    s['clock2'] = fig.text(1860 / W, 968 / H, '', fontsize=px(26), color=C.ink2, ha='right', va='baseline')
    s['cm'] = cmap_to(C.data)
    trk = []
    for tid in (0.0, 2.0, 4.0):
        m = np.where(TID == tid)[0]; sel = m[np.linspace(0, len(m) - 1, 120).astype(int)]
        trk.append((X[sel], Y[sel], ZZ[sel]))
    s['trk_xyz'] = trk
    s['hcol'] = s['cm'](np.clip((hq + 12) / 40, 0, 1))                     # keeps the weakest hits bright on a projector
    return s

def caption(t):
    for t0, t1, txt in ((0.4, T_TRK, 'Module-0: two drift volumes share a cathode. 62 × 124 cm anodes, 30.4 cm drift, 78 400 pixels of 4.43 mm'),
                        (T_TRK, T_CH, 'Three muons cross one drift volume: 36, 52 and 63 cm of track'),
                        (T_CH, T_DR, f'They leave {TOTAL / 1e6:.1f} million free electrons. Each dot = {PERDOT / 1e3:.0f} k electrons, sampled from the real segments'),
                        (T_DR, T_HOLD, f'The charge drifts to the anode at {vd * 10:.2f} mm/µs: {tdrift.max():.0f} µs at most in this event, {FULLD:.0f} µs for a full drift'),
                        (T_HOLD, T_UP, f'The pixels record {len(hq)} hits: which pixel, and when'),
                        (T_UP, DUR + 1, 'Pixel position + hit time × drift velocity = the event in 3-D')):
        if t0 <= t < t1:
            return txt, fade(t, t0, t1 - 0.35, 0.35)
    return '', 0.0

def draw(s, i, t):
    c = cam(t)
    fig, ax = s['fig'], s['ax']
    a_vol = ease(seg_(t, T_VOL, T_VOL + 1.2))
    # volume
    fl = rect(ZA, c); s['floor'].set_xy(fl); s['floor'].set_facecolor(to_rgba(C.spine, 0.6 * a_vol)); s['floor'].set_edgecolor(to_rgba(C.ink2, a_vol))
    ca = rect(ZC, c); s['cath'].set_xy(ca); s['cath'].set_facecolor(to_rgba(C.spine, 0.28 * a_vol)); s['cath'].set_edgecolor(to_rgba(C.ink3, a_vol))
    top = rect(-ZA, c)
    zoom = ease(seg_(t, 7.0, 10.0))
    a_top = a_vol * (1 - zoom)
    # tile grid: 70 x 70 pixels each
    lines = []
    for xv in (0.0,):
        u, v, _ = proj([xv, xv], [YL, YH], [ZA, ZA], c); lines.append(np.c_[u, v])
    for yv in (-31.038, 0.0, 31.038):
        u, v, _ = proj([XL, XH], [yv, yv], [ZA, ZA], c); lines.append(np.c_[u, v])
    s['tiles'].set_segments(lines); s['tiles'].set_alpha(a_vol)
    edges = []
    for k in range(4):
        edges.append(np.vstack([fl[k], ca[k]]))
    s['box'].set_segments(edges + [np.vstack([ca[k], top[k]]) for k in range(4)] + [np.vstack([top[k], top[(k + 1) % 4]]) for k in range(4)])
    al = np.array([a_vol] * 4 + [a_top] * 8)
    s['box'].set_alpha(None); s['box'].set_color([to_rgba(C.ink3, x) for x in al])
    # tracks
    g = ease(seg_(t, T_TRK, T_TRK + 2.5))
    dim = lerp(1.0, 0.6, ease(seg_(t, T_UP + 1.0, T_UP + 3.5)))
    for ln, (tx, ty, tz) in zip(s['trk'], s['trk_xyz']):
        n = int(round(g * len(tx)))
        if n < 2:
            ln.set_data([], []); continue
        u, v, _ = proj(tx[:n], ty[:n], tz[:n], c); ln.set_data(u, v); ln.set_alpha(dim * 0.9)
    # charge
    tc = lerp(0.0, CLOCK_MAX, seg_(t, T_DR, T_HOLD))
    n_on = int(round(NCH * ease(seg_(t, T_CH, T_CH + 1.6))))
    if n_on > 0:
        idx = iS[:n_on]
        z_now = np.minimum(ZZ[idx] + vd * tc, ZA)
        alive = tdrift[idx] > tc
        u, v, sc = proj(X[idx][alive], Y[idx][alive], z_now[alive], c)
        s['ch'].set_offsets(np.c_[u, v]); s['ch'].set_sizes(np.full(alive.sum(), px(6.5) ** 2 * lerp(1.0, 1.2, zoom)))
        s['ch'].set_alpha(0.75)
    else:
        s['ch'].set_offsets(np.empty((0, 2)))
    # hits
    on = (ht_us <= tc) & (t >= T_DR)
    r = ease(seg_(t, T_UP + 0.4, T_END - 1.0))
    if on.any():
        h_z = lerp(ZA, 1, 0) * np.ones(on.sum())
        stag = np.clip((r * 1.25 - 0.25 * ht_us[on] / ht_us.max()), 0, 1) if r > 0 else np.zeros(on.sum())
        zh = ZA + (hz[on] - ZA) * stag
        u, v, sc = proj(hx[on], hy[on], zh, c)
        age = (tc - ht_us[on]) / CLOCK_MAX * (T_HOLD - T_DR)                 # seconds since firing
        fl_ = np.exp(-np.clip(age, 0, None) / 0.28) * (t < T_HOLD + 0.8)
        s['hit'].set_offsets(np.c_[u, v]); s['hit'].set_sizes(px(10.5) ** 2 * (1 + 1.2 * fl_) * sc ** 2 * lerp(0.9, 1.1, r))
        s['hit'].set_facecolors(s['hcol'][on])
        s['glow'].set_offsets(np.c_[u, v]); s['glow'].set_sizes(px(36) ** 2 * fl_ * sc ** 2 + 0.01); s['glow'].set_alpha(0.22)
    else:
        s['hit'].set_offsets(np.empty((0, 2))); s['glow'].set_offsets(np.empty((0, 2)))
    # labels
    u, v, _ = proj([XL], [YH], [ZA], c); s['lab_an'].set_position((float(u[0]) + 14, float(v[0]) + 12))
    a_lab = a_vol * (1 - ease(seg_(t, T_UP - 0.2, T_UP + 0.8)))
    s['lab_an'].set_alpha(a_lab)
    u, v, _ = proj([XL], [YH], [ZC], c); s['lab_ca'].set_position((float(u[0]) + 14, float(v[0]))); s['lab_ca'].set_alpha(a_vol * (1 - ease(seg_(t, T_UP - 0.2, T_UP + 0.8))))
    u, v, _ = proj([XL], [YH], [-ZA], c); s['lab_an2'].set_position((float(u[0]) + 6, float(v[0]) + 10)); s['lab_an2'].set_alpha(a_top)
    # drift arrow on the near-left corner: from cathode down to the anode
    a_ar = ease(seg_(t, T_TRK + 3.5, T_DR)) * (1 - ease(seg_(t, T_UP - 0.2, T_UP + 0.8)))
    ux, vx, _ = proj([XH, XH], [YL - 5, YL - 5], [ZC + 6, ZA - 5], c)
    s['arrow'].set_data(ux, vx); s['arrow'].set_alpha(a_ar)
    d = np.array([ux[1] - ux[0], vx[1] - vx[0]]); d = d / max(np.hypot(*d), 1e-6); n_ = np.array([-d[1], d[0]])
    p = np.array([ux[1], vx[1]]); s['head'].set_xy(np.array([p + d * 18, p + n_ * 10, p - n_ * 10])); s['head'].set_alpha(a_ar)
    s['lab_dr'].set_position((float(np.mean(ux)) - 16, float(np.mean(vx)))); s['lab_dr'].set_alpha(a_ar)
    # text
    txt, al = caption(t); s['sub'].set_text(txt); s['sub'].set_alpha(al)
    a_clk = fade(t, T_DR - 0.5, T_UP - 0.4, 0.5)
    s['clock'].set_text(f'{tc:.0f} µs'); s['clock'].set_alpha(a_clk)
    s['clock2'].set_text('drift time'); s['clock2'].set_alpha(a_clk)

BEATS = [(0, 'detector'), (T_TRK, 'muon tracks'), (T_CH, 'ionisation charge'), (T_DR, 'drift to the anode'), (T_HOLD, 'hits on the pixels'), (T_UP, 'hits reassemble in 3-D'), (T_END, 'end hold')]
render(args, 'a01_muon_to_hits', duration=DUR, setup=setup, draw=draw, beats=BEATS)
