"""a03: how drifting charge becomes current. (1) The real field-response table response_44.npy at a real sub-pixel position: the current induced by
one electron on the pixel below it and on its neighbours, as it arrives. (2) One real pixel of the main track: the production simulate_signals evaluated
piece by piece (0.5 mm of track at a time; linear in the charge), adding up to the pixel's real waveform.
Table axes, from sim_jax.simulate_drift_new + _simulate_signals_impl: [i, j, t]; i, j = floor(|distance from the centre of the read pixel| / 0.4434 mm) in x and y,
t = 0.1 us samples, the charge arrives at the end of the array. Data: a01_03_sim.npz (a0x_extract.py), a03_pixel.npz (a0x_extract2.py)."""
from common import *
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle, Polygon

args = parse_args()
Z = np.load(f'{DATA}/a01_03_sim.npz'); G = np.load(f'{DATA}/a03_pixel.npz')
raw = Z['raw_tail']                        # (45, 45, 400): last 40 us of the table, 1/us per electron
PITCH = float(Z['pitch']) * 10             # mm
vd = float(Z['vd']) * 10                   # mm/us
TS = float(Z['t_sampling'])
wg = G['wg']; PX = G['px']; full = G['full']; gt, gn, gx, gy, gsx, gsy = G['gt'], G['gn'], G['gx'], G['gy'], G['gsx'], G['gsy']
tb = Z['tpc_borders']; X0, Y0 = tb[0, 0, 0], tb[0, 1, 0]
cx = X0 + (PX[0] + .5) * Z['pitch']; cy = Y0 + (PX[1] + .5) * Z['pitch']         # pixel centre, cm
ngr = len(gt)
# reference piece: the group landing closest to the centre of the pixel
g0 = int(np.argmin(np.hypot(gx - cx, gy - cy)))
dx0 = (gx[g0] - cx) * 10; dy0 = (gy[g0] - cy) * 10            # mm
b = lambda d: int(np.floor((d + PITCH / 2) / (PITCH / 10)))    # sub-pixel bin inside the pixel (0..9)
bx, by = b(dx0), b(dy0)
idx = lambda bb, k: int(abs(bb - 5 + 0.5 - 10 * k))            # sim_jax: |b % n - n/2 + 1/2 - n*k| -> int
print('reference piece', g0, 'offset mm', dx0, dy0, 'bins', bx, by, 'centre idx', idx(bx, 0), idx(by, 0), 'n_e', gn[g0], 't', gt[g0])
Ntab = raw.shape[-1]
tt = (np.arange(Ntab) - (Ntab - 1)) * TS                         # us relative to the last sample of the table (index 1890)
END = float(tt[np.max(np.nonzero(raw[idx(bx, 0), idx(by, 0)])[0])])                 # end of the pulse on the collecting pixel
tr = tt - END                                                     # time relative to the end of the undiffused pulse
i_c = raw[idx(bx, 0), idx(by, 0)]
i_n1 = raw[idx(bx, 1), idx(by, 0)]                                 # next pixel in +x
i_n1b = raw[idx(bx, -1), idx(by, 0)]
i_n2 = raw[idx(bx, 1), idx(by, 1)]                                 # diagonal
i_n3 = raw[idx(bx, 2), idx(by, 0)]
qc = round(float(Z['raw_sum'][idx(bx, 0), idx(by, 0)]) * TS, 2) + 0.0; qn = round(float(Z['raw_sum'][idx(bx, 1), idx(by, 0)]) * TS, 2) + 0.0   # full table, 189.1 us
print('peak central', i_c.max(), 'nearest neighbour min/max', i_n1.min(), i_n1.max(), 'other side', i_n1b.min(), 'diag', i_n2.min(), 'two away', i_n3.min(), 'net charges last 40 us', qc, qn, 'full table sums', Z['raw_sum'][idx(bx, 0), idx(by, 0)] * TS, Z['raw_sum'][idx(bx, 1), idx(by, 0)] * TS, 'END', END)
print('offsets of arrival: pulse end index', END)
TW0, TW1 = -12.0, 1.0
S25 = np.array([[raw[idx(bx, k), idx(by, l)] for l in range(-2, 3)] for k in range(-2, 3)])     # current on the 5 x 5 patch, real table indices
# pixel waveform
tk = np.arange(len(full)) * TS + TS   # wfs[:, 1:] dropped tick 0
pk = int(full.argmax()); W0, W1 = tk[pk] - 11, tk[pk] + 6
sel = (tk >= W0) & (tk <= W1)
QT = full.sum() * TS
above = tk[full > 0.05 * full.max()]; WID = above.max() - above.min()
print('width above 5% of peak (us)', WID, 'waveform min', full.min())
order = np.argsort(gt)
csum = np.cumsum(wg[order], axis=0)
contrib_peak = np.abs(wg).max(1)
contrib = contrib_peak > 0                                         # pieces beyond the +-4 neighbour block add exactly zero to this pixel
print('pieces contributing', contrib.sum(), 'of', ngr, 'gsx-px of the zero ones', np.unique(np.round(np.abs(gsx[~contrib] - PX[0]))))
print('pixel', PX, 'Q', QT, 'peak', full.max(), 'peak time', tk[pk], 'groups', ngr, 'peak contrib max', contrib_peak.max(), 'groups with >1% peak', (contrib_peak > 0.01 * full.max()).sum(), 'total ne', gn.sum())
landing = (np.round(gsx) == PX[0]) & (np.round(gsy) == PX[1])
print('groups landing on the pixel', landing.sum(), 'ne', gn[landing].sum(), 'rows near', len(G['idx']))

T1, T2, T3, T4, T5, T6, DUR = 0.0, 5.5, 12.0, 16.5, 19.0, 33.0, 38.0   # intro, neighbours, net charge, cut to pixel, build-up, end

def setup():
    fig = new_figure()
    a, sub = title(fig, 'Drifting charge induces current on every pixel around it')
    footer(fig, 'Simulation at truth parameters: field-response table and production waveform. Side view: charge height at uniform drift speed', 'response_44.npy; example event, main track, pixel (%d, %d)' % tuple(PX))
    s = dict(fig=fig, sub=sub, ttl=a)
    # left: side view of the pixel row
    axS = fig.add_axes(box_px(60, 130, 820, 760)); axS.axis('off')
    axS.set_xlim(-2.9 * PITCH, 2.9 * PITCH); axS.set_ylim(-2.2, 22.5)
    s['axS'] = axS
    s['pix'] = []
    for k in range(-2, 3):
        c = C.ink if k == 0 else (C.ink2 if abs(k) == 1 else C.ink3)
        r = axS.add_patch(Rectangle(((k - .44) * PITCH, -1.2), .88 * PITCH, 1.2, fc=c, ec='none', zorder=3)); s['pix'].append(r)
    s['anode'], = axS.plot([-2.9 * PITCH, 2.9 * PITCH], [-1.2, -1.2], color=C.spine, lw=px(2), zorder=2)
    s['dot'] = axS.scatter([0], [20], s=px(26) ** 2, color=C.ink, linewidths=0, zorder=5)
    s['lab_d'] = axS.text(0.5 * PITCH, 21.5, '', fontsize=px(26), color=C.ink2, va='top')
    axP = fig.add_axes(box_px(66, 590, 300, 300)); axP.set_aspect('equal'); axP.axis('off'); axP.set_xlim(-2.5 * PITCH, 2.5 * PITCH); axP.set_ylim(-2.5 * PITCH, 2.5 * PITCH)
    cellsP = [Rectangle(((k - .46) * PITCH, (l - .46) * PITCH), .92 * PITCH, .92 * PITCH) for k in range(-2, 3) for l in range(-2, 3)]
    s['cellsP'] = PatchCollection(cellsP, facecolors=C.panel, edgecolors='none'); axP.add_collection(s['cellsP'])
    axP.scatter([dx0], [dy0], s=px(10) ** 2, color=C.bg, linewidths=px(1.5), edgecolors=C.ink3, zorder=5)
    axP.text(-2.46 * PITCH, -2.62 * PITCH, 'top view of the patch:\nbrightness: |current|, compressed', fontsize=px(22), color=C.ink2, va='top', linespacing=1.3)
    s['axP'] = axP
    s['lab_p'] = [axS.text(k * PITCH, -2.0, t, fontsize=px(22), color=C.ink2, ha='center', va='top') for k, t in ((0, 'pixel'), (1, 'neighbour'), (2, 'two away'))]
    s['pitch'] = axS.annotate('', xy=(-2 * PITCH + .0, 1.2), xytext=(-1 * PITCH, 1.2), arrowprops=dict(arrowstyle='<->', color=C.ink3, lw=px(2)))
    s['pitcht'] = axS.text(-1.5 * PITCH, 1.7, f'{PITCH:.2f} mm', fontsize=px(22), color=C.ink3, ha='center', va='bottom')
    # right: current on the three pixels
    axC = fig.add_axes(box_px(1010, 560, 850, 340)); style_axes(axC)
    axN = fig.add_axes(box_px(1010, 130, 850, 340)); style_axes(axN)
    for a_, yl in ((axC, (-0.08 * i_c.max(), 1.18 * i_c.max())), (axN, (1.15 * min(i_n1.min(), i_n1b.min()), 0.35 * max(-i_n1.min(), 0.1)))):
        a_.set_xlim(TW0, TW1); a_.set_ylim(*yl); a_.set_xticks([-12, -8, -4, 0]); a_.axhline(0, color=C.spine, lw=px(1.5), zorder=1)
    axN.set_xlabel('time to arrival (µs)'); axC.tick_params(labelbottom=False)
    axC.set_ylabel('current per electron\n(e/µs)', fontsize=px(24)); axN.set_ylabel('current per electron\n(e/µs)', fontsize=px(24))
    s['axC'], s['axN'] = axC, axN
    s['lC'], = axC.plot([], [], color=C.ink, lw=px(3.5), zorder=4)
    s['lN1'], = axN.plot([], [], color=C.ink2, lw=px(3), zorder=4)
    s['lN3'], = axN.plot([], [], color=C.ink3, lw=px(3), zorder=4)
    s['tC'] = axC.text(0.0, 1.03, 'pixel under the charge', transform=axC.transAxes, fontsize=px(26), color=C.ink2, va='bottom')
    s['tQ'] = axC.text(1.0, 1.03, '', transform=axC.transAxes, fontsize=px(24), color=C.ink2, va='bottom', ha='right')
    s['tN'] = axN.text(0.0, 1.03, 'neighbours (y-axis zoomed)', transform=axN.transAxes, fontsize=px(26), color=C.ink2, va='bottom')
    s['tQn'] = axN.text(1.0, 1.03, '', transform=axN.transAxes, fontsize=px(24), color=C.ink2, va='bottom', ha='right')
    s['cur'] = [axC.axvline(0, color=C.ink3, lw=px(1.5), ls=(0, (2, 3))), axN.axvline(0, color=C.ink3, lw=px(1.5), ls=(0, (2, 3)))]
    # part 2: top view of the patch + waveform
    axT = fig.add_axes(box_px(60, 130, 760, 760)); axT.set_aspect('equal'); axT.axis('off')
    R = 4.5
    axT.set_xlim(-R * PITCH - 1, R * PITCH + 1); axT.set_ylim(-R * PITCH - 1, R * PITCH + 1)
    s['axT'] = axT; cells = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            cells.append(Rectangle(((i - .46) * PITCH, (j - .46) * PITCH), .92 * PITCH, .92 * PITCH))
    s['cells'] = PatchCollection(cells, facecolors=C.panel, edgecolors='none', zorder=1); axT.add_collection(s['cells'])
    s['ctr'] = axT.add_patch(Rectangle((-.46 * PITCH, -.46 * PITCH), .92 * PITCH, .92 * PITCH, fc='none', ec=C.ink, lw=px(3), zorder=4))
    s['seg'] = axT.scatter([], [], s=1, linewidths=0, zorder=5)
    axT.text(0, 0.56 * PITCH, 'this pixel', fontsize=px(24), color=C.ink2, ha='center', va='bottom', zorder=6)
    s['lt'] = axT.text(0, -4.7 * PITCH, '', fontsize=px(24), color=C.ink2, ha='center', va='top')
    axW = fig.add_axes(box_px(930, 450, 930, 450)); style_axes(axW); axG = fig.add_axes(box_px(930, 150, 930, 240)); style_axes(axG)
    axW.set_xlim(W0 - tk[pk], W1 - tk[pk]); axW.set_ylim(1.15 * min(full.min(), wg.cumsum(0).min()) / 1e3, 1.12 * full.max() / 1e3)
    axW.axhline(0, color=C.spine, lw=px(1.5), zorder=1); axW.tick_params(labelbottom=False); axW.set_ylabel('current on this pixel\n(ke/µs)', fontsize=px(24))
    gmax = wg.max() / 1e3; gmin = wg.min() / 1e3
    axG.set_xlim(W0 - tk[pk], W1 - tk[pk]); axG.set_ylim(1.2 * gmin, 1.2 * gmax); axG.axhline(0, color=C.spine, lw=px(1.5), zorder=1); axG.set_xticks([-10, -5, 0, 5]); axG.set_xlabel('time around the peak (µs)'); axG.set_ylabel('each 0.5 mm piece\n(ke/µs)', fontsize=px(24))
    axW.set_xticks([-10, -5, 0, 5])
    s['axW'], s['axG'] = axW, axG
    s['lc'] = LineCollection([], colors=C.ink3, linewidths=px(1.5), zorder=2, alpha=0.8); axG.add_collection(s['lc'])
    s['lW'], = axW.plot([], [], color=C.ink, lw=px(4), zorder=5)
    s['lW1'], = axG.plot([], [], color=C.ink, lw=px(3.5), zorder=4)
    s['tW'] = axW.text(0.985, 0.62, '', transform=axW.transAxes, fontsize=px(26), color=C.ink2, va='top', ha='right', linespacing=1.35)
    s['tsel'] = tk[sel] - tk[pk]
    wgt = np.sqrt(contrib_peak[order] / contrib_peak.max()) + 0.03          # pieces that matter for this pixel get more screen time
    s['arr'] = np.cumsum(wgt) / wgt.sum()
    s['veil'] = fig.add_artist(Rectangle((0, 60 / H), 1, 850 / H, transform=fig.transFigure, fc=C.bg, ec='none', zorder=50, alpha=0))
    s['lbN'] = axN.text(-3.3, -0.052, 'neighbour', fontsize=px(24), color=C.ink2, ha='right', va='center')
    s['lbA'] = axN.text(-0.9, 0.013, 'two away', fontsize=px(24), color=C.ink3, ha='center', va='bottom')
    return s

def set_vis(arts, on):
    for a in arts:
        a.set_visible(on)

def caption(t):
    for t0, t1, txt in ((0.3, T2, 'One electron, one position inside a real pixel: current from the field-response table'),
                        (T2, T3 - 0.6, 'Neighbouring pixels see it too, with a current that changes sign'),
                        (T3 - 0.6, T4, f'Net charge: 1 electron on the collecting pixel, 0 on every neighbour'),
                        (T4, T5, 'Now a real pixel on the muon track'),
                        (T5, T6 - 2, f'Every 0.5 mm piece within reach adds its own current: on its pixel, and on the neighbours'),
                        (T6 - 2, DUR + 1, f'The real waveform of one pixel: {QT / 1e3:.1f} ke collected in about {WID:.0f} µs')):
        if t0 <= t < t1:
            return txt, fade(t, t0, t1 - 0.3, 0.3)
    return '', 0.0

def draw(s, i, t):
    part1 = t < T4
    set_vis([s['axS'], s['axC'], s['axN'], s['axP']], part1)
    set_vis([s['axT'], s['axW'], s['axG']], not part1)
    txt, al = caption(t); s['sub'].set_text(txt); s['sub'].set_alpha(al)
    s['veil'].set_alpha(float(ease(seg(t, T4 - 0.35, T4)) * (1 - ease(seg(t, T4, T4 + 0.35)))))
    if part1:
        p = seg(t, T1 + 1.0, T3 - 0.8)
        tn = lerp(TW0, TW1, 1 - (1 - p) ** 1.8)                     # slows down through the signal
        m = tr <= tn
        s['lC'].set_data(tr[m], i_c[m]); s['lN1'].set_data(tr[m], i_n1[m]); s['lN3'].set_data(tr[m], i_n3[m])
        s['lbN'].set_alpha(float(ease(seg(tn, -3.2, -2.6))))           # inline labels appear once the cursor has passed them
        s['lbA'].set_alpha(float(ease(seg(tn, -0.1, 0.5))))
        for c in s['cur']:
            c.set_xdata([tn, tn])
        d = max(-tn * vd, 0.0)
        cur = np.array([abs(float(np.interp(tn, tr, S25[k + 2, l + 2]))) for k in range(-2, 3) for l in range(-2, 3)])
        br = np.clip(cur / i_c.max(), 0, 1) ** 0.35
        c0, c1 = np.array(to_rgba(C.panel)), np.array(to_rgba(C.ink))
        s['cellsP'].set_facecolors(c0[None, :] * (1 - br[:, None]) + c1[None, :] * br[:, None])
        s['dot'].set_offsets([[dx0, d]])
        s['dot'].set_alpha(1.0 if tn < 0 else 0.45)
        s['lab_d'].set_text(f'{-tn:.1f} µs before arrival' if tn < 0 else 'charge collected')
        a = ease(seg(t, T3 - 0.6, T3 + 0.1))
        s['tQ'].set_text(f'net charge = {qc:.2f} e'); s['tQ'].set_alpha(a)
        s['tQn'].set_text(f'net charge = {qn:.2f} e'); s['tQn'].set_alpha(a)
        return
    # part 2
    p = ease(seg(t, T5, T6 - 1.0))
    arr = s['arr']
    on = arr <= p if p > 0 else np.zeros(ngr, bool)
    n = int(on.sum()); nc = int(contrib[order[:n]].sum())               # the count shown: contributing pieces only
    tsel = s['tsel']
    ka = lambda a: a[sel] / 1e3
    segs = [np.c_[tsel, ka(wg[order[k]])] for k in range(n) if contrib_peak[order[k]] > 0.003 * full.max()]
    s['lc'].set_segments(segs)
    s['lW1'].set_data(tsel, ka(wg[order[n - 1]])) if 0 < n < ngr else s['lW1'].set_data([], [])
    if n > 0:
        s['lW'].set_data(tsel, ka(csum[n - 1]))
    else:
        s['lW'].set_data([], [])
    u = (gx[order] - cx) * 10; v = (gy[order] - cy) * 10
    show = np.abs(u) < 4.5 * PITCH
    s['seg'].set_offsets(np.c_[u[show], v[show]])
    fresh = np.clip(1 - (p - arr[show]) / 0.04, 0, 1) * (arr[show] <= p)
    col = np.array([to_rgba(C.ink3, 0.35)] * show.sum())
    col[arr[show] <= p] = to_rgba(C.ink2, 0.9)
    s['seg'].set_facecolors(col)
    s['seg'].set_sizes(px(12) ** 2 * (1 + 1.2 * fresh))
    s['lt'].set_text('top view: the 9 x 9 patch around the pixel, with the track above it')
    if t > T6 - 2:
        s['tW'].set_text(f'sum of {nc} pieces\n{QT / 1e3:.1f} ke on this pixel')
    else:
        s['tW'].set_text(f'{nc} pieces added')

BEATS = [(0, 'one electron approaches'), (T2, 'neighbours respond'), (T3, 'net charge'), (T4, 'real pixel on the track'), (T5, 'pieces add up'), (T6, 'real waveform: hand-off to electronics')]
render(args, 'a03_cloud_to_current', duration=DUR, setup=setup, draw=draw, beats=BEATS)
