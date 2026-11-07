"""a08: the centrepiece. One 63 cm simulated muon (sample 8, batch 35, event 2, track (2,4)): the joint fit moves calibration, path and
dE/dx together. Left: SAVED states only (never interpolated; they switch by a short crossfade when the cursor reaches them).
Right: per-iteration calibration traces of the same run (continuous). Data: $T/data/a08_pack.npz (from _a08_extract.py), which holds
hero.pkl states, the validated hit-seed state (_a08_seed.py) and the KN10N8C75H row of traces.npz."""
from common import *
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

args = parse_args()
Z = np.load(f'{DATA}/a08_pack.npz', allow_pickle=True)
Z = {k: Z[k] for k in Z.files}
STATES = ['straight', 'seed', 'iter10000', 'iter14000', 'iter20000']
ITER = [0, 0, 10000, 14000, 20000]

# ---- timeline (s) ----
T_SEED, T_RUN, T_N1, T_LEG2, T_N2, T_LEG3, T_N3, DUR = 5.0, 11.0, 19.5, 24.5, 27.0, 31.5, 34.5, 39.5
XF = 0.4
SW = [0.0, T_SEED, T_N1, T_N2, T_N3]                    # state k becomes current at SW[k]
BEATS = [(0.0, 'Input: straight line, initial calibration'), (T_SEED + XF, 'Iteration 0: path seeded from the hits'),
         (T_RUN, 'Fit runs: curves are every iteration, panels hold the last saved state'), (T_N1 + XF, 'Saved state at iteration 10 000'),
         (T_N2 + XF, 'Saved state at iteration 14 000'), (T_N3 + XF, 'Saved state at iteration 20 000, end of the fit')]
CAPTIONS = [(0.0, T_SEED, 'Input: straight line at the initial calibration. Not yet a fit iterate.'),
            (T_SEED, T_RUN, "Iteration 0: path seeded from the track's own hits. The time offset that remains is the starting electric field."),
            (T_RUN, T_N1, 'Left: this track at the last saved state. Right: the whole fit, every iteration.'),
            (T_N1, T_N2, 'Calibration, path and dE/dx have all moved in the same fit.'),
            (T_N2, T_N3, 'Longitudinal diffusion moves most in this stage; per-segment dE/dx stays noisy, about 3 % from truth.'),
            (T_N3, 1e9, 'One sample, one track. Ensemble results come later.')]


def cursor_iter(t):
    if t < T_RUN: return 0.0
    if t < T_N1: return 10000.0 * float(ease(seg(t, T_RUN, T_N1))) ** 1.7
    if t < T_LEG2: return 10000.0
    if t < T_N2: return 10000.0 + 4000.0 * float(ease(seg(t, T_LEG2, T_N2)))
    if t < T_LEG3: return 14000.0
    if t < T_N3: return 14000.0 + 6000.0 * float(ease(seg(t, T_LEG3, T_N3)))
    return 20000.0


def mix(t):
    """(previous state, current state, crossfade 0..1). Crossfade only: no shape, charge or dE/dx is ever interpolated."""
    k = max(j for j, s in enumerate(SW) if t >= s)
    return max(k - 1, 0), k, (1.0 if k == 0 else float(ease(seg(t, SW[k], SW[k] + XF))))


def fmt_pct(v):
    return ('+' if v >= 0 else '−') + f'{abs(v):.1f}'


def fmt_iter(n):
    return f'{int(n):,}'.replace(',', ' ')


# ---- derived data ----
PITCH = 0.4434
uxy, nxy, o = Z['uxy'], Z['nxy'], Z['o']
rot = lambda P: np.stack([(P - o) @ uxy, (P - o) @ nxy], -1)
corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * PITCH / 2
w0, w1 = float(Z['Lxy']) / 2 - 13, float(Z['Lxy']) / 2 + 13       # the event display's 26 cm strip around the middle
STRIP_X, STRIP_W, STRIP_H, STRIP_GAP = 205, 1095, 56, 8
hw = STRIP_H / (STRIP_W / (w1 - w0)) / 2                          # equal aspect: 1 cm is the same length along and across
pids = Z['D_pid']
for s in STATES: pids = np.union1d(pids, Z[f'{s}_pid'])
XY = np.zeros((len(pids), 2))
for src in ['D'] + STATES:
    k = np.searchsorted(pids, Z[f'{src}_pid']); XY[k, 0] = Z[f'{src}_x']; XY[k, 1] = Z[f'{src}_y']
AL = rot(XY); inwin = (AL[:, 0] > w0 - 1) & (AL[:, 0] < w1 + 1) & (np.abs(AL[:, 1]) < 1.55)
POLYS = rot(XY[:, None, :] + corners[None, :, :])[inwin]
Vd = np.full(len(pids), np.nan); Vd[np.searchsorted(pids, Z['D_pid'])] = Z['D_q']
VM, VR = [], []
for s in STATES:
    v = np.full(len(pids), np.nan); v[np.searchsorted(pids, Z[f'{s}_pid'])] = Z[f'{s}_q']
    VM.append(v[inwin]); VR.append((np.nan_to_num(Vd) - np.nan_to_num(v))[inwin])
Vd = Vd[inwin]
QMAX = float(np.nanpercentile(np.concatenate([Vd, VM[-1]]), 99))
RMAX = 8.0
_inx = (POLYS[:, :, 0].max(1) > w0) & (POLYS[:, :, 0].min(1) < w1)     # strips are centred on the pixels with real charge
_lit = _inx & ((np.nan_to_num(Vd) > 1.0) | np.any([np.nan_to_num(v) > 1.0 for v in VM], axis=0))
CY = 0.5 * float(POLYS[_lit, :, 1].min() + POLYS[_lit, :, 1].max())
if __name__ == '__main__':
    print(f'strip across range {POLYS[_lit, :, 1].min() - CY:.3f} .. {POLYS[_lit, :, 1].max() - CY:.3f} cm, half height {hw:.3f} cm; '
          f'any-charge range {POLYS[_inx, :, 1].min() - CY:.3f} .. {POLYS[_inx, :, 1].max() - CY:.3f}; QMAX {QMAX:.2f}', flush=True)
RATIO = [float(Z[f'{s}_q'].sum() / Z['D_q'].sum()) for s in STATES]
s_cm = Z['s_cm']; dev_true = Z['dev_true'][:, 1]; ded_true = Z['dedx_true']
DEV = [Z[f'{s}_dev'][:, 1] for s in STATES]; DED = [Z[f'{s}_dedx'] for s in STATES]
RMS = [float(np.sqrt(np.mean((d - dev_true) ** 2))) for d in DEV]
DRMS = [100 * float(np.sqrt(np.mean((d / ded_true - 1) ** 2))) for d in DED]
mask = Z['tmask']; sa = Z['sa'][mask]
RS = [Z[f'{s}_rs'][mask] for s in STATES]
TMEAN = [float(np.nanmean(r)) for r in RS]; TSD = [float(np.nanstd(r)) for r in RS]
true_cal = np.array([float(Z['true_' + p]) for p in PARAMS])
CAL = [100 * (Z[f'{s}_calib'] / true_cal - 1) for s in STATES]
stride = 5
IT = np.arange(0, 20001, stride)
TR = {p: 100 * (Z['tr_' + p][::stride] / float(Z['true_' + p]) - 1) for p in PARAMS}
CMIN, CMAX = min(float(TR[p].min()) for p in PARAMS) - 3, max(float(TR[p].max()) for p in PARAMS) + 3


RIGHT_L = 1300                          # right edge of the left (saved-state) column
AX_Y0, AX_Y1 = 128, 592                 # bottom panels
CX0, CW, CTOP = 1450, 410, 582          # calibration curves


def setup():
    fig = new_figure()
    a, b = title(fig, 'Calibration, path and dE/dx fitted together, seen on one muon')
    footer(fig, 'Simulated data: noisy simulation at known truth. Endpoints from simulation.',
           right='Run KN10N8C75 / FR / H, sample 8, batch 35, event 2, track (2,4). Hit seed: production code rerun on CPU')
    st = dict(fig=fig, cap=b)
    # ---- left: saved-state panels ----
    st['badge'] = fig.text(60 / W, 920 / H, '', fontsize=px(30), fontweight='semibold', color=C.ink, va='baseline')
    st['ratio'] = fig.text(RIGHT_L / W, 920 / H, '', fontsize=px(25), color=C.ink2, va='baseline', ha='right')
    y0 = 846
    st['strips'] = []
    for r, (lab, sub) in enumerate((('data', f'0 to {QMAX:.0f} ke'), ('model', 'same scale'), ('data − model', f'±{RMAX:.0f} ke'))):
        yr = y0 - r * (STRIP_H + STRIP_GAP)
        ax = fig.add_axes(box_px(STRIP_X, yr, STRIP_W, STRIP_H))
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(w0, w1); ax.set_ylim(CY - hw, CY + hw)
        fig.text(195 / W, (yr + STRIP_H / 2 + 13) / H, lab, fontsize=px(24), color=C.ink, ha='right', va='center')
        fig.text(195 / W, (yr + STRIP_H / 2 - 14) / H, sub, fontsize=px(22), color=C.ink3, ha='right', va='center')
        if r == 0: n, cm = Normalize(0, QMAX), cmap_to(C.data, 'd')
        elif r == 1: n, cm = Normalize(0, QMAX), cmap_to(C.model, 'm')
        else: n, cm = Normalize(-RMAX, RMAX), cmap_diverging()
        cm = cm.copy(); cm.set_bad((0, 0, 0, 0))
        pcs = []
        for _ in range(1 if r == 0 else 2):
            pc = PolyCollection(POLYS, cmap=cm, norm=n, lw=px(0.8), edgecolors=C.bg); pc.set_array(np.ma.masked_invalid(Vd)); ax.add_collection(pc); pcs.append(pc)
        st['strips'].append(pcs)
    # shared legend for the three panels below
    lx = 205; ly = 694
    leg = fig.add_axes([0, 0, 1, 1], label='legend'); leg.set_xlim(0, W); leg.set_ylim(0, H); leg.axis('off')
    for lw_, col, ls, text, dx in ((3, C.truth, (0, (3, 2)), 'truth', 60), (3.5, C.model, '-', 'model', 70), (3, C.start, '-', 'input: straight, flat dE/dx', 0)):
        leg.plot([lx, lx + 44], [ly, ly], color=col, lw=px(lw_), ls=ls); leg.text(lx + 54, ly, text, fontsize=px(24), color=C.ink2, va='center'); lx += 54 + dx + 44
    fig.text(RIGHT_L / W, ly / H, 'central 26 cm of the 63 cm track', fontsize=px(22), color=C.ink3, va='center', ha='right')
    # ---- trio: the path panel gets the most width ----
    A = [(60, 145, 380, 'Path, drift-carrying axis', 'deviation from straight (mm)', 'arc length (cm)'),
         (550, 635, 285, 'First-hit time', r'data $-$ model (ticks)', 'position along track (cm)'),
         (945, 1030, 270, 'Energy loss per segment', 'dE/dx (MeV/cm)', 'arc length (cm)')]
    axs = []; st['num'] = []
    for hx, x, w_, head, yl, xl in A:
        ax = fig.add_axes(box_px(x, AX_Y0, w_, AX_Y1 - AX_Y0)); style_axes(ax); ax.tick_params(labelsize=px(24))
        ax.set_ylabel(yl, fontsize=px(26), labelpad=px(3)); ax.set_xlabel(xl, fontsize=px(26), labelpad=px(1))
        fig.text(hx / W, 648 / H, head, fontsize=px(29), fontweight='semibold', color=C.ink, va='baseline')
        st['num'].append(fig.text(hx / W, 612 / H, '', fontsize=px(27), color=C.ink2, va='baseline'))
        ax.set_xticks([0, 30, 60]); axs.append(ax)
    ap, at, ad = axs
    ap.set_xlim(-2, 65); ap.set_ylim(-0.6, 6.0); ap.set_yticks([0, 2, 4])
    ap.plot(s_cm, DEV[0], color=C.start, lw=px(3))
    ap.plot(s_cm, dev_true, color=C.truth, lw=px(2.5), ls=(0, (3, 2)), zorder=4)
    st['pl'] = [ap.plot(s_cm, dev_true, color=C.model, lw=px(3.5), zorder=3)[0] for _ in range(2)]
    at.set_xlim(-2, 64); at.set_ylim(-30, 62); at.set_yticks([-20, 0, 20, 40, 60]); at.axhline(0, color=C.truth, lw=px(1.5), ls=(0, (3, 2)), zorder=1)
    at.scatter(sa, RS[0], s=px(30), color=C.start, lw=0, alpha=0.75, zorder=2)
    st['ts'] = [at.scatter(sa, RS[1], s=px(46), color=C.model, lw=0, zorder=3) for _ in range(2)]
    ad.set_xlim(-2, 65); ad.set_ylim(1.62, 2.24); ad.set_yticks([1.7, 1.9, 2.1])
    ad.plot(s_cm, DED[0], color=C.start, lw=px(3))
    ad.plot(s_cm, ded_true, color=C.truth, lw=px(2.2), ls=(0, (3, 2)), zorder=4)
    st['dl'] = [ad.plot(s_cm, ded_true, color=C.model, lw=px(2.8), zorder=3)[0] for _ in range(2)]
    # ---- right: calibration table (saved state), then the continuous curves ----
    RX = 1370
    fig.text(RX / W, 920 / H, 'Calibration, % from truth', fontsize=px(30), fontweight='semibold', color=C.ink, va='baseline')
    st['thead'] = fig.text(1860 / W, 888 / H, '', fontsize=px(23), color=C.ink2, ha='right', va='baseline')
    fig.text(1765 / W, 888 / H, 'start', fontsize=px(23), color=C.ink3, ha='right', va='baseline')
    st['tv'] = []
    tab = fig.add_axes([0, 0, 1, 1], label='tab'); tab.set_xlim(0, W); tab.set_ylim(0, H); tab.axis('off')
    for r, p in enumerate(PARAMS):
        y = 856 - r * 40
        tab.plot([RX, RX + 26], [y, y], color=C[p], lw=px(6), solid_capstyle='butt')
        fig.text((RX + 36) / W, y / H, PARAM_LABEL[p], fontsize=px(27), color=C.ink, va='center')
        fig.text((RX + 80) / W, y / H, PARAM_NAME[p], fontsize=px(24), color=C.ink2, va='center')
        fig.text(1765 / W, y / H, fmt_pct(CAL[1][r]), fontsize=px(26), color=C.ink3, va='center', ha='right')
        st['tv'].append(fig.text(1860 / W, y / H, '', fontsize=px(28), fontweight='semibold', color=C.ink, va='center', ha='right'))
    fig.text(RX / W, 658 / H, 'all 100 batches; now: mean of last 1000 iterations', fontsize=px(22), color=C.ink3, va='baseline')
    fig.text(RX / W, 612 / H, 'Every iteration', fontsize=px(30), fontweight='semibold', color=C.ink, va='baseline')
    st['ctr'] = fig.text(1860 / W, 612 / H, '', fontsize=px(30), color=C.ink, va='baseline', ha='right')
    ac = fig.add_axes(box_px(CX0, AX_Y0, CW, CTOP - AX_Y0)); style_axes(ac); ac.tick_params(labelsize=px(24))
    ac.set_xlim(0, 20000); ac.set_ylim(CMIN, CMAX); ac.set_yticks([-30, -15, 0, 15, 30]); ac.set_xticks([0, 10000, 14000, 20000])
    ac.set_xticklabels(['0', '10 000', '14 000', '20 000'])
    ac.get_xticklabels()[1].set_ha('right'); ac.get_xticklabels()[3].set_ha('right')
    ac.set_ylabel('% from truth', fontsize=px(26), labelpad=px(1)); ac.set_xlabel('iteration (diamonds: saved states)', fontsize=px(24), labelpad=px(1))
    ac.axhline(0, color=C.truth, lw=px(2), ls=(0, (3, 2)), zorder=5)
    st['cl'] = {p: ac.plot([], [], color=C[p], lw=px(3))[0] for p in PARAMS}
    for n in (10000, 14000, 20000): ac.axvline(n, color=C.spine, lw=px(1.5), zorder=0)
    cur = fig.add_axes([0, 0, 1, 1], label='cur'); cur.set_xlim(0, W); cur.set_ylim(0, H); cur.axis('off')
    st['cursor'] = cur.plot([CX0, CX0], [AX_Y0, CTOP], color=C.ink, lw=px(2.5), zorder=9)[0]
    st['nodes'] = [cur.plot([CX0 + CW * n / 20000], [CTOP + 8], marker='o' if n == 0 else 'D', ms=px(13), mfc=C.bg, mec=C.ink2, mew=px(2), ls='', zorder=10, clip_on=False)[0]
                   for n in (0, 10000, 14000, 20000)]
    return st


def draw(s, i, t):
    a, b, x = mix(t)
    show = b if x >= 0.5 else a
    s['badge'].set_text('Input, before the fit' if show == 0 else f'Saved state: iteration {fmt_iter(ITER[show])}' if show > 1 else 'Iteration 0: hit seed (reproduced offline)')
    # strips: under = previous state, over = current state fading in
    for r in (1, 2):
        V = VM if r == 1 else VR
        u, v = s['strips'][r]
        u.set_array(np.ma.masked_invalid(V[a])); v.set_array(np.ma.masked_invalid(V[b])); v.set_alpha(x)
    s['ratio'].set_text(f'charge in event, model / data = {RATIO[show]:.3f}')
    s['num'][0].set_text(f'rms to truth {RMS[show]:.2f} mm' if RMS[show] >= 0.1 else f'rms to truth {RMS[show]:.3f} mm')
    s['num'][1].set_text(f'mean {fmt_pct(TMEAN[show])}, sd {TSD[show]:.1f} ticks')
    s['num'][2].set_text(f'rms to truth {DRMS[show]:.1f} %')
    # lines and points: crossfade, the straight input stays as the grey reference
    vis = lambda k: 0.0 if k == 0 else 1.0
    for art, data in ((s['pl'], DEV), (s['dl'], DED)):
        art[0].set_ydata(data[a]); art[0].set_alpha(vis(a) * (1 - x)); art[1].set_ydata(data[b]); art[1].set_alpha(vis(b) * x)
    for k, (art, st_k, al) in enumerate(((s['ts'][0], a, vis(a) * (1 - x)), (s['ts'][1], b, vis(b) * x))):
        art.set_offsets(np.stack([sa, RS[st_k]], 1)); art.set_alpha(al)
    # calibration table: values of the saved state, cut at mid-crossfade
    for r in range(5): s['tv'][r].set_text(fmt_pct(CAL[show][r]))
    s['thead'].set_text('now' if show > 1 else '')
    for r in range(5):
        if show <= 1: s['tv'][r].set_text('')
    # continuous curves
    it = cursor_iter(t); n = int(it // stride) + 1; on = t >= T_RUN
    for p in PARAMS: s['cl'][p].set_data(IT[:n] if on else [], TR[p][:n] if on else [])
    xc = CX0 + CW * it / 20000
    s['cursor'].set_xdata([xc, xc])
    s['ctr'].set_text(f'iteration {fmt_iter(round(it))}' if t >= T_SEED else '')
    for nd, st_k in zip(s['nodes'], (1, 2, 3, 4)):
        done = b >= st_k
        nd.set_markerfacecolor(C.ink if done else C.bg)
        pulse = 1 - float(ease(seg(t, SW[st_k], SW[st_k] + 0.7))) if done else 0.0
        nd.set_markersize(px(13 + 6 * pulse))
    # one caption at a time
    for t0, t1, text in CAPTIONS:
        if t0 <= t < t1:
            s['cap'].set_text(text); s['cap'].set_alpha(fade(t, t0, 1e9, 0.4))


if __name__ == '__main__':
    render(args, 'a08_track_converging', duration=DUR, setup=setup, draw=draw, beats=BEATS)
