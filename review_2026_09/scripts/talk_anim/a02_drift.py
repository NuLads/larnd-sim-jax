"""a02: what happens to the charge on the way to the anode. One real 0.1 mm segment of the main track, at truth parameters. All numbers come from the production
quench + drift outputs saved in data/a01_03_sim.npz (a0x_extract.py); the live values use the same formulas as drifting_jax.drift and are asserted against the
saved production outputs at the end of the drift. The model is analytic: the electron dots are an illustrative draw of the real counts and of the real Gaussian widths."""
from common import *
from matplotlib.patches import Rectangle, Circle

args = parse_args()
Z = np.load(f'{DATA}/a01_03_sim.npz')
SEG = Z['seg']; X, ZZ, TID = SEG[:, 0], SEG[:, 2], SEG[:, 12]
main = np.where(TID == 4)[0]
ROW = int(main[300])
dE, dEdx, dx = SEG[ROW, 10], SEG[ROW, 11], SEG[ROW, 9]
n_ion, nq, nd, TD, sT, sL = [float(Z[k][ROW]) for k in ('n_ion', 'nq', 'nd', 'tdrift', 'sT', 'sL')]
tb = Z['tpc_borders']; ZA, ZC = float(tb[0, 2, 0]), float(tb[0, 2, 1])
vd = float(Z['vd']); TAU, DT, DL, AB, KB, EF, RHO = [float(Z[k]) for k in ('lifetime', 'DT', 'DL', 'Ab', 'kb', 'eField', 'rho')]
R = nq / n_ion
FULL = float(Z['drift_length']) / vd
N_of = lambda t: nq * np.exp(-t / TAU)
sT_of = lambda t: np.sqrt(2 * DT * t) * 10                     # mm
sL_of = lambda t: np.sqrt(2 * DL * t) * 10
assert abs(N_of(TD) - nd) / nd < 1e-3 and abs(sT_of(TD) - sT * 10) < 1e-3 and abs(sL_of(TD) - sL * 10) < 1e-3
assert abs(R - AB / (1 + KB * dEdx / (EF * RHO))) < 1e-3
print(f'row {ROW} dE {dE*1e3:.2f} keV dEdx {dEdx:.3f} dx {dx} n_ion {n_ion:.1f} nq {nq:.1f} R {R:.4f} nd {nd:.1f} TD {TD:.2f} sT {sT*10:.3f} sL {sL*10:.3f} mm; dist {vd*TD:.2f} cm; lost {100*(1-nd/nq):.2f}%; full drift {FULL:.1f}')

NI, NQ = int(round(n_ion)), int(round(nq))
rng = np.random.default_rng(5)
perm = rng.permutation(NI)
surv = np.zeros(NI, bool); surv[perm[:NQ]] = True
gD, gT = rng.standard_normal(NI), rng.standard_normal(NI)      # drift-direction and transverse draws
uL = rng.uniform(-.5, .5, NI)                                   # position along the 0.1 mm segment
uA = np.ones(NI); uA[surv] = rng.permutation((np.arange(NQ) + 0.5) / NQ)   # electron k is lost when exp(-t/tau) < uA[k]: the dots track the readout
ncol = int(np.ceil(np.sqrt(NI * 1.3))); nrow = int(np.ceil(NI / ncol))
gx, gy = np.meshgrid((np.arange(ncol) - (ncol - 1) / 2) * 0.095, (np.arange(nrow) - (nrow - 1) / 2) * 0.095)
GRID = np.c_[gx.ravel()[:NI], gy.ravel()[:NI]]
# the segment's direction, projected on (drift, transverse x)
ax_d = (SEG[ROW, 8] - SEG[ROW, 5]); ax_t = (SEG[ROW, 6] - SEG[ROW, 3])

T_ION, T_REC, T_DR, T_END_DR, T_SUM, DUR = 3.0, 7.0, 12.5, 29.0, 29.5, 37.0
TD0 = T_DR + 1.2                                                # drift clock starts after the dots collapse into the cloud
ACT = [(T_REC, T_DR, (0, 1)), (T_DR, 18.0, (1,)), (18.0, 23.0, (2,)), (23.0, T_SUM, (3, 4)), (T_SUM, 99, (0, 1, 2, 3, 4))]
KEYS = ['Ab', 'eField', 'lifetime', 'tran_diff', 'long_diff']
NAMES = [PARAM_NAME[k] for k in KEYS]
VALS = [f'{AB:.3f}', f'{EF:.3f} kV/cm', f'{TAU / 1e3:.2f} ms', f'{DT * 1e6:.1f} cm²/s', f'{DL * 1e6:.1f} cm²/s']
ROWY = [640, 540, 440, 340, 240]

def td_of(t):
    return TD * seg(t, TD0, T_END_DR)


def setup():
    fig = new_figure()
    a, sub = title(fig, 'Five parameters shape the charge on its way to the pixels')
    footer(fig, 'Simulation at truth parameters: one true segment through the production quench and drift code. Dots: illustrative draw of the analytic cloud',
           'Example event, main track')
    s = dict(fig=fig, sub=sub)
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(0, H); ov.axis('off'); ov.set_zorder(5)
    s['ov'] = ov
    # lane: drift coordinate across, lateral coordinate up (compressed)
    L = fig.add_axes(box_px(90, 730, 1740, 190)); L.axis('off'); L.set_xlim(ZC - 0.6, ZA + 0.6); L.set_ylim(-34, 34)
    s['L'] = L
    L.add_patch(Rectangle((ZC - 0.25, -32), 0.25, 64, fc=C.spine, ec='none'))
    L.add_patch(Rectangle((ZA, -32), 0.25, 64, fc=C.ink2, ec='none'))
    L.text(ZC + 0.25, 4, 'cathode', fontsize=px(24), color=C.ink2, va='center')
    L.text(ZA - 0.25, 4, 'anode (pixels)', fontsize=px(24), color=C.ink2, va='center', ha='right')
    L.plot(ZZ[main[::40]], X[main[::40]], color=C.truth, lw=px(2.5), ls=(0, (4, 2.5)), zorder=2)
    s['pc'] = L.scatter([ZZ[ROW]], [X[ROW]], s=px(20) ** 2, color=C.ink2, linewidths=0, zorder=4)
    s['pl'], = L.plot([], [], color=C.ink3, lw=px(2), ls=(0, (2, 3)), zorder=1)
    L.annotate('', xy=(11, -27), xytext=(5, -27), arrowprops=dict(arrowstyle='-|>', color=C.ink3, lw=px(3)))
    L.text(4.7, -27, 'drift', fontsize=px(24), color=C.ink2, ha='right', va='center')
    # zoom panel
    P = fig.add_axes(box_px(60, 130, 560, 540)); P.set_xlim(-1.76, 1.76); P.set_ylim(-1.70, 1.70); P.set_aspect('equal')
    P.set_facecolor(C.panel); P.set_xticks([]); P.set_yticks([])
    for sp in P.spines.values():
        sp.set_visible(False)
    s['P'] = P
    s['dots'] = P.scatter(np.zeros(NI), np.zeros(NI), s=px(6.5) ** 2, color=C.ink2, linewidths=0, zorder=3)
    s['ptxt'] = P.text(-1.68, 1.62, '', fontsize=px(24), color=C.ink2, va='top')
    s['brL'], = P.plot([], [], color=C.long_diff, lw=px(4), zorder=4, solid_capstyle='butt')
    s['brT'], = P.plot([], [], color=C.tran_diff, lw=px(4), zorder=4, solid_capstyle='butt')
    s['tL'] = P.text(0, -1.42, '', fontsize=px(24), color=C.ink2, ha='center', va='top')
    s['tT'] = P.text(-1.56, 0, '', fontsize=px(24), color=C.ink2, ha='left', va='center', rotation=90)
    s['barl'], = P.plot([1.64 - 1, 1.64], [-1.56, -1.56], color=C.ink3, lw=px(3)); s['bar'] = P.text(1.14, -1.50, '1 mm', fontsize=px(22), color=C.ink3, ha='center', va='bottom')
    # parameter rows
    s['rows'] = []
    for k, y in zip(KEYS, ROWY):
        d = ov.add_patch(Circle((672, y), 9, fc=C[k], ec='none'))
        n = ov.text(694, y, f'{PARAM_LABEL[k]}   {PARAM_NAME[k]}', fontsize=px(27), color=C.ink, va='center')
        v = ov.text(1250, y, VALS[KEYS.index(k)], fontsize=px(25), color=C.ink2, va='center', ha='right')
        l = ov.text(694, y - 38, '', fontsize=px(25), color=C.ink2, va='center')
        s['rows'].append((d, n, v, l))
    # charts
    tg = np.linspace(0, FULL, 300)
    A1 = fig.add_axes(box_px(1310, 440, 550, 200)); style_axes(A1); A1.set_xlim(0, FULL); A1.set_ylim(90, 100.5); A1.set_yticks([90, 95, 100]); A1.tick_params(labelbottom=False); A1.set_xticks([0, 50, 100, 150])
    A2 = fig.add_axes(box_px(1310, 190, 550, 200)); style_axes(A2); A2.set_xlim(0, FULL); A2.set_ylim(0, 0.7); A2.set_yticks([0, .3, .6]); A2.set_xticks([0, 50, 100, 150]); A2.set_xlabel('drift time (µs)')
    A1.text(0, 1.05, 'electrons left (%)', transform=A1.transAxes, fontsize=px(25), color=C.ink2)
    A2.text(0, 1.05, 'cloud width (mm)', transform=A2.transAxes, fontsize=px(25), color=C.ink2)
    A1.plot(tg, 100 * np.exp(-tg / TAU), color=C.lifetime, lw=px(2), ls=(0, (2, 3)), alpha=0.5)
    A2.plot(tg, sT_of(tg), color=C.tran_diff, lw=px(2), ls=(0, (2, 3)), alpha=0.5); A2.plot(tg, sL_of(tg), color=C.long_diff, lw=px(2), ls=(0, (2, 3)), alpha=0.5)
    A2.text(FULL * 0.985, sT_of(FULL) + 0.035, r'$\sigma_T$', fontsize=px(26), color=C.ink2, ha='right', va='bottom')
    A2.text(FULL * 0.985, sL_of(FULL) - 0.07, r'$\sigma_L$', fontsize=px(26), color=C.ink2, ha='right', va='top')
    s['l1'], = A1.plot([], [], color=C.lifetime, lw=px(4)); s['l2'], = A2.plot([], [], color=C.tran_diff, lw=px(4)); s['l3'], = A2.plot([], [], color=C.long_diff, lw=px(4))
    s['m1'] = A1.scatter([], [], s=px(13) ** 2, color=C.lifetime, zorder=5); s['m2'] = A2.scatter([], [], s=px(13) ** 2, color=C.tran_diff, zorder=5); s['m3'] = A2.scatter([], [], s=px(13) ** 2, color=C.long_diff, zorder=5)
    s['A1'], s['A2'] = A1, A2
    s['clock'] = fig.text(1860 / W, 1006 / H, '', fontsize=px(52), fontweight='semibold', color=C.ink, ha='right', va='baseline')
    s['clock2'] = fig.text(1860 / W, 968 / H, '', fontsize=px(26), color=C.ink2, ha='right', va='baseline')
    return s

def caption(t):
    items = ((0.3, T_ION, f'Follow one 0.1 mm segment of the main track, {vd * TD:.1f} cm from the anode'),
             (T_ION, T_REC, f'Ionisation: {dE * 1e3:.1f} keV deposited frees {NI} electrons'),
             (T_REC, T_DR, r'Recombination: a fraction $R = A_b\,/\,(1 + k_b\,(dE/dx)\,/\,E\rho)$ escapes the ions'),
             (T_DR, 18.0, f'The field sets the drift speed: {vd * 10:.2f} mm/µs'),
             (18.0, 23.0, r'Impurities capture electrons on the way: $e^{-t/\tau}$'),
             (23.0, T_SUM, r'Diffusion spreads the cloud: $\sigma = \sqrt{2Dt}$'),
             (T_SUM, DUR + 1, f'At the anode: {nd:.0f} electrons, ' + r'$\sigma$' + f' = {sT * 10:.2f} mm across the drift, {sL * 10:.2f} mm along it'))
    for t0, t1, txt in items:
        if t0 <= t < t1:
            return txt, fade(t, t0, t1 - 0.3, 0.3)
    return '', 0.0

def draw(s, i, t):
    td = td_of(t)
    txt, al = caption(t); s['sub'].set_text(txt); s['sub'].set_alpha(al)
    # lane
    zp = ZZ[ROW] + vd * td
    s['pc'].set_offsets([[zp, X[ROW]]]); s['pc'].set_sizes([px(20 * (1 + 0.5 * np.sin(np.pi * seg(t, 0.3, 2.2)) ** 2)) ** 2])
    s['pl'].set_data([ZZ[ROW], zp], [X[ROW], X[ROW]])
    # dots
    a_ion = ease(seg(t, T_ION, T_ION + 1.4))
    a_rec = ease(seg(t, T_REC + 1.0, T_DR - 1.0))
    col = np.tile(np.array(to_rgba(C.ink2)), (NI, 1))
    alpha = np.full(NI, a_ion)
    alpha[~surv] *= 1 - a_rec
    q = ease(seg(t, T_DR, TD0 + 0.2))
    st, sl = sT_of(td), sL_of(td)
    cloud = np.c_[uL * ax_d * 10 + gD * sl, uL * ax_t * 10 + gT * st]
    pos = GRID * (1 - q) + cloud * q
    lost = np.exp(-td / TAU) < uA
    alpha[lost & surv] = 0.0
    col[:, 3] = alpha
    s['dots'].set_offsets(pos); s['dots'].set_facecolors(col)
    s['ptxt'].set_text('' if t < T_ION else ('1 dot = 1 electron, arranged to count' if t < T_DR else '1 dot = 1 electron, drawn from the cloud'))
    s['ptxt'].set_alpha(float(a_ion))
    s['bar'].set_alpha(float(q)); s['barl'].set_alpha(float(q))                # no length scale while the dots are arranged for counting
    # brackets
    a_b = ease(seg(t, 23.0 - 3.0, 23.0 - 1.0))
    s['brL'].set_data([-sl, sl], [-1.36, -1.36]); s['brT'].set_data([-1.60, -1.60], [-st, st])
    s['brL'].set_alpha(a_b); s['brT'].set_alpha(a_b)
    s['tL'].set_text(r'$2\sigma_L$'); s['tL'].set_alpha(a_b); s['tT'].set_text(r'$2\sigma_T$'); s['tT'].set_alpha(a_b)
    s['tT'].set_position((-1.56, st + 0.08)); s['tT'].set_va('bottom')
    # parameter rows
    act = None
    for t0, t1, ids in ACT:
        if t0 <= t < t1:
            act = ids
    n_now = N_of(td)
    live = [f'{100 * R:.1f} % of the electrons escape' if t >= T_REC else '',
            (f'sets recombination, and speed {vd * 10:.2f} mm/µs' if t >= T_DR else ('sets how many escape' if t >= T_REC else '')),
            f'{n_now:.0f} of {nq:.0f} electrons left' if t >= T_DR else '',
            r'$\sigma_T$' + f' = {st:.2f} mm' if t >= T_DR else '',
            r'$\sigma_L$' + f' = {sl:.2f} mm' if t >= T_DR else '']
    for k, (d, n, v, l) in enumerate(s['rows']):
        on = act is not None and k in act
        a = 1.0 if on else (0.35 if t >= 0 else 0)
        for art in (n, v, l):
            art.set_alpha(a)
        d.set_alpha(a)
        l.set_text(live[k])
    # charts
    tl = np.linspace(0, td, 80) if td > 0 else np.array([0.0])
    s['l1'].set_data(tl, 100 * np.exp(-tl / TAU)); s['l2'].set_data(tl, sT_of(tl)); s['l3'].set_data(tl, sL_of(tl))
    s['m1'].set_offsets([[td, 100 * np.exp(-td / TAU)]]); s['m2'].set_offsets([[td, st]]); s['m3'].set_offsets([[td, sl]])
    a_cl = fade(t, T_DR, 99, 0.5)
    s['clock'].set_text(f'{td:.0f} µs'); s['clock'].set_alpha(a_cl); s['clock2'].set_text('drift time'); s['clock2'].set_alpha(a_cl)

BEATS = [(0, 'one segment of the track'), (T_ION, 'ionisation'), (T_REC, 'recombination: A_b, E'), (T_DR, 'drift speed: E'), (18.0, 'attenuation: tau'), (23.0, 'diffusion: D_T, D_L'), (T_SUM, 'five parameters at the anode')]
render(args, 'a02_drift', duration=DUR, setup=setup, draw=draw, beats=BEATS)
