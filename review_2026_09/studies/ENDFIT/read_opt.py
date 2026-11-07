"""27 Oct readout of options 1 / 2 of the "why 44000" follow-up (rule: REVIEW.md 27 Oct "ENDFIT (i)", written before any
result). Arms (env RO_ARMS, default 'S,G'), each = the (h) fast continuation <X>FR @ 14000 -> 20000 plus one flag:
  S = LARND_CALIB_SAGA=1 (option 2); G = LARND_GEOM_SWEEP=14000,16500,18500 (option 1); GS = both;
  SL = LARND_CALIB_SAGA=1 on the LONG run's schedule (calibration 1e-3), 14000 -> 20000, compared with the long run's
  own 20000 window (L20, same schedule and iterations, no SAGA) -- amendment after the smoke (REVIEW.md 27 Oct).
X = EXS<s> (1 mm smear, ends held) and KN10N<s> (true ends), s = 1, 2, 4; references <X>H @ 20000 and <X>L @ 44000.
Errors as read_h3.py (mean of the last 1000 iterations / target - 1); c = nuisance relaxation along FR -> L.
usage (container, PYTHONPATH = live tree): python3 read_opt.py      [test: RO_ARMS=H3, an existing arm as stand-in]"""
import os
exec(open('/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/read_h3.py').read().split('\nC = {}')[0].split('\nSUF = ')[0])
ARMS = tuple(os.environ.get('RO_ARMS', 'S,G').split(',')); RUNS = ('KN10N', 'EXS'); EH, EL = 20000, 44000


def ck(run, it):
    fs = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))
    if not fs: raise SystemExit(f'{run}: no checkpoint at {it} -> block not complete, no readout')
    return pickle.load(open(fs[0], 'rb'))


def series(d, p, it, end):
    a = np.array([f1(x) for x in d[f'{p}_iter']]); return a[:len(a) - (end - it)], f1(d[f'{p}_target'][0])


def win(d, it, end, n=1000):
    r = {}
    for p in P:
        a, t = series(d, p, it, end); r[p] = 100 * (a[-n:] / t - 1).mean()
    pr = np.array([f1(x) for x in d['pos_residual_iter']]); r['pos'] = 1e4 * float(np.nanmean(pr[:len(pr) - (end - it)][-n:]))
    return r


def jitter(d, p, lo=15500, hi=16500, end=EH):
    a, t = series(d, p, hi, end); return 100 * float(np.std(a[-(hi - lo):] / t))


def nuis(d):
    g = {}
    for bi in d['chain_cache']:
        for st, c in zip(d['chain_cache'][bi], d['chain_contexts'][bi]):
            t = tuple(int(x) for x in c['track_id'])
            if t != (-1, -1): g[bi, t] = np.asarray(st['angles'], np.float64)
    return g, {bi: np.asarray(v['log_dedx'], np.float64).ravel() for bi, v in d['dedx_cache'].items()}


def relax(f, y, l):
    out = []
    for a, b, c in zip(f, y, l):
        ks = sorted(set(a) & set(b) & set(c), key=str); assert len(ks) == len(a) == len(b) == len(c)
        va, vb, vc = (np.concatenate([x[k] for k in ks]) for x in (a, b, c)); dl = vc - va
        out.append(float((vb - va) @ dl / (dl @ dl)))
    return out


C = {}
for a in RUNS:
    for s in S:
        C[a, s] = {'FR': ck(f'{a}{s}FR', 14000), 'H': ck(f'{a}{s}H', EH), 'L': ck(f'{a}{s}L', EL)}
        for y in ARMS:
            C[a, s][y] = ck(f'{a}{s}{y}', EH); assert int(C[a, s][y]['total_iter']) == EH
TAGS = ('FR', 'H') + ARMS + ('L',)
END = {t: EH for t in TAGS}; END.update(FR=EL, L=EL); AT = dict(END); AT['FR'] = 14000
SRC = {t: t for t in TAGS}; SRC['FR'] = 'L'          # the 14000 window is read from the L history (as read_fast.py)
R = {(a, s, t): win(C[a, s][SRC[t]], AT[t], END[t]) for a in RUNS for s in S for t in TAGS}
print(f"{'run':11s} " + ' '.join(f'{q:>9s}' for q in Q) + f'   (FR @ 14000 / H, {", ".join(ARMS)} @ 20000 / L @ 44000)')
for s in S:
    for a in RUNS:
        for t in TAGS:
            print(f'{a}{s}{t}        '[:11] + ' ' + ' '.join(f'{R[a, s, t][q]:+9.2f}' for q in Q))
M = {}
print('\npaired shift EXS - KN10N (s = 1 / 2 / 4; mean)')
for t in TAGS:
    for q in ('lifetime', 'long_diff', 'tran_diff', 'Ab', 'pos'):
        v = np.array([R['EXS', s, t][q] - R['KN10N', s, t][q] for s in S]); M[t, q] = v
        print(f'  {t:3s} {q:9s} ' + ' / '.join(f'{x:+7.2f}' for x in v) + f'   mean {v.mean():+7.2f}')
for y in ARMS:
    print(f'\ntrajectory, {y}: paired mean shift lifetime | long_diff (1000-iteration windows), and H for comparison')
    for it in range(15000, EH + 1, 1000):
        v = {(t, q): np.mean([win(C['EXS', s][t], it, EH)[q] - win(C['KN10N', s][t], it, EH)[q] for s in S]) for t in ('H', y) for q in ('lifetime', 'long_diff')}
        print(f'  {it:6d}  {y} {v[y, "lifetime"]:+7.2f} | {v[y, "long_diff"]:+7.2f}    H {v["H", "lifetime"]:+7.2f} | {v["H", "long_diff"]:+7.2f}')
print('\nnuisance relaxation c (geometry | log dE/dx) along FR -> L')
CG = {}
for a in RUNS:
    for s in S:
        n = {t: nuis(C[a, s][t]) for t in ('FR', 'H', 'L') + ARMS}; row = []
        for t in ('H',) + ARMS:
            c = relax(n['FR'], n[t], n['L']); CG[a, s, t] = c; row.append(f'{t} {c[0]:.2f} | {c[1]:.2f}')
        print(f'  {a}{s:<2d}  ' + '    '.join(row))
print('\ndistance to the long run end state |X @ 20000 - L @ 44000| (lifetime, long_diff), per run')
DL = {}
for t in ('H',) + ARMS:
    for q in ('lifetime', 'long_diff'):
        DL[t, q] = np.array([abs(R[a, s, t][q] - R[a, s, 'L'][q]) for a in RUNS for s in S])
    print(f'  {t:3s} lifetime ' + ' '.join(f'{x:5.2f}' for x in DL[t, "lifetime"]) + f'  mean {DL[t, "lifetime"].mean():.2f}'
          f'   long_diff ' + ' '.join(f'{x:5.2f}' for x in DL[t, "long_diff"]) + f'  mean {DL[t, "long_diff"].mean():.2f}')
print('\ncalibration jitter: sd of the iterates over 15500-16500 (LR held at 1e-2), % of target')
JT = {}
for t in ('H',) + ARMS:
    for q in ('lifetime', 'long_diff'):
        JT[t, q] = np.array([jitter(C[a, s][t], q) for a in RUNS for s in S])
    print(f'  {t:3s} lifetime ' + ' '.join(f'{x:5.2f}' for x in JT[t, "lifetime"]) + '   long_diff ' + ' '.join(f'{x:5.2f}' for x in JT[t, "long_diff"]))
print('\nposition residual um (KN10N | EXS), s = 1 / 2 / 4')
for t in ('H',) + ARMS + ('L',):
    print(f'  {t:3s} ' + ' / '.join(f'{R["KN10N", s, t]["pos"]:.0f}' for s in S) + '  |  ' + ' / '.join(f'{R["EXS", s, t]["pos"]:.0f}' for s in S))
for y in ARMS:
    gs = C['EXS', S[0]][y].get('geom_sweep')
    if not gs: continue
    print(f'\nsweep log ({y}): per sweep, sum dJ over batches | accepted Newton steps | median max|step| um | hours')
    for a in RUNS:
        for s in S:
            rec = C[a, s][y]['geom_sweep']; its = sorted({r['iter'] // 100 * 100 for r in rec}); row = []
            for k0 in sorted({min(r['iter'] for r in rec if r['iter'] >= t0 and r['iter'] < t0 + 100) for t0 in (14000, 16500, 18500) if any(t0 <= r['iter'] < t0 + 100 for r in rec)}):
                rr = [r for r in rec if k0 <= r['iter'] < k0 + 100]; nw = [n for r in rr for n in r['newton'] if n.get('accepted')]
                row.append(f'@{k0}: {sum(n["dJ"] for n in nw):+.0f} | {len(nw)}/{sum(len(r["newton"]) for r in rr)} | '
                           f'{np.median([n["max_step_um"] for n in nw]) if nw else 0:.0f} | {sum(r["time_s"] for r in rr) / 3600:.2f}')
            print(f'  {a}{s:<2d} ' + '   '.join(row))
    after = {(a, s): sorted(glob.glob(f'{FR}/sci_full_{a}{s}{y}/history_iter14100_*.pkl')) for a in RUNS for s in S}
    if all(after.values()):
        print(f'c right after the first sweep (checkpoint 14100, calibration still at FR), geometry | dE/dx:')
        for (a, s), f in after.items():
            c = relax(nuis(C[a, s]['FR']), nuis(pickle.load(open(f[0], 'rb'))), nuis(C[a, s]['L'])); print(f'  {a}{s}  {c[0]:.2f} | {c[1]:.2f}')
if 'SL' in ARMS:
    for a in RUNS:
        for s in S:
            R[a, s, 'L20'] = win(C[a, s]['L'], 20000, EL)
    print('\nSL (SAGA, long schedule) vs L20 (the long run at 20000, no SAGA) vs L @ 44000')
    for t in ('L20', 'SL', 'L'):
        for q in ('lifetime', 'long_diff', 'pos'):
            v = np.array([R['EXS', s, t][q] - R['KN10N', s, t][q] for s in S]); M[t, q] = v
            print(f'  paired {t:3s} {q:9s} ' + ' / '.join(f'{x:+7.2f}' for x in v) + f'   mean {v.mean():+7.2f}')
    for t in ('L20', 'SL'):
        for q in ('lifetime', 'long_diff'):
            DL[t, q] = np.array([abs(R[a, s, t][q] - R[a, s, 'L'][q]) for a in RUNS for s in S])
            JT[t, q] = np.array([jitter(C[a, s]['L' if t == 'L20' else 'SL'], q, 19000, 20000, EL if t == 'L20' else EH) for a in RUNS for s in S])
        print(f'  {t:3s} |X - L| lifetime ' + ' '.join(f'{x:5.2f}' for x in DL[t, "lifetime"]) + f' mean {DL[t, "lifetime"].mean():.2f}'
              f'  long_diff mean {DL[t, "long_diff"].mean():.2f}   jitter (sd 19000-20000) lifetime '
              + ' '.join(f'{x:5.2f}' for x in JT[t, "lifetime"]) + '  long_diff ' + ' '.join(f'{x:5.2f}' for x in JT[t, "long_diff"]))
print('\nVERDICTS')
for y in ARMS:
    if y == 'SL':
        rj = np.array([max(JT['SL', 'lifetime'][k] / JT['L20', 'lifetime'][k], JT['SL', 'long_diff'][k] / JT['L20', 'long_diff'][k]) for k in range(6)])
        rd = DL['SL', 'lifetime'].mean() / DL['L20', 'lifetime'].mean(); dl = M['SL', 'lifetime'] - M['L20', 'lifetime']
        print(f'SL: paired lifetime {M["SL", "lifetime"].mean():+.2f} vs L20 {M["L20", "lifetime"].mean():+.2f} (change ' + ' / '.join(f'{x:+.2f}' for x in dl)
              + f') -> ' + ('HELPS' if (dl < 0).all() and dl.mean() <= -2 else 'NO RESOLVED GAIN'))
        print(f'SL: jitter ratio vs L20 ' + ' '.join(f'{x:.2f}' for x in rj) + f' -> {"REMOVES JITTER" if (rj <= 0.5).sum() >= 5 else "NO"};'
              f'  |X - L| lifetime mean {DL["SL", "lifetime"].mean():.2f} vs L20 {DL["L20", "lifetime"].mean():.2f} (x{rd:.2f}) -> '
              + ('CLOSER TO CONVERGED' if rd <= 0.7 else 'NOT CLOSER'))
        continue
    lx, dx = M[y, 'lifetime'].mean(), M[y, 'long_diff'].mean(); dl = M[y, 'lifetime'] - M['H', 'lifetime']
    v_rec = ('RECOVERS' if abs(lx) <= 3 and abs(dx) <= 5 else 'HELPS' if (dl < 0).all() and dl.mean() <= -2 else 'NO RESOLVED GAIN')
    print(f'{y}: paired EXS - KN10N @ 20000 lifetime {lx:+.2f} (H {M["H", "lifetime"].mean():+.2f}, L {M["L", "lifetime"].mean():+.2f}), '
          f'long_diff {dx:+.2f} (H {M["H", "long_diff"].mean():+.2f}); change vs H ' + ' / '.join(f'{x:+.2f}' for x in dl) + f' -> {v_rec}')
    cg = np.array([CG[a, s, y][0] for a in RUNS for s in S]); cH = np.array([CG[a, s, 'H'][0] for a in RUNS for s in S])
    pr = np.array([R[a, s, y]['pos'] / R[a, s, 'H']['pos'] for a in RUNS for s in S])
    v_g = ('RELAXES GEOMETRY' if (cg >= 0.7).all() and (pr <= 1.1).all() else 'PARTIAL' if cg.mean() >= cH.mean() + 0.2 and (pr <= 1.1).all()
           else 'POSITION PENALTY' if (pr > 1.1).any() else 'NO')
    print(f'{y}: geometry c min {cg.min():.2f} mean {cg.mean():.2f} (H mean {cH.mean():.2f}); position / H max {pr.max():.2f} -> {v_g}')
    rj = np.array([max(JT[y, "lifetime"][k] / JT["H", "lifetime"][k], JT[y, "long_diff"][k] / JT["H", "long_diff"][k]) for k in range(6)])
    v_j = 'REMOVES JITTER' if (rj <= 0.5).sum() >= 5 else 'NO'
    rd = DL[y, 'lifetime'].mean() / DL['H', 'lifetime'].mean()
    print(f'{y}: jitter ratio vs H (worse of lifetime / long_diff) ' + ' '.join(f'{x:.2f}' for x in rj) + f' -> {v_j};'
          f'  |X - L| lifetime mean {DL[y, "lifetime"].mean():.2f} vs H {DL["H", "lifetime"].mean():.2f} (x{rd:.2f}) -> '
          + ('CLOSER TO CONVERGED' if rd <= 0.7 else 'NOT CLOSER'))
