"""Ablation ladder analysis. Rung 0 = MAP arm of the Laplace test (target-file segments, correct
binned prior, no mean constraint), restricted to the same batches. Rungs L1..L3 from ladder_test.py.
Also: REF on the 1 cm grid (piece-averaged true dE/dx) vs REF on the target grid."""
import glob, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

D = '/sdf/scratch/neutrino/pgranger/review_2026_09/LAD'
DL = '/sdf/scratch/neutrino/pgranger/review_2026_09/L_laplace'
VALLEY = 20.3

# ---------------- ladder rungs
lad, rungs = {}, None
for f in sorted(glob.glob(f'{D}/out/lad_*.npz')):
    z = np.load(f, allow_pickle=True)
    rungs = [r[0] for r in json.loads(str(z['rungs']))]
    LTl, DAl = z['lt_grid'], z['dab_grid']
    for j, b in enumerate(z['batch']):
        d = {}
        for r in rungs:
            for k in ['G', 'decr', 'n_negeig', 'scale_err', 'mean_dedx', 'fresh']:
                d[f'{r}_{k}'] = z[f'{r}_{k}'][j]
        d['REF1'] = z[f'{rungs[0]}_nll_ref'][j]
        lad[int(b)] = d
# ---------------- rung 0 (Laplace-test MAP arm, merged grid, row 0.92 of main run dropped)
LT0 = np.array([0.84, 0.88, 0.92, 0.96, 1.0, 1.04, 1.08, 1.12]); DA0 = np.array([-0.0015, 0.0, 0.0015, 0.003, 0.0045])
r0 = {}
for pat, drop in [(f'{DL}/out/main_*.npz', (0.92,)), (f'{DL}/out/low_*.npz', ())]:
    for f in sorted(glob.glob(pat)):
        z = np.load(f, allow_pickle=True)
        for j, b in enumerate(z['batch']):
            d = r0.setdefault(int(b), {k: np.full((len(LT0), len(DA0)), np.nan) for k in ['G', 'REF', 'decr', 'neg']})
            for il, l in enumerate(z['lt_grid']):
                if any(abs(l - x) < 1e-9 for x in drop): continue
                rr = int(np.argmin(np.abs(LT0 - l)))
                for ia, a in enumerate(z['dab_grid']):
                    c = int(np.argmin(np.abs(DA0 - a)))
                    d['G'][rr, c] = z['G_map'][j][il, ia]; d['REF'][rr, c] = z['nll_ref'][j][il, ia]
                    d['decr'][rr, c] = z['decr'][j][il, ia]; d['neg'][rr, c] = z['n_negeig'][j][il, ia]

def ok0(b):
    d = r0.get(b)
    return d is not None and not np.isnan(d['G']).any() and not (d['decr'] > 0.05).any() and not (d['neg'] > 0).any()

def okL(b):
    d = lad[b]
    for r in rungs:
        dec = d[f'{r}_decr']; neg = d[f'{r}_n_negeig']
        if np.isnan(d[f'{r}_G']).any() or (dec > 0.05).any() or (np.nan_to_num(neg) > 0).any():
            return False
    return True

bl = sorted(lad)
bad = [b for b in bl if not (okL(b) and ok0(b))]
bs = [b for b in bl if b not in bad]
print(f'ladder batches {len(bl)}; excluded {len(bad)} {bad}; used {len(bs)}')

def profile(S, LT, DA):
    X = 100 * (LT - 1)
    P, Y = np.zeros(len(LT)), np.zeros(len(LT))
    for r in range(len(LT)):
        c = np.polyfit(DA, S[r], 2); y = -c[1] / (2 * c[0]); P[r] = np.polyval(c, y); Y[r] = y
    k = int(np.argmin(P)); k0 = min(max(k - 1, 0), len(LT) - 3); idx = np.arange(k0, k0 + 3)
    c = np.polyfit(X[idx], P[idx], 2); xm = -c[1] / (2 * c[0])
    return xm, np.sqrt(1 / (2 * c[0])), P - P.min(), np.interp(xm, X, Y) * 100 - xm / VALLEY, k in (0, len(LT) - 1)

ARMS = {'R0 REF (true dE/dx, target seg)': (lambda s: sum(r0[b]['REF'] for b in s), LT0, DA0),
        'R0 MAP (target seg, binned prior)': (lambda s: sum(r0[b]['G'] for b in s), LT0, DA0),
        'REF1 (true dE/dx, 1 cm pieces)': (lambda s: sum(lad[b]['REF1'] for b in s), LTl, DAl)}
for r in rungs:
    ARMS[f'{r}'] = (lambda s, r=r: sum(lad[b][f'{r}_G'] for b in s), LTl, DAl)

res = {}
for name, (fn, LT, DA) in ARMS.items():
    xm, w, P, ab, edge = profile(fn(bs), LT, DA)
    jk = np.array([profile(fn([c for c in bs if c != b]), LT, DA)[0] for b in bs]); n = len(jk)
    se = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
    res[name] = dict(lt=xm, se=se, width=w, ab=ab, edge=edge, P=P.tolist(), jk=jk.tolist(), X=(100 * (LT - 1)).tolist())
    print(f'{name:40s} lifetime {xm:+6.2f} +- {se:.2f} %   width {w:.2f} %   Ab {ab:+.3f} %  {"EDGE!" if edge else ""}')

def paired(a, b_):
    ja, jb = np.array(res[a]['jk']), np.array(res[b_]['jk']); n = len(ja)
    dj = ja - jb; d0 = res[a]['lt'] - res[b_]['lt']
    return d0, np.sqrt((n - 1) / n * np.sum((dj - dj.mean()) ** 2))
names = list(ARMS)
print('\npaired differences (jackknife on the difference):')
chain = ['R0 MAP (target seg, binned prior)'] + rungs
for a, b_ in list(zip(chain[1:], chain[:-1])) + [(rungs[-1], 'R0 MAP (target seg, binned prior)'),
                                                 ('REF1 (true dE/dx, 1 cm pieces)', 'R0 REF (true dE/dx, target seg)')]:
    d, s = paired(a, b_); res[f'{a} - {b_}'] = dict(d=d, se=s)
    print(f'  {a:34s} - {b_:34s}: {d:+6.2f} +- {s:.2f} points')

print('\nfitted dE/dx length-weighted scale error at the truth-nearest grid point (mean over batches):')
il = int(np.argmin(np.abs(LTl - 1.0))); ia = int(np.argmin(np.abs(DAl - 0.0)))
for r in rungs:
    se_ = np.array([lad[b][f'{r}_scale_err'][il, ia] for b in bs]) * 100
    print(f'  {r}: {se_.mean():+.3f} +- {se_.std(ddof=1)/np.sqrt(len(se_)):.3f} %')
json.dump(res, open(f'{D}/ladder_result.json', 'w'), indent=1)

# figure
lab = {'R0 REF (true dE/dx, target seg)': 'true dE/dx\n(target segments)',
       'R0 MAP (target seg, binned prior)': 'rung 0: fitted,\ntarget seg, correct prior',
       'REF1 (true dE/dx, 1 cm pieces)': 'true dE/dx\n(1 cm pieces)'}
cfg = {r: c for r, c in zip(rungs, ['L1: 1 cm pieces,\nproduction-shape prior', 'L2: + exact production\nprior form',
                                     'L3: + mean constraint\n(production dE/dx config)'])}
lab.update(cfg)
order = ['R0 REF (true dE/dx, target seg)', 'REF1 (true dE/dx, 1 cm pieces)', 'R0 MAP (target seg, binned prior)'] + rungs
col = ['#8a8983', '#8a8983', '#2a78d6', '#1baf7a', '#eda100', '#eb6834']
fig, ax = plt.subplots(figsize=(10, 4.4))
for k, (nm, c) in enumerate(zip(order, col)):
    r = res[nm]
    ax.errorbar([r['lt']], [k], xerr=[r['se']], fmt='o', color=c, ms=8, capsize=4, zorder=3)
    ax.plot([r['lt'] - r['width'], r['lt'] + r['width']], [k - 0.2] * 2, color=c, lw=5, alpha=0.3, solid_capstyle='butt')
    ax.text(r['lt'] + r['se'] + 0.4, k + 0.05, f"{r['lt']:+.1f} ± {r['se']:.1f}%  (Ab {r['ab']:+.2f}%)", va='center', fontsize=8.5)
ax.axvline(0, color='#52514e', lw=0.8, ls='--')
ax.axvline(8.85, color='#e34948', lw=1, ls=':'); ax.text(8.95, len(order) - 0.6, 'production\nfits +8.85%', fontsize=8, color='#e34948')
ax.set_yticks(range(len(order))); ax.set_yticklabels([lab[n] for n in order], fontsize=8.5); ax.invert_yaxis()
ax.set_xlabel('lifetime at the profiled minimum − truth (%)   [point ± jackknife SE; bar = 1σ statistical width]')
ax.set_title(f'Which production dE/dx ingredient moves the lifetime? (true geometry, {len(bs)} batches × 400 cm)', loc='left', fontsize=10)
ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', color='#e5e4df', lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{D}/fig_ladder.png', dpi=150)
print('figure written')
