"""Decisive comparison: full production dE/dx configuration (L3: 1 cm pieces on the TRUE trajectory,
exact production prior, mean constraint w=2500) vs rung 0 (target segments, correct binned prior,
no mean constraint), both MAP-profiled over dE/dx and Ab, on the SAME batches and the SAME grid
points: lifetime x{0.92..1.16}, Ab offset {-0.15, +0.15, +0.45}% from the valley line."""
import glob, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

D = '/sdf/scratch/neutrino/pgranger/review_2026_09/LAD'
DL = '/sdf/scratch/neutrino/pgranger/review_2026_09/L_laplace'
VALLEY = 20.3
LT = np.array([0.92, 0.96, 1.00, 1.04, 1.08, 1.12, 1.16]); DA = np.array([-0.0015, 0.0015, 0.0045])

def put(store, b, key, val, lt, da):
    d = store.setdefault(b, {})
    arr = d.setdefault(key, np.full((len(LT), len(DA)), np.nan))
    for il, l in enumerate(lt):
        r = np.where(np.abs(LT - l) < 1e-9)[0]
        if not len(r): continue
        for ia, a in enumerate(da):
            c = np.where(np.abs(DA - a) < 1e-9)[0]
            if len(c): arr[r[0], c[0]] = val[il, ia]

L3 = {}
for f in sorted(glob.glob(f'{D}/out/lad_*.npz')) + sorted(glob.glob(f'{D}/out/l3_*.npz')):
    z = np.load(f, allow_pickle=True)
    first = json.loads(str(z['rungs']))[0][0]
    for j, b in enumerate(z['batch']):
        b = int(b)
        for k in ['G', 'decr', 'n_negeig']:
            put(L3, b, k, z[f'L3_{k}'][j], z['lt_grid'], z['dab_grid'])
        put(L3, b, 'REF1', z[f'{first}_nll_ref'][j], z['lt_grid'], z['dab_grid'])
R0 = {}
for pat, drop in [(f'{DL}/out/main_*.npz', (0.92,)), (f'{DL}/out/low_*.npz', ())]:
    for f in sorted(glob.glob(pat)):
        z = np.load(f, allow_pickle=True)
        keep = np.array([not any(abs(l - x) < 1e-9 for x in drop) for l in z['lt_grid']])
        for j, b in enumerate(z['batch']):
            for k, zk in [('G', 'G_map'), ('REF', 'nll_ref'), ('decr', 'decr'), ('n_negeig', 'n_negeig')]:
                v = z[zk][j].astype(float).copy(); v[~keep] = np.nan
                # only write rows kept from this file (do not overwrite with NaN)
                d = R0.setdefault(int(b), {}); arr = d.setdefault(k, np.full((len(LT), len(DA)), np.nan))
                for il, l in enumerate(z['lt_grid']):
                    if not keep[il]: continue
                    r = np.where(np.abs(LT - l) < 1e-9)[0]
                    if not len(r): continue
                    for ia, a in enumerate(z['dab_grid']):
                        c = np.where(np.abs(DA - a) < 1e-9)[0]
                        if len(c): arr[r[0], c[0]] = v[il, ia]

def good(d, rows):
    G = d['G'][rows]
    return (not np.isnan(G).any()) and not (np.nan_to_num(d['decr'][rows]) > 0.05).any() \
        and not (np.nan_to_num(d['n_negeig'][rows]) > 0).any()

rowsL3 = np.arange(len(LT)); rowsR0 = np.arange(len(LT) - 1)          # R0 has no x1.16 row
bs = sorted(b for b in set(L3) & set(R0) if good(L3[b], rowsL3) and good(R0[b], rowsR0))
dropped = sorted((set(L3) | set(R0)) - set(bs))
print(f'L3 batches {len(L3)}, R0 batches {len(R0)}; common & converged: {len(bs)}; dropped {len(dropped)}')

def profile(S, rows):
    X = 100 * (LT[rows] - 1); S = S[rows]
    P, Y = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(len(rows)):
        c = np.polyfit(DA, S[r], 2); y = -c[1] / (2 * c[0]); P[r] = np.polyval(c, y); Y[r] = y
    k = int(np.argmin(P)); k0 = min(max(k - 1, 0), len(rows) - 3); idx = np.arange(k0, k0 + 3)
    c = np.polyfit(X[idx], P[idx], 2); xm = -c[1] / (2 * c[0])
    return xm, np.sqrt(1 / (2 * c[0])), P - P.min(), np.interp(xm, X, Y) * 100 - xm / VALLEY, k in (0, len(rows) - 1), X

ARMS = {'REF (true dE/dx, target segments)': (lambda s: sum(R0[b]['REF'] for b in s), rowsR0),
        'REF1 (true dE/dx, 1 cm pieces)': (lambda s: sum(L3[b]['REF1'] for b in s), rowsL3),
        'R0: fitted dE/dx, clean setup': (lambda s: sum(R0[b]['G'] for b in s), rowsR0),
        'L3: fitted dE/dx, production config': (lambda s: sum(L3[b]['G'] for b in s), rowsL3)}
res = {}
for nm, (fn, rows) in ARMS.items():
    xm, w, P, ab, edge, X = profile(fn(bs), rows)
    jk = np.array([profile(fn([c for c in bs if c != b]), rows)[0] for b in bs]); n = len(jk)
    se = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
    res[nm] = dict(lt=xm, se=se, width=w, ab=ab, edge=bool(edge), P=P.tolist(), X=X.tolist(), jk=jk.tolist())
    print(f'{nm:38s} lifetime {xm:+6.2f} +- {se:.2f} %   width {w:.2f} %   Ab {ab:+.3f} %  {"EDGE!" if edge else ""}')

def paired(a, b_):
    ja, jb = np.array(res[a]['jk']), np.array(res[b_]['jk']); n = len(ja); dj = ja - jb
    return res[a]['lt'] - res[b_]['lt'], np.sqrt((n - 1) / n * np.sum((dj - dj.mean()) ** 2))
print()
for a, b_ in [('L3: fitted dE/dx, production config', 'R0: fitted dE/dx, clean setup'),
              ('R0: fitted dE/dx, clean setup', 'REF (true dE/dx, target segments)'),
              ('L3: fitted dE/dx, production config', 'REF1 (true dE/dx, 1 cm pieces)'),
              ('REF1 (true dE/dx, 1 cm pieces)', 'REF (true dE/dx, target segments)')]:
    d, s = paired(a, b_); res[f'{a} MINUS {b_}'] = dict(d=d, se=s)
    print(f'  {a:38s} - {b_:36s}: {d:+6.2f} +- {s:.2f} points  ({d/s if s > 0 else np.nan:+.1f} sigma)')
json.dump(res, open(f'{D}/l3_result.json', 'w'), indent=1)

C = {'REF (true dE/dx, target segments)': '#8a8983', 'REF1 (true dE/dx, 1 cm pieces)': '#b4b2a9',
     'R0: fitted dE/dx, clean setup': '#2a78d6', 'L3: fitted dE/dx, production config': '#eb6834'}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw=dict(width_ratios=[1.1, 1]))
for k, nm in enumerate(ARMS):
    r = res[nm]
    ax[0].plot(r['X'], r['P'], 'o-', color=C[nm], lw=2, ms=5, label=nm)
    ax[1].errorbar([r['lt']], [k], xerr=[r['se']], fmt='o', color=C[nm], ms=8, capsize=4, zorder=3)
    ax[1].plot([r['lt'] - r['width'], r['lt'] + r['width']], [k - 0.22] * 2, color=C[nm], lw=5, alpha=0.3, solid_capstyle='butt')
    ax[1].text(r['lt'] + r['se'] + 0.3, k, f"{r['lt']:+.1f} ± {r['se']:.1f}%", va='center', fontsize=8.5)
ax[0].set_xlabel('lifetime − truth (%)'); ax[0].set_ylabel('Δ objective, Ab profiled (nats)')
ax[0].set_title(f'(a) profiled objective ({len(bs)} batches × 400 cm, true geometry)', loc='left', fontsize=10)
ax[0].legend(frameon=False, fontsize=8); ax[0].set_ylim(-3, 80)
d, s = res['L3: fitted dE/dx, production config MINUS R0: fitted dE/dx, clean setup'].values()
ax[1].set_yticks(range(len(ARMS))); ax[1].set_yticklabels([n.split(':')[0].split(' (')[0] for n in ARMS]); ax[1].invert_yaxis()
ax[1].axvline(0, color='#52514e', lw=0.8, ls='--'); ax[1].axvline(8.85, color='#e34948', lw=1, ls=':'); ax[1].text(8.7, len(ARMS) - 0.55, 'production\nfits +8.85%', ha='right', fontsize=8, color='#e34948')
ax[1].set_xlabel('lifetime at the minimum − truth (%)')
ax[1].set_title(f'(b) production config − clean: {d:+.1f} ± {s:.1f} points (paired)', loc='left', fontsize=10)
for a_ in ax:
    a_.spines[['top', 'right']].set_visible(False); a_.grid(axis='x', color='#e5e4df', lw=0.8); a_.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{D}/fig_l3_vs_r0.png', dpi=150)
print('figure written')
