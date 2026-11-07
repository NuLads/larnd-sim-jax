"""Analyse the Laplace test: profiled lifetime minimum for REF / MAP / LAP, jackknife errors,
convergence diagnostics, and the MAP dE/dx field vs truth (scale and drift trend)."""
import glob, json, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

D = '/sdf/scratch/neutrino/pgranger/review_2026_09/L_laplace'
files = sorted(glob.glob(f'{D}/out/main_*.npz'))
B = {}   # batch -> dict of per-batch arrays
for f in files:
    z = np.load(f, allow_pickle=True)
    lt, dab, valley = z['lt_grid'], z['dab_grid'], float(z['valley'])
    for j, b in enumerate(z['batch']):
        B[int(b)] = {k: z[k][j] for k in ['G_map', 'nll_map', 'logdet', 'nll_ref', 'decr', 'n_negeig', 'mineig', 'niter']}
        B[int(b)].update({k: z[f'{k}_{b}'] for k in ['u_map', 'u_true', 'seg_dx', 'seg_z'] if f'{k}_{b}' in z})
bs = sorted(B)
print(f'files {len(files)}  batches {len(bs)}  grid lt {lt} dab {dab}')
X = lt - 1.0

def arms(sel):
    ref = sum(B[b]['nll_ref'] for b in sel)
    mp = sum(B[b]['G_map'] for b in sel)
    lap = mp + 0.5 * sum(B[b]['logdet'] for b in sel)
    return dict(REF=ref, MAP=mp, LAP=lap)

def profile_min(S):
    """S[il, ia]: quadratic in dab per lifetime row -> profile P(x); then local parabola on P."""
    P = np.zeros(len(X)); Y = np.zeros(len(X))
    for il in range(len(X)):
        c = np.polyfit(dab, S[il], 2)
        y = np.clip(-c[1] / (2 * c[0]), dab.min() * 1.5, dab.max() * 1.5) if c[0] > 0 else dab[np.argmin(S[il])]
        P[il] = np.polyval(c, y); Y[il] = y
    k = int(np.argmin(P))
    k0 = min(max(k - 1, 0), len(X) - 3)
    idx = np.arange(k0, k0 + 3)
    c = np.polyfit(X[idx], P[idx], 2)
    xm = -c[1] / (2 * c[0]) if c[0] > 0 else X[k]
    edge = (k == 0) or (k == len(X) - 1)
    ym = np.interp(xm, X, Y)
    ab_off = -xm / valley + ym
    return xm, ab_off, P, edge

full = arms(bs)
res = {}
for a, S in full.items():
    xm, abo, P, edge = profile_min(S)
    # delete-1 jackknife over batches
    jk = []
    for b in bs:
        xs, *_ = profile_min(arms([c for c in bs if c != b])[a]); jk.append(xs)
    jk = np.array(jk); n = len(jk)
    se = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
    res[a] = dict(lt=100 * xm, se=100 * se, ab=100 * abo, P=P - P.min(), edge=edge)
    print(f'{a:4s}: lifetime min {100*xm:+.2f} +- {100*se:.2f} %   Ab {100*abo:+.3f} %   {"(AT GRID EDGE)" if edge else ""}')

# paired differences (same data): MAP-REF, LAP-REF, LAP-MAP via jackknife of the difference
def pdiff(a1, a2):
    d0 = res[a1]['lt'] - res[a2]['lt']; jk = []
    for b in bs:
        s = arms([c for c in bs if c != b])
        jk.append(100 * (profile_min(s[a1])[0] - profile_min(s[a2])[0]))
    jk = np.array(jk); n = len(jk)
    return d0, np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
for a1, a2 in [('MAP', 'REF'), ('LAP', 'REF'), ('LAP', 'MAP')]:
    d, s = pdiff(a1, a2); print(f'  {a1}-{a2}: {d:+.2f} +- {s:.2f} points')

# diagnostics
decr = np.array([B[b]['decr'] for b in bs]); neg = np.array([B[b]['n_negeig'] for b in bs])
print(f'Newton decrement: median {np.median(decr):.1e}, max {decr.max():.2e}, frac>0.05 {np.mean(decr>0.05):.3f};'
      f' points with negative eigenvalues {int((neg>0).sum())}/{neg.size}')
ldt = np.array([B[b]['logdet'] for b in bs]).sum(0)
print('sum logdet (rows lt, cols dab) minus min:\n', np.round(ldt - ldt.min(), 2))

# MAP dE/dx vs truth at the truth-nearest grid point
il0 = int(np.argmin(np.abs(X))); ia0 = int(np.argmin(np.abs(dab)))
num = den = 0.; zs, rs, ws = [], [], []
for b in bs:
    if 'u_map' not in B[b]: continue
    um = B[b]['u_map'][il0, ia0]; ut = B[b]['u_true']; dx = B[b]['seg_dx']; z = B[b]['seg_z'].mean(1)
    num += np.sum(np.exp(um) * dx); den += np.sum(np.exp(ut) * dx)
    zs.append(np.abs(z)); rs.append(um - ut); ws.append(dx)
zs, rs, ws = map(np.concatenate, (zs, rs, ws))
sl = np.polyfit(zs, rs, 1, w=np.sqrt(ws))[0]
print(f'MAP dE/dx at truth theta: length-weighted scale {100*(num/den-1):+.3f} %; '
      f'log-residual trend vs |z| {100*sl*30:+.3f} % over 30 cm; residual rms {np.sqrt(np.average(rs**2, weights=ws)):.3f}')

json.dump({a: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for a, r in res.items()},
          open(f'{D}/laplace_result.json', 'w'), indent=1)

# figure
C = {'REF': '#8a8983', 'MAP': '#eb6834', 'LAP': '#2a78d6'}
LBL = {'REF': 'dE/dx at truth (reference)', 'MAP': 'dE/dx profiled (MAP, production-style)',
       'LAP': 'dE/dx marginalised (MAP + ½ log det H)'}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a in ['REF', 'MAP', 'LAP']:
    r = res[a]
    ax[0].plot(100 * X, r['P'], 'o-', color=C[a], lw=2, ms=5, label=LBL[a])
    ax[1].errorbar([r['lt']], [['REF', 'MAP', 'LAP'].index(a)], xerr=[r['se']], fmt='o', color=C[a], ms=8, capsize=4)
    ax[1].text(r['lt'], ['REF', 'MAP', 'LAP'].index(a) + 0.18, f"{r['lt']:+.2f} ± {r['se']:.2f}%", ha='center', fontsize=9)
ax[0].set_xlabel('lifetime − truth (%)'); ax[0].set_ylabel('Δ objective, Ab profiled (nats)')
ax[0].set_title('(a) profiled objective vs lifetime (100 batches)', loc='left', fontsize=10)
ax[0].legend(frameon=False, fontsize=8); ax[0].set_ylim(bottom=-2)
ax[1].axvline(0, color='#52514e', lw=0.8, ls='--')
ax[1].set_yticks([0, 1, 2]); ax[1].set_yticklabels(['reference', 'MAP', 'Laplace']); ax[1].set_ylim(-0.6, 2.7)
ax[1].set_xlabel('profiled lifetime minimum − truth (%)')
ax[1].set_title('(b) where each objective puts the lifetime (± jackknife SE)', loc='left', fontsize=10)
for a_ in ax:
    a_.spines[['top', 'right']].set_visible(False); a_.grid(axis='x', color='#e5e4df', lw=0.8)
fig.tight_layout(); fig.savefig(f'{D}/fig_laplace_test.png', dpi=150)
print('figure', f'{D}/fig_laplace_test.png')
