"""Merged analysis of the Laplace test (main run lt 0.96..1.12 + low run lt 0.84..0.96).
Row lt=0.92 of the main run is dropped (it held each batch's cold start; 38/100 unconverged).
Batches with any unconverged (decrement > 0.05) or indefinite point are excluded."""
import glob, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

D = '/sdf/scratch/neutrino/pgranger/review_2026_09/L_laplace'
KEYS = ['G_map', 'logdet', 'nll_ref', 'decr', 'n_negeig']
LT = np.array([0.84, 0.88, 0.92, 0.96, 1.0, 1.04, 1.08, 1.12])
DAB = np.array([-0.0015, 0.0, 0.0015, 0.003, 0.0045])
VALLEY = 20.3

def load(pattern, drop_lt=()):
    out = {}
    for f in sorted(glob.glob(pattern)):
        z = np.load(f, allow_pickle=True)
        for j, b in enumerate(z['batch']):
            d = out.setdefault(int(b), {k: np.full((len(LT), len(DAB)), np.nan) for k in KEYS})
            for il, l in enumerate(z['lt_grid']):
                if any(abs(l - x) < 1e-9 for x in drop_lt): continue
                r = int(np.argmin(np.abs(LT - l)))
                for ia, a in enumerate(z['dab_grid']):
                    c = int(np.argmin(np.abs(DAB - a)))
                    for k in KEYS: d[k][r, c] = z[k][j][il, ia]
    return out

M = load(f'{D}/out/main_*.npz', drop_lt=(0.92,))
L = load(f'{D}/out/low_*.npz')
B, repro = {}, []
for b in sorted(set(M) | set(L)):
    if b not in M or b not in L: continue
    d = {k: np.where(np.isnan(M[b][k]), L[b][k], M[b][k]) for k in KEYS}
    r96 = int(np.argmin(np.abs(LT - 0.96)))
    repro.append(np.nanmax(np.abs(M[b]['G_map'][r96] - L[b]['G_map'][r96])))
    B[b] = d
bs_all = sorted(B)
bad = [b for b in bs_all if np.isnan(B[b]['G_map']).any() or (B[b]['decr'] > 0.05).any() or (B[b]['n_negeig'] > 0).any()]
bs = [b for b in bs_all if b not in bad]
print(f'batches with both runs: {len(bs_all)}; excluded {len(bad)} {bad}; used {len(bs)}')
print(f'row 0.96 reproducibility main vs low: max |dG| over batches {np.max(repro):.3f} nats (median {np.median(repro):.4f})')
X = 100 * (LT - 1.0)

def arms(sel):
    R = sum(B[b]['nll_ref'] for b in sel); G = sum(B[b]['G_map'] for b in sel)
    return dict(REF=R, MAP=G, LAP=G + 0.5 * sum(B[b]['logdet'] for b in sel))

def profile(S):
    P, Y = np.zeros(len(LT)), np.zeros(len(LT))
    for r in range(len(LT)):
        c = np.polyfit(DAB, S[r], 2); y = -c[1] / (2 * c[0]); P[r] = np.polyval(c, y); Y[r] = y
    k = int(np.argmin(P)); k0 = min(max(k - 1, 0), len(LT) - 3); idx = np.arange(k0, k0 + 3)
    c = np.polyfit(X[idx], P[idx], 2); xm = -c[1] / (2 * c[0])
    sig = np.sqrt(1 / (2 * c[0]))                        # dNLL = 1/2 width, per %
    return xm, sig, P - P.min(), np.interp(xm, X, Y) * 100 - xm / VALLEY, k in (0, len(LT) - 1)

full = arms(bs); res = {}
for a in ['REF', 'MAP', 'LAP']:
    xm, sig, P, abo, edge = profile(full[a])
    jk = np.array([profile(arms([c for c in bs if c != b])[a])[0] for b in bs]); n = len(jk)
    se = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
    res[a] = dict(lt=xm, se=se, width=sig, ab=abo, P=P.tolist(), edge=bool(edge))
    print(f'{a}: lifetime min {xm:+.2f} +- {se:.2f} % (jackknife)   curvature 1-sigma width {sig:.2f} %   '
          f'Ab offset {abo:+.3f} %   {"EDGE!" if edge else ""}')
for a1, a2 in [('MAP', 'REF'), ('LAP', 'REF'), ('LAP', 'MAP')]:
    d0 = res[a1]['lt'] - res[a2]['lt']
    jk = []
    for b in bs:
        s = arms([c for c in bs if c != b]); jk.append(profile(s[a1])[0] - profile(s[a2])[0])
    jk = np.array(jk); n = len(jk); se = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
    res[f'{a1}-{a2}'] = dict(d=d0, se=se)
    print(f'  paired {a1}-{a2}: {d0:+.2f} +- {se:.2f} points')
print('information loss: MAP width / REF width =', round(res['MAP']['width'] / res['REF']['width'], 2),
      '; LAP/REF =', round(res['LAP']['width'] / res['REF']['width'], 2))
json.dump(res, open(f'{D}/laplace_result_merged.json', 'w'), indent=1)

C = {'REF': '#8a8983', 'MAP': '#eb6834', 'LAP': '#2a78d6'}
LBL = {'REF': 'dE/dx fixed at truth (reference)', 'MAP': 'dE/dx fitted to best point (MAP, production-style)',
       'LAP': 'dE/dx integrated out (MAP + ½ log det H)'}
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
for i, a in enumerate(['REF', 'MAP', 'LAP']):
    r = res[a]
    ax[0].plot(X, r['P'], 'o-', color=C[a], lw=2, ms=5, label=LBL[a], zorder=3)
    ax[1].errorbar([r['lt']], [i], xerr=[r['se']], fmt='o', color=C[a], ms=8, capsize=4, zorder=3)
    ax[1].plot([r['lt'] - r['width'], r['lt'] + r['width']], [i - 0.18] * 2, color=C[a], lw=5, alpha=0.35, solid_capstyle='butt')
    ax[1].text(r['lt'], i + 0.22, f"{r['lt']:+.1f} ± {r['se']:.1f}%   (Ab {r['ab']:+.2f}%)", ha='center', fontsize=8.5, color='#0b0b0b')
ax[0].set_xlabel('lifetime − truth (%)'); ax[0].set_ylabel('Δ objective, Ab profiled (nats)')
ax[0].set_title(f'(a) profiled objective vs lifetime ({len(bs)} batches × 400 cm)', loc='left', fontsize=10)
ax[0].legend(frameon=False, fontsize=8, loc='upper center'); ax[0].set_ylim(-3, 60)
ax[1].axvline(0, color='#52514e', lw=0.8, ls='--')
ax[1].set_yticks([0, 1, 2]); ax[1].set_yticklabels(['reference', 'MAP', 'Laplace']); ax[1].set_ylim(-0.6, 2.7)
ax[1].set_xlabel('lifetime at the minimum − truth (%)')
ax[1].set_title('(b) minimum ± jackknife SE; bar = statistical 1σ width', loc='left', fontsize=10)
for a_ in ax:
    a_.spines[['top', 'right']].set_visible(False); a_.grid(axis='x', color='#e5e4df', lw=0.8); a_.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{D}/fig_laplace_test.png', dpi=150)
print('figure written')
