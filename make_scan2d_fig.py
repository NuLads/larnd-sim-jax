"""Fig 40: 2-D likelihood scans — the slice-to-profile correction.

Each panel is the summed NLL over a (lifetime, A_b) grid. Three things are marked:

  * TRUTH            — where the answer should be.
  * SLICE minimum    — A_b frozen at nominal; this is what every 1-D scan in the campaign measured.
  * PROFILE minimum  — A_b minimised at each lifetime; the quantity a displaced minimum must be
                       measured on before it can be called a bias.

The white curve is A_b_hat(lifetime), the ridge the fit can slide along. Its slope IS the
degeneracy: where it is steep, a charge-normalisation error is absorbed by A_b rather than by
lifetime, which is exactly why the 1-D slice over-states lifetime's sensitivity by ~25x.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_scan2d import load2d, summarise, find, NOM

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})

# maps for the three conditions that actually move A_b; ALL FIVE appear in the summary panel
CONDS = [('true', 'true geometry'), ('guess', 'straight-line guess'),
         ('dmeanp2', 'dE/dx mean +2%')]
TAGS = {'true': 'true_s2d', 'guess': 'guess_s2d', 'dmeanp2': 'dmeanp2_s2d'}
ALL5 = [('true_s2d', 'true'), ('dedx040_s2d', 'dE/dx f=0.40'), ('pos880_s2d', 'pos 880µm'),
        ('dmeanp2_s2d', 'dE/dx mean +2%'), ('guess_s2d', 'guess file')]
if len(sys.argv) > 1:                       # allow a smoke-test override
    TAGS = {'true': sys.argv[1]}
    CONDS = [('true', 'smoke test')]

got = []
for key, lab in CONDS:
    f = find(TAGS[key])
    if not f:
        print(f'  {key}: no scan yet'); continue
    r = load2d(f)
    if r is None:
        print(f'  {key}: no complete batch'); continue
    a, b, L = r
    got.append((key, lab, a, b, L, summarise(a, b, L)))
if not got:
    raise SystemExit('no 2-D scans available yet')

n = len(got)
fig, axes = plt.subplots(1, n + 1, figsize=(4.2 * (n + 1), 4.4))
if n + 1 == 1:
    axes = [axes]

for k, (key, lab, a, b, L, s) in enumerate(got):
    ax = axes[k]
    S = s['S'] - s['S'].min()
    ax_pct = 100 * (a - NOM['lifetime']) / NOM['lifetime']
    b_pct = 100 * (b - NOM['Ab']) / NOM['Ab']
    m = ax.pcolormesh(ax_pct, b_pct, S.T, shading='nearest', cmap='viridis_r',
                      vmin=0, vmax=np.percentile(S, 70))
    lv = [0.5, 2, 8, 32]
    ax.contour(ax_pct, b_pct, S.T, levels=[l for l in lv if l < S.max()],
               colors='white', linewidths=.7, alpha=.65)
    ax.plot(ax_pct, 100 * (s['bhat'] - NOM['Ab']) / NOM['Ab'], color='white', lw=2.0,
            label='ridge  $\\hat{A_b}$(lifetime)')
    ax.scatter([0], [0], marker='*', s=200, color=C['red'], zorder=6, label='truth')
    ax.axvline(s['slice_pct'], color=C['orange'], lw=1.8, ls='--',
               label=f"slice min  {s['slice_pct']:+.1f}%")
    ax.axvline(s['prof_pct'], color=C['green'], lw=2.0,
               label=f"profile min  {s['prof_pct']:+.1f}%")
    ax.set_xlabel('lifetime offset from truth (%)')
    if k == 0:
        ax.set_ylabel('A$_b$ offset from truth (%)')
    ax.set_title(f'({chr(97+k)}) {lab}', fontsize=9.5)
    ax.legend(fontsize=7.0, frameon=True, framealpha=.75, loc='upper right')
    fig.colorbar(m, ax=ax, label='Δ NLL' if k == n - 1 else '')

ax = axes[-1]
summary = []
for tg, lab in ALL5:
    f = find(tg)
    if not f: continue
    r = load2d(f)
    if r is None: continue
    aa, bb, LL = r
    summary.append((lab, summarise(aa, bb, LL)))
got_bar = summary if summary else [(g[1], g[5]) for g in got]
xs = np.arange(len(got_bar))
ax.grid(True, alpha=.25)
ax.bar(xs - .18, [g[1]['slice_pct'] for g in got_bar], width=.34, color=C['orange'],
       edgecolor='white', label='1-D SLICE (A$_b$ frozen)')
ax.bar(xs + .18, [g[1]['prof_pct'] for g in got_bar], width=.34, color=C['green'],
       edgecolor='white', label='PROFILE (A$_b$ minimised)')
for k, g in enumerate(got_bar):
    if abs(g[1]['slice_pct'] - g[1]['prof_pct']) < 0.05:
        ax.text(k, max(g[1]['slice_pct'], 0) + 1.2, 'no\nchange', ha='center',
                fontsize=6.6, color=C['grey'])
ax.axhline(0, color=C['grey'], lw=1.3)
ax.set_xticks(xs); ax.set_xticklabels([g[0] for g in got_bar], fontsize=7.4, rotation=25, ha='right')
ax.set_ylabel('lifetime minimum offset (%)')
ax.set_title(f'({chr(97+n)}) profiling over A$_b$ is the correction', fontsize=9.5)
ax.legend(fontsize=7.6, frameon=False)

fig.suptitle('Fig 40 — profiling over A$_b$ removes the apparent lifetime bias wherever the defect is a '
             'CHARGE-NORMALISATION error\n(guess file +13.9% → +0.4%), and changes nothing where it '
             'is not (dE/dx spread, geometry). Orange = every 1-D result in this report.',
             fontsize=9.8)
fig.tight_layout(rect=[0, 0, 1, .90])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig40_scan2d_profile.png'); plt.close(fig)
print('wrote', f'{OUT}/fig40_scan2d_profile.png')
for key, lab, a, b, L, s in got:
    print(f"  {lab:22s} nbatch={L.shape[0]:2d} grid {L.shape[1]}x{L.shape[2]}  "
          f"slice {s['slice_pct']:+7.2f}%  profile {s['prof_pct']:+7.2f}%  "
          f"Ab at joint min {s['bmin_pct']:+.2f}%")
