"""Figure: is lifetime degenerate with the drift position? (a) mean fitted-minus-true drift displacement vs
drift distance, (b) its scatter vs drift distance, (c) run-by-run long_diff bias vs local drift-slope error,
(d) run-by-run lifetime bias vs long_diff bias."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = '/sdf/scratch/neutrino/pgranger/review_2026_09/GEOFRZ'
J = json.load(open(f'{OUT}/drift_residuals.json'))
E = np.array(J['edges']); X = 0.5 * (E[1:] + E[:-1])
A = J['arms']
STY = {'DSREP': ('production (geometry + dE/dx fitted)', '#2a78d6', 'o'),
       'DXTRU': ('geometry fitted, dE/dx at truth', '#1baf7a', 's'),
       'DRIFTFRZ': ('drift frozen at straight line', '#eb6834', '^'),
       'DRIFTTRU': ('drift at truth (3 modes), rest fitted', '#4a3aa7', 'D'),
       'DRIFTTRUK6': ('drift at truth (6 modes), rest fitted', '#e87ba4', 'v')}
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for a, (lab, c, mk) in STY.items():
    p = A[a]['pooled']; m, se, r = map(np.array, (p['mean'], p['se'], p['rms']))
    ax[0, 0].plot(X, m, '-', color=c, lw=2, marker=mk, ms=5, label=lab)
    ax[0, 0].fill_between(X, m - se, m + se, color=c, alpha=0.18, lw=0)
    ax[0, 1].plot(X, r, '-', color=c, lw=2, marker=mk, ms=5, label=lab)
ax[0, 0].axhline(0, color='#52514e', lw=0.8, ls=':')
ax[0, 0].set_xlabel('drift distance (cm)'); ax[0, 0].set_ylabel('mean fitted − true drift position (µm)\n(+ = charge placed farther from the anode)')
ax[0, 0].set_title('(a) systematic drift displacement vs drift distance', loc='left', fontsize=10)
ax[0, 0].text(0.02, 0.04, 'faking the lifetime bias through attenuation alone\nwould need ~3000 µm at the far end', transform=ax[0, 0].transAxes, fontsize=8.5, color='#52514e')
ax[0, 1].set_xlabel('drift distance (cm)'); ax[0, 1].set_ylabel('RMS fitted − true drift position (µm)')
ax[0, 1].set_title('(b) random drift error vs drift distance', loc='left', fontsize=10); ax[0, 1].set_yscale('log')
ax[0, 1].legend(frameon=False, fontsize=8, loc='upper left')
xs, ys, ls = [], [], []
for a, (lab, c, mk) in STY.items():
    for s, v in A[a]['per_sample'].items():
        ax[1, 0].plot(v['slope_err_rms'], v['errs']['long_diff'], mk, color=c, ms=8, mec='white')
        ax[1, 1].plot(v['errs']['long_diff'], v['errs']['lifetime'], mk, color=c, ms=8, mec='white')
        xs.append(v['slope_err_rms']); ys.append(v['errs']['long_diff']); ls.append(v['errs']['lifetime'])
xs, ys, ls = map(np.array, (xs, ys, ls))
r1 = np.corrcoef(np.log(xs), ys)[0, 1]; r2 = np.corrcoef(ys, ls)[0, 1]
b2 = np.polyfit(ys, ls, 1)
xx = np.linspace(ys.min(), ys.max(), 10); ax[1, 1].plot(xx, np.polyval(b2, xx), '--', color='#8a8983', lw=1.2)
ax[1, 0].set_xscale('log'); ax[1, 0].axhline(0, color='#52514e', lw=0.8, ls=':')
ax[1, 0].set_xlabel('RMS local drift-slope error of the fitted geometry (µm per cm of track)'); ax[1, 0].set_ylabel('long_diff error (%)')
ax[1, 0].set_title(f'(c) longitudinal diffusion follows the drift-geometry error (r = {r1:+.2f}, log x)', loc='left', fontsize=10)
ax[1, 1].axhline(0, color='#52514e', lw=0.8, ls=':'); ax[1, 1].axvline(0, color='#52514e', lw=0.8, ls=':')
ax[1, 1].set_xlabel('long_diff error (%)'); ax[1, 1].set_ylabel('lifetime error (%)')
ax[1, 1].set_title(f'(d) lifetime follows longitudinal diffusion (r = {r2:+.2f}; slope {b2[0]:.2f})', loc='left', fontsize=10)
for a_ in ax.ravel():
    a_.spines[['top', 'right']].set_visible(False); a_.grid(color='#e5e4df', lw=0.6); a_.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_drift_residuals.png', dpi=150)
print(f'r(log slope-err, long_diff) = {r1:+.3f};  r(long_diff, lifetime) = {r2:+.3f}, slope {b2[0]:.3f} %/%;  n runs = {len(xs)}')
# within the six production fits alone
p = A['DSREP']['per_sample']
ld = np.array([v['errs']['long_diff'] for v in p.values()]); lt = np.array([v['errs']['lifetime'] for v in p.values()]); se = np.array([v['slope_err_rms'] for v in p.values()]); rf = np.array([v['rms_far'] for v in p.values()])
print(f'production only (n=6): r(long_diff, lifetime) = {np.corrcoef(ld, lt)[0,1]:+.2f}; r(slope-err, long_diff) = {np.corrcoef(se, ld)[0,1]:+.2f}; r(rms_far, lifetime) = {np.corrcoef(rf, lt)[0,1]:+.2f}')
