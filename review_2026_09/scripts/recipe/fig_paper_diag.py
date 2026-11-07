"""Paper figures: (1) coherent drift offset vs lifetime error across configurations; (2) timing calibration at truth."""
import sys, json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/GEOFRZ')
import drift_residuals as dr
OUT = '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/figs/'
CLASSES = [  # (label, colour, marker, [(arm, samples)])
    ('production: straight start, free geometry', 'C3', 'o', [('DSREP', (1, 2, 3, 4, 7, 8)), ('K6FREE', (1, 2, 4)), ('LDPIN', (3, 7, 8))]),
    ('true start, free geometry', 'C1', 's', [('GTRU', (1, 2, 4, 8))]),
    ('true start, free, self-consistent targets', 'C5', 'P', [('PSTGTRU', (1, 2, 4))]),
    ('hit-seeded start, free geometry', 'C4', 'D', [('HITSEED', (1, 2, 4))]),
    ('drift frozen at truth (± noise)', 'C7', '^', [('DRIFTTRUK6', (1, 2, 4)), ('DRIFTNZ50', (1, 2, 4)), ('DRIFTNZ150D', (1, 2, 4)), ('DRIFTTRU', (1, 2, 3, 4, 7, 8))]),
    ('recipe: hit seed + freeze straight tracks', 'C2', '*', [('STRFRZ50', (1, 2, 4, 5, 6, 9))]),
]
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
for lab, col, mk, arms in CLASSES:
    xs, ys = [], []
    for arm, ss in arms:
        for s in ss:
            tracks, errs = dr.run_residuals(f'sci_full_{arm}{s}')
            m = np.array([t[1].mean() for t in tracks]); m = m[np.abs(m) < 2000]   # drop diverged tracks (> 2 mm mean)
            xs.append(m.mean()); ys.append(errs['lifetime'])
    ax[0].plot(xs, ys, mk, color=col, ms=9 if mk == '*' else 6, label=lab, alpha=0.85)
    print(lab, np.round(xs, 1), np.round(ys, 2), flush=True)
ax[0].axhline(0, color='k', lw=0.6); ax[0].axvline(0, color='k', lw=0.6)
ax[0].set_xlabel('coherent drift offset of fitted tracks vs truth [µm]  (+ = away from anode)')
ax[0].set_ylabel('lifetime error [%]'); ax[0].legend(fontsize=7.5, loc='upper left')
ax[0].set_title('fitted geometry offset and lifetime bias, one point per fit')
# right: drift-distance profile of the offset, truth start free vs frozen
for arm, col, lab in (('GTRU', 'C1', 'true start, free (stochastic targets)'), ('PSTGTRU', 'C5', 'true start, free (self-consistent targets)'), ('HITSEED', 'C4', 'hit seed, free'), ('STRFRZ50', 'C2', 'recipe')):
    prof = []
    for s in (1, 2, 4):
        tracks, _ = dr.run_residuals(f'sci_full_{arm}{s}'); prof.append(dr.summarise([t for t in tracks if abs(t[1].mean()) < 2000])['mean'])
    prof = np.nanmean(np.array(prof, float), axis=0); c = 0.5 * (dr.EDGES[1:] + dr.EDGES[:-1])
    ax[1].plot(c, prof, '-o', color=col, ms=3, label=lab)
ax[1].axhline(0, color='k', lw=0.6); ax[1].set_xlabel('drift distance [cm]'); ax[1].set_ylabel('mean fitted − true drift [µm]')
ax[1].set_title('offset profile (samples 1/2/4)'); ax[1].legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT + 'fig21_drift_offset_vs_lifetime.png', dpi=140)
# --- figure 2: timing calibration at truth
z = np.load('/sdf/scratch/neutrino/pgranger/review_2026_09/E_truthdiag/timing/timing_pit_b0_20.npz'); R = z['rec']
k, pe, pz, pit = R[:, 2], R[:, 6], R[:, 7], R[:, 8]; D = 30.58975 - np.abs(pz)
sh = np.load('/sdf/scratch/neutrino/pgranger/review_2026_09/E_truthdiag/timing/shift_b0_20.npz')
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
edges = np.arange(0, 33, 4)
for lab, m0, col in (('first hits (P>0.95)', (k == 0) & (pe > .95), 'C0'), ('all hits', pe > 0, 'C1')):
    c, mu, er = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = m0 & (D >= lo) & (D < hi)
        if m.sum() > 100: c.append(0.5 * (lo + hi)); mu.append(pit[m].mean()); er.append(pit[m].std(ddof=1) / np.sqrt(m.sum()))
    ax[0].errorbar(c, mu, er, fmt='o-', color=col, label=lab, capsize=2)
ax[0].axhline(0.5, color='k', lw=0.6); ax[0].set_xlabel('drift distance [cm]'); ax[0].set_ylabel('mean PIT of observed tick')
ax[0].set_title('timing calibration at truth (0.5 = calibrated)'); ax[0].legend(fontsize=8)
def argmax_parab(L, D_=sh['deltas']):
    j = int(np.argmax(L)); j = min(max(j, 1), len(L) - 2); y0, y1, y2 = L[j - 1], L[j], L[j + 1]; den = y0 - 2 * y1 + y2
    return D_[j] + (0.1 * 0.5 * (y0 - y2) / den if den else 0.0)
cen = 0.5 * (sh['edges'][1:] + sh['edges'][:-1])
for key, col, lab in (('selfcons_TW0', 'C2', 'self-consistent hits, exact likelihood'), ('selfcons_TW2', 'C0', 'self-consistent hits, production likelihood'),
                      ('stoch_TW0', 'C3', 'stochastic hits, exact likelihood'), ('stoch_TW2', 'C1', 'stochastic hits, production likelihood')):
    ax[1].plot(cen, [argmax_parab(L) * 158.7 for L in sh[key]], 'o-', color=col, label=lab)
ax[1].axhline(0, color='k', lw=0.6); ax[1].set_xlabel('drift distance [cm]'); ax[1].set_ylabel('timing shift maximising the likelihood [µm]')
ax[1].set_title('where the likelihood puts the drift coordinate'); ax[1].legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT + 'fig22_timing_closure.png', dpi=140)
print('saved both')
