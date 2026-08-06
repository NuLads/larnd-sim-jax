"""Fig 36: WHY the standard lifetime fit fails on real hits — the two mechanisms, shown directly.

Fig 35 showed the outcome (tau spanning -14.5% to +123.5% with the fit range). This shows the
causes on the same hits:

 (a) THRESHOLD SELECTION. The hit-charge spectrum in three drift-time slices. The whole spectrum
     slides down with drift, but it is truncated at a FIXED 5 ke threshold, so the low tail is
     eaten progressively -- the survivors at long drift are a biased-high subset.
 (b) CHARGE SHARING. Hits per pixel against drift time. Near the anode there has been almost no
     diffusion, so a segment's charge is concentrated in fewer, larger hits; as diffusion sets in
     the same charge is split across more hits. Per-hit Q therefore falls for a reason that has
     nothing to do with attenuation.
 (c) EXPOSURE. Hits per unit drift time. This is NOT threshold loss -- a 3.6x fall cannot come
     from an 8.3% total attenuation -- it is the sample's own drift-time exposure, and it weights
     the fit very unevenly (the long-drift end, where the lifetime signal lives, is the sparsest).
 (d) THE FIX. Three candidate observables against drift time, each normalised to its own first
     bin, with the true exp(-t/tau) overlaid. Per-hit Q tracks neither; charge summed per pixel
     undoes the sharing and tracks truth much better.
"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

exec(open('analyze_lifetime_hits.py').read().split('if __name__')[0])   # constants + adc2charge

TAU = 2200.0
RAMP = ['#9ecae1', '#3182bd', '#08306b']          # sequential: early -> late drift


def load_full(d):
    Q, t, ev, pid = [], [], [], []
    for k, f in enumerate(sorted(glob.glob(os.path.join(d, 'batch*_target.npz')))):
        z = np.load(f)
        adc = np.asarray(z['adcs'], float).ravel()
        tk = np.asarray(z['ticks'], float).ravel()
        e = np.asarray(z['event'], float).ravel()
        p = np.asarray(z['pixel_id'], float).ravel()
        q = adc2charge(adc)
        m = np.isfinite(q) & (q > 0) & (tk > 0)
        Q.append(q[m]); t.append(tk[m] * T_SAMPLING)
        ev.append(e[m] + 1e6 * k)                  # make event ids unique across batches
        pid.append(p[m])
    return (np.concatenate(Q), np.concatenate(t),
            np.concatenate(ev), np.concatenate(pid))


d = sorted(glob.glob('target_hitdump_hitdump_noise_*'))[0]
Q, t, ev, pid = load_full(d)
key = ev.astype(np.int64) * 10 ** 7 + pid.astype(np.int64)
uk, inv = np.unique(key, return_inverse=True)
nhit_per_pix = np.bincount(inv)[inv]
sumQ_per_pix = np.bincount(inv, weights=Q)
tmin_per_pix = np.full(len(uk), np.inf); np.minimum.at(tmin_per_pix, inv, t)
print(f'{len(Q)} hits, {len(uk)} pixels')

fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))

# ---- (a) threshold selection
ax = axes[0, 0]
SL = [(0, 25), (80, 105), (165, 190)]
for (lo, hi), col in zip(SL, RAMP):
    m = (t >= lo) & (t < hi)
    ax.hist(Q[m], bins=np.linspace(4, 30, 60), histtype='step', lw=1.9, color=col,
            density=True, label=f'{lo}–{hi} µs  (n={m.sum()/1e3:.0f}k)')
ax.axvline(THRESHOLD_E / 1e3, color=C['red'], ls='--', lw=1.8)
ax.text(THRESHOLD_E / 1e3, ax.get_ylim()[1] * .93, '  5 ke threshold', color=C['red'], fontsize=8)
ax.set_xlabel('hit charge Q (ke)'); ax.set_ylabel('normalised hits')
ax.set_title('(a) threshold selection: the spectrum slides down\n'
             'into a FIXED cut, eating the low tail', fontsize=9.5)
ax.legend(fontsize=7.8, frameon=False, title='drift time', title_fontsize=7.8)

# ---- (b) charge sharing
ax = axes[0, 1]
e = np.linspace(0, 190, 20); i = np.digitize(t, e) - 1
tc, nh, qh = [], [], []
for b in range(19):
    m = i == b
    if m.sum() < 100:
        continue
    tc.append(.5 * (e[b] + e[b + 1])); nh.append(nhit_per_pix[m].mean()); qh.append(Q[m].mean())
tc, nh, qh = map(np.asarray, (tc, nh, qh))
ax.plot(tc, nh, color=C['purple'], lw=2, marker='o', ms=5)
ax.set_xlabel('drift time (µs)'); ax.set_ylabel('mean hits per pixel', color=C['purple'])
ax.tick_params(axis='y', labelcolor=C['purple'])
ax.axvspan(0, 20, color=C['grey'], alpha=.15)
ax.text(9, nh.min(), ' anode\n edge', fontsize=7.5, va='bottom', color=C['grey'])
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(tc, qh, color=C['orange'], lw=2, marker='s', ms=5)
ax2.set_ylabel('mean Q per hit (ke)', color=C['orange'])
ax2.tick_params(axis='y', labelcolor=C['orange'])
ax.set_title('(b) charge sharing: as diffusion sets in the same\n'
             'charge is split over MORE hits, so Q/hit drops', fontsize=9.5)

# ---- (c) hits lost to the threshold
ax = axes[1, 0]
cnt, _ = np.histogram(t, bins=e)
w = np.diff(e)
ax.plot(.5 * (e[:-1] + e[1:]), cnt / w, color=C['green'], lw=2, marker='o', ms=5)
ax.set_xlabel('drift time (µs)'); ax.set_ylabel('hits per µs')
# NOT threshold loss: a 3.6x fall cannot come from an 8.3% total attenuation. This is the
# sample's own drift-time exposure, and it matters because it weights the fit very unevenly.
ax.set_title('(c) the sample is strongly non-uniform in drift:\n3.6x fewer hits at the cathode',
             fontsize=9.5)
ax.axvspan(0, 20, color=C['grey'], alpha=.15)

# ---- (d) which observable tracks truth
ax = axes[1, 1]
def prof(tt, vv, est=np.mean):
    ii = np.digitize(tt, e) - 1
    a, b_ = [], []
    for k in range(19):
        m = ii == k
        if m.sum() < 100:
            continue
        a.append(.5 * (e[k] + e[k + 1])); b_.append(est(vv[m]))
    return np.asarray(a), np.asarray(b_)

for lab, (tt, vv), col in [
        ('per-hit Q', (t, Q), C['orange']),
        ('per-hit Q,  Q > 7 ke', (t[Q > 7], Q[Q > 7]), C['red']),
        ('Q summed per pixel', (tmin_per_pix, sumQ_per_pix), C['blue'])]:
    a, b_ = prof(tt, vv)
    k0 = np.argmin(np.abs(a - 30))            # normalise beyond the anode edge
    ax.plot(a, b_ / b_[k0], color=col, lw=2, marker='o', ms=4.5, label=lab)
tt = np.linspace(0, 190, 60)
ax.plot(tt, np.exp(-(tt - 30) / TAU), color=C['ink'], ls='--', lw=1.8,
        label=f'truth  exp(−t/{TAU:.0f} µs)')
ax.axvspan(0, 20, color=C['grey'], alpha=.15)
ax.set_xlabel('drift time (µs)'); ax.set_ylabel('charge, normalised at 30 µs')
ax.set_title('(d) per-hit Q tracks neither; summing over the\n'
             'pixel undoes the sharing', fontsize=9.5)
ax.legend(fontsize=7.6, frameon=False)

fig.suptitle('Fig 36 — the two front-end mechanisms that break the standard lifetime fit\n'
             'both are comparable to or larger than the 8.3% attenuation signal being measured',
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, .93])
fig.savefig(f'{OUT}/fig36_hit_mechanisms.png'); plt.close(fig)
print('wrote', f'{OUT}/fig36_hit_mechanisms.png')

for lab, (tt, vv) in [('per-hit Q', (t, Q)), ('per-hit Q>7ke', (t[Q > 7], Q[Q > 7])),
                      ('per-pixel sum', (tmin_per_pix, sumQ_per_pix))]:
    m = tt >= 20
    _, _, _, tau, dtau = fit_exp(tt[m], vv[m], est='mean')
    print(f'  {lab:16s} t>20us -> tau {tau:9.1f} +/- {dtau:7.1f} us ({100*(tau/TAU-1):+7.1f}%)')
