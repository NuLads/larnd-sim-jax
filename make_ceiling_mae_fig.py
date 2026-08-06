"""Fig 39: what the dE/dx block can actually recover when geometry is perfect.

WHY CEILING RUNS ARE THE RIGHT PLACE TO ASK. In ceiling mode the sim input IS the target, so the
long-standing `dedx_mae_iter` -- which is computed against `--input_file_sim` and is therefore NOT
a truth error in any S3/S4 run -- happens to coincide with the truth error. Every historical
ceiling run is therefore directly readable without the arc-length matcher.

WHAT THE NUMBERS MEAN. The per-segment dE/dx parameters are initialised at the prior centre
(`dedx_student_loc`); `_get_or_init_dedx_state` states explicitly that no true dE/dx information is
used. So every run starts at the same place -- MAE ~ 0.129, the distance from the prior to truth --
and "gap closed" is the fraction of that distance the fit removes. A perfect block would reach 0.

TWO EFFECTS, NOT ONE. The full 17-64% spread has two causes and they must not be conflated:
  * NOISE. Noiseless runs reach 53-64%; noise-ON runs 40-49%. Every historical run used
    prior_w = 0.5, so the prior explains none of this part.
  * THE dE/dx PRIOR WEIGHT. Against the matched comparison -- noise-ON, prior_w = 0.5 -- the
    historical band is 40-49% and the recent arms reach 17.6%. Those arms inherited
    `SCIDEDXPRIOR=5` from the ANNEALLONG (fitted-geometry) config against the 0.5 default.
    It is the only dE/dx knob that differs; `ceil_p05` is the clean single-variable test.

All configurations here are READ FROM THE CHECKPOINTS: every history pickle carries the full
argparse Namespace under `config`, old runs included. (An earlier version of this figure claimed
the historical prior weight was unrecoverable because those runs predate the `provenance` block --
wrong: `config` was always there.)
"""
import glob, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})

def read_cfg(h):
    c = h.get('config')
    c = c.item() if hasattr(c, 'item') else c
    if c is None:
        return None, None
    return getattr(c, 'dedx_prior_weight', None), ('OFF' if getattr(c, 'no_noise', False) else 'ON')


def series(d):
    out = []
    for f in sorted(glob.glob(d + '/history_iter*.pkl')):
        try:
            h = pickle.load(open(f, 'rb'))
        except Exception:
            continue
        a = np.asarray([x for x in h.get('dedx_mae_iter', []) if x == x], float)
        if len(a) > 200:
            out.append(a)
    return out


rows, curves = [], {}
for d in sorted(glob.glob('fit_result/sci_ceiling*')):
    if not os.path.isdir(d):
        continue
    ss = series(d)
    if not ss:
        continue
    h = pickle.load(open(sorted(glob.glob(d + '/history_iter*.pkl'))[-1], 'rb'))
    pw, noise = read_cfg(h)
    if pw is None:
        continue
    st = np.mean([a[:50].mean() for a in ss]); en = np.mean([a[-50:].mean() for a in ss])
    name = os.path.basename(d)
    rows.append((name, len(ss), st, en, 100 * (1 - en / st), pw, noise))
    curves[name] = ss[0]
rows.sort(key=lambda r: -r[4])

fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.6))

a = ax[0]
y = np.arange(len(rows))
# colour by the two real variables: dE/dx prior weight, and noise
def colr(r):
    if r[5] and r[5] > 1: return C['red']          # stiff prior
    return C['blue'] if r[6] == 'ON' else C['green']
cols = [colr(r) for r in rows]
a.barh(y, [r[4] for r in rows], color=cols, edgecolor='white', linewidth=.7)
a.set_yticks(y); a.set_yticklabels(
    [f'{r[0].replace("sci_ceiling","") or "(base)"}  n={r[1]}  w={r[5]:g} {r[6]}' for r in rows],
    fontsize=6.8)
a.invert_yaxis()
for k, r in enumerate(rows):
    a.text(r[4] + 1, k, f'{r[4]:.0f}%', va='center', fontsize=7.4, color=C['ink'])
a.set_xlabel('fraction of the prior→truth gap closed (%)')
a.set_title('(a) with PERFECT geometry, how much of the per-segment\n'
            'dE/dx does the block actually recover?', fontsize=9.5)
a.legend(handles=[plt.Line2D([], [], color=C['green'], lw=7, label='prior w=0.5, noise OFF'),
                  plt.Line2D([], [], color=C['blue'], lw=7, label='prior w=0.5, noise ON'),
                  plt.Line2D([], [], color=C['red'], lw=7, label='prior w=5, noise ON  (recent)')],
         fontsize=7.4, frameon=False, loc='lower right')

a = ax[1]
SHOW = [('sci_ceiling_400', C['green'], 'w=0.5, noise OFF  (62.7%)'),
        ('sci_ceiling_noise_s2_thr2500', C['blue'], 'w=0.5, noise ON  (49.0%)'),
        ('sci_ceiling_CEILBASE', C['red'], 'w=5,   noise ON  (17.6%)'),
        ('sci_ceiling_CEILW', C['orange'], 'w=5,   noise ON + mean_w 3000')]
for name, col, lab in SHOW:
    if name not in curves:
        continue
    v = curves[name]; k = 200
    a.plot(np.convolve(v, np.ones(k) / k, 'valid'), color=col, lw=2, label=lab)
a.axhline(np.mean([r[2] for r in rows]), color=C['grey'], ls='--', lw=1.4)
a.text(200, np.mean([r[2] for r in rows]), ' prior centre → truth (all runs start here)',
       fontsize=7.6, color=C['grey'], va='bottom')
a.set_xlabel('iteration (200-step rolling mean)'); a.set_ylabel('per-segment dE/dx MAE (= truth error)')
a.set_title('(b) noise costs ~14 points; the stiff prior\ncosts ~25 more and stalls within ~1000 iters', fontsize=9.5)
a.legend(fontsize=7.6, frameon=False)

fig.suptitle('Fig 39 — even with TRUE geometry and noise ON the dE/dx block recovers only 40–49% '
             'of the gap to truth.\nThe recent arms reach 17.6% because they inherited a 10× '
             'stiffer prior. Objective-limited, not geometry-limited.', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .90])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig39_ceiling_dedx_recovery.png'); plt.close(fig)
print('wrote', f'{OUT}/fig39_ceiling_dedx_recovery.png')
for r in rows:
    print(f'  {r[0]:32s} n={r[1]} gap {r[4]:5.1f}%  prior_w={r[5]:<5g} noise={r[6]}')
