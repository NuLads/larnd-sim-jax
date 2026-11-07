"""ENDFIT (k) readout (rule: REVIEW.md 28 Oct "ENDFIT (k)", written before any result): do epoch acceleration (EA,
option 1) or joint Newton steps (JN, option 2) close the legacy shared-smear fast-vs-long gap?
  EXS<s>EA / EXS<s>JN = the (h) fast continuation of EXS<s>FR @ 14000 -> 20000 (as EXS<s>H) + LARND_EPOCH_ACCEL_BETA=0.8
  (14600-19000) / LARND_JOINT_NEWTON=14000,14400,14800 (10-candidate ladder); s = 1, 2, 4.
Metric (read_j.py): mean over s of |X - EXS<s>L @ 44000| (lifetime, long_diff). Rule per arm: CLOSES THE GAP if
<= 2.5 and <= 3; HELPS if not that but both <= 0.8 x H's; HARMFUL if either >= 1.25 x H's; else NO GAIN.
usage (container, PYTHONPATH = live tree): python3 read_k.py"""
import os
os.environ['RJ_BASE'] = 'EXS'; os.environ['RJ_ARMS'] = 'H:20000,EA:20000,JN:20000'
exec(open('/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/read_j.py').read())
hl, hd = V['H']
print(f'\n(k) VERDICTS vs EXS H ({hl:.2f} / {hd:.2f}; HELPS <= {0.8 * hl:.2f} / {0.8 * hd:.2f}, HARMFUL >= {1.25 * hl:.2f} / {1.25 * hd:.2f})')
for a in ('EA', 'JN'):
    l, d = V[a]
    v = ('CLOSES THE GAP' if l <= 2.5 and d <= 3 else 'HELPS' if l <= 0.8 * hl and d <= 0.8 * hd
         else 'HARMFUL' if l >= 1.25 * hl or d >= 1.25 * hd else 'NO GAIN')
    print(f'  EXS{a} @ 20000: |X - L| lifetime {l:.2f}, long_diff {d:.2f} -> {v}')
print('\nper run: position (um), epoch-mean loss of the last epoch, and the option logs')
for s in S:
    for a in ('H', 'EA', 'JN'):
        d = ck(f'EXS{s}{a}', 20000); lo = np.array(d['losses_iter'], float)
        line = f'  EXS{s}{a:3s} pos {R["EXS", s, a]["pos"]:7.1f}  last-epoch loss {lo[-100:].mean():9.1f}'
        if a == 'EA':
            ea = d.get('epoch_accel', [])
            pu = [r for r in ea if 'geom' in r or 'dedx' in r]; rs = [r for r in ea if r.get('restart')]
            g = [r['geom']['rms_um'] for r in pu if 'geom' in r]
            line += f' | pushes {len(pu)}, restarts {len(rs)}, geom push rms median {np.median(g) if g else float("nan"):.1f} um'
        if a == 'JN':
            for r in d.get('joint_newton', []):
                if r.get('accepted'):
                    dc = next(e['d_cal'] for e in r['ladder'] if (e['kappa_c'], e['kappa']) == (r['kappa_c'], r['kappa']))
                    line += (f"\n      step {r['step']} @{r['it_end']}: kappa {r['kappa_c']:g}/{r['kappa']:g} dJ {r['dJ']:+.4g} (model {r['model']:+.4g}) "
                             f"d_cal(%) Ab {100 * dc['Ab']:+.2f} long_diff {100 * dc['long_diff']:+.2f} lifetime {100 * dc['lifetime']:+.2f} "
                             f"geom rms {r['geom_rms_um']:.0f} um | {r['time_s'] / 60:.0f} min")
                else:
                    line += f"\n      step {r['step']} @{r['it_end']}: no candidate lowers the objective | {r['time_s'] / 60:.0f} min"
        print(line)
