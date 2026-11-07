"""DATASZ (3 Nov; REVIEW.md "DATASZ: accuracy vs data volume and batch size"): every job as 6 dependency chains, chain c
starting after (afterany) the last job of UQ-test chain c, so <= 6 run at once and nothing runs before the UQ test ends.
Units (a pipeline fit -> FR -> H via submit_ds.sh, optionally followed in the same chain by its calibration-only E1 dump)
are assigned greedily by estimated GPU-h to the least loaded chain: all V-arm units first, then the B arms.
Job ids -> DATASZ/ds_jobs.txt. usage: python3 submit_dsall.py [--dry]"""
import subprocess, sys, os, time
DS = '/sdf/scratch/neutrino/pgranger/review_2026_09/DATASZ'
S = '/sdf/scratch/neutrino/pgranger/fit_result_staging'; FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
UQ_TAILS = [42929678, 42929683, 42929688, 42929693, 42929697, 42929701]
DRY = '--dry' in sys.argv
SEEDS = range(17, 25)
CFG = {'V025': (600, 25, 400, 1.2, (17, 19, 21), 0.15), 'V050': (600, 50, 400, 2.0, (17, 19, 21), 0.25),
       'V200': (1200, 200, 400, 6.2, (17, 18, 19), 0.85), 'B800': (600, 50, 800, 3.0, (), 0),
       'B200': (600, 200, 200, 4.0, (), 0)}
LOG = f'{DS}/ds_jobs.txt'


def units(arms):
    u = []
    for a in arms:
        dsz, nb, L, h, dumps, hd = CFG[a]
        for s in SEEDS:
            u.append((h + (hd if s in dumps else 0), a, s))
    return sorted(u, key=lambda x: -x[0])


def sh(cmd, env=None):
    if DRY:
        print('  DRY:', ' '.join(cmd) if isinstance(cmd, list) else cmd, env or ''); return '0 1 2 3'.split()[0:1] and 'X 1 2 3'
    e = dict(os.environ, **(env or {}))
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, env=e, universal_newlines=True).stdout.strip()


def link(r):
    if os.path.exists(f'{FR}/{r}') and not os.path.islink(f'{FR}/{r}'):
        raise SystemExit(f'{FR}/{r} exists and is not a link')
    if not DRY:
        os.makedirs(f'{S}/{r}', exist_ok=True)
        if not os.path.islink(f'{FR}/{r}'):
            os.symlink(f'{S}/{r}', f'{FR}/{r}')


load = [0.0] * 6; last = list(UQ_TAILS); plan = [[] for _ in range(6)]
for phase in (('V200', 'V050', 'V025'), ('B200', 'B800')):
    for h, a, s in units(phase):
        c = min(range(6), key=lambda i: load[i]); load[c] += h; plan[c].append((a, s))
log = open(LOG, 'a') if not DRY else sys.stdout
log.write(f'# submitted {time.strftime("%F %T")}; chains after UQ tails {UQ_TAILS}; est. GPU-h per chain '
          + ' '.join(f'{x:.1f}' for x in load) + '\n')
for c in range(6):
    for a, s in plan[c]:
        dsz, nb, L, h, dumps, hd = CFG[a]; name = f'{a}s{s}'
        out = sh([f'{DS}/submit_ds.sh', name, str(s), str(dsz), str(nb), str(L)], env={'DEP': f'afterany:{last[c]}'})
        n, j1, j2, j3 = out.split()[-4:]
        log.write(f'chain {c} prod {name} {j1} {j2} {j3} :: after {last[c]}\n'); last[c] = j3
        if s in dumps:
            dn = f'D_{name}'; it = int(20000 * nb / 100)
            if os.path.exists(f'/sdf/scratch/neutrino/pgranger/review_2026_09/UQ/dumps/{dn}.npz'):
                raise SystemExit(f'{dn} exists')
            link(f'sci_full_UQ_{dn}')
            j = sh(['sbatch', '--parsable', '--account=neutrino:dune-ml', '--exclude=sdfampere038,sdfampere003',
                    f'--job-name=UQ_{dn}', f'--dependency=afterany:{last[c]}', f'{DS}/run_ds_dump.sbatch',
                    f'{name}H', dn, str(it), 'LARND_JOINT_NEWTON_DUMP_CALIB=1'])
            j = j.split()[-1]
            log.write(f'chain {c} dump {dn} {j} :: {name}H @ {it} after {last[c]}\n'); last[c] = j
    log.flush()
log.write(f'# chain tails {last}\n')
print('est. GPU-h per chain', ' '.join(f'{x:.1f}' for x in load), '| plan:', [[f'{a}s{s}' for a, s in p] for p in plan])
