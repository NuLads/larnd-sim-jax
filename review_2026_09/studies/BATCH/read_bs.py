"""28 Oct batch-size probe readout: per LEN (cm per batch), the actual number of batches and total track length (log),
time per epoch for epochs 1-5 (training_history step_time), steady s/it and s/epoch (median of epochs 3-5), throughput
(cm per s), GPU peak (max peak_bytes_in_use) and host MaxRSS (sacct).
usage (container, PYTHONPATH = live tree): python3 read_bs.py"""
import glob, pickle, re, subprocess, numpy as np
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'; LG = '/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/logs'
JOBS = dict(l.split() for l in open('/sdf/scratch/neutrino/pgranger/review_2026_09/BATCH/bs_jobs.txt'))
print(f"{'LEN':>5s} {'NB':>4s} {'cm':>7s} | epoch 1..5 (min) | {'s/it':>6s} {'s/epoch':>8s} {'cm/s':>6s} | GPU peak GiB | MaxRSS | state")
for name, j in JOBS.items():
    L = int(name[2:]); log = open(f'{LG}/{name}-{j}.out', errors='replace').read()
    nb = re.findall(r'number of simulation batches is (\d+)', log); tl = re.findall(r'simulation data includes a total track length of ([0-9.]+)', log)
    try:   # sacct exists on the host, not in the container: then read the waiter's sacct dump
        sa = subprocess.run(['sacct', '-j', j, '-X', '-n', '-o', 'State,Elapsed'], capture_output=True, text=True).stdout.split()
        rss = subprocess.run(['sacct', '-j', j, '-n', '-o', 'MaxRSS'], capture_output=True, text=True).stdout.split()
    except FileNotFoundError:
        rows = [l.split() for l in open('/sdf/scratch/neutrino/pgranger/review_2026_09/BATCH/sacct.txt') if l.split() and l.split()[0].split('.')[0] == j]
        sa = next((r[1:3] for r in rows if r[0] == j), ['?']); rss = [r[3] for r in rows if len(r) > 3]
    fs = sorted(glob.glob(f'{FR}/sci_full_{name}/history_iter*_*.pkl'))
    oom = 'RESOURCE_EXHAUSTED' in log or 'out of memory' in log.lower()
    if not fs:
        print(f'{L:5d} {nb[0] if nb else "?":>4s} {tl[0] if tl else "?":>7s} | no checkpoint | {" ".join(sa)}{" OOM" if oom else ""}'); continue
    d = pickle.load(open(fs[-1], 'rb')); st = np.array(d['step_time'], float); n = int(nb[0]) if nb else len(st) // 5
    ep = [st[k * n:(k + 1) * n].sum() / 60 for k in range(len(st) // n)]
    sit = float(np.median(st[2 * n:])) if len(st) >= 3 * n else float('nan')
    pk = max(m.get('peak_bytes_in_use', 0) for m in d['memory']) / 2 ** 30
    cm = float(tl[0]) if tl else float('nan')
    print(f'{L:5d} {n:4d} {cm:7.0f} | ' + ' '.join(f'{e:5.1f}' for e in ep) + f' | {sit:6.3f} {sit * n:8.1f} {cm / (sit * n):6.1f} | {pk:6.1f} | '
          f'{max(rss, key=lambda x: float(x[:-1]) if x[:-1].replace(".", "").isdigit() else 0) if rss else "?"} | {" ".join(sa)}{" OOM" if oom else ""}')
