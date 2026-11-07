"""Benchmark v2 (19 Oct 2026, review R023): time and peak GPU memory per batch, probabilistic (PPP likelihood) vs
stochastic (MMD + charge loss), for the forward pass alone (--part fwd) or forward + gradient (--part grad), each in its
own process so that the peak-memory counter (which cannot be reset) belongs to that part only. Same data, models and
losses as paper_bench.py; one mode / length / part per process. Records the code version.
usage: python3 paper_bench2.py --mode prob|stoch --len CM --part fwd|grad --out FILE.json"""
import argparse, json, subprocess
from paper_common import *
ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=['prob', 'stoch'], required=True)
ap.add_argument('--len', type=float, required=True); ap.add_argument('--part', choices=['fwd', 'grad'], required=True)
ap.add_argument('--nb', type=int, default=3); ap.add_argument('--reps', type=int, default=5)
ap.add_argument('--out', required=True); ap.add_argument('--no_target', action='store_true'); ap.add_argument('--load_cpu', action='store_true')
a = ap.parse_args()
mem = lambda: {k: jax.devices()[0].memory_stats().get(k, -1) for k in ('bytes_in_use', 'peak_bytes_in_use')}
stages = {}
git = subprocess.run(['git', '-C', REPO, 'log', '-1', '--format=%h'], stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
dirty = subprocess.run(['git', '-C', REPO, 'status', '--porcelain', 'src/larndsim', 'optimize/strategies.py', 'optimize/dataio.py'],
                       stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
if a.load_cpu:   # build the dataset and the detector tables on the CPU, so that their transient does not set the GPU peak
    with jax.default_device(jax.devices('cpu')[0]):
        ds_sim, ds_tgt, fields = load(a.len, a.nb); truth, response = params()
    stages['after_load_cpu'] = mem()
    gpu = jax.devices()[0]; truth = jax.device_put(truth, gpu); response = jax.device_put(response, gpu)
else:
    ds_sim, ds_tgt, fields = load(a.len, a.nb); truth, response = params()
stages['after_load'] = mem()
sim = LUTProbabilisticSimulation(response) if a.mode == 'prob' else LUTSimulation(response)
loss = ProbabilisticLossStrategy(loss_fn=llhd_loss) if a.mode == 'prob' else GenericLossStrategy(mse_adc)
res = []
for i in range(min(a.nb, len(ds_tgt))):
    tracks = jax.device_put(jnp.asarray(ds_tgt[i]), jax.devices()[0]); L = track_length_cm(tracks, fields)
    tgt = None if a.no_target else target_hits(truth, response, tracks, fields, i + 1)
    stages[f'batch{i}_before'] = mem()
    rk = None if a.mode == 'prob' else (i + 1)
    fwd = lambda P: sim.predict(P, tracks, fields, rk)
    if a.part == 'fwd':
        run = lambda: tick(jax.tree_util.tree_leaves(fwd(truth))[0])
    else:
        vg = jax.value_and_grad(lambda P: loss.compute(P, fwd(P), tgt), has_aux=True)
        run = lambda: tick(vg(truth)[0][0])
    t0 = time.perf_counter(); first = run() - t0
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter(); ts.append(run() - t0)
    stages[f'batch{i}_after'] = mem()
    res.append(dict(batch=i, length_cm=L, first_s=first, time_s=float(np.median(ts))))
    print(f'[{a.mode} {a.part} len {a.len}] batch {i}: {L:.0f} cm  {np.median(ts):.3f} s  (first {first:.1f} s)', flush=True)
peak = jax.devices()[0].memory_stats().get('peak_bytes_in_use', -1)
json.dump(dict(stages=stages, no_target=a.no_target, load_cpu=a.load_cpu, mode=a.mode, part=a.part, max_batch_len=a.len, peak_bytes=peak, batches=res, git=git, code_dirty=dirty,
               device=str(jax.devices()[0])), open(a.out, 'w'), indent=1)
print(f'[{a.mode} {a.part} len {a.len}] peak GPU memory {peak/1e9:.2f} GB  (code {git}{" dirty" if dirty else ""})', flush=True)
