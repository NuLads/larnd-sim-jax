"""Benchmark: time and peak GPU memory of one forward and one forward+gradient evaluation per batch, probabilistic
(PPP likelihood) vs stochastic (MMD + charge loss), as a function of batch track length. One mode/length per process."""
import argparse, json
from paper_common import *
ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=['prob', 'stoch'], required=True)
ap.add_argument('--len', type=float, required=True); ap.add_argument('--nb', type=int, default=3); ap.add_argument('--reps', type=int, default=5)
ap.add_argument('--out', required=True); a = ap.parse_args()
ds_sim, ds_tgt, fields = load(a.len, a.nb); truth, response = params()
sim = LUTProbabilisticSimulation(response) if a.mode == 'prob' else LUTSimulation(response)
loss = ProbabilisticLossStrategy(loss_fn=llhd_loss) if a.mode == 'prob' else GenericLossStrategy(mse_adc)
res = []
for i in range(min(a.nb, len(ds_tgt))):
    tracks = jnp.asarray(ds_tgt[i]); L = track_length_cm(tracks, fields)
    tgt = target_hits(truth, response, tracks, fields, i + 1)
    rk = None if a.mode == 'prob' else (i + 1)
    fwd = lambda P: sim.predict(P, tracks, fields, rk)
    lfn = lambda P: loss.compute(P, fwd(P), tgt)
    vg = jax.value_and_grad(lfn, has_aux=True)
    t0 = time.perf_counter(); out = fwd(truth); t1 = tick(jax.tree_util.tree_leaves(out)[0]); first_fwd = t1 - t0
    t0 = time.perf_counter(); (v, _), g = vg(truth); t1 = tick(v); first_vg = t1 - t0
    tf, tg = [], []
    for _ in range(a.reps):
        t0 = time.perf_counter(); out = fwd(truth); tf.append(tick(jax.tree_util.tree_leaves(out)[0]) - t0)
        t0 = time.perf_counter(); (v, _), g = vg(truth); tg.append(tick(v) - t0)
    res.append(dict(batch=i, length_cm=L, first_fwd_s=first_fwd, first_fwd_grad_s=first_vg, fwd_s=float(np.median(tf)), fwd_grad_s=float(np.median(tg)),
                    grads={k: float(getattr(g, k)) for k in PARAMS}))
    print(f'[{a.mode} len {a.len}] batch {i}: {L:.0f} cm  fwd {np.median(tf):.3f} s  fwd+grad {np.median(tg):.3f} s  (first: {first_fwd:.1f} / {first_vg:.1f} s)', flush=True)
peak = jax.devices()[0].memory_stats().get('peak_bytes_in_use', -1)
json.dump(dict(mode=a.mode, max_batch_len=a.len, peak_bytes=peak, batches=res), open(a.out, 'w'), indent=1)
print(f'[{a.mode} len {a.len}] peak GPU memory {peak/1e9:.2f} GB', flush=True)
