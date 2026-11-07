"""Input-quality ladder as PROFILES rather than 1-D slices (paper query: the ladder results are one-dimensional likelihood
slices with the other parameters held at truth). For each rung (a perturbed copy of the true input file, same events and
segmentation) compute the score of the production likelihood at the TRUE parameters, against the same stochastic
targets (generated from the true segments, seed batch+1) and on the same batches as paper_hessian.py. With H from
hessian.json (true inputs, same batches), the linear-response optimum shift is
    profile:  delta = -H^-1 g          (all five parameters free)
    slice  :  delta_j = -g_j / H_jj    (parameter j alone, others at truth; comparable with the scans)
in relative units. Rung 'true' reproduces the Hessian job's own score (sanity check)."""
import argparse, json
from paper_common import *
LAD = '/sdf/group/neutrino/pgranger/lads-data/quality_ladder'
RUNGS = ['dedx_f0.75', 'dedx_f0.5', 'dedx_f0.25', 'dedx_f0.0', 'pos_50um', 'pos_170um', 'pos_400um', 'pos_880um',
         'chord_c0.5', 'chord_c1.0', 'dedxmean0.02', 'dedxmean-0.02']
ap = argparse.ArgumentParser(); ap.add_argument('--len', type=float, default=200.); ap.add_argument('--nb', type=int, default=400)
ap.add_argument('--rungs', default=','.join(RUNGS)); ap.add_argument('--out', required=True); a = ap.parse_args()
rungs = a.rungs.split(',')
ds_sim, ds_tgt, fields = load(a.len, a.nb, 1); truth, response = params()
ds_r = {r: TgtTracksDataset(filename=f'{LAD}/{r}.h5', dataset_sim=ds_sim, chopped=True, pad=True, electron_sampling_resolution=0.01,
                            print_input=False) for r in rungs}
sim = LUTProbabilisticSimulation(response); loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)
res = dict(params=PARAMS, len=a.len, rungs={r: dict(g=np.zeros(5).tolist(), n=0, length_cm=0.) for r in ['true'] + rungs}, failed=[])
for i in range(min(a.nb, len(ds_tgt))):
    t0 = time.perf_counter(); tracks = jnp.asarray(ds_tgt[i]); Lb = track_length_cm(tracks, fields)
    tgt = target_hits(truth, response, tracks, fields, i + 1)
    for r in ['true'] + rungs:
        try:
            tr = tracks if r == 'true' else jnp.asarray(ds_r[r][i])
            g = jax.grad(lambda P: loss.compute(P, sim.predict(P, tr, fields, None), tgt)[0])(truth)
            gv = np.array([float(getattr(g, k)) * TRUTH[k] for k in PARAMS])
            e = res['rungs'][r]; e['g'] = (np.array(e['g']) + gv).tolist(); e['n'] += 1; e['length_cm'] += Lb
        except Exception as ex:
            res['failed'].append((i, r, repr(ex)[:200])); print(f'[batch {i}] {r} FAILED {repr(ex)[:150]}', flush=True)
    print(f'[batch {i}] {Lb:.0f} cm  {time.perf_counter()-t0:.1f} s', flush=True)
    if i % 10 == 0: json.dump(res, open(a.out, 'w'))
json.dump(res, open(a.out, 'w'))
print('done', flush=True)
