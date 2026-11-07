"""Fig. 4 (larndsim_comparison) noise-on differences, decomposed (12 Oct): which reference hit pixels our simulation cannot
produce (not in its active-pixel list), and the hits/ADC distributions on the pixels both simulate."""
import sys, numpy as np
sys.argv = sys.argv[:1]; sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/paperfigs')
import importlib.util
spec = importlib.util.spec_from_file_location('g', '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/paperfigs/fig_larndsim_comparison_geo.py')
src = open(spec.origin).read().split('nums = {}')[0]          # only the loaders, not the plotting
g = {}; exec(compile(src, spec.origin, 'exec'), g)
for stage in (2, 3):
    ref, jx, jrow = g['load'](stage)
    nref, njax, tref, tjax, J = g['per_pixel_hits'](ref, jx, jrow)
    valid = ref['ticks'] != 0
    hit_ref = nref > 0
    outside = hit_ref & (jrow < 0)                 # reference hit pixels our simulation does not simulate at all
    inlist = hit_ref & (jrow >= 0)
    up = np.asarray(jx['unique_pixels']); hp = np.asarray(jx['hit_pixels'])
    jhit_pix = set(int(p) for p in hp if p >= 0)
    print(f'=== stage {stage} ({"noise" if stage == 3 else "no noise"}) ===')
    print(f'reference: {hit_ref.sum()} hit pixels, {nref.sum()} hits; ours: {len(jhit_pix)} hit pixels, {int((hp >= 0).sum())} hits')
    print(f'  reference hit pixels NOT in our active list: {outside.sum()} pixels, {nref[outside].sum()} hits, '
          f'hits/pixel {np.bincount(nref[outside])[1:].tolist()}')
    # on pixels in our list
    print(f'  on pixels we simulate: reference {nref[inlist].sum()} hits on {inlist.sum()} pixels; ours {njax[inlist].sum()} hits')
    hr = np.bincount(nref[inlist], minlength=7)[:7]; hj = np.bincount(njax[inlist], minlength=7)[:7]
    print(f'    hits-per-pixel histogram (0..6) reference {hr.tolist()}  ours {hj.tolist()}')
    same = np.mean(nref[inlist] == njax[inlist]); print(f'    same number of hits on {same:.1%} of these pixels')
    # ADC on reference-only pixels vs common
    ra = ref['adc']
    a_out = ra[outside][valid[outside]]; a_in = ra[inlist][valid[inlist]]
    ja = np.asarray(jx['adc'])[hp >= 0]
    for name, a in (('ref, pixels we lack', a_out), ('ref, pixels we simulate', a_in), ('ours, all', ja)):
        if len(a): print(f'    ADC {name:24s}: n {len(a):4d} mean {a.mean():.1f} sd {a.std():.1f}  frac < 84: {np.mean(a < 84):.2f}')
