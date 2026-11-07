"""Integrated induced charge per pixel, NuLads against the reference larnd-sim (stage 2 of the aligned-geometry
comparison: diffusion on, noise off), binned in the reference charge, for production NuLads and (if present) the exact
neighbor induction (out_stage2_jax_geo_exact.h5). usage: python compare_geo.py [OUT.json]"""
import sys, os, json, numpy as np
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP'; N = '/sdf/scratch/neutrino/pgranger/review_2026_09/NEIGH'
sys.argv = sys.argv[:2]; OUT = sys.argv[1] if len(sys.argv) > 1 else N + '/compare_geo.json'
sys.path.insert(0, D); sys.argv = sys.argv[:1]
import analyze_comparison as ac
DX, DY = -0.2217, -0.2217
ref = ac.load_reference_flat(f'{D}/out_stage2_ref_geo.h5'); rx, ry = ac.ref_pixel_physical_coords(ref['pixels'])[:2]
qref = ref['pixel_signals'].sum(1) * 0.1; nref = (ref['ticks'] != 0).sum(1)
res = {}
for tag, f in (('prod', f'{D}/out_stage2_jax_geo.h5'), ('exact', f'{N}/out_stage2_jax_geo_exact.h5')):
    if not os.path.exists(f): continue
    jx = ac.load_jax_direct(f); up = np.asarray(jx['unique_pixels']); keep = up >= 0; idx = np.where(keep)[0]
    jxx = np.asarray(jx['pix_x_decoded'])[keep] + DX; jyy = np.asarray(jx['pix_y_decoded'])[keep] + DY
    jrow = np.full(len(rx), -1)
    for i in range(len(rx)):
        m = np.where((np.abs(jxx - rx[i]) < 0.01) & (np.abs(jyy - ry[i]) < 0.01))[0]
        if len(m): jrow[i] = idx[m[0]]
    w = np.asarray(jx['wfs']); qj = np.array([w[j].sum() * 0.1 if j >= 0 else np.nan for j in jrow])
    sel = jrow >= 0; r = {'matched': int(sel.sum()), 'ref_pixels': len(rx)}
    for lo, hi in ((0, 2e3), (2e3, 5e3), (5e3, 1e4), (1e4, 2e4), (2e4, 1e9)):
        m = sel & (qref >= lo) & (qref < hi)
        if m.sum(): r[f'{lo:.0f}-{hi:.0g}'] = dict(n=int(m.sum()), ratio_median=float(np.median(qj[m] / qref[m])),
                                               ratio_iqr=[float(np.percentile(qj[m] / qref[m], 25)), float(np.percentile(qj[m] / qref[m], 75))],
                                               qref_median=float(np.median(qref[m])), qjax_median=float(np.median(qj[m])))
    r['unmatched_ref_pixels_qref_median'] = float(np.median(qref[~sel])) if (~sel).any() else None
    res[tag] = r; print(tag, json.dumps(r, indent=0), flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
