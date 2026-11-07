"""Round-5 readout (generalised read_r3.py: ARM:JOBID[:ITERS]; RSS slope from iteration RSS_FROM = 300, or 600 for
1500-iteration arms; also losses_iter[0:5] vs KN10S1 bitwise, the 1000-1500 median and the final cache fill). Round-3 readout (rule: REVIEW.md, 20 Oct, stated before any result). usage: python read_r3.py BENCHR3A:41429512 BENCHR3B:41429513
Per arm: median step_time over iterations 300-600 vs the KN10S ensemble over the same window; RSS slope over 300->600
(log timestamps mapped to iterations through the cumulative step_time) extrapolated to 1e4 iterations; iteration-0 loss
vs KN10S1; epoch-1 time vs KN10S1 and its payback; gen-2 GC share of epoch 1; caches with persistent misses at 300-600."""
import sys, re, glob, pickle, collections, numpy as np
FR = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
LOG = '/sdf/scratch/neutrino/pgranger/review_2026_09/SPEED/logs'
ld = lambda arm, it: pickle.load(open(glob.glob(f'{FR}/sci_full_{arm}/history_iter{it}_*.pkl')[0], 'rb'))
K = [np.ravel(ld(f'KN10S{s}', 10000)['step_time']).astype(float) for s in range(1, 17)]
ref = float(np.median([np.median(k[300:600]) for k in K])); K1 = ld('KN10S1', 10000)
ref_e1 = float(K[0][:100].sum())
print(f'KN10S ensemble median step 300-600: {ref:.2f} s (limit 0.7x = {0.7 * ref:.2f}); KN10S1 epoch 1 {ref_e1 / 60:.1f} min')
for a in sys.argv[1:]:
    arm, jid, *nit = a.split(':'); NIT = int(nit[0]) if nit else 600; H = ld(arm, NIT); RF = 600 if NIT > 600 else 300; st = np.ravel(H['step_time']).astype(float); L = np.ravel(H['losses_iter'])
    s = open(f'{LOG}/{arm}-{jid}.out', errors='ignore').read().replace('\r', '\n')
    m = np.median(st[300:600]); e1 = st[:100].sum()
    print(f'\n{arm}: median step 300-600 {m:.2f} s = {m / ref:.2f}x  (100-200 {np.median(st[100:200]):.2f}, 200-300 {np.median(st[200:300]):.2f}, '
          f'300-400 {np.median(st[300:400]):.2f}, 400-500 {np.median(st[400:500]):.2f}, 500-600 {np.median(st[500:600]):.2f})')
    LK = np.ravel(K1["losses_iter"]); n_eq = next((i for i in range(len(L)) if L[i] != LK[i]), len(L))
    print(f'  losses_iter equal to KN10S1 bitwise for the first {n_eq} iterations -> losses[0:5] {"PASS" if n_eq >= 5 else "FAIL"}')
    if NIT >= 1500: print(f'  median 600-1000 {np.median(st[600:1000]):.2f} s, 1000-1500 {np.median(st[1000:1500]):.2f} s = {np.median(st[1000:1500]) / ref:.2f}x')
    print(f'  (i) {"PASS" if m <= 0.7 * ref else "FAIL"}   iteration-0 loss {L[0]:.4f} vs KN10S1 {np.ravel(K1["losses_iter"])[0]:.4f} '
          f'(iii) {"PASS" if L[0] == np.ravel(K1["losses_iter"])[0] else "FAIL"}')
    # log reports are mapped to iterations through the last progress-bar line printed before each report
    seq = [(m_.start(), int(n), sum(int(v) * 60 ** i for i, v in enumerate(reversed(e.split(':')))))
           for m_, n, e in [(mm, mm.group(1), mm.group(2)) for mm in re.finditer(r'(\d+)/%d \[([\d:]+)<' % NIT, s)]]
    rep = [(mm.start(), int(mm.group(1)), float(mm.group(2))) for mm in re.finditer(r'\[LARND_JAX_LRU\] t=(\d+) rss_gb ([\d.]+)', s)]
    pts = []
    for pos, t, r in rep:
        prev = [n for p, n, e in seq if p < pos]
        if prev: pts.append((prev[-1], r))
    pts = np.array(pts)
    sel = pts[(pts[:, 0] >= RF)] if len(pts) else pts
    if len(sel) >= 2:
        slope = np.polyfit(sel[:, 0], sel[:, 1], 1)[0]; r600 = sel[-1, 1]; r1e4 = r600 + slope * (1e4 - sel[-1, 0])
        print(f'  RSS at ~{sel[-1, 0]:.0f} {r600:.1f} GB, slope {slope * 100:.3f} GB/100 it over {RF}-{NIT} -> {r1e4:.1f} GB at 1e4 '
              f'(ii) {"PASS" if r1e4 < 60 else "FAIL"}')
    gc_rows = [(mm.start(), [float(x) for x in mm.group(1).split('/')]) for mm in re.finditer(r'gc_s ([\d/]+) ', s)]
    gc_e1 = [g for pos, g in gc_rows if [n for p, n, e in seq if p < pos][-1:] and [n for p, n, e in seq if p < pos][-1] <= 100]
    if gc_e1:
        print(f'  epoch 1: {e1 / 60:.1f} min (KN10S1 {ref_e1 / 60:.1f}); GC by end of epoch 1 (s, gen0/1/2) {gc_e1[-1]}; '
              f'gen-2 share {gc_e1[-1][2] / e1 * 100:.0f}% -> {"GC" if gc_e1[-1][2] > 0.3 * e1 else "not GC (unexplained)"}')
    gc_end = gc_rows[-1][1] if gc_rows else None
    print(f'  GC at end (s, gen0/1/2) {gc_end}; total wall {st.sum() / 60:.1f} min')
    saving = (ref - m) * (1e4 - 600); extra = e1 - ref_e1
    print(f'  (iv) extra epoch-1 time {extra / 60:+.1f} min vs saving {saving / 3600:.1f} h over 1e4 -> {"PASS" if extra < saving else "FAIL"}')
    names = collections.Counter(); last = {}
    for mm in re.finditer(r'\[LARND_JAX_LRU\]   (\S+): size (\d+)/(\d+) \+hits (\d+) \+misses (\d+)', s):
        pos = mm.start(); prev = [n for p, n, e in seq if p < pos]
        if prev and prev[-1] >= RF: names[mm.group(1)] += int(mm.group(5)); last[mm.group(1)] = f'{mm.group(2)}/{mm.group(3)}'
    print(f'  caches with new misses at {RF}-{NIT} (top 6 per report, summed):')
    last_tot = re.findall(r'\[LARND_JAX_LRU\] t=\d+ .*?entries (\d+) at-capacity (\d+) hits (\d+) misses (\d+)', s)
    if last_tot: print(f'  final report: entries {last_tot[-1][0]}, at-capacity {last_tot[-1][1]} caches, hits {last_tot[-1][2]}, misses {last_tot[-1][3]}')
    for n, c in names.most_common(8): print(f'    {n}: +{c} misses, size {last[n]}')
