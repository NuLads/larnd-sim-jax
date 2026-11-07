"""GNREAL post hoc (AFTER the readout; descriptive): per-sample widths of the UNBOUNDED GN-C full bread at ridges 0 / 1e-3 /
1e-2, and how much of each sample's bounded-vs-unbounded difference comes from the rho > 1 (Delta_b > F_cc,b) batches."""
import sys, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/GNREAL')
from read_gr import load, sample, breads, width, J_of, fmt, LAMS, SAMPLES, clipped
for s in SAMPLES:
    recs, truth = load(f'GR_EXP{s}'); o = sample(recs, truth); J = J_of(o['G'])
    W = {l: width(breads(o, 'C', l)['GN-C full (F - D)'], J, o['jac']) for l in LAMS}
    Fs, Ds, Hs, c = clipped(recs, 'C', '0'); wb = width(Fs - Ds, J, o['jac'])
    print(f's{s} unbounded w0 {fmt(W["0"])} | w1e-3/w0 {fmt(W["0.001"] / W["0"], "{:6.3f}")} | w1e-2/w0 {fmt(W["0.01"] / W["0"], "{:6.3f}")}')
    print(f'    bounded  w0 {fmt(wb)} | unbounded/bounded {fmt(W["0"] / wb, "{:6.3f}")}')
