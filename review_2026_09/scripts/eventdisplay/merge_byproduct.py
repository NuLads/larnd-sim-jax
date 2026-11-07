"""Merge byproduct.py outputs run on disjoint sample subsets (parallel parts) into one byproduct_arrays.npz.
T_* arrays are concatenated in part order; S_trk (index into the T arrays) is offset by the tracks of earlier parts.
usage: python merge_byproduct.py <out_dir> <part_dir> [<part_dir> ...]"""
import sys, os, numpy as np
out, parts = sys.argv[1], sys.argv[2:]
Z = [np.load(f'{p}/byproduct_arrays.npz') for p in parts]
M, off = {}, 0
for z in Z:
    nt = len(z['T_L'])
    for k in z.files:
        v = z[k] + off if k == 'S_trk' else z[k]
        M.setdefault(k, []).append(v)
    off += nt
os.makedirs(out, exist_ok=True)
np.savez(f'{out}/byproduct_arrays.npz', **{k: np.concatenate(v) for k, v in M.items()})
print('merged', len(parts), 'parts:', off, 'tracks,', sum(len(z['S_dx']) for z in Z), 'segments, samples', sorted(set(np.concatenate(M['T_sample']).tolist())))
