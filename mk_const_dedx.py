"""Write a copy of the linear-guess segments with dEdx held CONSTANT at the file mean.
dE is set to dEdx*dx so the two stay consistent; n_electrons is recomputed by the
simulation (quenching_jax.py:266 overwrites it), so it is left untouched."""
import h5py, numpy as np, shutil, sys
SRC='/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5'
DST='/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments_constdedx.h5'
print('copying', SRC, '->', DST, flush=True)
shutil.copyfile(SRC, DST)
with h5py.File(DST,'r+') as f:
    s=f['segments']
    d=s['dEdx'][:]
    mean=float(d.mean())
    print(f'original dEdx: mean {mean:.6f} std {d.std():.6f} min {d.min():.4f} max {d.max():.4f}', flush=True)
    arr=s[:]
    arr['dEdx']=np.float32(mean)
    arr['dE']=(np.float32(mean)*arr['dx']).astype(arr['dE'].dtype)
    s[...]=arr
    print(f"written: dEdx const={mean:.6f}, dE=dEdx*dx (mean {arr['dE'].mean():.6f})", flush=True)
print('done')
