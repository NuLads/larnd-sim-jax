"""Rerun the JAX side of the larnd-sim comparison (stage 2: diffusion on, noise off; batch_350cm_input.h5, 127 segments,
438 cm) with the exact neighbor induction (variant E of neigh_exact.py), to check whether the excess of integrated charge
on weakly lit pixels seen against the reference comes from the neighbor approximation. Runs LARNDSIM_CMP/run_jax_direct.py
unchanged, with larndsim.sim_jax.simulate_wfs replaced by the exact variant. usage: python run_geo_exact.py OUT.h5"""
import sys
CMP = '/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP'
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/src'); sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/NEIGH'); sys.path.insert(0, CMP)
import larndsim.sim_jax as sj
from neigh_exact import simulate_wfs_variant
sj.simulate_wfs = lambda p, r, t, f, unique_size=None: simulate_wfs_variant(p, r, t, f, 'E', chunk=512)
import run_jax_direct
sys.argv = ['run_jax_direct.py', '--stage', '2', '--input', CMP + '/batch_350cm_input.h5', '--output', sys.argv[1], '--max-batch-len', '1000']
run_jax_direct.main()
