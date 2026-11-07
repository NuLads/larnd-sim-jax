"""Shared setup for the paper's short GPU jobs: production data selection and detector configuration
(identical to E_truthdiag/truthdiag.py, which was validated against the production fits)."""
import os, sys, time
REPO = '/sdf/group/neutrino/pgranger/larnd-sim-jax'
sys.path.insert(0, REPO + '/src'); sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
import jax, jax.numpy as jnp
from optimize.dataio import TracksDataset, TgtTracksDataset
from optimize.strategies import LUTProbabilisticSimulation, LUTSimulation, ProbabilisticLossStrategy, GenericLossStrategy
from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from larndsim.losses_jax import llhd_loss, mse_adc
from larndsim.sim_jax import simulate_wfs, simulate_stochastic

INPUT_SIM = '/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5'
INPUT_TGT = '/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5'
TRUTH = dict(Ab=0.83488135, eField=0.52151894, tran_diff=1.00276338e-05, long_diff=5.81418228e-06, lifetime=2406.44659703)
PARAMS = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']

def load(max_batch_len=400., max_nbatch=3, data_seed=1):
    ds_sim = TracksDataset(filename=INPUT_SIM, nevents=600, max_nbatch=max_nbatch, random_nevents=True,
                           data_seed=data_seed, track_len_sel=2., max_abs_costheta_sel=0.966, min_abs_segz_sel=15.,
                           track_z_bound=28., max_batch_len=max_batch_len, print_input=False, chopped=True, pad=True,
                           electron_sampling_resolution=0.01, live_selection=False,
                           use_dedx_density=False, dedx_density_mode='histogram')
    ds_tgt = TgtTracksDataset(filename=INPUT_TGT, dataset_sim=ds_sim, chopped=True, pad=True,
                              electron_sampling_resolution=0.01, print_input=False)
    return ds_sim, ds_tgt, ds_tgt.get_track_fields()

def params():
    Params = build_params_class(PARAMS)
    ref = load_detector_properties(Params, 'src/larndsim/detector_properties/module0.yaml',
                                   'src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml')
    ref = ref.replace(electron_sampling_resolution=0.01, number_pix_neighbors=4, signal_length=150, time_window=150)
    response, ref = load_lut('src/larndsim/detector_properties/response_44.npy', ref)
    ref = ref.replace(diffusion_in_current_sim=False, mc_diff=False, use_dedx_density=False, dedx_density_mode='histogram')
    return ref.replace(**TRUTH), response

def target_hits(truth, response, tracks, fields, seed):
    wfs, up = simulate_wfs(truth, response, tracks, fields)
    adcs, x, y, z, ticks, hp, ev, pix = simulate_stochastic(truth, wfs, up, rngseed=seed)
    return dict(adcs=adcs, pixel_x=x, pixel_y=y, pixel_z=z, ticks=ticks, hit_prob=hp, event=ev, pixel_id=pix)

def track_length_cm(tracks, fields):
    return float(np.asarray(tracks)[:, fields.index('dx')].sum())

def tick(x):
    jax.block_until_ready(x); return time.perf_counter()
