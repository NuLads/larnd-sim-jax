#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import os
import sys
import argparse
import traceback

# Enforce CPU if --gpu is not specified
if '--gpu' not in sys.argv:
    os.environ['JAX_PLATFORMS'] = 'cpu'

from larndsim.consts_jax import (
    build_params_class,
    load_detector_properties,
    load_lut,
    apply_diffusion_link,
)
from larndsim.sim_jax import simulate_stochastic, simulate_parametrized, simulate_probabilistic, simulate_wfs
from larndsim.losses_jax import get_hits_space_coords
from larndsim.detsim_jax import validate_event_ids_for_packing, validate_local_event_ids, id2pixel, get_hit_z
from pprint import pprint
import numpy as np
import h5py
import jax
from tqdm import tqdm
from numpy.lib import recfunctions as rfn
from larndsim.sim_jax import pad_size
from .dataio import TracksDataset
from .strategies import pad_to_closest_multiple
import jax.numpy as jnp
from larndsim.fee_jax import digitize
from larndsim.losses_jax import adc2charge

# from ctypes import cdll
# libcudart = cdll.LoadLibrary('libcudart.so')


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# jax.config.update('jax_log_compiles', True)


def _write_batch_h5(path, ibatch, event_arr, local_to_global,
                    datasets, unmasked_datasets=None):
    """Split arrays by event_id and write one h5 group per event under batch_<ibatch>.

    `datasets` values are indexed by (event_arr == local_id); `unmasked_datasets`
    values are written verbatim into every event group (used for the non-prob wfs blob).
    """
    with h5py.File(path, 'a') as f:
        batch_group = f.create_group(f"batch_{ibatch}")
        for local_event_id in np.unique(event_arr).astype(int):
            if local_event_id < 0:  # skip padding
                continue
            global_event_id = local_to_global.get(local_event_id, local_event_id)
            m = (event_arr == local_event_id)
            event_group = batch_group.create_group(f"event_{global_event_id}")
            for name, arr in datasets.items():
                event_group.create_dataset(name, data=np.asarray(arr)[m])
            event_group.create_dataset(
                'eventID',
                data=np.full(int(m.sum()), global_event_id, dtype=np.int64),
            )
            if unmasked_datasets:
                for name, arr in unmasked_datasets.items():
                    event_group.create_dataset(name, data=np.asarray(arr))


def main(config):
    output_filename = config.output_file
    if not config.out_np:
        if not output_filename.endswith('.h5'):
            output_filename += '.h5'

    if os.path.isfile(output_filename):
        os.remove(output_filename)
    if config.lut_file == "" and config.mode == 'lut':
        return 1, 'Error: LUT file is required for mode "lut"'
    if config.probabilistic_sim and config.mode != 'lut':
        return 1, 'Error: --probabilistic_sim is only supported for mode "lut"'
    if config.probabilistic_sim and config.jac:
        return 1, 'Error: --probabilistic_sim is not compatible with --jac'
    if config.probabilistic_sim and config.out_np:
        return 1, 'Error: --probabilistic_sim is not compatible with --out_np (distribution shape differs)'

    if not config.gpu:
        jax.config.update('jax_platform_name', 'cpu')

    pars = []
    if config.jac:
        def sim_wrapper(params, tracks):
            adcs, pixel_x, pixel_y, pixel_z, ticks, event, unique_pixels, pix_renumbering, electrons, _ = simulate_parametrized(params, tracks, fields, rngseed=config.seed)
            return jnp.stack([adcs, ticks], axis=-1)
        pars = ['Ab', 'kb', 'eField', 'long_diff', 'tran_diff', 'lifetime', 'shift_z']
    Params = build_params_class(pars)
    ref_params = load_detector_properties(Params, config.detector_props, config.pixel_layouts)

    if config.mode == 'lut':
        response, ref_params = load_lut(config.lut_file, ref_params)

    params_to_apply = [
        'diffusion_in_current_sim',
        'mc_diff',
        'electron_sampling_resolution',
        'number_pix_neighbors',
        'signal_length',
        'use_dedx_density',
        'dedx_density_mode',
    ]


    ref_params = ref_params.replace(**{k: getattr(config, k) for k in params_to_apply}, time_window=config.signal_length)

    if config.link_diffusion:
        ref_params = apply_diffusion_link(ref_params, anchor='long_diff')
        logger.info(
            "Diffusion link enabled: long_diff=%s, tran_diff=%s",
            float(ref_params.long_diff),
            float(ref_params.tran_diff),
        )

    if not config.noise:
        ref_params = ref_params.replace(RESET_NOISE_CHARGE=0, UNCORRELATED_NOISE_CHARGE=0)

    dataset = TracksDataset(
        filename=config.input_file,
        nevents=config.n_events,
        max_nbatch=None,
        swap_xz=True,
        random_nevents=False,
        data_seed=config.seed if config.seed is not None else 42,
        max_batch_len=config.max_batch_len,
        print_input=False,
        chopped=config.chop,
        pad=False,
        electron_sampling_resolution=config.electron_sampling_resolution,
        live_selection=False,
        use_dedx_density=config.use_dedx_density,
        dedx_density_mode=config.dedx_density_mode,
    )
    fields = dataset.get_track_fields()
    evt_col = fields.index("eventID")

    if config.out_np:
        l_adc, l_Q, l_ticks, l_eventID, l_pix_x, l_pix_y, l_pix_z, l_hit_prob = [], [], [], [], [], [], [], []

    # libcudart.cudaProfilerStart()
    for ibatch in tqdm(range(len(dataset)), desc="Loading tracks", total=len(dataset)):
        batch = dataset[ibatch]
        size = pad_size(batch.shape[0], "batch_size", 0.5)
        batch = dataset.pad_batch(batch, size, ibatch)

        global_event_ids = dataset.get_batch_global_event_ids(ibatch)

        event_ids = batch[:, evt_col].astype(np.int64)

        # Validate local event ID namespace before overflow checks
        validate_local_event_ids(event_ids, context=f"simulate batch {ibatch}")
        validate_event_ids_for_packing(ref_params, event_ids, kind="pixel", context=f"simulate batch {ibatch}")
        validate_event_ids_for_packing(ref_params, event_ids, kind="bin", context=f"simulate batch {ibatch}")

        # Get mapping from local event IDs back to global IDs
        local_to_global = {i: int(gid) for i, gid in enumerate(global_event_ids)}

        tracks = jax.device_put(batch)
        rngseed = ibatch if config.seed is None else config.seed
        if config.mode == 'lut':
            wfs, unique_pixels = simulate_wfs(ref_params, response, tracks, fields)
            if config.probabilistic_sim:
                unique_pixels = pad_to_closest_multiple(unique_pixels, multiple=128, pad_value=-1, pad_front=True)
                wfs = pad_to_closest_multiple(wfs, dims_to_pad=(0,), multiple=128, pad_value=0.0, pad_front=True)
                adcs_distrib, pix_x_prob, pix_y_prob, ticks_prob, event_prob = simulate_probabilistic(ref_params, wfs, unique_pixels)
                _, _, pixel_plane_prob, _ = id2pixel(ref_params, unique_pixels)
            else:
                adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event, hit_pixels = simulate_stochastic(ref_params, wfs, unique_pixels, rngseed=rngseed)
        else:

            adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event, hit_pixels = simulate_parametrized(ref_params, tracks, fields, rngseed=rngseed)
            wfs = None
        if config.jac:
            jac_res = jax.jacfwd(sim_wrapper)(ref_params, tracks)

        if config.probabilistic_sim:
            ds = {
                'adcs_distrib': adcs_distrib,
                'ticks_prob':   ticks_prob,
                'pix_x':        pix_x_prob,
                'pix_y':        pix_y_prob,
                'pixel_plane':  pixel_plane_prob,
                'pixels':       unique_pixels,
            }
            if config.save_wfs:
                ds['wfs'] = wfs
            _write_batch_h5(output_filename, ibatch,
                            np.asarray(event_prob), local_to_global, ds)
            continue

        adc_lowest = digitize(ref_params, ref_params.DISCRIMINATION_THRESHOLD)
        adcs_clean = adcs - adc_lowest
        mask = (adcs_clean.flatten() != 0) & (event.flatten() != -1)
        Q = adc2charge(adcs.flatten()[mask], ref_params)

        if not config.out_np:
            event_flat = event.flatten()[mask]
            ds = {
                'adc_clean': adcs_clean.flatten()[mask],
                'adc':       adcs.flatten()[mask],
                'Q':         Q,
                'pixels':    hit_pixels[mask],
                'ticks':     ticks.flatten()[mask],
                'pix_x':     pixel_x[mask],
                'pix_y':     pixel_y[mask],
                'pix_z':     pixel_z.flatten()[mask],
            }
            if config.jac:
                for par in pars:
                    jac_par = getattr(jac_res, par)
                    ds[f'jac_{par}_adc']   = jac_par[:, :, 0].flatten()[mask]
                    ds[f'jac_{par}_ticks'] = jac_par[:, :, 1].flatten()[mask]
            unmasked = {'wfs': wfs} if config.save_wfs else None
            _write_batch_h5(output_filename, ibatch, event_flat,
                            local_to_global, ds, unmasked_datasets=unmasked)


        else:
            l_adc.append(adcs.flatten()[mask])
            l_Q.append(Q)
            l_ticks.append(ticks.flatten()[mask])
            l_eventID.append(event.flatten()[mask])
            l_pix_x.append(pixel_x.flatten()[mask])
            l_pix_y.append(pixel_y.flatten()[mask])
            l_pix_z.append(pixel_z.flatten()[mask])
            l_hit_prob.append(hit_prob.flatten()[mask])

    if config.out_np:
        jnp.savez(config.output_file, adcs=np.concatenate(l_adc), Q=np.concatenate(l_Q), x=np.concatenate(l_pix_x), y=np.concatenate(l_pix_y), z=np.concatenate(l_pix_z), ticks=np.concatenate(l_ticks), hit_prob=np.concatenate(l_hit_prob), event_id=np.concatenate(l_eventID))

    # libcudart.cudaProfilerStop()
    return 0, 'Success'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", dest="input_file",
                        default="/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5",
                        help="Input data file")
    parser.add_argument("--output_file", dest="output_file",
                        help="Output data file", required=True)
    parser.add_argument("--detector_props", dest="detector_props",
                        default="src/larndsim/detector_properties/module0.yaml",
                        help="Path to detector properties YAML file")
    parser.add_argument("--pixel_layouts", dest="pixel_layouts",
                        default="src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml",
                        help="Path to pixel layouts YAML file")
    parser.add_argument('--mode', type=str, help='Mode used to simulate the induced current on the pixels', choices=['lut', 'parametrized'], default='lut')
    parser.add_argument('--electron_sampling_resolution', type=float, required=True, default=0.1, help='Electron sampling resolution')
    parser.add_argument('--number_pix_neighbors', type=int, required=True, help='Number of pixel neighbors')
    parser.add_argument('--signal_length', type=int, required=True, help='Signal length')
    parser.add_argument('--lut_file', type=str, required=False, default="", help='Path to the LUT file')
    parser.add_argument('--noise', action='store_true', help='Add noise to the simulation')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--diffusion_in_current_sim', action='store_true', help='Use diffusion in current simulation')
    parser.add_argument('--batch_size', type=float, default=500, help='Batch size for simulation')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for simulation')
    parser.add_argument('--jac', action='store_true', help='Compute jacobian')
    parser.add_argument('--mc_diff', action='store_true', help='Use Monte Carlo diffusion')
    parser.add_argument('--use_dedx_density', action='store_true', default=False,
                        help='Use dE/dx density propagation in quenching (default: disabled).')
    parser.add_argument('--dedx_density_mode', type=str, default='histogram', choices=['histogram', 'flow'],
                        help='dE/dx density model when --use_dedx_density is enabled.')
    parser.add_argument('--save_wfs', action='store_true', help='Save waveforms')
    parser.add_argument('--n_events', type=int, default=-1, help='Number of events to be simulated')
    parser.add_argument('--out_np', action='store_true', default=False, help='store target-like output in npz')
    parser.add_argument('--max_batch_len', type=float, default=50., help='Maximum trajectory length budget used while preparing tracks')
    parser.add_argument('--chop', action='store_true', default=False, help='Enable segment chopping in data loading (default: disabled)')
    parser.add_argument('--probabilistic_sim', '--probabilistic-sim', default=False, action='store_true',
                        help='Use probabilistic sim: output full ADC/tick distribution per pixel (LUT mode only).')
    parser.add_argument('--link_diffusion', action='store_true', default=False,
                        help='Link long_diff and tran_diff using mobility/field transport relation (default: off).')

    try:
        args = parser.parse_args()
        if args.save_wfs and args.jac:
            raise ValueError("Cannot save waveforms and compute jacobian at the same time. Please choose one of the two options.")

        retval, status_message = main(args)
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = 'Error: Fitting failed.'

    logger.info(status_message)
    exit(retval)
