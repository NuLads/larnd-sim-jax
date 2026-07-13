#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import argparse
import yaml
import sys, os
import traceback
import json
import cProfile
import jax

from .fit_params import GradientDescentFitter, LikelihoodProfiler, MinuitFitter, HessianCalculator, GaussNewtonCalibFitter
from .dataio import TgtTracksDataset, TracksDataset, DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# jax.config.update('jax_log_compiles', True)

def make_param_list(config):
    if len(config.param_list) == 1 and os.path.splitext(config.param_list[0])[1] == ".yaml":
        with open(config.param_list[0], 'r') as config_file:
            config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
        for key in config_dict.keys():
            logger.info(f"Setting lr {config_dict[key]} for {key}")
        param_list = config_dict
    else:
        param_list = config.param_list
    return param_list


def main(config):
    if config.cpu_only:
        jax.config.update('jax_platform_name', 'cpu')
    else:
        jax.config.update('jax_platform_name', 'gpu')

    if config.debug_nans:
        jax.config.update("jax_debug_nans", True)
    else:
        jax.config.update("jax_debug_nans", False)

    if config.non_deterministic:
        os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=false'
    else:
        os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

    # Persist ALL compiled kernels to the on-disk cache. JAX's default only caches kernels
    # that take >1 s to compile, so the many fast (~40 ms) sim kernels are never persisted and
    # recompile on every cold start. Lowering the threshold makes the persistent cache complete,
    # so recompilation happens only on the very first run for a given shape set.
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)

    logger.info(f"Jax devices: {jax.devices()}")

    logger.info(f"fit label: {config.out_label}")

    if config.lut_file == "" and config.mode == 'lut':
        return 1, 'Error: LUT file is required for mode "lut"'

    iterations = config.iterations
    max_nbatch = config.max_nbatch

    if config.resume_from is not None and config.fit_type != "chain":
        raise ValueError("--resume_from is only supported for fit_type=chain")

    if iterations is not None and config.fit_type != "gn_calib":
        # For per-batch optimizers, loading more batches than iterations is wasted.
        # gn_calib is FULL-batch (every iteration consumes all batches), so the cap
        # must not apply — otherwise iterations silently shrinks the dataset.
        if max_nbatch is None or iterations < max_nbatch or max_nbatch <= 0:
            max_nbatch = iterations

    dataset_sim = TracksDataset(filename=config.input_file_sim, nevents=config.data_sz, max_nbatch=max_nbatch, random_nevents=config.random_nevents, data_seed=config.data_seed,
                            track_len_sel=config.track_len_sel, max_abs_costheta_sel=config.max_abs_costheta_sel, min_abs_segz_sel=config.min_abs_segz_sel, track_z_bound=config.track_z_bound, max_batch_len=config.max_batch_len, print_input=config.print_input, chopped=(not config.no_chop), pad=(not config.no_pad), electron_sampling_resolution=config.electron_sampling_resolution, live_selection=config.live_selection,
                            use_dedx_density=config.use_dedx_density, dedx_density_mode=config.dedx_density_mode)

    if ".np" in config.input_file_tgt:
        if not config.read_target:
            logger.warning("read_target is not activated but a ready target are provided. Changing read_target to TRUE")
            config.read_target = True
    elif ".h5" in config.input_file_tgt or ".hdf5" in config.input_file_tgt:
        if config.read_target:
            logger.warning("read_target is activated but target is provided as a simulation input. Changing read_target to FALSE")
            config.read_target = False
    if not config.read_target:
        # Get the same events for target

        dataset_target = TgtTracksDataset(filename=config.input_file_tgt, dataset_sim = dataset_sim, chopped=(not config.no_chop), pad=(not config.no_pad), electron_sampling_resolution=config.electron_sampling_resolution, print_input=config.print_input)

        # check if the track in sim and target are consistent
        if len(dataset_sim) != len(dataset_target):
            raise Exception("target and sim inputs are different in size.")

    batch_sz = config.batch_sz
    if config.max_batch_len is not None and batch_sz != 1:
        logger.warning("Need batch size == 1 for splitting in dx chunks. Setting now...")
        batch_sz = 1

    tracks_dataloader_sim = DataLoader(dataset_sim,
                                  batch_size=batch_sz)
    sim_track_fields = dataset_sim.get_track_fields()
    tgt_track_fields = dataset_sim.get_track_fields()

    if not config.read_target:
        tgt_track_fields = dataset_target.get_track_fields()
        tracks_dataloader_target = DataLoader(dataset_target,
                                      batch_size=batch_sz)

        # check if tracks_dataloader_sim and tracks_dataloader_target have the same size
        if len(tracks_dataloader_sim) != len(tracks_dataloader_target):
            raise Exception("target and sim inputs are different in size.")

    # For readout noise: no_noise overrides if explicitly set to True. Otherwise, turn on noise
    # individually for target and guess
    param_list = make_param_list(config)
    logger.info(f"Param list: {param_list}")

    if config.fit_type == "chain":
        param_fit = GradientDescentFitter(relevant_params=param_list,
                                sim_track_fields=sim_track_fields, tgt_track_fields=tgt_track_fields,
                                detector_props=config.detector_props, pixel_layouts=config.pixel_layouts,
                                lr=config.lr, readout_noise_target=(not config.no_noise) and (not config.no_noise_target),
                                readout_noise_guess=(not config.no_noise) and (not config.no_noise_guess),
                                out_label=config.out_label, test_name=config.test_name,
                                max_clip_norm_val=config.max_clip_norm_val, clip_from_range=config.clip_from_range,
                                optimizer_fn=config.optimizer_fn,
                                lr_scheduler=config.lr_scheduler, lr_kw=config.lr_kw,
                                loss_fn=config.loss_fn, loss_fn_kw=config.loss_fn_kw, shift_no_fit=config.shift_no_fit,
                                set_target_vals=config.set_target_vals, set_init_params=config.set_init_params, vary_init=config.vary_init, compute_target_hessian=config.compute_target_hessian,
                                config = config, epoch_size=len(tracks_dataloader_sim), keep_in_memory=config.keep_in_memory,
                                diffusion_in_current_sim=config.diffusion_in_current_sim,
                                mc_diff=config.mc_diff,
                                adc_norm=config.chamfer_adc_norm, match_z=config.chamfer_match_z,
                                sim_seed_strategy=config.sim_seed_strategy, target_seed=config.seed, target_fixed_range = config.fixed_range, read_target=config.read_target,
                                probabilistic_sim=config.probabilistic_sim,
                                probabilistic_sampling_sim=config.probabilistic_sampling_sim,
                                probabilistic_sampling_target=config.probabilistic_sampling_target,
                                normalization_scheme=config.normalization_scheme,
                                normalization_scale_sigmoid=config.normalization_scale_sigmoid,
                                normalization_scale_exp_log=config.normalization_scale_exp_log,
                                sz_mini_bt=config.sz_mini_bt, shuffle_bt=config.shuffle_bt, shuffle_seed=config.shuffle_seed,
                                resume_from=config.resume_from,
                                fit_dedx=config.fit_dedx, dedx_prior_weight=config.dedx_prior_weight, dedx_lr=config.dedx_lr,
                                dedx_start_iter=config.dedx_start_iter, dedx_freeze_iter=config.dedx_freeze_iter,
                                dedx_student_loc=config.dedx_student_loc,
                                dedx_soft_barrier_threshold=config.dedx_soft_barrier_threshold,
                                dedx_soft_barrier_weight=config.dedx_soft_barrier_weight,
                                dedx_use_split_t=config.dedx_use_split_t,
                                dedx_student_nu_l=config.dedx_student_nu_l,
                                dedx_student_nu_r=config.dedx_student_nu_r,
                                dedx_student_scale_l=config.dedx_student_scale_l,
                                dedx_student_scale_r=config.dedx_student_scale_r,
                                dedx_mean_constraint_weight=config.dedx_mean_constraint_weight,
                                dedx_mean_constraint_target=config.dedx_mean_constraint_target,
                                dedx_drift_profile_weight=config.dedx_drift_profile_weight,
                                fit_track_positions=config.fit_track_positions,
                                track_max_iter=config.track_max_iter,
                                track_alignment_freq=config.track_alignment_freq,
                                fit_chain_positions=config.fit_chain_positions,
                                chain_lr=config.chain_lr,
                                chain_start_iter=config.chain_start_iter,
                                chain_update_freq=config.chain_update_freq,
                                chain_decay_rate=config.chain_decay_rate,
                                chain_decay_start=config.chain_decay_start,
                                pos_residual_freq=config.pos_residual_freq,
                                mcs_prior_weight=config.mcs_prior_weight,
                                chain_step_len=config.chain_step_len,
                                chain_momentum_GeV=config.chain_momentum_GeV,
                                chain_basis=config.chain_basis, chain_spline_knot_cm=config.chain_spline_knot_cm)
    elif config.fit_type == "gn_calib":
        param_fit = GaussNewtonCalibFitter(relevant_params=param_list,
                                sim_track_fields=sim_track_fields, tgt_track_fields=tgt_track_fields,
                                detector_props=config.detector_props, pixel_layouts=config.pixel_layouts,
                                lr=config.lr, readout_noise_target=(not config.no_noise) and (not config.no_noise_target),
                                readout_noise_guess=(not config.no_noise) and (not config.no_noise_guess),
                                out_label=config.out_label, test_name=config.test_name,
                                max_clip_norm_val=config.max_clip_norm_val, clip_from_range=config.clip_from_range,
                                optimizer_fn=config.optimizer_fn,
                                lr_scheduler=config.lr_scheduler, lr_kw=config.lr_kw,
                                loss_fn=config.loss_fn, loss_fn_kw=config.loss_fn_kw, shift_no_fit=config.shift_no_fit,
                                set_target_vals=config.set_target_vals, set_init_params=config.set_init_params, vary_init=config.vary_init,
                                config = config, epoch_size=len(tracks_dataloader_sim), keep_in_memory=config.keep_in_memory,
                                diffusion_in_current_sim=config.diffusion_in_current_sim,
                                mc_diff=config.mc_diff,
                                adc_norm=config.chamfer_adc_norm, match_z=config.chamfer_match_z,
                                sim_seed_strategy=config.sim_seed_strategy, target_seed=config.seed, target_fixed_range = config.fixed_range, read_target=config.read_target,
                                probabilistic_sim=config.probabilistic_sim,
                                probabilistic_sampling_sim=config.probabilistic_sampling_sim,
                                probabilistic_sampling_target=config.probabilistic_sampling_target,
                                normalization_scheme=config.normalization_scheme,
                                normalization_scale_sigmoid=config.normalization_scale_sigmoid,
                                normalization_scale_exp_log=config.normalization_scale_exp_log,
                                gn_max_iter=(config.iterations if config.iterations is not None else 40),
                                gn_curvature=config.gn_curvature,
                                gn_validate=config.gn_validate,
                                gn_degeneracy=config.gn_degeneracy,
                                gn_warmup_iters=config.gn_warmup_iters,
                                gn_resume_joint=config.gn_resume_joint,
                                fit_dedx=config.fit_dedx, dedx_prior_weight=config.dedx_prior_weight, dedx_lr=config.dedx_lr,
                                dedx_start_iter=config.dedx_start_iter, dedx_freeze_iter=config.dedx_freeze_iter,
                                dedx_student_loc=config.dedx_student_loc,
                                dedx_soft_barrier_threshold=config.dedx_soft_barrier_threshold,
                                dedx_soft_barrier_weight=config.dedx_soft_barrier_weight,
                                dedx_use_split_t=config.dedx_use_split_t,
                                dedx_student_nu_l=config.dedx_student_nu_l,
                                dedx_student_nu_r=config.dedx_student_nu_r,
                                dedx_student_scale_l=config.dedx_student_scale_l,
                                dedx_student_scale_r=config.dedx_student_scale_r,
                                dedx_mean_constraint_weight=config.dedx_mean_constraint_weight,
                                dedx_mean_constraint_target=config.dedx_mean_constraint_target,
                                dedx_drift_profile_weight=config.dedx_drift_profile_weight,
                                fit_chain_positions=config.fit_chain_positions,
                                chain_lr=config.chain_lr,
                                chain_start_iter=config.chain_start_iter,
                                chain_update_freq=config.chain_update_freq,
                                chain_decay_rate=config.chain_decay_rate,
                                chain_decay_start=config.chain_decay_start,
                                mcs_prior_weight=config.mcs_prior_weight,
                                chain_step_len=config.chain_step_len,
                                chain_momentum_GeV=config.chain_momentum_GeV,
                                chain_basis=config.chain_basis, chain_spline_knot_cm=config.chain_spline_knot_cm)

    elif config.fit_type == "scan":
        param_fit = LikelihoodProfiler(relevant_params=param_list,
                                sim_track_fields=sim_track_fields, tgt_track_fields=tgt_track_fields,
                                detector_props=config.detector_props, pixel_layouts=config.pixel_layouts,
                                readout_noise_target=(not config.no_noise) and (not config.no_noise_target),
                                readout_noise_guess=(not config.no_noise) and (not config.no_noise_guess),
                                out_label=config.out_label, test_name=config.test_name,
                                loss_fn=config.loss_fn, loss_fn_kw=config.loss_fn_kw, shift_no_fit=config.shift_no_fit,
                                set_target_vals=config.set_target_vals, set_init_params=config.set_init_params, vary_init=config.vary_init,
                                config = config, keep_in_memory=config.keep_in_memory,
                                diffusion_in_current_sim=config.diffusion_in_current_sim,
                                mc_diff=config.mc_diff,
                                adc_norm=config.chamfer_adc_norm, match_z=config.chamfer_match_z,
                                sim_seed_strategy=config.sim_seed_strategy, target_seed=config.seed, target_fixed_range = config.fixed_range, read_target=config.read_target,
                                scan_tgt_nom=config.scan_tgt_nom, probabilistic_sim=config.probabilistic_sim,
                                probabilistic_sampling_sim=config.probabilistic_sampling_sim,
                                probabilistic_sampling_target=config.probabilistic_sampling_target,
                                normalization_scheme=config.normalization_scheme,
                                normalization_scale_sigmoid=config.normalization_scale_sigmoid,
                                normalization_scale_exp_log=config.normalization_scale_exp_log)
    elif config.fit_type == "minuit":
        param_fit = MinuitFitter(relevant_params=param_list,
                                sim_track_fields=sim_track_fields, tgt_track_fields=tgt_track_fields,
                                detector_props=config.detector_props, pixel_layouts=config.pixel_layouts,
                                readout_noise_target=(not config.no_noise) and (not config.no_noise_target),
                                readout_noise_guess=(not config.no_noise) and (not config.no_noise_guess),
                                out_label=config.out_label, test_name=config.test_name,
                                loss_fn=config.loss_fn, loss_fn_kw=config.loss_fn_kw, shift_no_fit=config.shift_no_fit,
                                set_target_vals=config.set_target_vals, set_init_params=config.set_init_params, vary_init=config.vary_init,
                                config = config, keep_in_memory=config.keep_in_memory,
                                diffusion_in_current_sim=config.diffusion_in_current_sim,
                                mc_diff=config.mc_diff,
                                adc_norm=config.chamfer_adc_norm, match_z=config.chamfer_match_z,
                                sim_seed_strategy=config.sim_seed_strategy, target_seed=config.seed, target_fixed_range = config.fixed_range, read_target=config.read_target,
                                minimizer_strategy=config.minimizer_strategy, minimizer_tol=config.minimizer_tol, separate_fits=config.separate_fits, probabilistic_sim=config.probabilistic_sim,
                                probabilistic_sampling_sim=config.probabilistic_sampling_sim,
                                probabilistic_sampling_target=config.probabilistic_sampling_target,
                                normalization_scheme=config.normalization_scheme,
                                normalization_scale_sigmoid=config.normalization_scale_sigmoid,
                                normalization_scale_exp_log=config.normalization_scale_exp_log)

    elif config.fit_type == "hess":
        param_fit = HessianCalculator(relevant_params=param_list, set_init_params=config.set_init_params,
                                sim_track_fields=sim_track_fields, tgt_track_fields=tgt_track_fields,
                                detector_props=config.detector_props, pixel_layouts=config.pixel_layouts,
                                readout_noise_target=(not config.no_noise) and (not config.no_noise_target),
                                readout_noise_guess=(not config.no_noise) and (not config.no_noise_guess),
                                out_label=config.out_label, test_name=config.test_name,
                                loss_fn=config.loss_fn, loss_fn_kw=config.loss_fn_kw, shift_no_fit=config.shift_no_fit,
                                set_target_vals=config.set_target_vals, vary_init=config.vary_init,
                                config = config, keep_in_memory=config.keep_in_memory,
                                diffusion_in_current_sim=config.diffusion_in_current_sim,
                                mc_diff=config.mc_diff,
                                adc_norm=config.chamfer_adc_norm, match_z=config.chamfer_match_z,
                                sim_seed_strategy=config.sim_seed_strategy, target_seed=config.seed, target_fixed_range = config.fixed_range, read_target=config.read_target,
                                probabilistic_sim=config.probabilistic_sim,
                                probabilistic_sampling_sim=config.probabilistic_sampling_sim,
                                probabilistic_sampling_target=config.probabilistic_sampling_target,
                                normalization_scheme=config.normalization_scheme,
                                normalization_scale_sigmoid=config.normalization_scale_sigmoid,
                                normalization_scale_exp_log=config.normalization_scale_exp_log,
                                compute_target_hessian=True)

    else:
        raise Exception(f"Unknown fit type: {config.fit_type}. Supported types are 'chain', 'scan', 'minuit', and 'hess'.")

    # with cProfile.Profile() as pr:

    trace_started = False
    if config.profile:
        profile_dir = "./profiling/"
        logger.info(f"Profiling execution. Output in {profile_dir}")
        profile_options_cls = getattr(jax.profiler, "ProfileOptions", None)
        try:
            if profile_options_cls is not None:
                options = profile_options_cls()
                options.host_tracer_level = 2
                jax.profiler.start_trace(profile_dir, profiler_options=options)
            else:
                logger.warning("jax.profiler.ProfileOptions not available; using default start_trace settings")
                jax.profiler.start_trace(profile_dir)
            trace_started = True
        except TypeError:
            # Older/newer JAX variants may accept different start_trace kwargs.
            logger.warning("Falling back to jax.profiler.start_trace(profile_dir) for this JAX version")
            jax.profiler.start_trace(profile_dir)
            trace_started = True
        jax.profiler.save_device_memory_profile("memory.prof")

    if getattr(config, 'benchmark_chain_opt', -1) is not None and config.benchmark_chain_opt >= 0:
        _tgt = config.input_file_tgt if config.read_target else tracks_dataloader_target
        param_fit.benchmark_chain_optimizers(tracks_dataloader_sim, _tgt,
                                             batch_idx=config.benchmark_chain_opt,
                                             n_steps=(iterations or 400))
        return 0, 'chain-optimizer benchmark done'

    if getattr(config, 'lbfgs_position_fit', False):
        _tgt = config.input_file_tgt if config.read_target else tracks_dataloader_target
        param_fit.run_lbfgs_position_fit(tracks_dataloader_sim, _tgt,
                                         max_steps=(iterations or 80))
        return 0, 'lbfgs position fit done'

    if getattr(config, 'profile_geom_compile', -1) is not None and config.profile_geom_compile >= 0:
        _tgt = config.input_file_tgt if config.read_target else tracks_dataloader_target
        param_fit.profile_geom_compile(tracks_dataloader_sim, _tgt, batch_idx=config.profile_geom_compile)
        return 0, 'compile profile done'

    if getattr(config, 'test_jitted_geom', -1) is not None and config.test_jitted_geom >= 0:
        _tgt = config.input_file_tgt if config.read_target else tracks_dataloader_target
        param_fit.test_jitted_geom_lbfgs(tracks_dataloader_sim, _tgt, batch_idx=config.test_jitted_geom, max_iter=(iterations or 60))
        return 0, 'jitted geom lbfgs done'

    if getattr(config, 'validate_static_unique', -1) is not None and config.validate_static_unique >= 0:
        param_fit.validate_static_unique(tracks_dataloader_sim, batch_idx=config.validate_static_unique)
        return 0, 'static-unique validation done'

    if getattr(config, 'test_chain_lbfgs', -1) is not None and config.test_chain_lbfgs >= 0:
        _tgt = config.input_file_tgt if config.read_target else tracks_dataloader_target
        param_fit.test_chain_geometry_lbfgs(tracks_dataloader_sim, _tgt,
                                            batch_idx=config.test_chain_lbfgs, max_iter=(iterations or 60))
        return 0, 'chain-geometry L-BFGS test done'

    if config.read_target:
        param_fit.fit(tracks_dataloader_sim, config.input_file_tgt, epochs=config.epochs, iterations=iterations, save_freq=config.save_freq)
    else:
        param_fit.fit(tracks_dataloader_sim, tracks_dataloader_target, epochs=config.epochs, iterations=iterations, save_freq=config.save_freq)

    
    if config.profile and trace_started:
        logger.info("Ending profiling")
        jax.profiler.stop_trace()

    # pr.dump_stats('prof.prof')
    # jax.profiler.stop_trace()
    return 0, 'Fitting successful'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", dest="param_list", default=[], nargs="+", required=True,
                        help="List of parameters to optimize. See consts_ep.py")
    parser.add_argument("--input_file_sim", dest="input_file_sim",
                        default="/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5",
                        help="Input sim data file")
    parser.add_argument("--input_file_tgt", dest="input_file_tgt",
                        default="/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5",
                        help="Input target data file")
    parser.add_argument("--detector_props", dest="detector_props",
                        default="src/larndsim/detector_properties/module0.yaml",
                        help="Path to detector properties YAML file")
    parser.add_argument("--pixel_layouts", dest="pixel_layouts",
                        default="src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml",
                        help="Path to pixel layouts YAML file")
    parser.add_argument("--lr", dest="lr", default=1, type=float,
                        help="Learning rate -- used for all params")
    parser.add_argument("--batch_sz", dest="batch_sz", default=1, type=int,
                        help="Batch size for fitting (tracks).")
    parser.add_argument("--epochs", dest="epochs", default=100, type=int,
                        help="Number of epochs")
    parser.add_argument("--seed", dest="seed", default=2, type=int,
                        help="Random seed for target construction")
    parser.add_argument("--data_seed", dest="data_seed", default=3, type=int,
                        help="Random seed for data picking if not using the whole set")
    parser.add_argument("--shuffle_seed", dest="shuffle_seed", default=42, type=int,
                        help="Random seed for shuffling the batch order within an epoch")
    parser.add_argument("--vary-init", dest="vary_init", default=False, action="store_true",
                        help="Randomly sample initial guess (vs starting at nominal value)")
    parser.add_argument("--data_sz", dest="data_sz", default=None, type=int,
                        help="Data size for fitting (number of tracks); input negative values to run on the whole dataset")
    parser.add_argument("--no-noise", dest="no_noise", default=False, action="store_true",
                        help="Flag to turn off readout noise (both target and guess)")
    parser.add_argument("--no-noise-target", dest="no_noise_target", default=False, action="store_true",
                        help="Flag to turn off readout noise (just target, guess has noise)")
    parser.add_argument("--no-noise-guess", dest="no_noise_guess", default=False, action="store_true",
                        help="Flag to turn off readout noise (just guess, target has noise)")
    parser.add_argument("--save_freq", dest="save_freq", default=10, type=int,
                        help="Save frequency of the result")
    parser.add_argument("--random_nevents", dest="random_nevents", default=False, action="store_true",
                        help="Flag of whether sampling the events randomly or sequentially")
    parser.add_argument("--track_len_sel", dest="track_len_sel", default=2., type=float,
                        help="Track selection requirement on track length.")
    parser.add_argument("--max_abs_costheta_sel", dest="max_abs_costheta_sel", default=0.966, type=float,
                        help="Theta is the angle of track wrt to the z axis. Remove tracks which are very colinear with z.")
    parser.add_argument("--min_abs_segz_sel", dest="min_abs_segz_sel", default=15., type=float,
                        help="Remove track segments that are close to the cathode.")
    parser.add_argument("--track_z_bound", dest="track_z_bound", default=28., type=float,
                        help="Set z bound to keep healthy set of tracks")
    parser.add_argument("--out_label", dest="out_label", default="",
                        help="Label for output pkl file")
    parser.add_argument("--test_name", dest="test_name", default="",
                        help="Name of the test")
    parser.add_argument("--fixed_range", dest="fixed_range", default=None, type=float,
                        help="Construct target by sampling in a certain range (fraction of nominal)")
    parser.add_argument("--max_clip_norm_val", dest="max_clip_norm_val", default=None, type=float,
                        help="If passed, does gradient clipping (norm)")
    parser.add_argument("--clip_from_range", dest="clip_from_range", default=False, action="store_true",
                        help="Flag to clip the fitted parameters at the nominal ranges")
    parser.add_argument("--optimizer_fn", dest="optimizer_fn", default="Adam",
                        help="Choose optimizer function (here Adam vs SGD")
    parser.add_argument("--lr_scheduler", dest="lr_scheduler", default=None,
                        help="Schedule learning rate, e.g. ExponentialLR")
    parser.add_argument("--lr_kw", dest="lr_kw", default=None, type=json.loads,
                        help="kwargs for learning rate scheduler, as string dict")
    parser.add_argument("--iterations", dest="iterations", default=None, type=int,
                        help="Number of iterations to run. Overrides epochs.")
    parser.add_argument("--loss_fn", dest="loss_fn", default=None,
                        help="Loss function to use. Named options are SDTW, space_match, distribution and markov_llhd.")
    parser.add_argument("--dist_feature", dest="dist_feature", default='charge',
                        choices=['charge', 'time', 'x', 'y', 'pix_counts'],
                        help="Feature to use for distribution loss.")
    parser.add_argument("--loss_fn_kw", dest="loss_fn_kw", default=None, type=json.loads,
                        help="Loss function keyword arguments.")
    parser.add_argument("--max_batch_len", dest="max_batch_len", default=None, type=float,
                        help="Max dx [cm] per batch. If passed, will add tracks to batch until overflow, splitting where needed")
    parser.add_argument("--max_nbatch", dest="max_nbatch", default=None, type=int,
                        help="Upper number of different batches taken from the data, given the max_batch_len. Overrides data_sz.")
    parser.add_argument("--print_input", dest="print_input", default=False, action="store_true",
                        help="print the event and track id per batch.")
    parser.add_argument("--shift_no_fit", dest="shift_no_fit", default=[], nargs="+",
                        help="Set of params to shift in target sim without fitting them (robustness/separability check).")
    parser.add_argument("--set_target_vals", dest="set_target_vals", default=[], nargs="+",
                        help="Explicitly set values of target. Syntax is <param1> <val1> <param2> <val2>...")
    parser.add_argument("--set_init_params", dest="set_init_params", default=[], nargs="+",
                        help="Init parameter values. Syntax is <param1> <val1> <param2> <val2>...")
    parser.add_argument("--scan_tgt_nom", dest="scan_tgt_nom", default=False, action="store_true",
                        help="Set the gradient and loss scan target to the parameter nominal value, otherwise there will be a target throw.")
    parser.add_argument('--mode', type=str, help='Mode used to simulate the induced current on the pixels', choices=['lut', 'parametrized', 'markov'], default='lut')
    parser.add_argument('--electron_sampling_resolution', type=float, required=True, help='Electron sampling resolution')
    parser.add_argument('--number_pix_neighbors', type=int, required=True, help='Number of pixel neighbors')
    parser.add_argument('--signal_length', type=int, required=True, help='Signal length')
    parser.add_argument('--lut_file', type=str, required=False, default="src/larndsim/detector_properties/response_44_v2a_full_tick.npz", help='Path to the LUT file')
    parser.add_argument('--keep_in_memory', default=False, action="store_true", help='Keep the expected output of each batch in memory')
    parser.add_argument('--compute_target_hessian', default=False, action="store_true", help='Computes the Hessian at the target for every batch')
    parser.add_argument('--non_deterministic', default=False, action="store_true", help='Make the computation slightly non-deterministic for faster computation')
    parser.add_argument('--debug_nans', default=False, action="store_true", help='Debug NaNs (much slower)')
    parser.add_argument('--cpu_only', default=False, action="store_true", help='Run on CPU only')
    parser.add_argument('--fit_type', type=str, choices=['chain', 'scan', 'minuit','hess','gn_calib'], required=True)
    parser.add_argument('--gn_curvature', type=str, choices=['exact', 'ggn'], default='exact',
                        help='Curvature model for gn_calib: exact Hessian (2nd-order) or Fisher/GGN (1st-order, PSD)')
    parser.add_argument('--gn_validate', default=False, action='store_true',
                        help='gn_calib: only compare exact Hessian vs Fisher/GGN on batch 0, then exit')
    parser.add_argument('--gn_degeneracy', default=False, action='store_true',
                        help='gn_calib: run the degeneracy-valley study (line scan, 2D grid, Hessian rotation), then exit')
    parser.add_argument('--gn_warmup_iters', default=0, type=int,
                        help='gn_calib hybrid mode: per-batch Adam warmup iterations before each GN takeover attempt; GN falls back to Adam when it stalls or saturates (0 = pure GN)')
    parser.add_argument('--dedx_drift_profile_weight', default=0.0, type=float,
                        help='Weight of the drift-profile penalty on per-segment dEdx: penalizes the WLS trend of log(dEdx) vs |z| (the dEdx<->lifetime degenerate mode). 0 = off')
    parser.add_argument('--chain_basis', type=str, choices=['angle','spline'], default='angle',
                        help='Track-shape parametrization for --fit_chain_positions: tangent angles (cumsum) or smooth transverse-displacement sine spline (length-independent conditioning)')
    parser.add_argument('--chain_spline_knot_cm', type=float, default=40.0,
                        help='Spline mode spacing: K = clip(round(total_len/this),3,12) sine modes per track')
    parser.add_argument('--gn_resume_joint', default=None, type=str,
                        help='gn_calib: joint-fit checkpoint pkl; freeze its fitted dEdx + chain angles and GN-polish only the 5 calibration params (conditional covariance reported)')
    parser.add_argument('--minimizer_strategy', type=int, choices=[0, 1, 2], default=1, help='Minimizer strategy for Minuit')
    parser.add_argument('--minimizer_tol', type=float, default=1e-4, help='Minimizer tolerance for Minuit')
    parser.add_argument('--separate_fits', default=False, action="store_true", help='Separate fits for each batch')
    parser.add_argument('--diffusion_in_current_sim', action='store_true', help='Use diffusion in current simulation')
    parser.add_argument('--sim_seed_strategy', default="different", type=str, choices=['same', 'different', 'different_epoch', 'random', 'constant'],
                        help='Strategy to choose the seed for the simulation (the seed for target is the batch id). It can be "same" (same for target and sim), "different" (different for target and sim but constant across epochs), "different_epoch" (different for target and sim, and in the simulation the key is different per epoch)"random" (different between target and sim and random across epochs), "constant" (the seed is constant across batches).')
    parser.add_argument('--chamfer_adc_norm', default=1., type=float, help='ADC normalisation wrt to position (cm)')
    parser.add_argument('--chamfer_match_z', default=False, action="store_true", help='match z (converted using the iterated simulation v_drift value for both the target and simulation) instead of t')
    parser.add_argument('--mc_diff', default=False, action="store_true", help='Use MC diffusion')
    parser.add_argument('--live_selection', default=False, action="store_true", help='Whether to run live selection or not')
    parser.add_argument('--read_target', default=False, action="store_true", help='read data(-like) target')
    parser.add_argument('--probabilistic_sim', '--probabilistic-sim', default=False, action="store_true", help='Use probabilistic sim')
    parser.add_argument('--probabilistic_sampling_sim', '--probabilistic-sampling-sim',
                        default=False, action="store_true",
                        help='Draw stochastic hits from the probabilistic distribution (fake-stochastic '
                             'mode). Output format matches real stochastic. '
                             'Mutually exclusive with --probabilistic_sim.')
    parser.add_argument('--probabilistic_sampling_target', '--probabilistic-sampling-target',
                        default=False, action="store_true",
                        help='When used together with --probabilistic_sim, draw the *target* from the '
                             'probabilistic distribution (via LUTProbabilisticSamplingSimulation) '
                             'instead of the real stochastic simulator. The guess remains the full '
                             'probabilistic distribution, keeping gradients differentiable. '
                             'Requires --probabilistic_sim. Mutually exclusive with --probabilistic_sampling_sim.')
    parser.add_argument('--shuffle_bt', default=False, action="store_true", help='shuffle the batch order within an epoch')
    parser.add_argument('--sz_mini_bt', type=int, default=1, help='Number of mini-batch for one update')
    parser.add_argument('--normalization_scheme', type=str, default='sigmoid', choices=['sigmoid', 'exp_log', 'divide'],
                        help='Parameter normalization scheme used during fitting')
    parser.add_argument('--normalization_scale_sigmoid', type=float, default=1.0,
                        help='Sigmoid normalization slope scale (<1 makes mapping flatter)')
    parser.add_argument('--normalization_scale_exp_log', type=float, default=1.0,
                        help='exp_log normalization slope scale (<1 makes mapping flatter in norm space)')
    parser.add_argument('--use_dedx_density', dest='use_dedx_density', action='store_true', default=False,
                        help='Use dE/dx density propagation in quenching.')
    parser.add_argument('--dedx_density_mode', type=str, default='histogram',
                        choices=['histogram', 'flow'],
                        help='dE/dx density model. histogram (default) or flow (trained CNF).')
    parser.add_argument('--profile', default=False, action='store_true', help='Should run some xprof execution profiling')
    parser.add_argument('--no_chop', default=False, action='store_true', help='Disable chopping in data loading')
    parser.add_argument('--no_pad', default=False, action='store_true', help='Disable padding in data loading')
    parser.add_argument("--resume_from", dest="resume_from", default=None, type=str, help="Resume chain fit from a saved history_iter*.pkl checkpoint")
    parser.add_argument("--fit_dedx", dest="fit_dedx", default=False, action="store_true",
                        help="Jointly optimise per-segment dEdx together with global physics params (chain fit only)")
    parser.add_argument("--dedx_prior_weight", dest="dedx_prior_weight", default=0.1, type=float,
                        help="Weight of Student-t NLL prior on per-segment dEdx values")
    parser.add_argument("--dedx_lr", dest="dedx_lr", default=None, type=float,
                        help="Learning rate for per-segment dEdx Adam optimiser (defaults to global --lr)")
    parser.add_argument("--dedx_start_iter", dest="dedx_start_iter", default=0, type=int,
                        help="Iteration at which to start dEdx fitting (0 = from the beginning). Use e.g. 200 to let calibration params warm up first.")
    parser.add_argument("--dedx_freeze_iter", dest="dedx_freeze_iter", default=None, type=int,
                        help="Iteration at which to freeze the per-segment dEdx values and only fine-tune the calibration parameter(s). Default: None (never freeze).")
    parser.add_argument("--dedx_student_loc", dest="dedx_student_loc", default=None, type=float,
                        help="dEdx Student-t prior location parameter (default: 1.864 MeV/cm)")
    parser.add_argument("--dedx_soft_barrier_threshold", dest="dedx_soft_barrier_threshold", default=8.5, type=float,
                        help="Threshold for dEdx soft barrier penalty (MeV/cm)")
    parser.add_argument("--dedx_soft_barrier_weight", dest="dedx_soft_barrier_weight", default=1.0, type=float,
                        help="Weight of one-sided soft barrier penalty on per-segment dEdx values")
    parser.add_argument("--dedx_use_split_t", dest="dedx_use_split_t", default=True, type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="Use bifurcated Split Student-t prior instead of symmetric Student-t prior (default: True)")
    parser.add_argument("--dedx_student_nu_l", dest="dedx_student_nu_l", default=4.785, type=float,
                        help="Left-side degrees of freedom for Split Student-t prior (default: 4.785)")
    parser.add_argument("--dedx_student_nu_r", dest="dedx_student_nu_r", default=2.073, type=float,
                        help="Right-side degrees of freedom for Split Student-t prior (default: 2.073)")
    parser.add_argument("--dedx_student_scale_l", dest="dedx_student_scale_l", default=0.1204, type=float,
                        help="Left-side scale for Split Student-t prior (default: 0.1204)")
    parser.add_argument("--dedx_student_scale_r", dest="dedx_student_scale_r", default=0.1058, type=float,
                        help="Right-side scale for Split Student-t prior (default: 0.1058)")
    parser.add_argument("--dedx_mean_constraint_weight", dest="dedx_mean_constraint_weight", default=0.0, type=float,
                        help="Weight of the global length-weighted mean dEdx constraint (default: 0.0, disabled)")
    parser.add_argument("--dedx_mean_constraint_target", dest="dedx_mean_constraint_target", default=1.887, type=float,
                        help="Target value for the global mean dEdx constraint in MeV/cm (default: 1.887)")
    parser.add_argument("--fit_track_positions", "--fit-track-positions", dest="fit_track_positions", default=False, action="store_true",
                        help="Jointly optimize local track translation and rotation (chain fit only)")
    parser.add_argument("--track_max_iter", "--track-max-iter", dest="track_max_iter", default=20, type=int,
                        help="Maximum L-BFGS-B iterations per local track alignment step")
    parser.add_argument("--track_alignment_freq", "--track-alignment-freq", dest="track_alignment_freq", default=1, type=int,
                        help="Frequency (in steps) of track alignment optimization")
    parser.add_argument("--fit_chain_positions", "--fit-chain-positions", dest="fit_chain_positions", default=False, action="store_true",
                        help="Jointly optimize per-track deflection chain geometry with global Ab and per-segment dEdx (chain fit only)")
    parser.add_argument("--chain_lr", dest="chain_lr", default=3e-5, type=float,
                        help="Learning rate for per-track chain angle Adam optimiser (default: 3e-5)")
    parser.add_argument("--chain_start_iter", dest="chain_start_iter", default=0, type=int,
                        help="Iteration at which to start chain position fitting (default: 0)")
    parser.add_argument("--chain_update_freq", dest="chain_update_freq", default=1, type=int,
                        help="Run chain gradient backward only every N steps; intermediate steps use argnums=(0,1) only (default: 1 = every step)")
    parser.add_argument("--chain_decay_rate", dest="chain_decay_rate", default=1.0, type=float,
                        help="Per-iteration exponential decay of the chain-position LR (default: 1.0 = no decay). <1 lets positions settle and damps the position/calibration oscillation.")
    parser.add_argument("--chain_decay_start", dest="chain_decay_start", default=None, type=int,
                        help="Iteration at which chain-LR decay begins (default: chain_start_iter)")
    parser.add_argument("--pos_residual_freq", dest="pos_residual_freq", default=1, type=int,
                        help="Compute the (diagnostic) chain position residual every N steps (default 1). Higher values speed up steps at no cost to the fit.")
    parser.add_argument("--mcs_prior_weight", dest="mcs_prior_weight", default=1.0, type=float,
                        help="Weight of Gaussian MCS prior on chain deflection angles (default: 1.0)")
    parser.add_argument("--chain_step_len", dest="chain_step_len", default=2.0, type=float,
                        help="Target chain segment length in cm (default: 2.0)")
    parser.add_argument("--chain_momentum_GeV", dest="chain_momentum_GeV", default=3.0, type=float,
                        help="Assumed muon momentum in GeV/c for Highland MCS formula (default: 3.0)")
    parser.add_argument("--benchmark_chain_opt", dest="benchmark_chain_opt", default=-1, type=int,
                        help="If >=0, benchmark chain optimizers (Adam vs L-BFGS) on this batch index instead of fitting.")
    parser.add_argument("--lbfgs_position_fit", dest="lbfgs_position_fit", action="store_true",
                        help="Run end-to-end position-only geometry fit with L-BFGS + early stopping instead of fitting.")
    parser.add_argument("--geom_optimizer", dest="geom_optimizer", default="adam", type=str,
                        help="Geometry-block optimiser: 'adam' (default, unchanged) or 'lbfgs' (jitted L-BFGS geometry solve).")
    parser.add_argument("--geom_lbfgs_steps", dest="geom_lbfgs_steps", default=10, type=int,
                        help="Max L-BFGS iterations per geometry solve (geom_optimizer=lbfgs).")
    parser.add_argument("--geom_lbfgs_unique_size", dest="geom_lbfgs_unique_size", default=0, type=int,
                        help="Static unique-pixel size for the jitted geometry loss (0 = auto from batch).")
    parser.add_argument("--profile_geom_compile", dest="profile_geom_compile", default=-1, type=int,
                        help="If >=0, profile geometry-loss compile time vs unique_size on this batch.")
    parser.add_argument("--test_jitted_geom", dest="test_jitted_geom", default=-1, type=int,
                        help="If >=0, run the jitted LLHD geometry L-BFGS on this batch index.")
    parser.add_argument("--validate_static_unique", dest="validate_static_unique", default=-1, type=int,
                        help="If >=0, validate static-size unique (numerical identity + jittable) on this batch.")
    parser.add_argument("--test_chain_lbfgs", dest="test_chain_lbfgs", default=-1, type=int,
                        help="If >=0, validate the jitted chain-geometry L-BFGS solver on this batch index.")

    try:
        args = parser.parse_args()
        retval, status_message = main(args)
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = 'Error: Fitting failed.'

    logger.info(status_message)
    exit(retval)
