#!/usr/bin/env python
"""
Prepare a reference-larnd-sim-compatible input h5 from one of our JAX-sim
input files (e.g. /sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
or /sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5).

Finding: no field renaming is actually needed. The reference checkout at
/sdf/home/p/pgranger/reference (DUNE/larnd-sim develop, Sep 2024) already
defaults to the snake_case schema (EVENT_SEPARATOR='event_id',
TRACKS_DSET_NAME='segments', and reads 'segment_id'/'pdg_id' etc. directly
-- see larndsim/consts/sim.py and cli/simulate_pixels.py's tracks-loading
block). Our input files' 'segments' dtype is a strict superset of what the
reference's cli/simulate_pixels.py actually reads (it only ever indexes
tracks['segment_id'/'tran_diff'/'pixel_plane'/'n_electrons'/'n_photons'/
't'/'t0'/'t0_end'/'t0_start'/'t_end'/'t_start'/'x'/'x_end'/'x_start'/'y'/
'y_end'/'y_start'/'z'/'z_end'/'z_start'/'dE'/'dEdx'/'dx']); the extra
columns we carry (vertex_id, file_vertex_id, traj_id, file_traj_id) are
simply ignored. So this script does NOT need to rename or drop fields --
it only needs to slice out a small/matched subset of events and write a
standalone h5 (segments [+ trajectories if present]) that the reference
CLI can read directly with --tracks_dset_name=segments (the default).

Two selection modes:
  --mode tiny   : first `--n-segments` segments of the first event only
                   (for a fast CPU/CUDASIM plumbing smoke test).
  --mode batch  : greedily accumulate whole events, in file order starting
                   at --start-event, until the summed `dx` reaches
                   --target-cm (e.g. 350 cm, to match one of our ~3-4 m
                   fit batches). Mirrors (approximately -- it does not
                   reproduce TracksDataset's track_len_sel /
                   max_abs_costheta_sel / min_abs_segz_sel / track_z_bound
                   cuts) the scale of a paper_common.py batch; it is meant
                   for a timing/plumbing comparison, not a bit-exact
                   replay of a specific fit batch.

Usage:
  venv/bin/python prepare_reference_input.py \
      --input /sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5 \
      --output tiny_input.h5 --mode tiny --n-segments 20

  venv/bin/python prepare_reference_input.py \
      --input /sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5 \
      --output batch_350cm_input.h5 --mode batch --target-cm 350
"""
import argparse
import numpy as np
import h5py

REQUIRED_FIELDS = [
    'segment_id', 'event_id', 'tran_diff', 'long_diff', 'pixel_plane',
    'n_electrons', 'n_photons', 't', 't0', 't0_end', 't0_start', 't_end',
    't_start', 'x', 'x_end', 'x_start', 'y', 'y_end', 'y_start', 'z',
    'z_end', 'z_start', 'dE', 'dEdx', 'dx',
]


def check_fields(segments):
    missing = [f for f in REQUIRED_FIELDS if f not in segments.dtype.names]
    if missing:
        raise ValueError(f"Input 'segments' dtype is missing fields the reference "
                          f"cli/simulate_pixels.py reads directly: {missing}. "
                          f"A rename step would be needed for these.")


def write_subset(input_path, output_path, event_mask):
    with h5py.File(input_path, 'r') as fin:
        segments = np.array(fin['segments'])
        check_fields(segments)
        sub = segments[event_mask(segments)]
        with h5py.File(output_path, 'w') as fout:
            fout.create_dataset('segments', data=sub)
            if 'trajectories' in fin:
                traj = np.array(fin['trajectories'])
                traj_mask = np.isin(traj['event_id'], np.unique(sub['event_id']))
                fout.create_dataset('trajectories', data=traj[traj_mask])
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--mode', choices=['tiny', 'batch'], default='tiny')
    ap.add_argument('--n-segments', type=int, default=20)
    ap.add_argument('--target-cm', type=float, default=350.0)
    ap.add_argument('--start-event', type=int, default=0)
    args = ap.parse_args()

    if args.mode == 'tiny':
        with h5py.File(args.input, 'r') as fin:
            segments = np.array(fin['segments'])
            check_fields(segments)
            first_event = segments['event_id'][0]
        mask = lambda seg: seg['event_id'] == first_event
        sub = write_subset(args.input, args.output, mask)
        sub = sub[:args.n_segments] if len(sub) > args.n_segments else sub
        # rewrite trimmed to n_segments
        with h5py.File(args.output, 'w') as fout:
            fout.create_dataset('segments', data=sub)
        print(f"Wrote {len(sub)} segments (1 event) -> {args.output}, "
              f"total track length = {sub['dx'].sum():.3f} cm")
    else:
        with h5py.File(args.input, 'r') as fin:
            segments = np.array(fin['segments'])
            check_fields(segments)
        uniq_events = np.unique(segments['event_id'])
        uniq_events = uniq_events[uniq_events >= args.start_event] if args.start_event else uniq_events
        chosen = []
        total_cm = 0.0
        for ev in uniq_events:
            seg_ev = segments[segments['event_id'] == ev]
            chosen.append(ev)
            total_cm += float(seg_ev['dx'].sum())
            if total_cm >= args.target_cm:
                break
        mask = lambda seg: np.isin(seg['event_id'], chosen)
        sub = write_subset(args.input, args.output, mask)
        print(f"Wrote {len(sub)} segments ({len(chosen)} events) -> {args.output}, "
              f"total track length = {total_cm:.3f} cm (target {args.target_cm} cm)")


if __name__ == '__main__':
    main()
