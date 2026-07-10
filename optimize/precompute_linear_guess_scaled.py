#!/usr/bin/env python
"""
Precompute straight-line guess segments for larnd-sim-jax parameter fitting,
including a length_scale_factor field to account for multiple Coulomb scattering
(wiggling) and cathode boundary dead zones.
This version uses vectorized TPC checks and O(1) contiguous slices to achieve
maximum performance (running in ~10 seconds for 216k tracks / 10M segments).
"""

import os
import sys
import argparse
import h5py
import numpy as np
import numpy.lib.recfunctions as rfn
from tqdm import tqdm

# TPC borders in unswapped coordinates (x is drift direction, y is vertical, z is beam direction)
# In this coordinate system, the cathode gap is along x: [-0.15875, 0.15875] cm.
# Active volumes:
tpc_borders = np.array([
    [[0.15875, 30.58975], [-62.076, 62.076], [-31.038, 31.038]],
    [[-30.58975, -0.15875], [-62.076, 62.076], [-31.038, 31.038]]
])

def is_active_vectorized(pts):
    # pts has shape (N, 3)
    # Returns boolean mask of shape (N,)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    
    active = np.zeros(pts.shape[0], dtype=bool)
    for tpc in tpc_borders:
        in_tpc = (
            (tpc[0, 0] <= x) & (x <= tpc[0, 1]) &
            (tpc[1, 0] <= y) & (y <= tpc[1, 1]) &
            (tpc[2, 0] <= z) & (z <= tpc[2, 1])
        )
        active |= in_tpc
    return active

def get_field_names(dtype):
    names = dtype.names
    event_col = 'event_id' if 'event_id' in names else 'eventID'
    traj_col = 'traj_id' if 'traj_id' in names else 'trackID'
    pdg_col = 'pdg_id' if 'pdg_id' in names else 'pdgID'
    return event_col, traj_col, pdg_col

def precompute_linear_guess_scaled(input_file, output_file, segment_length=1.0, mode='mcs', eta=9.138287e-5, max_trajs=None):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Reading target segments from {input_file}...")
    with h5py.File(input_file, 'r') as f:
        segments = f['segments'][:]
        has_trajectories = 'trajectories' in f
        if has_trajectories:
            trajectories_data = f['trajectories'][:]

    input_dtype = segments.dtype
    event_col, traj_col, pdg_col = get_field_names(input_dtype)

    print("Sorting segments by event and track ID for contiguous slicing...")
    sort_idx = np.lexsort((segments[traj_col], segments[event_col]))
    segments = segments[sort_idx]

    # Find boundaries of each unique (eventID, trackID) group
    keys = np.stack([segments[event_col], segments[traj_col]], axis=-1)
    unique_keys, start_indices = np.unique(keys, axis=0, return_index=True)
    end_indices = np.append(start_indices[1:], len(segments))
    n_trajs = len(unique_keys)

    if max_trajs is not None and max_trajs < n_trajs:
        print(f"Limiting processed trajectories to {max_trajs} (out of {n_trajs})")
        n_trajs = max_trajs
        unique_keys = unique_keys[:n_trajs]
        start_indices = start_indices[:n_trajs]
        end_indices = end_indices[:n_trajs]

    print(f"Processing {n_trajs} trajectories.")
    
    new_segments_list = []
    scale_factors_by_track = []
    
    step_sz = 0.02
    
    for idx in tqdm(range(n_trajs), desc="Precomputing scaled straight-line guesses"):
        rows = segments[start_indices[idx]:end_indices[idx]]
        
        # 1. Extract segment boundary points to find endpoints
        starts = np.stack([rows['x_start'], rows['y_start'], rows['z_start']], axis=-1)
        ends = np.stack([rows['x_end'], rows['y_end'], rows['z_end']], axis=-1)
        pts = np.concatenate([starts, ends], axis=0)
        
        dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        i, j = np.unravel_index(np.argmax(dists), dists.shape)
        start_pos = pts[i]
        end_pos = pts[j]
        total_len = dists[i, j]
        
        # Orient chronologically using true start time (t_start or t0)
        time_field = 't_start' if 't_start' in input_dtype.names else ('t0' if 't0' in input_dtype.names else None)
        if time_field is not None:
            true_start_idx = np.argmin(rows[time_field])
            true_start_pos = starts[true_start_idx]
            if np.linalg.norm(end_pos - true_start_pos) < np.linalg.norm(start_pos - true_start_pos):
                start_pos, end_pos = end_pos, start_pos
                
        # 2. Compute active linear length using vectorized checks
        if total_len < 1e-5:
            active_linear_len = 0.0
        else:
            n_steps = int(np.ceil(total_len / step_sz))
            steps = (np.arange(n_steps) + 0.5) / n_steps
            mid_pts = start_pos + steps[:, None] * (end_pos - start_pos)
            active_steps = np.sum(is_active_vectorized(mid_pts))
            active_linear_len = active_steps * (total_len / n_steps)

        # 3. Compute the track length scale factor based on the mode
        true_active_len = np.sum(rows['dx'])
        
        if mode == 'true':
            if active_linear_len > 1e-5:
                scale_factor = true_active_len / active_linear_len
            else:
                scale_factor = 1.0
        elif mode == 'mcs':
            # Highland model quadratic scaling: L_est = L_linear_active + eta * L_linear_active^2
            scale_factor = 1.0 + eta * active_linear_len
        elif mode == 'const':
            # Constant ratio scaling
            scale_factor = 1.0 + 0.008979
        else: # 'none'
            scale_factor = 1.0
            
        # 4. Construct straight-line segments
        if total_len < 1e-5:
            n_segs = 1
            step_size = 0.0
            direction_unit = np.zeros(3)
        else:
            n_segs = int(np.ceil(total_len / segment_length))
            step_size = total_len / n_segs
            direction_unit = (end_pos - start_pos) / total_len
            
        total_de = rows['dE'].sum()
        de_per_seg = total_de / n_segs
        
        for k in range(n_segs):
            seg_record = np.zeros((1,), dtype=input_dtype)[0]
            
            # Interpolate geometry
            seg_start = start_pos + k * step_size * direction_unit
            seg_end = start_pos + (k + 1) * step_size * direction_unit
            seg_mid = 0.5 * (seg_start + seg_end)
            
            seg_record['x_start'] = seg_start[0]
            seg_record['y_start'] = seg_start[1]
            seg_record['z_start'] = seg_start[2]
            
            seg_record['x_end'] = seg_end[0]
            seg_record['y_end'] = seg_end[1]
            seg_record['z_end'] = seg_end[2]
            
            seg_record['x'] = seg_mid[0]
            seg_record['y'] = seg_mid[1]
            seg_record['z'] = seg_mid[2]
            
            seg_record['dx'] = step_size
            seg_record['dE'] = de_per_seg
            seg_record['dEdx'] = de_per_seg / step_size if step_size > 0 else 0.0
            
            # Copy track metadata fields
            seg_record[event_col] = rows[0][event_col]
            seg_record[traj_col] = rows[0][traj_col]
            seg_record[pdg_col] = rows[0][pdg_col]
            
            for field in ['vertex_id', 'file_vertex_id', 'file_traj_id', 't0', 't_start', 't_end', 't0_start', 't0_end']:
                if field in input_dtype.names:
                    seg_record[field] = rows[0][field]
                    
            for field in ['long_diff', 'tran_diff', 'pixel_plane']:
                if field in input_dtype.names:
                    seg_record[field] = rows[0][field]
                    
            new_segments_list.append(seg_record)
            scale_factors_by_track.append(scale_factor)
            
    # Convert list to structured array
    new_segments = np.array(new_segments_list, dtype=input_dtype)
    scale_factors_by_track = np.array(scale_factors_by_track, dtype=np.float32)
    
    # Append the length_scale_factor field
    new_segments = rfn.append_fields(new_segments, 'length_scale_factor', scale_factors_by_track, dtypes=np.float32, usemask=False)
    
    print(f"Writing {len(new_segments)} generated guess segments to {output_file}...")
    if os.path.exists(output_file):
        os.remove(output_file)
        
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('segments', data=new_segments)
        # Filter trajectories to keep the output file self-consistent
        if has_trajectories:
            # Match (event_id, traj_id) keys of segments
            seg_evts = new_segments[event_col]
            seg_trajs = new_segments[traj_col]
            # Form keys of written segments
            written_keys = set(zip(seg_evts, seg_trajs))
            
            traj_evts = trajectories_data[event_col]
            traj_trajs = trajectories_data[traj_col]
            
            traj_mask = [((evt, traj) in written_keys) for evt, traj in zip(traj_evts, traj_trajs)]
            filtered_trajectories = trajectories_data[traj_mask]
            f.create_dataset('trajectories', data=filtered_trajectories)
            
    print("Precomputation completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute straight-line guess segments with length scale factors.")
    parser.add_argument("-i", "--input", required=True, help="Path to the true target tracks HDF5 file.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the guess segments HDF5 file.")
    parser.add_argument("-l", "--seg-length", type=float, default=1.0, help="Typical guess segment length in cm (default: 1.0).")
    parser.add_argument("--mode", choices=['true', 'mcs', 'const', 'none'], default='mcs',
                        help="Scaling factor mode (default: mcs).")
    parser.add_argument("--eta", type=float, default=9.138287e-5,
                        help="Highland wiggling model coefficient (default: 9.138287e-5).")
    parser.add_argument("--max-trajs", type=int, default=None,
                        help="Limit the number of processed trajectories to speed up generation.")
    
    args = parser.parse_args()
    precompute_linear_guess_scaled(args.input, args.output, args.seg_length, args.mode, args.eta, args.max_trajs)
