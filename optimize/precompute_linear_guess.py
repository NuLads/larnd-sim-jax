#!/usr/bin/env python
"""
Precompute straight-line guess segments for larnd-sim-jax parameter fitting.

This script reads a true tracks (target) HDF5 file containing detailed stochastic
trajectories, groups segments by individual trajectory (eventID, trackID), locates
their chronological endpoints, and generates a straight-line segmented track
approximation with typical segment size (e.g. 1.0 cm). The output is saved in
structured HDF5 format, ready to be passed to example_run.py.
"""

import os
import sys
import argparse
import h5py
import numpy as np
from tqdm import tqdm

def get_field_names(dtype):
    """Normalize eventID/trackID and x_start/y_start/z_start field names."""
    names = dtype.names
    event_col = 'event_id' if 'event_id' in names else 'eventID'
    traj_col = 'traj_id' if 'traj_id' in names else 'trackID'
    pdg_col = 'pdg_id' if 'pdg_id' in names else 'pdgID'
    return event_col, traj_col, pdg_col

def precompute_linear_guess(input_file, output_file, segment_length=1.0):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Reading target segments from {input_file}...")
    with h5py.File(input_file, 'r') as f:
        segments = f['segments'][:]
        # Keep track of trajectories dataset if it exists
        has_trajectories = 'trajectories' in f
        if has_trajectories:
            trajectories_data = f['trajectories'][:]
            trajectories_dtype = f['trajectories'].dtype

    input_dtype = segments.dtype
    event_col, traj_col, pdg_col = get_field_names(input_dtype)

    # Group segments by unique (eventID, trackID)
    keys = np.stack([segments[event_col], segments[traj_col]], axis=-1)
    unique_keys, inverse_idx = np.unique(keys, axis=0, return_inverse=True)
    n_trajs = len(unique_keys)

    print(f"Found {n_trajs} unique trajectories.")
    
    # Preallocate output records list
    new_segments_list = []
    
    # Progress bar
    for idx in tqdm(range(n_trajs), desc="Precomputing straight-line guesses"):
        rows = segments[inverse_idx == idx]
        
        # 1. Extract segment boundary points
        starts = np.stack([rows['x_start'], rows['y_start'], rows['z_start']], axis=-1)
        ends = np.stack([rows['x_end'], rows['y_end'], rows['z_end']], axis=-1)
        pts = np.concatenate([starts, ends], axis=0)
        
        # 2. Find furthest pair of points to get endpoints
        dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        i, j = np.unravel_index(np.argmax(dists), dists.shape)
        start_pos = pts[i]
        end_pos = pts[j]
        total_len = dists[i, j]
        
        # 3. Orient chronologically using true start time (t_start or t0)
        time_field = 't_start' if 't_start' in input_dtype.names else ('t0' if 't0' in input_dtype.names else None)
        if time_field is not None:
            true_start_idx = np.argmin(rows[time_field])
            true_start_pos = starts[true_start_idx]
            # If end_pos is closer to the chronological start, swap them
            if np.linalg.norm(end_pos - true_start_pos) < np.linalg.norm(start_pos - true_start_pos):
                start_pos, end_pos = end_pos, start_pos
                
        # 4. Construct straight-line segments
        if total_len < 1e-5:
            # Handle point-like tracks
            n_segs = 1
            step_size = 0.0
            direction_unit = np.zeros(3)
        else:
            n_segs = int(np.ceil(total_len / segment_length))
            step_size = total_len / n_segs
            direction_unit = (end_pos - start_pos) / total_len
            
        total_de = rows['dE'].sum()
        de_per_seg = total_de / n_segs
        
        # Create segments
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
                    
            # Copy remaining fields with defaults if appropriate
            for field in ['long_diff', 'tran_diff', 'pixel_plane']:
                if field in input_dtype.names:
                    seg_record[field] = rows[0][field]
                    
            new_segments_list.append(seg_record)
            
    # Convert list to structured array
    new_segments = np.array(new_segments_list, dtype=input_dtype)
    
    print(f"Writing {len(new_segments)} generated guess segments to {output_file}...")
    
    # Write to output file
    if os.path.exists(output_file):
        os.remove(output_file)
        
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('segments', data=new_segments)
        if has_trajectories:
            f.create_dataset('trajectories', data=trajectories_data)
            
    print("Precomputation completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute straight-line guess segments for larnd-sim-jax.")
    parser.add_argument("-i", "--input", required=True, help="Path to the true target tracks HDF5 file.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the precomputed guess segments HDF5 file.")
    parser.add_argument("-l", "--seg-length", type=float, default=1.0, help="Typical guess segment length in cm (default: 1.0).")
    
    args = parser.parse_args()
    
    precompute_linear_guess(args.input, args.output, args.seg_length)
