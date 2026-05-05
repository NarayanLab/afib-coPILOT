"""Illustrative CLI: run coPILOT on a single recording.

Inputs (NumPy / pickle for portability):
    --activations_path : .npy file, shape (num_nodes, T_total). Binary
        matrix where 1 marks an activation sample. See the companion
        coMAP repository for the activation-time annotator that
        produces this matrix from raw electrograms.
    --catheter_config_path : .npz file with the catheter geometry
        (num_splines, num_electrodes_per_spline, electrode_spacing_mm).

Outputs:
    --directions_path : .npy of dtype=object containing the list of
        per-segment direction dicts returned by ``stage_coPILOT``.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

from catheter import GridCatheter
from copilot_wrapper import stage_coPILOT


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_path", type=str, required=True)
    parser.add_argument("--catheter_config_path", type=str, required=True)
    parser.add_argument("--directions_path", type=str, required=True)
    parser.add_argument("--fs", type=int, default=1000, help="sampling frequency (Hz)")
    parser.add_argument(
        "--segment_seconds",
        type=float,
        default=4.0,
        help="segment length over which gradients are aggregated",
    )
    parser.add_argument(
        "--min_CV",
        type=float,
        default=200.0,
        help="minimum conduction velocity (mm/s) for wave grouping",
    )
    args = parser.parse_args()

    activations = np.load(args.activations_path)

    config = np.load(args.catheter_config_path, allow_pickle=True)
    cath = GridCatheter(
        num_splines=int(config["num_splines"]),
        num_electrodes_per_spline=int(config["num_electrodes_per_spline"]),
        electrode_spacing_mm=float(config["electrode_spacing_mm"]),
    )

    seg_len = int(args.segment_seconds * args.fs)
    start_times = list(range(0, activations.shape[1] - seg_len + 1, seg_len))

    directions = stage_coPILOT(
        activations=activations,
        cath=cath,
        start_times=start_times,
        seg_len_samples=seg_len,
        fs=args.fs,
        min_CV=args.min_CV,
    )

    np.save(args.directions_path, np.asarray(directions, dtype=object))


if __name__ == "__main__":
    main()
