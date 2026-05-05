"""coPILOT pipeline wrapper.

Orchestrates the three steps described in the manuscript:

    Step 1: Estimation of AF waves
            (the activation matrix is an *input*; wave segmentation is
            performed by ``step_waves`` using a CV_min constraint).
    Step 2: Per-wave LEAP gradients
            (``step_wleap`` produces a per-wave LEAP map and a wave-
            averaged map).
    Step 3: Aggregation to a predominant compass direction
            (``step_direction`` maps the segment-level wLEAP map to a
            compass angle using catheter-specific masks).

The activation-time annotator that produces the input ``activations``
matrix is described in the companion repository (coMAP); see the
README. This module assumes the matrix is already available.
"""
from __future__ import annotations

import logging

import numpy as np

from step_direction import grid_masks, step_direction
from step_waves import step_waves
from step_wleap import step_wleap

logger = logging.getLogger(__name__)


def stage_coPILOT(
    activations: np.ndarray,
    cath,
    start_times: list[int],
    seg_len_samples: int,
    min_CV: float = 200.0,
    fs: int = 1000,
    use_diagonal: bool = True,
    min_activations_per_wave: int | None = None,
):
    """Run coPILOT over a sequence of 4 s segments.

    Args:
        activations (ndarray): binary 2D array, shape
            ``(num_nodes, T_total)``, marking activation samples per
            electrode for the full recording.
        cath: catheter object (see :class:`catheter.GridCatheter`).
        start_times (list[int]): start sample of each segment to analyse.
            The manuscript uses non-overlapping 4 s windows.
        seg_len_samples (int): segment length in samples (e.g. 4 * fs).
        min_CV (float): minimum conduction velocity in mm/s used in
            :func:`step_waves.step_waves`.
        fs (int): sampling frequency in Hz.
        use_diagonal (bool): include diagonal neighbors when grouping
            activations into waves.
        min_activations_per_wave (int or None): drop tiny waves below
            this size (helps suppress noise).

    Returns:
        list[dict]: one entry per segment. Each entry contains
            ``angle`` (compass angle in degrees, or -1 for "centre", or
            None when ambiguous), ``score`` (float, or None) and
            ``all_angle_scores`` (per-direction scores).
    """
    masks = grid_masks(cath.num_splines, cath.num_electrodes_per_spline)

    results = []
    for start in start_times:
        end = start + seg_len_samples
        seg_activations = activations[:, start:end]

        ############################################
        ## Step 1. Estimation of AF Waves
        ############################################
        unique_activations, waves = step_waves(
            cath,
            seg_activations,
            use_diagonal=use_diagonal,
            min_CV=min_CV,
            fs=fs,
            min_activations_per_wave=min_activations_per_wave,
        )

        logger.info(
            "Step 1. Estimation of AF Waves: %d activation samples → %d AF waves",
            len(unique_activations),
            len(waves),
        )

        if len(waves) == 0:
            logger.info("Step 1. Estimation of AF Waves: no waves found; segment skipped")
            results.append({"angle": None, "score": None, "all_angle_scores": {}, "wleap_map": None, "n_waves": 0})
            continue

        ############################################
        ## Step 2. Assigning Activation Gradients for AF Waves
        ############################################
        wleap_avg, _wleap_per_wave = step_wleap(
            cath, unique_activations, waves
        )

        grid_str = "\n".join(
            "  " + "  ".join(f"{v:+6.1f}" for v in row) for row in wleap_avg
        )
        logger.info(
            "Step 2. Assigning Activation Gradients for AF Waves: wLEAP map (ms, positive = leads peers):\n%s", grid_str
        )

        ############################################
        ## Step 3. Aggregating Local Activation Gradients over Time
        ############################################
        result = step_direction(wleap_avg, masks)

        scores_str = "  ".join(
            f"{'ctr' if a == -1 else f'{a}°'}: {s:+.2f}"
            for a, s in sorted(
                result["all_angle_scores"].items(),
                key=lambda x: (x[0] == -1, x[0]),
            )
        )
        logger.info("Step 3. Aggregating Local Activation Gradients over Time: direction scores: %s", scores_str)
        logger.info(
            "Step 3. Aggregating Local Activation Gradients over Time: predicted direction: %s (score %s)",
            result["angle"],
            f"{result['score']:.3f}" if result["score"] is not None else "n/a",
        )

        result["wleap_map"] = wleap_avg
        result["n_waves"] = len(waves)
        results.append(result)

    return results
