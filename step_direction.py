"""Step 3. Aggregating Local Activation Gradients over Time.

This step estimates the predominant direction across successive AF waves
by aggregating local activation gradients within each local domain,
smoothing fluctuations from transient asynchronous activity. Aggregation
used a 4 s window.

Given a wave-averaged wLEAP map for a 4 s segment (output of
:func:`step_wleap.step_wleap`), LEAP rankings are combined across waves
to assign a single predominant direction, displayed on its corresponding
catheter edge. The direction is expressed as a compass angle, with the
convention used in the manuscript:

      0 deg = right (east)        180 deg = left (west)
     45 deg = top-right            225 deg = bottom-left
     90 deg = top (north)          270 deg = bottom (south)
    135 deg = top-left             315 deg = bottom-right
     -1     = "centre" - earliest activation lies inside the array

Waves whose earliest activation fell within the interior of the array
are labelled "centre". Waves contributing less spatial support within
the array were down-weighted. No predominant direction is assigned when
gradients were divergent.

Only the generic 4 x 4 / 3 x 3 grid masks are exposed in this
pseudo-code release. Other catheter geometries (basket, Optrell, sphere,
etc.) follow the same recipe with different per-direction electrode
subsets.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Compass direction masks for a generic rectangular grid
# ---------------------------------------------------------------------------


def grid_masks(num_splines: int, num_electrodes_per_spline: int) -> dict:
    """Return the per-compass-direction electrode-index masks.

    The 4 x 4 case below is the default reported in Fig. 2 of the
    manuscript. For a generic R x C grid each direction is mapped to the
    corresponding edge / corner / centre slab of the array.

    Returns:
        dict[int, list[int]]: mapping from compass angle (in degrees, or
        -1 for "centre") to flat electrode indices.
    """
    R, C = num_splines, num_electrodes_per_spline

    if (R, C) == (4, 4):
        # Layout (row-major flattening):
        #   0  1  2  3
        #   4  5  6  7
        #   8  9 10 11
        #  12 13 14 15
        return {
            -1:  [5, 6, 9, 10],     # centre
              0: [7, 11],           # right
             45: [2, 3, 7],         # top-right
             90: [1, 2],            # top
            135: [0, 1, 4],         # top-left
            180: [4, 8],            # left
            225: [8, 12, 13],       # bottom-left
            270: [13, 14],          # bottom
            315: [11, 14, 15],      # bottom-right
        }

    if (R, C) == (3, 3):
        # Layout:
        #   0 1 2
        #   3 4 5
        #   6 7 8
        return {
            -1:  [4],
              0: [2, 5, 8],
             45: [2],
             90: [0, 1, 2],
            135: [0],
            180: [0, 3, 6],
            225: [6],
            270: [6, 7, 8],
            315: [8],
        }

    # Generic fallback for other rectangular grids: pick the corresponding
    # edge / corner / centre slab.
    total = R * C
    centre = [
        r * C + c
        for r in range(max(0, R // 2 - 1), min(R, R // 2 + 1))
        for c in range(max(0, C // 2 - 1), min(C, C // 2 + 1))
    ]
    return {
        -1:  centre,
          0: [r * C + (C - 1) for r in range(R)],   # rightmost column
         90: list(range(C)),                        # top row
        180: [r * C for r in range(R)],             # leftmost column
        270: list(range(total - C, total)),         # bottom row
        # corner directions left empty by default; users should supply
        # geometry-specific masks when relevant.
         45: [],
        135: [],
        225: [],
        315: [],
    }


# ---------------------------------------------------------------------------
# Direction inference from the wLEAP map
# ---------------------------------------------------------------------------


def step_direction(
    wleap_avg: np.ndarray,
    masks: dict,
    ambiguity_score_tolerance: float = 0.01,
    ambiguity_min_angle_separation: float = 45.0,
):
    """Compute the predominant compass direction for a 4 s segment.

    Args:
        wleap_avg (ndarray): wave-averaged LEAP map for the segment as
            produced by :func:`step_wleap.step_wleap`. Shape is the
            grid layout (R, C) but it is flattened internally.
        masks (dict): output of :func:`grid_masks`.
        ambiguity_score_tolerance (float): tolerance for declaring the
            result ambiguous when a competing direction scores similarly.
        ambiguity_min_angle_separation (float): minimum angular separation
            (degrees) between competing directions to trigger ambiguity.

    Returns:
        dict with keys
            - ``angle`` (int or None): predominant compass angle.
            - ``score`` (float or None): score of that direction.
            - ``all_angle_scores`` (dict[int, float]): per-direction
              scores, useful for visualisation and debugging.
    """
    # Pseudo-code: aggregate wLEAP scores within each compass-direction
    # mask to identify the predominant propagation direction across the
    # 4 s segment.
    raise NotImplementedError(
        "Direction aggregation is not implemented in this pseudo-code release."
    )
