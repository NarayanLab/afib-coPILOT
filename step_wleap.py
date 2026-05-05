"""Step 2. Assigning Activation Gradients for AF Waves.

For each candidate wave produced by ``step_waves``, this step computes
local gradients within each estimated wave using a relative-timing
metric: the Leading Earliest Activity Potential (LEAP) score.
Electrodes are then ranked from earliest to latest by LEAP.

It is rank-based; this rank-based formulation yields a
robust estimate of the dominant activation-time gradient across the
electrode neighbourhood and reduces sensitivity to outlying pairwise
delays and variations in inter-electrode spacing (cf. rank-based
sequencing in optical mapping; Laughner 2012, Jolliffe 2016). The
aggregation across waves within a 4 s segment is done in
:mod:`step_direction`.
"""
import numpy as np


def step_wleap(
    cath,
    unique_activations: np.ndarray,
    waves: list,
    buffer_ms: float = 0.0,
):
    """Compute per-wave wLEAP scores and a wave-averaged map.

    Args:
        cath: catheter object exposing ``num_nodes``, ``num_splines`` and
            ``num_electrodes_per_spline``.
        unique_activations (ndarray): (n_acts, 2) array of (electrode,
            time) returned by :func:`step_waves.step_waves`.
        waves (list[list[int]]): wave membership lists from
            :func:`step_waves.step_waves`.
        mode (str): ``"time_difference"`` (default, signed mean delay)
            or ``"binary"`` (fraction of peers led).
        buffer_ms (float): samples-equivalent buffer used by the
            ``"binary"`` mode to declare an electrode as leading.

    Returns:
        wleap_avg (ndarray): wave-averaged LEAP map reshaped to the grid
            (num_splines, num_electrodes_per_spline).
        wleap_per_wave (ndarray): (n_waves, num_nodes) raw scores.
    """
    if len(waves) == 0:
        empty_grid = np.full(
            (cath.num_splines, cath.num_electrodes_per_spline), np.nan
        )
        empty_raw = np.empty((0, cath.num_nodes))
        return empty_grid, empty_raw

    wleap_per_wave = np.vstack(
        [
            compute_wleap_for_wave(
                cath.num_nodes, unique_activations, wave, buffer_ms
            )
            for wave in waves
        ]
    )

    wleap_avg = np.nanmean(wleap_per_wave, axis=0).reshape(
        cath.num_splines, cath.num_electrodes_per_spline
    )
    return wleap_avg, wleap_per_wave


def compute_wleap_for_wave(
    num_nodes: int,
    unique_activations: np.ndarray,
    wave: list,
) -> np.ndarray:
    """LEAP score for every electrode within a single wave.

    Pseudo-code (time_difference mode):
        wave_acts = unique_activations[wave]                # (k, 2)
        for each electrode i in [0, num_nodes):
            t_i = activation time of i in wave_acts (or NaN)
            scores[i] = mean over j != i of (t_j - t_i)
        return scores                                       # (num_nodes,)

    Electrodes that did not participate in the wave receive ``NaN``.
    """
    scores = np.full(num_nodes, np.nan)

    # Build per-electrode activation times for this wave (first if multiple)
    times = {}
    for row in unique_activations[wave]:
        e, t = int(row[0]), int(row[1])
        times.setdefault(e, t)  # keep earliest occurrence

    if not times:
        return scores

    for i in range(num_nodes):
        if i not in times:
            continue
        t_i = times[i]

        peer_diffs = []
        for j, t_j in times.items():
            if j == i:
                continue

            # signed mean delay: positive = i is earlier than peers
            peer_diffs.append(t_j - t_i)

        if peer_diffs:
            scores[i] = float(np.mean(peer_diffs))

    return scores
