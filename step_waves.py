"""Step 1. AI-Based Estimation of AF Waves.

Activation times per electrode are assumed to be available as a binary
matrix (electrodes x time) where 1 marks an activation. In the published
system these are produced by an upstream AI annotator validated against
expert review and simultaneous monophasic action potentials; see the
companion repository

    https://github.com/NarayanLab/afib-tissue-activation 

for the activation-time annotator.

The algorithm analyzed short windows (~one AF cycle) from 2–8 cm²
catheter footprints, capturing propagation dominated by a small number
of waves while limiting averaging from concurrent, spatially distinct
wavefronts.

Given the activation matrix, activations are grouped across electrodes
into candidate waves by requiring time delays (Δt) between electrodes
with known spacing (d) to be consistent with a minimum conduction
velocity (CV_min):

        Δt  <  d / CV_min

Physiologic CV_min values are drawn from preclinical and human studies
(Harrild 2000, Boyle 2019, Lalani 2012, Fukumoto 2016). Electrode pairs
with slower CV were assigned into separate waves, while near-simultaneous
pairs were retained as they could reflect a wave traveling orthogonal to
the inter-electrode axis.
"""
import numpy as np
from scipy.spatial import distance


def step_waves(
    cath,
    activations: np.ndarray,
    use_diagonal: bool = True,
    min_CV: float = 200.0,
    fs: int = 1000,
    min_activations_per_wave: int | None = None,
):
    """Wrapper around the wave-segmentation routine.

    Args:
        cath: catheter object exposing ``get_electrode_pairs`` and
            ``num_nodes``.
        activations (ndarray): binary 2D array, shape (num_nodes, T),
            with 1 marking activation samples.
        use_diagonal (bool): include diagonal neighbors when grouping.
        min_CV (float): minimum conduction velocity in mm/s used to
            bound the maximum inter-electrode delay within a wave.
            Choose a physiologic value from preclinical/human studies.
        fs (int): sampling frequency in Hz.
        min_activations_per_wave (int or None): drop waves with fewer
            activations than this when set.

    Returns:
        unique_activations (ndarray): (n_acts, 2) array; columns are
            (electrode_index, activation_sample).
        waves (list[list[int]]): each sublist contains row indices into
            ``unique_activations`` that belong to the same wave.
    """
    positions, pairs = cath.get_electrode_pairs(use_diagonal=use_diagonal)

    compatible_pairs = _pair_compatible_activations(
        activations, pairs, positions, min_CV=min_CV, fs=fs
    )
    unique_activations, waves = _group_into_waves(compatible_pairs)

    if min_activations_per_wave is not None:
        waves = [w for w in waves if len(w) >= min_activations_per_wave]

    return unique_activations, waves


def _pair_compatible_activations(
    activations: np.ndarray,
    electrode_pairs: np.ndarray,
    positions: np.ndarray,
    min_CV: float,
    fs: int,
) -> np.ndarray:
    """For every neighbor pair (a, b), find activation-time pairs whose
    delay is consistent with at least ``min_CV``.

    Each accepted pair becomes a row of the form
    ``[earlier_electrode, earlier_time, later_electrode, later_time]``.

    For each electrode pair (a, b):
        d_ab = || pos(b) - pos(a) ||
        max_delay_samples = (d_ab / min_CV) * fs
        for each activation time t_a on electrode a:
            find nearest activation time t_b on electrode b
            if |t_a - t_b| < max_delay_samples:
                record (a, t_a, b, t_b) ordered by time
    """
    accepted = []
    for a, b in electrode_pairs:
        d_ab = np.linalg.norm(positions[b] - positions[a])
        max_delay = d_ab / min_CV * fs

        t_a = np.where(activations[a, :])[0]
        t_b = np.where(activations[b, :])[0]
        if t_a.size == 0 or t_b.size == 0:
            continue

        delays = distance.cdist(t_a.reshape(-1, 1), t_b.reshape(-1, 1))
        nearest = np.argmin(delays, axis=1)
        within = delays[np.arange(len(nearest)), nearest] < max_delay

        for i, ok in enumerate(within):
            if not ok:
                continue
            ti, tj = int(t_a[i]), int(t_b[nearest[i]])
            if ti <= tj:
                accepted.append([a, ti, b, tj])
            else:
                accepted.append([b, tj, a, ti])

    if not accepted:
        return np.empty((0, 4), dtype=int)
    return np.asarray(accepted, dtype=int)


def _group_into_waves(compatible_pairs: np.ndarray):
    """Group compatible activation pairs into candidate waves.

    Each unique (electrode, activation_time) pair is a node; each row of
    ``compatible_pairs`` links two nodes. Nodes that are transitively
    connected form a single candidate wave.

    Returns:
        unique_activations (ndarray): (n_acts, 2) (electrode, time).
        waves (list[list[int]]): row indices into ``unique_activations``.
    """
    raise NotImplementedError(
        "Wave grouping is not implemented in this pseudo-code release."
    )
