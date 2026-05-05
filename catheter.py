"""Minimal mapping-catheter placeholder.

The production system supports several catheter geometries. For this
pseudo-code, we expose only a regular planar grid catheter (e.g. a 4x4
multielectrode array as in Fig. 2 of the manuscript). Extending the
algorithm to other geometries reduces to providing:

  1. Electrode (x, y) coordinates.
  2. A list of neighbor pairs over which inter-electrode delays are
     evaluated when grouping activations into waves.
  3. A direction-mask dictionary mapping compass angles to electrode
     subsets to be averaged (see ``step_direction.grid_masks``).
"""

import numpy as np


class GridCatheter:
    """Regular rectangular grid of electrodes.

    Attributes:
        num_splines (int): rows in the grid (e.g. 4).
        num_electrodes_per_spline (int): columns in the grid (e.g. 4).
        num_nodes (int): total number of electrodes (rows * cols).
        electrode_spacing_mm (float): inter-electrode spacing in mm.
        data (dict): per-recording data such as the activation matrix and
            the list of segment start times. The wrapper writes/reads
            entries here using string keys ("activations", "start_times",
            "seg_len").
    """

    def __init__(
        self,
        num_splines: int = 4,
        num_electrodes_per_spline: int = 4,
        electrode_spacing_mm: float = 3.0,
    ):
        self.num_splines = num_splines
        self.num_electrodes_per_spline = num_electrodes_per_spline
        self.num_nodes = num_splines * num_electrodes_per_spline
        self.electrode_spacing_mm = electrode_spacing_mm
        self.data: dict = {}

    def get_electrode_positions(self) -> np.ndarray:
        """Returns an (num_nodes, 2) array of electrode positions in mm.

        Indexing matches the row-major flattening used everywhere else
        in this codebase: electrode k is at (row=k // cols, col=k % cols).
        """
        cols = self.num_electrodes_per_spline
        rows = self.num_splines
        return np.array(
            [
                [c * self.electrode_spacing_mm, r * self.electrode_spacing_mm]
                for r in range(rows)
                for c in range(cols)
            ]
        )

    def get_electrode_pairs(self, use_diagonal: bool = True):
        """Returns (positions, pairs).

        Pairs are the neighbor edges on which inter-electrode delays will
        be tested for wave membership. For a planar grid these are the
        4-connected (up/down/left/right) and, optionally, the 4-connected
        diagonal neighbors.

        Args:
            use_diagonal (bool): include diagonal neighbors when True.

        Returns:
            positions (ndarray): (num_nodes, 2) electrode coordinates.
            pairs (ndarray): (n_pairs, 2) integer pairs of electrode
                indices that are neighbors.
        """
        rows, cols = self.num_splines, self.num_electrodes_per_spline
        positions = self.get_electrode_positions()

        offsets = [(0, 1), (1, 0)]
        if use_diagonal:
            offsets += [(1, 1), (1, -1)]

        pairs = []
        for r, c in product(range(rows), range(cols)):
            idx = r * cols + c
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    pairs.append((idx, rr * cols + cc))

        return positions, np.array(pairs)
