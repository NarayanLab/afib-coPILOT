# afib-tracking-copilot

This repository provides high-level pseudo-code for **coPILOT**, the AI
system described in *AI-Based Tracking of Fibrillatory Propagation in
the Heart*. 

### What's included

- **Pipeline wrapper**: stitches together wave segmentation, per-wave
  LEAP scoring, and aggregation to a predominant compass direction.
- **CLI entry point**: shows how to invoke the pipeline on a single
  recording.
- **Per-step modules**: one Python module per step
  (Estimation of AF Waves; Assigning Activation Gradients for
  AF Waves; Aggregating Local Activation Gradients over Time), with
  readable pseudo-code for the algorithmic steps described in the paper
  and minimal placeholders for catheter-specific geometry.

### Key files

- `run_analysis.py`: minimal CLI to run coPILOT on a single recording;
  loads inputs, calls `stage_coPILOT`, and saves per-segment directions.
- `demo.py`: renders the pre-computed wLEAP map for an example basket
  segment as a labelled 3×3 grid with a compass-direction arrow.
- `download_data.py`: fetches the two example EGM segment JSON files from
  a public S3 bucket into `data/`.
- `copilot_wrapper.py`: orchestrates the pipeline over a sequence of
  4 s segments: wave segmentation → per-wave LEAP → predominant
  compass direction.
- `catheter.py`: minimal `GridCatheter` placeholder; provides electrode
  coordinates and neighbor pairs for a regular planar grid.
- `step_waves.py`: Step 1. AI-Based Estimation of AF Waves. Analyzes
  short windows (~one AF cycle) from catheter footprints, grouping
  activations into candidate AF waves by requiring time delays (Δt)
  between electrodes with known spacing (d) to be consistent with a
  minimum conduction velocity (Δt < d / CV<sub>min</sub>). Near-
  simultaneous pairs are retained as they may reflect orthogonal
  propagation. Connected components of the activation graph define the
  waves.
- `step_wleap.py`: Step 2. Assigning Activation Gradients for AF Waves.
  Computes local gradients within each estimated wave using a relative-
  timing metric: the Leading Earliest Activity Potential (LEAP) score
  (activation time minus the mean activation time of the other
  electrodes in the local field). Electrodes are ranked from earliest
  to latest by LEAP. This rank-based formulation reduces sensitivity to
  outlying pairwise delays and variations in inter-electrode spacing.
- `step_direction.py`: Step 3. Aggregating Local Activation Gradients
  over Time. Estimates the predominant direction across successive AF
  waves by aggregating local activation gradients within each local
  domain using a 4 s window, smoothing fluctuations from transient
  asynchronous activity. LEAP rankings are combined across waves to
  assign a single predominant direction displayed on the corresponding
  catheter edge. Only the generic rectangular-grid masks are exposed in
  this release.

### Inputs

The pipeline assumes that an activation matrix is already available:

- shape: `(num_electrodes, T_samples)`
- dtype: binary (1 marks an activation sample, 0 otherwise)
- typical sampling: 1 kHz

This matrix is produced by an upstream AI annotator. Pseudo-code for that annotator is published as a separate
repository:

> https://github.com/NarayanLab/afib-tissue-activation

### Notes and limitations

- Several functions are placeholders. In particular, only a generic
  rectangular grid catheter is exposed here.
- Trained-model artefacts and proprietary preprocessing are **not**
  included; they live in the upstream coMAP annotator.
- This repository is intended for educational/demo purposes only.

### System requirements

Hardware requirements
- A standard computer with enough RAM to hold NumPy arrays for one
  recording (a few hundred MB is typical at 1 kHz over a few minutes
  for a 16-electrode grid).
- No non-standard hardware is required.

OS requirements
- macOS, Linux, Windows. Example environments:
  - macOS: Ventura (13) or later
  - Linux: Ubuntu 20.04 or later
  - Windows: Windows 10 or 11

Python
- Python 3.12+ recommended.

Python dependencies
- numpy (≥1.24)
- scipy (≥1.10)
- matplotlib (≥3.7)

Note: this repository provides pseudo-code; end-to-end execution
requires the upstream activation-time annotator (coMAP) and the
catheter geometry library that are not included here.

### Installation guide

Important
- This repository is pseudo-code; there is nothing to install to run
  the demo. Use the web app instead.

If you still want a local environment for reading/experimentation:

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Typical install time
- ~5 minutes on a normal desktop with a stable internet connection.

### Example

Two example EGM segments from basket recordings are provided to
illustrate the input format and expected output. Each JSON file contains
the raw electrograms and a pre-computed wLEAP map. The `output/` folder
contains the corresponding direction images.

> **Note**: the pre-computed wLEAP maps were produced by the full system.
> Wave segmentation (Step 1) and direction aggregation (Step 3) are not
> implemented in this pseudo-code release, so the demo renders the
> pre-computed result rather than deriving it from the EGM signals.

**Step 1 — download data** (one-time, requires internet access):

```bash
python download_data.py
```

**Step 2 — render the output**:

```bash
python demo.py pt_0068  # basket recording, central electrode F4
python demo.py pt_0469  # basket recording, central electrode H7
```

Expected output
- Stdout: the 3×3 wLEAP map (ms) and a note on what is pre-computed.
- `output/<case>_direction.png`: 3×3 electrode grid coloured by wLEAP
  score (red = early, blue = late), with an arrow pointing toward the
  earliest-activating electrode.

Expected run time
- A few seconds on a standard laptop.

**Run on your own activation matrix**:

- Prepare inputs:
  - `activations.npy`: binary activation matrix from the coMAP
    annotator.
  - `catheter_config.npz` with keys `num_splines`,
    `num_electrodes_per_spline`, `electrode_spacing_mm`.
- Run:

```bash
python run_analysis.py \
  --activations_path path/to/activations.npy \
  --catheter_config_path path/to/catheter_config.npz \
  --directions_path path/to/directions.npy \
  --fs 1000 \
  --segment_seconds 4 \
  --min_CV 200
```

Expected output
- A `.npy` file containing a list of per-segment direction dicts:
  `{ "angle": int | None, "score": float | None }`.

Expected run time
- A few seconds, depending on recording length and number of
  electrodes.

### Instructions for use

Adapting locally (advanced; pseudo-code)
- Annotate raw electrograms to obtain the binary activation matrix.
- Provide a catheter geometry compatible with `catheter.GridCatheter`
  (or extend it with new electrode layouts and direction masks; see
  `step_direction.grid_masks`).
- Call `copilot_wrapper.stage_coPILOT` with the activation matrix and
  the list of segment start times.

### Reproduction instructions (optional)

Not applicable
- Full reproduction is not possible from this repository because the
  upstream activation annotator and the proprietary catheter geometry
  library are not included.

### License

This repository is licensed under the Apache License, Version 2.0. See
`LICENSE` for details.
