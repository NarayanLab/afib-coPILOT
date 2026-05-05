"""Render the pre-computed coPILOT output for an example basket segment.

> **Note**: the wLEAP map shown here is pre-computed and included in the
> example data file. It is not derived from the EGM signals at runtime:
> wave segmentation (Step 1) and direction aggregation (Step 3) are not
> implemented in this pseudo-code release. The visualization illustrates
> the expected output format of the full system.

Usage:
    python demo.py pt_0068
    python demo.py pt_0469

Data must be placed under ``data/`` before running. Download it with::

    python download_data.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CASES = {
    "pt_0068": {"file": "pt_0068_example_segment.json", "central": "F4"},
    "pt_0469": {"file": "pt_0469_example_segment.json", "central": "H7"},
}


def render(case_id: str, segment: dict, out_path: Path):
    labels = segment["electrode_positions"]
    central = segment["central_electrode"]
    sq = segment.get("signal_quality", {})
    wleap_map = np.array(segment["wleap_map"])  # (3, 3)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    vmax = float(np.nanmax(np.abs(wleap_map)))
    im = ax.imshow(wleap_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    plt.colorbar(im, ax=ax, label="wLEAP score (ms)", fraction=0.046, pad=0.04)

    for idx, label in enumerate(labels):
        row, col = idx // 3, idx % 3
        text = f"[{label}]" if label == central else label
        if not sq.get(label, True):
            text = f"{text}*"
        ax.text(col, row, text, ha="center", va="center",
                fontsize=11, fontweight="bold", color="black")

    flat_idx = int(np.nanargmax(wleap_map))
    target_row, target_col = flat_idx // 3, flat_idx % 3
    cx, cy = 1.0, 1.0
    dx, dy = target_col - cx, target_row - cy
    if dx != 0 or dy != 0:
        scale = 0.45 / max(abs(dx), abs(dy))
        ax.annotate(
            "",
            xy=(cx + dx * scale, cy + dy * scale),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", color="black", lw=2.5),
            zorder=4,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{case_id}  —  central electrode {central}\n(pre-computed output)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"figure saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Render pre-computed coPILOT output for one example segment.")
    parser.add_argument("case", choices=sorted(CASES.keys()))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    args = parser.parse_args()

    json_path = args.data_dir / CASES[args.case]["file"]
    if not json_path.exists():
        print(f"Data file not found: {json_path}")
        print("Run  python download_data.py  first.")
        sys.exit(1)

    segment = json.loads(json_path.read_text())

    print(
        "\nNote: the wLEAP map below is pre-computed and embedded in the example "
        "data file.\nIt is not derived from the EGM signals at runtime — wave "
        "segmentation (Step 1)\nand direction aggregation (Step 3) are not "
        "implemented in this pseudo-code release.\n"
    )
    print(f"=== {args.case} — central electrode {CASES[args.case]['central']} ===")
    wleap = np.array(segment["wleap_map"])
    print("wLEAP map (ms, positive = leads peers):")
    for row in wleap:
        print("  " + "  ".join(f"{v:+6.1f}" for v in row))

    args.out.mkdir(parents=True, exist_ok=True)
    render(args.case, segment, args.out / f"{args.case}_direction.png")


if __name__ == "__main__":
    main()
