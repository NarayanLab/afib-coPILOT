"""Download example segment data for the demo.

Files are fetched from a public S3 bucket and saved to ``data/``.
No AWS credentials are required.

Usage:
    python download_data.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

BASE_URL = "https://copilot-demo-data.s3.us-west-1.amazonaws.com"

FILES = [
    "pt_0068_example_segment.json",
    "pt_0469_example_segment.json",
]

DATA_DIR = Path(__file__).resolve().parent / "data"


def download():
    DATA_DIR.mkdir(exist_ok=True)
    for filename in FILES:
        dest = DATA_DIR / filename
        if dest.exists():
            print(f"already present: {dest.name}")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"downloading {filename} …", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        print("done")
    print(f"\ndata saved to {DATA_DIR}")


if __name__ == "__main__":
    download()
