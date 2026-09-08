"""Prepare the local demo subset from the original Barcelona sample workspace.

Copies 12 sample panoramas, a manifest and the corresponding *derived* CV/VLM tables so the
framework demo notebook and the `pyrestore.run_task` walkthrough execute fully offline. Raw imagery stays
local (gitignored): the Barcelona source download supplied no redistribution licence, so the
repository ships derived tables and this regeneration script instead.

This is a one-time data-preparation script kept for provenance; it reads from the original,
unshipped source workspace (not part of this repository) and its output already ships under
`data/demo/`. It is not part of the reproducible pipeline a fresh clone needs to run.

Usage (from the repository root):
    .venv/bin/python scripts/prepare_demo.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SAMPLE_DIR = ROOT.parent / "barcelona_sample_source" / "data" / "demo" / "barcelona_sample"
DEFAULT_DERIVED_DIR = ROOT.parent / "barcelona_sample_source" / "outputs" / "barcelona"
N_IMAGES = 12


def main() -> int:
    sample_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SAMPLE_DIR
    derived_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DERIVED_DIR

    manifest_src = sample_dir / "manifest.csv"
    cv_src = derived_dir / "cv_features.csv"
    vlm_src = derived_dir / "vlm_scores.csv"
    missing = [str(p) for p in (manifest_src, cv_src, vlm_src) if not p.exists()]
    if missing:
        print("Required source files not found (run from the GProject workspace):")
        for p in missing:
            print("  -", p)
        return 1

    import pandas as pd

    demo_dir = ROOT / "data" / "demo"
    images_dir = demo_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_src)
    cv = pd.read_csv(cv_src)
    vlm = pd.read_csv(vlm_src)

    subset = manifest.head(N_IMAGES).copy()
    copied = 0
    for row in subset.itertuples():
        src = sample_dir / row.image_path
        if not src.exists():
            print(f"missing image, skipping {row.image}: {src}")
            subset = subset[subset.image != row.image]
            continue
        shutil.copy2(src, images_dir / Path(row.image_path).name)
        copied += 1
    subset["image_path"] = subset["image_path"].map(lambda p: f"images/{Path(p).name}")
    subset.to_csv(demo_dir / "manifest.csv", index=False)

    ids = set(subset["image"])
    # The source derived tables are keyed by the legacy `image` column and use the old
    # ART dimension names; rename both so the demo tables match the current prs11 task contract.
    vlm_rename = {
        "image": "capture_id",
        "vlm_Being-away": "vlm_being_away",
        "vlm_Coherence": "vlm_coherence",
        "vlm_Scope": "vlm_scope",
        "vlm_Fascination": "vlm_fascination",
        "vlm_restorative": "vlm_restorative_average",
    }
    cv_rename = {"image": "capture_id"}
    cv[cv["image"].isin(ids)].rename(columns=cv_rename).to_csv(
        demo_dir / "cv_features.csv", index=False
    )
    vlm[vlm["image"].isin(ids)].rename(columns=vlm_rename).to_csv(
        demo_dir / "vlm_scores.csv", index=False
    )

    print(f"demo subset ready: {copied} images -> {demo_dir}")
    print("notebook 01 and `pyrestore.run_task(..., cv_features=..., vlm_scores=...)` now work offline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
