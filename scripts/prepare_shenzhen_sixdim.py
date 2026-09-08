"""Prepare the real Shenzhen six-dimension validation package from local source data.

The source label table contains 566 complete six-dimension ratings on a 0-5 scale. Image names in
that table and the local image export use slightly different coordinate precision, so records are
matched through rounded coordinates (five decimal places), following the established case-study
procedure. Raw images are never copied into the repository.

This is a one-time data-preparation script kept for provenance; it reads from an unshipped local
source workspace (not part of this repository) and its output already ships under
`data/shenzhen_sixdim/`. It is not part of the reproducible pipeline a fresh clone needs to run --
notebook 03 reads the shipped tables directly.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from pyrestore import load_config
from pyrestore.cv import derive_from_precomputed
from pyrestore.manifest import index_images_by_coord, parse_lonlat, resolve_image_path

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parent
LABELS_PATH = WORKSPACE / "VLM_restorative-quality" / "ground_truth.csv"
IMAGES_DIR = WORKSPACE / "dataset"
OUTPUT_DIR = ROOT / "data" / "shenzhen_sixdim"
DIMENSIONS = ["safe", "lively", "beautiful", "wealthy", "boring", "depressing"]
PRECISION = 5


def main() -> int:
    if not LABELS_PATH.is_file():
        raise FileNotFoundError(f"Shenzhen reference table not found: {LABELS_PATH}")
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Shenzhen image directory not found: {IMAGES_DIR}")

    labels = pd.read_csv(LABELS_PATH)
    missing = [column for column in ["image", *DIMENSIONS] if column not in labels.columns]
    if missing:
        raise ValueError(f"Reference table is missing columns: {missing}")
    numeric_labels = labels[DIMENSIONS].apply(pd.to_numeric, errors="coerce")
    if numeric_labels.isna().any().any() or not numeric_labels.stack().between(0, 5).all():
        raise ValueError("Six-dimension reference values must be complete and within [0, 5].")

    coordinate_index = index_images_by_coord(str(IMAGES_DIR), precision=PRECISION)
    matched_rows: list[dict[str, object]] = []
    unmatched_ids: list[str] = []
    for source_index, row in labels.iterrows():
        capture_id = str(row["image"])
        image_path = resolve_image_path(capture_id, coordinate_index, precision=PRECISION)
        if image_path is None:
            unmatched_ids.append(capture_id)
            continue
        lon, lat = parse_lonlat(capture_id)
        matched_rows.append(
            {
                "source_index": int(source_index),
                "site_id": capture_id,
                "capture_id": capture_id,
                "image_path": str(Path(image_path).resolve()),
                "lon": lon,
                "lat": lat,
                "source": "Ma and Kwan (2026) local author image export",
                "licence": "restricted-local; raw image not redistributed",
            }
        )

    matched = pd.DataFrame(matched_rows)
    if matched.empty:
        raise ValueError("No Shenzhen labels matched the local image collection.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = matched.drop(columns="source_index").copy()
    manifest["image_path"] = manifest["image_path"].map(
        lambda value: os.path.relpath(value, start=OUTPUT_DIR)
    )
    manifest.to_csv(OUTPUT_DIR / "manifest.csv", index=False)

    reference = matched[["source_index", "capture_id"]].merge(
        labels.reset_index(names="source_index")[["source_index", *DIMENSIONS]],
        on="source_index",
        validate="one_to_one",
    )
    reference[DIMENSIONS] = reference[DIMENSIONS].astype(float) / 5.0
    reference.drop(columns="source_index").to_csv(OUTPUT_DIR / "reference.csv", index=False)

    cfg = load_config()
    cv_rows: list[dict[str, object]] = []
    for record in matched.to_dict("records"):
        source_row = labels.iloc[int(record["source_index"])]
        features = derive_from_precomputed(source_row, cfg)
        cv_rows.append(
            {
                "capture_id": record["capture_id"],
                **features,
                "cv_status": "ok",
                "cv_error": "",
            }
        )
    pd.DataFrame(cv_rows).to_csv(OUTPUT_DIR / "cv_features.csv", index=False)

    pd.DataFrame(
        [
            {
                "n_reference_rows": int(len(labels)),
                "n_complete_sixdim_rows": int(numeric_labels.notna().all(axis=1).sum()),
                "n_local_images_indexed": int(len(coordinate_index)),
                "coordinate_precision_decimals": PRECISION,
                "n_matched": int(len(matched)),
                "n_unmatched": int(len(unmatched_ids)),
                "reference_scale_original": "0-5",
                "reference_scale_exported": "0-1",
            }
        ]
    ).to_csv(OUTPUT_DIR / "matching_report.csv", index=False)
    pd.DataFrame({"capture_id": unmatched_ids}).to_csv(
        OUTPUT_DIR / "unmatched_capture_ids.csv", index=False
    )

    print(
        f"prepared Shenzhen sixdim package: {len(matched)}/{len(labels)} labels matched "
        f"at {PRECISION} decimal places -> {OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
