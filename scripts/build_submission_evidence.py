"""Build the frozen submission evidence from verified local case-study tables.

This script replays the historical Shenzhen PRS scores through the current PyRestore contract and
recomputes spatial statistics with projected KNN weights and FDR-adjusted local pseudo-p values.
It also creates the equivalent spatial-statistics product for the newly completed six-dimension
run. Cartographic rendering remains a QGIS step; this script writes GIS-ready GeoPackages only.

This is a one-time data-preparation script kept for provenance; it reads from the original,
unshipped source workspace (not part of this repository) and its output already ships under
`data/shenzhen_prs11/`. It is not part of the reproducible pipeline a fresh clone needs to run.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from pyrestore import load_config, run_task
from pyrestore.harmonise import to_geodataframe
from pyrestore.map import local_moran

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parent
PRS_SOURCE = WORKSPACE / "restorative_quality_source" / "outputs" / "shenzhen_indicators.csv"
SHENZHEN_SIXDIM = ROOT / "data" / "shenzhen_sixdim"
PRS_DATA = ROOT / "data" / "shenzhen_prs11"
SUBMISSION_OUTPUT = ROOT / "outputs" / "submission"
SOURCE_MODEL = "gpt-5.4-mini"
SOURCE_PROMPT_HASH = "35128c8cfbadece8"


def _prepare_prs_package() -> None:
    source = pd.read_csv(PRS_SOURCE)
    manifest = pd.read_csv(SHENZHEN_SIXDIM / "manifest.csv")
    capture_ids = set(manifest["capture_id"].astype(str))
    source = source[source["image"].astype(str).isin(capture_ids)].copy()
    if len(source) != len(manifest):
        raise ValueError(
            f"PRS source/manifest mismatch: source={len(source)}, manifest={len(manifest)}"
        )

    PRS_DATA.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SHENZHEN_SIXDIM / "manifest.csv", PRS_DATA / "manifest.csv")
    shutil.copy2(SHENZHEN_SIXDIM / "cv_features.csv", PRS_DATA / "cv_features.csv")

    vlm = source[
        [
            "image",
            "vlm_Being-away",
            "vlm_Coherence",
            "vlm_Scope",
            "vlm_Fascination",
            "vlm_restorative",
        ]
    ].rename(
        columns={
            "image": "capture_id",
            "vlm_Being-away": "vlm_being_away",
            "vlm_Coherence": "vlm_coherence",
            "vlm_Scope": "vlm_scope",
            "vlm_Fascination": "vlm_fascination",
            "vlm_restorative": "vlm_restorative_average",
        }
    )
    vlm["vlm_model"] = SOURCE_MODEL
    vlm["prompt_hash"] = SOURCE_PROMPT_HASH
    vlm["cache_hit"] = pd.NA
    vlm["vlm_status"] = "ok"
    vlm["vlm_error"] = ""
    vlm.to_csv(PRS_DATA / "vlm_scores.csv", index=False)

    reference = source[
        [
            "image",
            "human_Being-away",
            "human_Coherence",
            "human_Scope",
            "human_Fascination",
            "human_Average",
        ]
    ].rename(
        columns={
            "image": "capture_id",
            "human_Being-away": "Being-away",
            "human_Coherence": "Coherence",
            "human_Scope": "Scope",
            "human_Fascination": "Fascination",
            "human_Average": "Average",
        }
    )
    reference.to_csv(PRS_DATA / "reference.csv", index=False)


def _run_prs() -> dict:
    cfg = load_config(
        {
            "vlm": {
                "model": SOURCE_MODEL,
                "offline": True,
                "allow_remote_images": False,
                "cache_db": "",
            },
            "provenance": {
                "data_status": "real Shenzhen case; historical PyRestore VLM scores replayed",
                "run_note": (
                    "465 matched human PRS-11 records; source prompt hash retained and "
                    "current-task contract mismatch exposed"
                ),
            },
        }
    )
    return run_task(
        PRS_DATA / "manifest.csv",
        ROOT / "tasks" / "prs11.yaml",
        cfg,
        cv_features=PRS_DATA / "cv_features.csv",
        vlm_scores=PRS_DATA / "vlm_scores.csv",
        reference=PRS_DATA / "reference.csv",
        make_map=False,
        out_dir=SUBMISSION_OUTPUT / "restorative_quality_prs11",
    )


def _spatial_product(case_name: str, value_column: str, k: int = 8) -> dict[str, object]:
    case_dir = SUBMISSION_OUTPUT / case_name
    table = pd.read_csv(case_dir / "indicators.csv")
    valid = table[value_column].notna() & table[["lon", "lat"]].notna().all(axis=1)
    gdf = to_geodataframe(table.loc[valid].copy())
    screened, global_i, global_p = local_moran(
        gdf,
        value_column,
        k=k,
        seed=42,
        p_threshold=0.05,
        p_adjust="fdr_bh",
    )
    screened.to_file(case_dir / "spatial_screening.gpkg", driver="GPKG", layer="screening")
    stats = {
        "case": case_name,
        "value_column": value_column,
        "n": int(len(screened)),
        "k": k,
        "permutations": 999,
        "local_p": "two-sided permutation pseudo-p",
        "multiple_testing": "Benjamini-Hochberg FDR",
        "global_moran_i": float(global_i),
        "global_moran_p": float(global_p),
        "n_candidate_low_clusters": int(screened["candidate_low_cluster"].sum()),
    }
    pd.DataFrame([stats]).to_csv(case_dir / "spatial_stats.csv", index=False)
    return stats


def main() -> int:
    _prepare_prs_package()
    prs_result = _run_prs()
    prs_stats = _spatial_product(
        "restorative_quality_prs11", "vlm_restorative_average"
    )
    sixdim_stats = _spatial_product(
        "urban_perception_sixdim", "vlm_beautiful"
    )
    print("PRS validation status", prs_result["validation_status"])
    print("PRS spatial", prs_stats)
    print("sixdim spatial", sixdim_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
