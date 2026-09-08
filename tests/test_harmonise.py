"""Harmonisation tests: manifest-left joins, paired suffixes, deltas, exports."""
from __future__ import annotations

import pandas as pd

from pyrestore.harmonise import (
    export_dataset,
    harmonise_paired,
    harmonise_single,
    quality_summary,
    to_geodataframe,
)
from pyrestore.pairs import make_pairs


def _cv_table(manifest):
    return pd.DataFrame(
        [{"capture_id": c, "GVI": 0.3, "SVF": 0.2, "cv_status": "ok", "cv_error": ""} for c in manifest["capture_id"]]
    )


def _vlm_table(manifest):
    return pd.DataFrame(
        [{"capture_id": c, "vlm_being_away": 0.4, "vlm_status": "ok", "vlm_error": ""} for c in manifest["capture_id"]]
    )


def test_legacy_image_column_alias_for_precomputed_tables(manifest):
    """PyRestore-derived tables are keyed by `image`; they must join without preprocessing."""
    cv = _cv_table(manifest).rename(columns={"capture_id": "image"})
    vlm = _vlm_table(manifest).rename(columns={"capture_id": "image"})
    table = harmonise_single(manifest, cv, vlm)
    assert len(table) == 6
    assert "GVI" in table.columns and "vlm_being_away" in table.columns
    assert (table["cv_status"] == "ok").all()


def test_harmonise_single_keeps_failed_rows(manifest):
    cv = _cv_table(manifest).head(5)  # one record has no CV result
    vlm = _vlm_table(manifest).head(4)  # two records have no VLM result
    table = harmonise_single(manifest, cv, vlm)
    assert len(table) == 6
    assert table["cv_status"].isna().sum() == 1
    assert table["vlm_status"].isna().sum() == 2
    assert "vlm_being_away" in table.columns and "GVI" in table.columns
    assert not any(c.endswith("_delta") for c in table.columns)  # single-scene has no temporal fields


def test_labels_get_human_prefix(manifest):
    labels = pd.DataFrame({"capture_id": manifest["capture_id"], "Being-away": 0.5})
    table = harmonise_single(manifest, _cv_table(manifest), _vlm_table(manifest), labels=labels)
    assert "human_Being-away" in table.columns
    assert "Being-away" not in table.columns


def test_harmonise_paired_with_deltas(paired_manifest):
    pairs = make_pairs(paired_manifest)
    ok = pairs[pairs["comparability_status"] == "ok"]
    vlm = pd.DataFrame(
        [{"pair_id": p, "vlm_renewal_detected": "yes", "vlm_status": "ok", "vlm_error": ""} for p in ok["pair_id"]]
    )
    table = harmonise_paired(ok, _cv_table(paired_manifest), vlm, cv_delta_columns=("GVI", "SVF"))
    assert len(table) == 4
    assert {"GVI_before", "GVI_after", "GVI_delta"}.issubset(table.columns)
    assert (table["GVI_delta"] == 0.0).all()
    assert table["lon"].notna().all()
    assert "vlm_renewal_detected" in table.columns


def test_quality_summary_and_export(manifest, tmp_path):
    table = harmonise_single(manifest, _cv_table(manifest), _vlm_table(manifest))
    summary = quality_summary(table)
    assert summary["n_input_rows"].iloc[0] == 6
    assert summary["n_cv_success"].iloc[0] == 6
    assert summary["n_vlm_success"].iloc[0] == 6

    gdf = to_geodataframe(table)
    paths = export_dataset(gdf, str(tmp_path / "out"), name="indicators")
    assert len(paths) == 2
    import geopandas as gpd

    roundtrip = gpd.read_file(paths[0])
    assert len(roundtrip) == 6
    csv = pd.read_csv(paths[1])
    assert "geometry" not in csv.columns and "lon" in csv.columns
