"""Harmonise analyzer outputs into analysis-ready geo-referenced tables.

The manifest (or pair table) is the left-hand data contract: every input record stays visible
even when one analyzer failed — missing values and ``*_status='error'`` rows are part of the
product, not noise to drop (framework invariant). CV columns keep their names/family and VLM
columns keep the ``vlm_`` prefix; the two families are never merged into one type of evidence.

``harmonise_single`` serves ``single_scene`` tasks; ``harmonise_paired`` serves ``paired_scene``
tasks, renaming the shared CV columns per leg (``*_before`` / ``*_after``) and optionally adding
per-leg differences (``*_delta``).
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import pandas as pd


def _check_ids(table: pd.DataFrame, name: str, id_col: str) -> pd.DataFrame:
    """Return a copy with a clean id column; raise on duplicates.

    PyRestore-derived tables are keyed by the legacy ``image`` column; for ``capture_id`` joins
    that column is accepted as an alias and renamed, so precomputed case-study tables feed the
    pipeline without a preprocessing step.
    """
    if id_col not in table.columns:
        if id_col == "capture_id" and "image" in table.columns:
            table = table.rename(columns={"image": "capture_id"})
        elif not isinstance(table.index, pd.RangeIndex) and table.index.name == id_col:
            table = table.reset_index()
        else:
            raise ValueError(
                f"{name} must contain a {id_col!r} column "
                "(the legacy PyRestore 'image' column is accepted as an alias)"
            )
    if table[id_col].isna().any():
        raise ValueError(f"{name} contains missing {id_col!r} values")
    table = table.copy()
    table[id_col] = table[id_col].astype(str)
    if table[id_col].duplicated().any():
        raise ValueError(f"{name} contains duplicate {id_col!r} values")
    return table


def harmonise_single(
    manifest: pd.DataFrame,
    cv_features: pd.DataFrame,
    vlm_scores: pd.DataFrame,
    *,
    labels: pd.DataFrame | None = None,
    id_col: str = "capture_id",
    label_prefix: str = "human_",
) -> pd.DataFrame:
    """Join manifest + CV + VLM (+ optional labels) by capture id; manifest rows are all kept.

    Human label columns are prefixed with ``label_prefix`` so reference data is never confused
    with extracted indicators.
    """
    required = {id_col, "image_path", "lon", "lat"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest must contain columns {sorted(required)}")

    manifest_table = _check_ids(manifest, "manifest", id_col)
    cv_table = _check_ids(cv_features, "cv_features", id_col)
    vlm_table = _check_ids(vlm_scores, "vlm_scores", id_col)

    table = manifest_table.merge(cv_table, on=id_col, how="left", validate="one_to_one")
    table = table.merge(vlm_table, on=id_col, how="left", validate="one_to_one")
    if labels is not None:
        label_table = _check_ids(labels, "labels", id_col)
        rename = {
            column: f"{label_prefix}{column}"
            for column in label_table.columns
            if column != id_col and not column.startswith(label_prefix)
        }
        label_table = label_table.rename(columns=rename)
        table = table.merge(label_table, on=id_col, how="left", validate="one_to_one")
    return table


def harmonise_paired(
    pairs: pd.DataFrame,
    cv_features: pd.DataFrame | None,
    vlm_scores: pd.DataFrame,
    *,
    labels: pd.DataFrame | None = None,
    id_col: str = "pair_id",
    label_prefix: str = "human_",
    cv_delta_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Join pairs + CV (both legs) + VLM pair scores (+ optional labels) by pair id.

    ``cv_features`` is one table keyed by capture id; its columns are attached twice, suffixed
    ``_before`` / ``_after``. ``cv_delta_columns`` names CV columns for which a ``_delta``
    (after − before) column is added. Geometry columns ``lon``/``lat`` are taken from the
    baseline leg.
    """
    required = {
        id_col,
        "site_id",
        "before_capture_id",
        "after_capture_id",
        "before_lon",
        "before_lat",
    }
    if not required.issubset(pairs.columns):
        raise ValueError(f"Pairs table must contain columns {sorted(required)}")

    pairs_table = _check_ids(pairs, "pairs", id_col)
    table = pairs_table.copy()
    table["lon"] = pd.to_numeric(table["before_lon"], errors="coerce")
    table["lat"] = pd.to_numeric(table["before_lat"], errors="coerce")

    if cv_features is not None and len(cv_features) > 0:
        cv = _check_ids(cv_features, "cv_features", "capture_id")
        value_cols = [c for c in cv.columns if c != "capture_id"]
        before = cv.rename(columns={c: f"{c}_before" for c in value_cols}).rename(
            columns={"capture_id": "before_capture_id"}
        )
        after = cv.rename(columns={c: f"{c}_after" for c in value_cols}).rename(
            columns={"capture_id": "after_capture_id"}
        )
        table = table.merge(before, on="before_capture_id", how="left", validate="many_to_one")
        table = table.merge(after, on="after_capture_id", how="left", validate="many_to_one")
        if cv_delta_columns:
            for column in cv_delta_columns:
                b, a = f"{column}_before", f"{column}_after"
                if b in table.columns and a in table.columns:
                    table[f"{column}_delta"] = pd.to_numeric(table[a], errors="coerce") - pd.to_numeric(
                        table[b], errors="coerce"
                    )

    vlm_table = _check_ids(vlm_scores, "vlm_scores", id_col)
    table = table.merge(vlm_table, on=id_col, how="left", validate="one_to_one")

    if labels is not None:
        label_table = _check_ids(labels, "labels", id_col)
        rename = {
            column: f"{label_prefix}{column}"
            for column in label_table.columns
            if column != id_col and not column.startswith(label_prefix)
        }
        label_table = label_table.rename(columns=rename)
        table = table.merge(label_table, on=id_col, how="left", validate="one_to_one")
    return table


def quality_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row processing summary for a harmonised table.

    Counts valid coordinates, per-analyzer success/failure (any ``*_status`` column) and the
    number of rows carrying human labels — failures stay visible in the product.
    """
    summary: dict[str, Any] = {"n_input_rows": int(len(table))}
    if {"lon", "lat"}.issubset(table.columns):
        summary["n_valid_coordinates"] = int(table[["lon", "lat"]].notna().all(axis=1).sum())
    status_columns = []
    for column in table.columns:
        if column in {"cv_status", "vlm_status"}:
            status_columns.append((column, column.removesuffix("_status")))
        elif column.startswith("cv_status_"):
            status_columns.append((column, f"cv_{column.removeprefix('cv_status_')}"))
        elif column.startswith("vlm_status_"):
            status_columns.append((column, f"vlm_{column.removeprefix('vlm_status_')}"))
    for status_col, name in sorted(status_columns):
        values = table[status_col]
        summary[f"n_{name}_success"] = int((values == "ok").sum())
        summary[f"n_{name}_error"] = int((values == "error").sum())
    human_columns = [c for c in table.columns if c.startswith("human_")]
    if human_columns:
        summary["n_with_human_labels"] = int(table[human_columns].notna().any(axis=1).sum())
    return pd.DataFrame([summary])


def to_geodataframe(table: pd.DataFrame) -> Any:
    """Convert a harmonised table to a GeoDataFrame (EPSG:4326 points from lon/lat)."""
    import geopandas as gpd
    from shapely.geometry import Point

    geometry = [Point(lon, lat) for lon, lat in zip(table["lon"], table["lat"], strict=True)]
    return gpd.GeoDataFrame(table.copy(), geometry=geometry, crs="EPSG:4326")


def export_dataset(
    gdf: Any,
    out_dir: str,
    name: str = "indicators",
    formats: tuple[str, ...] = ("gpkg", "csv"),
) -> list[str]:
    """Write the analysis-ready dataset to disk (GeoPackage + CSV); return written paths.

    GeoPackage = GIS-ready geometry + attributes (QGIS reads it directly); CSV = flat table with
    lon/lat kept and geometry dropped.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []
    if "gpkg" in formats:
        p = os.path.join(out_dir, f"{name}.gpkg")
        gdf.to_file(p, driver="GPKG", layer=name)
        paths.append(p)
    if "csv" in formats:
        p = os.path.join(out_dir, f"{name}.csv")
        gdf.drop(columns="geometry").to_csv(p, index=False)
        paths.append(p)
    return paths
