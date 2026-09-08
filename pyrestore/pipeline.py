"""``run_task`` — the single orchestration entry point for every task.

One function, two input signatures (single_scene | paired_scene). Nothing here knows the
semantics of any particular task: prompts, output fields, validation mappings, pair contracts and
spatial screening all come from the task definition. Every run exports the same product family:

``observations.csv`` (manifest or audited pairs), ``cv_features.csv``, ``vlm_scores.csv``,
``indicators.gpkg`` + ``indicators.csv`` (the analysis-ready dataset), ``quality_report.csv``,
``validation_report.csv`` and ``provenance.csv``.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .config import load_config
from .cv import extract_cv_batch
from .harmonise import (
    export_dataset,
    harmonise_paired,
    harmonise_single,
    quality_summary,
    to_geodataframe,
)
from .manifest import load_manifest
from .map import local_moran
from .map import make_map as render_map
from .pairs import audit_pairs, load_pairs, make_pairs
from .validate import validate_classification, validate_continuous
from .vlm import (
    apply_derived_fields,
    load_task,
    request_hash,
    score_vlm_batch,
    task_prompt_hash,
    validate_response_object,
)

_SPATIAL_COLUMNS = [
    "local_I",
    "lisa_q",
    "lisa_p",
    "lisa_p_adj",
    "lisa_label",
    "candidate_low_cluster",
    "candidate_high_cluster",
]

# A task's ``spatial.screening`` value selects which LISA quadrant is the relevant "candidate"
# for that task's own value_col — e.g. a low restorative-quality score or a high renewal-priority
# score. Unrecognised or missing values fall back to the historical Low-Low behaviour.
_SCREENING_CANDIDATE_COLUMNS = {
    "lisa_low_low": "candidate_low_cluster",
    "lisa_high_high": "candidate_high_cluster",
}


def _overall_validation_status(report: pd.DataFrame) -> str:
    """Collapse target-level validation states without overstating partial/empty comparisons."""
    if report.empty or "status" not in report.columns:
        return "not_validated"
    statuses = set(report["status"].dropna().astype(str))
    if statuses == {"validated"}:
        return "validated"
    if "validated" in statuses:
        return "partially_validated"
    if "no_overlap" in statuses:
        return "no_overlap"
    if statuses:
        return sorted(statuses)[0] if len(statuses) == 1 else "validation_failed"
    return "not_validated"


def _read_table(value: str | os.PathLike[str] | pd.DataFrame | None, what: str) -> pd.DataFrame | None:
    if value is None or isinstance(value, pd.DataFrame):
        return None if value is None else value.copy()
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return pd.read_csv(path)


def _id_alias(table: pd.DataFrame, id_col: str, what: str) -> pd.DataFrame:
    """Return a copy using the canonical id name accepted by analyzer-table contracts."""
    out = table.copy()
    if id_col not in out.columns and id_col == "capture_id" and "image" in out.columns:
        out = out.rename(columns={"image": "capture_id"})
    if id_col not in out.columns:
        raise ValueError(f"{what} must contain {id_col!r}.")
    if out[id_col].isna().any() or out[id_col].astype(str).str.strip().eq("").any():
        raise ValueError(f"{what} contains missing or empty {id_col!r} values.")
    out[id_col] = out[id_col].astype(str).str.strip()
    if out[id_col].duplicated().any():
        raise ValueError(f"{what} contains duplicate {id_col!r} values.")
    return out


def _validate_precomputed_cv(table: pd.DataFrame) -> pd.DataFrame:
    """Validate the common CV table contract before it can enter harmonisation."""
    out = _id_alias(table, "capture_id", "Precomputed CV features")
    required = {"cv_status", "cv_error"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Precomputed CV features is missing contract columns: {missing}")
    invalid_status = sorted(set(out["cv_status"].dropna().astype(str)) - {"ok", "error"})
    if invalid_status:
        raise ValueError(f"Precomputed CV features has invalid cv_status values: {invalid_status}")
    feature_cols = [
        column
        for column in out.columns
        if column not in {"capture_id", "cv_status", "cv_error"}
    ]
    if not feature_cols:
        raise ValueError("Precomputed CV features contains no feature columns.")
    bounded = [
        column
        for column in feature_cols
        if column.startswith("seg_") or column in {"GVI", "SVF", "enclosure"}
    ]
    ok = out["cv_status"].eq("ok")
    for column in bounded:
        numeric = pd.to_numeric(out.loc[ok, column], errors="coerce")
        if numeric.isna().any() or not numeric.between(0.0, 1.0).all():
            raise ValueError(
                f"Precomputed CV feature {column!r} must be finite and within [0, 1] on ok rows."
            )
        out.loc[ok, column] = numeric
    return out


def _decode_list_value(value: Any, field_name: str) -> Any:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        try:
            decoded = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Precomputed VLM list field {field_name!r} is not a JSON/Python list."
            ) from exc
    return decoded


def _validate_precomputed_vlm(
    table: pd.DataFrame, task: dict[str, Any], id_col: str
) -> pd.DataFrame:
    """Apply the same task output contract to offline/precomputed VLM records."""
    out = _id_alias(table, id_col, "Precomputed VLM scores")
    metadata = {"vlm_status", "vlm_error", "vlm_model", "prompt_hash"}
    missing_metadata = sorted(metadata - set(out.columns))
    if missing_metadata:
        raise ValueError(
            f"Precomputed VLM scores is missing contract columns: {missing_metadata}"
        )
    invalid_status = sorted(set(out["vlm_status"].dropna().astype(str)) - {"ok", "error"})
    if invalid_status:
        raise ValueError(f"Precomputed VLM scores has invalid vlm_status values: {invalid_status}")
    for column in ("vlm_model", "prompt_hash"):
        if out[column].isna().any() or out[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Precomputed VLM scores contains missing {column!r} values.")
        if out[column].astype(str).nunique() != 1:
            raise ValueError(f"Precomputed VLM scores contains multiple {column!r} values.")
    if "task_id" in out.columns:
        task_ids = set(out["task_id"].dropna().astype(str))
        if task_ids != {task["id"]}:
            raise ValueError(
                f"Precomputed VLM scores task_id values {sorted(task_ids)} do not match "
                f"task {task['id']!r}."
            )

    output_columns = {f"vlm_{name}" for name in task["output_fields"]}
    missing_outputs = sorted(output_columns - set(out.columns))
    if missing_outputs and out["vlm_status"].eq("ok").any():
        raise ValueError(
            f"Precomputed VLM scores is missing task output columns: {missing_outputs}"
        )

    for index in out.index[out["vlm_status"].eq("ok")]:
        raw: dict[str, Any] = {}
        for name, spec in task["output_fields"].items():
            value = out.at[index, f"vlm_{name}"]
            if spec["type"] == "list":
                value = _decode_list_value(value, name)
            raw[name] = value
        validated = validate_response_object(raw, task)
        for name, value in validated.items():
            out.at[index, f"vlm_{name}"] = value
        derived = apply_derived_fields(validated, task)
        for name, value in derived.items():
            column = f"vlm_{name}"
            if column in out.columns and pd.notna(out.at[index, column]):
                existing = float(out.at[index, column])
                if abs(existing - float(value)) > 1e-9:
                    raise ValueError(
                        f"Precomputed VLM derived field {column!r} does not match its inputs."
                    )
            else:
                out.loc[index, column] = value
    return out


def _manifest_from_pairs(pairs_ok: pd.DataFrame) -> pd.DataFrame:
    """Build a capture-level pseudo-manifest from a pair table so live CV extraction can run."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in pairs_ok.to_dict("records"):
        for leg in ("before", "after"):
            capture_id = str(record[f"{leg}_capture_id"])
            if capture_id in seen:
                continue
            seen.add(capture_id)
            rows.append(
                {
                    "site_id": record["site_id"],
                    "capture_id": capture_id,
                    "image_path": record[f"{leg}_image_path"],
                    "lon": record[f"{leg}_lon"],
                    "lat": record[f"{leg}_lat"],
                    "captured_at": record.get(f"{leg}_captured_at"),
                    "heading": None,
                }
            )
    return pd.DataFrame(rows)


def _portable_image_paths(table: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    """Write image paths relative to the product directory when possible.

    Internal analysis continues to use resolved absolute paths. Only shared artifacts are made
    relocatable, avoiding author-machine path leakage when the project tree is moved as a unit.
    """
    out = table.copy()
    path_columns = [column for column in out.columns if column.endswith("image_path")]
    for column in path_columns:
        out[column] = out[column].map(
            lambda value: (
                os.path.relpath(str(value), start=output_root)
                if pd.notna(value) and Path(str(value)).is_absolute()
                else value
            )
        )
    return out


def _attach_image_hashes(table: pd.DataFrame) -> pd.DataFrame:
    """Attach full SHA-256 input identities without re-reading repeated pair legs."""
    out = table.copy()
    cache: dict[str, str] = {}
    for column in [name for name in out.columns if name.endswith("image_path")]:
        prefix = column.removesuffix("image_path")
        hash_column = f"{prefix}image_sha256"

        def digest(value: object) -> str:
            path = str(value)
            if path not in cache:
                file_path = Path(path)
                if not file_path.is_file():
                    cache[path] = ""
                else:
                    hasher = hashlib.sha256()
                    with open(file_path, "rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            hasher.update(chunk)
                    cache[path] = hasher.hexdigest()
            return cache[path]

        out[hash_column] = out[column].map(digest)
    return out


def _resolve_source(
    source: str | os.PathLike[str] | pd.DataFrame,
    task: dict[str, Any],
    cfg: dict[str, Any],
    check_files: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return ``(pairs_ok, manifest_or_none)`` for a paired task.

    Accepts a pair table (its own CSV or DataFrame) or an image manifest; in the latter case
    candidate pairs are derived with the task's ``pair_contract``. Excluded candidates are counted
    and kept in the returned observations table as records.
    """
    contract = task.get("pair_contract") or {}
    if isinstance(source, pd.DataFrame) and "pair_id" in source.columns:
        pairs_table = load_pairs(source, check_files=check_files)
        pairs_table = audit_pairs(
            pairs_table,
            max_distance_m=float(contract.get("max_distance_m", 20.0)),
            max_heading_diff_deg=float(contract.get("max_heading_diff_deg", 30.0)),
            require_captured_at=bool(contract.get("require_captured_at", True)),
        )
        return pairs_table, None
    if not isinstance(source, pd.DataFrame):
        header = pd.read_csv(Path(source).expanduser(), nrows=0).columns
        if "pair_id" in header:
            pairs_table = load_pairs(source, check_files=check_files)
            pairs_table = audit_pairs(
                pairs_table,
                max_distance_m=float(contract.get("max_distance_m", 20.0)),
                max_heading_diff_deg=float(contract.get("max_heading_diff_deg", 30.0)),
                require_captured_at=bool(contract.get("require_captured_at", True)),
            )
            return pairs_table, None
    manifest = load_manifest(source, check_files=check_files)
    pairs_all = make_pairs(
        manifest,
        max_distance_m=float(contract.get("max_distance_m", 20.0)),
        max_heading_diff_deg=float(contract.get("max_heading_diff_deg", 30.0)),
        require_captured_at=bool(contract.get("require_captured_at", True)),
    )
    if pairs_all.empty:
        raise ValueError(
            "No candidate pairs could be derived from the manifest: sites need at least two captures."
        )
    return pairs_all, manifest


def run_task(
    source: str | os.PathLike[str] | pd.DataFrame,
    task: str | os.PathLike[str] | dict[str, Any],
    config: str | os.PathLike[str] | dict[str, Any] | None = None,
    *,
    cv_features: str | os.PathLike[str] | pd.DataFrame | None = None,
    vlm_scores: str | os.PathLike[str] | pd.DataFrame | None = None,
    labels: str | os.PathLike[str] | pd.DataFrame | None = None,
    reference: str | os.PathLike[str] | pd.DataFrame | None = None,
    check_files: bool = True,
    make_map: bool = True,
    out_dir: str | os.PathLike[str] | None = None,
    vlm_client: Any = None,
    cv_extractor: Any = None,
) -> dict[str, Any]:
    """Run a task into a staging directory, then publish a complete product family.

    A failed run leaves the previous successful output untouched. On success, files owned by the
    previous run but not produced this time (for example a skipped map) are removed using the run
    manifest, preventing stale products from being mistaken for current results.
    """
    task_def = load_task(task)
    cfg = load_config(config)
    final_output_root = Path(
        out_dir
        or Path(cfg["output"].get("dir", "outputs")).expanduser()
        / task_def["id"]
    ).expanduser().resolve()
    final_output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{final_output_root.name}.staging-", dir=final_output_root.parent
    ) as staging_dir:
        staging_root = Path(staging_dir)
        result = _run_task_impl(
            source,
            task_def,
            cfg,
            cv_features=cv_features,
            vlm_scores=vlm_scores,
            labels=labels,
            reference=reference,
            check_files=check_files,
            make_map=make_map,
            out_dir=staging_root,
            vlm_client=vlm_client,
            cv_extractor=cv_extractor,
        )
        _publish_staged_outputs(staging_root, final_output_root)
        result["paths"] = _remap_paths(result["paths"], staging_root, final_output_root)
        result["paths"]["run_manifest"] = str(final_output_root / "run_manifest.json")
        return result


def _publish_staged_outputs(staging_root: Path, final_output_root: Path) -> None:
    final_output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = final_output_root / "run_manifest.json"
    previous_files: set[str] = set()
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_files = {str(name) for name in previous.get("files", [])}
        except (json.JSONDecodeError, OSError, TypeError):
            previous_files = set()
    standard_files = {
        "observations.csv",
        "cv_features.csv",
        "vlm_scores.csv",
        "indicators.gpkg",
        "indicators.csv",
        "quality_report.csv",
        "validation_report.csv",
        "provenance.csv",
        "screening_map.html",
        "spatial_stats.csv",
    }
    staged_files = [path for path in staging_root.iterdir() if path.is_file()]
    for name in previous_files | standard_files:
        target = final_output_root / name
        if target.is_file():
            target.unlink()
    for path in staged_files:
        os.replace(path, final_output_root / path.name)
    manifest_payload = {
        "files": sorted(path.name for path in staged_files),
        "managed_by": "pyrestore",
    }
    temp_manifest = final_output_root / ".run_manifest.json.tmp"
    temp_manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_manifest, manifest_path)


def _remap_paths(value: Any, staging_root: Path, final_output_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _remap_paths(item, staging_root, final_output_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_remap_paths(item, staging_root, final_output_root) for item in value]
    if isinstance(value, str):
        path = Path(value)
        try:
            relative = path.relative_to(staging_root)
        except ValueError:
            return value
        return str(final_output_root / relative)
    return value


def _run_task_impl(
    source: str | os.PathLike[str] | pd.DataFrame,
    task: str | os.PathLike[str] | dict[str, Any],
    config: str | os.PathLike[str] | dict[str, Any] | None = None,
    *,
    cv_features: str | os.PathLike[str] | pd.DataFrame | None = None,
    vlm_scores: str | os.PathLike[str] | pd.DataFrame | None = None,
    labels: str | os.PathLike[str] | pd.DataFrame | None = None,
    reference: str | os.PathLike[str] | pd.DataFrame | None = None,
    check_files: bool = True,
    make_map: bool = True,
    out_dir: str | os.PathLike[str] | None = None,
    vlm_client: Any = None,
    cv_extractor: Any = None,
) -> dict[str, Any]:
    """Run one task end-to-end and export the full product family; return tables and paths.

    Precomputed ``cv_features`` / ``vlm_scores`` tables may be supplied for offline or reproduced
    runs (they must carry the analyzer status columns); otherwise the corresponding analyzer is
    executed live. ``vlm_client`` and ``cv_extractor`` may be injected for testing without API
    keys or torch. Validation runs only when a reference table is available or the task declares
    one; without a reference the report records ``not_validated``.
    """
    task_def = load_task(task)
    cfg = load_config(config)
    signature = task_def["input_signature"]
    output_root = Path(
        out_dir
        or Path(cfg["output"].get("dir", "outputs")).expanduser()
        / task_def["id"]
    )
    output_root.mkdir(parents=True, exist_ok=True)

    cv_table = _read_table(cv_features, "Precomputed CV features")
    vlm_table = _read_table(vlm_scores, "Precomputed VLM scores")
    label_table = _read_table(labels, "Labels")
    ref_table = _read_table(reference, "Validation reference")
    # Execution modes recorded in provenance: a precomputed table means the run reproduces an
    # earlier analysis; live means the analyzer actually executed during this run.
    cv_execution = (
        "precomputed" if cv_table is not None else "injected" if cv_extractor is not None else "live"
    )
    vlm_execution = (
        "precomputed" if vlm_table is not None else "injected" if vlm_client is not None else "live"
    )

    if cv_table is not None:
        cv_table = _validate_precomputed_cv(cv_table)
    if vlm_table is not None:
        vlm_id_col = "capture_id" if signature == "single_scene" else "pair_id"
        vlm_table = _validate_precomputed_vlm(vlm_table, task_def, vlm_id_col)

    if signature == "single_scene":
        manifest = load_manifest(source, check_files=check_files)
        if cv_table is None:
            cv_table = extract_cv_batch(manifest, cfg, extractor=cv_extractor)
        if vlm_table is None:
            vlm_table = score_vlm_batch(manifest, task_def, cfg, client=vlm_client)
        table = harmonise_single(manifest, cv_table, vlm_table, labels=label_table)
        observations = manifest
        excluded = 0
    else:
        pairs_all, manifest = _resolve_source(source, task_def, cfg, check_files)
        excluded = int((pairs_all["comparability_status"] != "ok").sum())
        pairs_ok = pairs_all[pairs_all["comparability_status"] == "ok"].reset_index(drop=True)
        if pairs_ok.empty:
            reasons = pairs_all["exclusion_reason"].value_counts().to_dict()
            raise ValueError(
                "No candidate pairs passed the pair contract — nothing to process. "
                f"Exclusion reasons: {reasons}. "
                "Most common cause: captured_at is missing (or identical) for some captures."
            )
        if cv_table is None:
            cv_source = manifest if manifest is not None else _manifest_from_pairs(pairs_ok)
            cv_table = extract_cv_batch(cv_source, cfg, extractor=cv_extractor)
        if vlm_table is None:
            vlm_table = score_vlm_batch(pairs_ok, task_def, cfg, client=vlm_client)
        delta_columns = (task_def.get("cv_change") or {}).get("delta_columns")
        table = harmonise_paired(
            pairs_ok, cv_table, vlm_table, labels=label_table, cv_delta_columns=delta_columns
        )
        observations = pairs_all

    # ------------------------------------------------------------- validation (optional)
    validation_status = "not_validated"
    validation_df: pd.DataFrame | None = None
    ref_source = ref_table
    if ref_source is None and (task_def.get("validation") or {}).get("reference"):
        declared = task_def["validation"]["reference"]
        if declared:
            ref_path = Path(str(declared)).expanduser()
            if not ref_path.is_absolute():
                ref_path = Path(task_def["_base_dir"]) / ref_path
            if ref_path.exists():
                ref_source = pd.read_csv(ref_path)
    if ref_source is not None:
        validation_cfg = task_def.get("validation") or {}
        if signature == "single_scene":
            mapping = validation_cfg.get("mapping")
            if not mapping:
                raise ValueError(
                    f"Task {task_def['id']!r} declares a reference but no validation.mapping."
                )
            validation_df = validate_continuous(
                table,
                ref_source,
                mapping,
                reference_type=validation_cfg.get("reference_type", "unspecified_reference"),
            )
        else:
            validation_df = validate_classification(
                table,
                ref_source,
                label_column=validation_cfg["label_column"],
                prediction_column=validation_cfg["prediction_column"],
                positive_value=validation_cfg["positive_value"],
                abstain_values=tuple(validation_cfg.get("abstain_values", ("uncertain",))),
                negative_values=tuple(validation_cfg.get("negative_values", ("no",))),
                reference_type=validation_cfg.get("reference_type", "unspecified_reference"),
            )
        validation_status = _overall_validation_status(validation_df)

    # ------------------------------------------------------------- spatial screening (single)
    spatial_note = ""
    spatial_cfg = task_def.get("spatial") or {}
    map_path: str | None = None
    stats_path: str | None = None
    if signature == "single_scene" and make_map and spatial_cfg.get("screening"):
        value_col = spatial_cfg.get("value_field")
        k = int(spatial_cfg.get("k", 8))
        if value_col not in table.columns:
            spatial_note = f"configured value column {value_col!r} not produced by this task"
        else:
            valid = table[value_col].notna() & table[["lon", "lat"]].notna().all(axis=1)
            if int(valid.sum()) <= k:
                spatial_note = (
                    f"too few valid {value_col!r} values ({int(valid.sum())}) for k={k} LISA screening"
                )
            else:
                gdf = to_geodataframe(table.loc[valid].copy())
                seed = int(spatial_cfg.get("seed", 42))
                p_threshold = float(spatial_cfg.get("p_threshold", 0.05))
                p_adjust = str(spatial_cfg.get("p_adjust", "fdr_bh"))
                try:
                    screened, global_i, global_p = local_moran(
                        gdf,
                        value_col,
                        k=k,
                        seed=seed,
                        p_threshold=p_threshold,
                        p_adjust=p_adjust,
                    )
                except ValueError as exc:
                    spatial_note = str(exc)
                    screened = None
                if screened is None:
                    pass
                else:
                    table = table.drop(columns=[c for c in _SPATIAL_COLUMNS if c in table.columns])
                    table = table.merge(
                        screened[["capture_id"] + _SPATIAL_COLUMNS],
                        on="capture_id",
                        how="left",
                        validate="many_to_one",
                    )
                    boundary = None
                    boundary_path = spatial_cfg.get("boundary_path")
                    if boundary_path:
                        import geopandas as gpd

                        boundary = gpd.read_file(
                            Path(boundary_path).expanduser(), layer=spatial_cfg.get("boundary_layer")
                        )
                    map_path = str(output_root / spatial_cfg.get("map_name", "screening_map.html"))
                    candidate_col = _SCREENING_CANDIDATE_COLUMNS.get(
                        str(spatial_cfg.get("screening")), "candidate_low_cluster"
                    )
                    render_map(
                        screened,
                        value_col,
                        map_path,
                        candidate_col=candidate_col,
                        boundary=boundary,
                        boundary_name=spatial_cfg.get("boundary_name", "Administrative boundary"),
                    )
                    stats_path = str(output_root / "spatial_stats.csv")
                    pd.DataFrame(
                        [
                            {
                                "value_col": value_col,
                                "n_valid": int(valid.sum()),
                                "k": k,
                                "p_adjust": p_adjust,
                                "global_moran_i": global_i,
                                "global_moran_p": global_p,
                                "screening": spatial_cfg.get("screening", "lisa_low_low"),
                                "candidate_col": candidate_col,
                                "n_candidate_low_clusters": int(
                                    screened["candidate_low_cluster"].sum()
                                ),
                                "n_candidate_high_clusters": int(
                                    screened["candidate_high_cluster"].sum()
                                ),
                            }
                        ]
                    ).to_csv(stats_path, index=False)

    # ------------------------------------------------------------- exports
    observations = _attach_image_hashes(observations)
    table = _attach_image_hashes(table)
    paths: dict[str, Any] = {}
    observations_path = output_root / "observations.csv"
    portable_observations = _portable_image_paths(observations, output_root)
    portable_observations.to_csv(observations_path, index=False)
    paths["observations"] = str(observations_path)

    if cv_table is not None:
        cv_path = output_root / "cv_features.csv"
        cv_table.to_csv(cv_path, index=False)
        paths["cv_features"] = str(cv_path)
    if vlm_table is not None:
        vlm_path = output_root / "vlm_scores.csv"
        vlm_table.to_csv(vlm_path, index=False)
        paths["vlm_scores"] = str(vlm_path)

    portable_table = _portable_image_paths(table, output_root)
    gdf = to_geodataframe(portable_table)
    dataset_paths = export_dataset(gdf, str(output_root), name="indicators")
    paths["datasets"] = dataset_paths

    quality_path = output_root / "quality_report.csv"
    quality_summary(table).to_csv(quality_path, index=False)
    paths["quality"] = str(quality_path)

    if validation_df is None:
        validation_df = pd.DataFrame([{"kind": "none", "status": "not_validated"}])
    validation_path = output_root / "validation_report.csv"
    validation_df.to_csv(validation_path, index=False)
    paths["validation"] = str(validation_path)

    vlm_status_col = "vlm_status" if "vlm_status" in table.columns else None
    cv_status_col = "cv_status" if "cv_status" in cv_table.columns else None
    current_task_hash = task_prompt_hash(task_def)
    current_request_hash = request_hash(task_def, cfg)
    if vlm_execution == "precomputed":
        source_vlm_model = str(vlm_table["vlm_model"].iloc[0])
        source_prompt_hash = str(vlm_table["prompt_hash"].iloc[0])
        precomputed_contract_match: bool | str = source_prompt_hash in {
            current_task_hash,
            current_request_hash,
        }
    else:
        source_vlm_model = str(cfg["vlm"].get("model", ""))
        source_prompt_hash = current_request_hash
        precomputed_contract_match = "not_applicable"
    provenance = pd.DataFrame(
        [
            {
                "task_id": task_def["id"],
                "task_version": task_def.get("version", ""),
                "input_signature": signature,
                "vlm_model": source_vlm_model,
                "prompt_hash": source_prompt_hash,
                "configured_vlm_model": cfg["vlm"].get("model", ""),
                "task_prompt_hash": current_task_hash,
                "request_hash": current_request_hash,
                "precomputed_contract_match": precomputed_contract_match,
                "cache_db": (
                    cfg["vlm"].get("cache_db", "")
                    if vlm_execution != "precomputed"
                    else ""
                ),
                "cv_model": cfg["cv"].get("segmenter", ""),
                "pyrestore_version": __version__,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_records": int(len(table)),
                "n_pairs_excluded": excluded,
                "cv_execution": cv_execution,
                "vlm_execution": vlm_execution,
                "remote_images_authorized": bool(cfg["vlm"].get("allow_remote_images", False)),
                "n_cv_success": int((cv_table[cv_status_col] == "ok").sum()) if cv_status_col else 0,
                "n_cv_error": int((cv_table[cv_status_col] == "error").sum()) if cv_status_col else 0,
                "n_vlm_success": int((table[vlm_status_col] == "ok").sum()) if vlm_status_col else 0,
                "n_vlm_error": int((table[vlm_status_col] == "error").sum()) if vlm_status_col else 0,
                "validation_status": validation_status,
                "spatial_note": spatial_note,
                "data_status": (cfg.get("provenance") or {}).get("data_status", "unspecified"),
                "run_note": (cfg.get("provenance") or {}).get("run_note", ""),
            }
        ]
    )
    provenance_path = output_root / "provenance.csv"
    provenance.to_csv(provenance_path, index=False)
    paths["provenance"] = str(provenance_path)
    if map_path:
        paths["map"] = map_path
    if stats_path:
        paths["spatial_stats"] = stats_path

    return {
        "task_id": task_def["id"],
        "input_signature": signature,
        "observations": observations,
        "cv_features": cv_table,
        "vlm_scores": vlm_table,
        "indicators": table,
        "quality": quality_summary(table),
        "provenance": provenance,
        "validation": validation_df,
        "validation_status": validation_status,
        "paths": paths,
    }
