"""End-to-end pipeline tests: the full product family with fake analyzers and offline tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pyrestore.pipeline import run_task
from pyrestore.vlm import load_task, request_hash, task_prompt_hash
from tests.conftest import FakeVLMClient, make_image, prs11_reply, renewal_reply


def _grid_manifest(tmp_path: Path, n: int = 12) -> pd.DataFrame:
    """12 captures on a 4x3 grid so LISA screening has enough valid points."""
    rows = []
    for i in range(n):
        image = make_image(tmp_path / f"g{i}.png", color=(30 + 10 * i, 120, 200))
        rows.append(
            {
                "site_id": f"g{i}",
                "capture_id": f"g{i}",
                "image_path": str(image),
                "lon": 2.150 + (i % 4) * 0.004,
                "lat": 41.390 + (i // 4) * 0.004,
                "captured_at": "2022-11",
                "heading": 0,
            }
        )
    return pd.DataFrame(rows)


def test_run_task_end_to_end_with_fakes(tmp_path, manifest, prs11_task_file, fake_cv_extractor, offline_cfg):
    client = FakeVLMClient(prs11_reply())
    cfg = {**offline_cfg, "vlm": {**offline_cfg["vlm"], "offline": False}}
    result = run_task(
        manifest, prs11_task_file, cfg,
        cv_extractor=fake_cv_extractor, vlm_client=client, make_map=False,
        out_dir=tmp_path / "single-live",
    )
    assert result["validation_status"] == "not_validated"
    table = result["indicators"]
    assert len(table) == 6
    assert (table["cv_status"] == "ok").all() and (table["vlm_status"] == "ok").all()
    assert table["vlm_being_away"].iloc[0] == 0.4
    assert table["vlm_restorative_average"].iloc[0] == pytest.approx(0.45)
    assert not any(c.startswith("lisa_") for c in table.columns)

    for key in ("observations", "cv_features", "vlm_scores", "datasets", "quality", "validation", "provenance"):
        assert key in result["paths"]
    for path in result["paths"]["datasets"]:
        assert Path(path).exists()
    exported = pd.read_csv(result["paths"]["datasets"][1])
    assert not Path(exported.loc[0, "image_path"]).is_absolute()
    assert len(exported.loc[0, "image_sha256"]) == 64
    provenance = result["provenance"].iloc[0]
    assert provenance["prompt_hash"] == request_hash(load_task(prs11_task_file), cfg)
    assert provenance["task_prompt_hash"] == task_prompt_hash(load_task(prs11_task_file))
    assert provenance["cv_execution"] == "injected" and provenance["vlm_execution"] == "injected"
    assert provenance["n_cv_error"] == 0


def test_run_task_offline_with_validation_and_map(tmp_path, offline_cfg, prs11_task_file):
    manifest = _grid_manifest(tmp_path)
    reference_values = [0.1 + 0.05 * i for i in range(12)]
    average_values = [(value + 0.001 + 1.5) / 4 for value in reference_values]
    ref = pd.DataFrame(
        {
            "capture_id": [f"g{i}" for i in range(12)],
            "Being-away": reference_values,
            "Coherence": [0.5] * 12,
            "Scope": [0.5] * 12,
            "Fascination": [0.5] * 12,
            "Average": average_values,
        }
    )
    vlm = pd.DataFrame(
        {
            "capture_id": [f"g{i}" for i in range(12)],
            "vlm_being_away": [v + 0.001 for v in reference_values],
            "vlm_coherence": [0.5] * 12,
            "vlm_scope": [0.5] * 12,
            "vlm_fascination": [0.5] * 12,
            "vlm_restorative_average": average_values,
            "vlm_status": "ok",
            "vlm_error": "",
            "vlm_model": "recorded-model",
            "prompt_hash": "recorded-hash",
        }
    )
    cv = pd.DataFrame(
        [{"capture_id": f"g{i}", "GVI": 0.3, "SVF": 0.2, "cv_status": "ok", "cv_error": ""} for i in range(12)]
    )
    result = run_task(
        manifest, prs11_task_file, offline_cfg,
        cv_features=cv, vlm_scores=vlm, reference=ref,
        out_dir=tmp_path / "single-validated",
    )
    assert result["validation_status"] == "partially_validated"
    report = result["validation"]
    being_away_row = report[report["target"] == "vlm_being_away"].iloc[0]
    assert being_away_row["n"] == 12
    assert being_away_row["pearson_r"] > 0.99
    assert being_away_row["status"] == "validated"
    assert (report[report["target"] == "vlm_coherence"]["status"] == "constant_input").all()

    assert "lisa_label" in result["indicators"].columns
    assert Path(result["paths"]["map"]).exists()
    stats = pd.read_csv(result["paths"]["spatial_stats"])
    assert "global_moran_i" in stats.columns


def test_run_task_high_high_screening_selects_candidate_high_cluster(
    tmp_path, offline_cfg, renewal_priority_task_file
):
    """A task declaring ``screening: lisa_high_high`` (renewal-priority style) must be screened
    on the High-High quadrant, not the historical Low-Low default used by prs11-style tasks."""
    manifest = _grid_manifest(tmp_path)
    # First 4 captures (one grid row) form a tight high-value cluster; the rest are low/flat, so
    # a High-High -- not a Low-Low -- cluster is the one with real signal to detect.
    values = [0.9, 0.92, 0.88, 0.91] + [0.1] * 8
    vlm = pd.DataFrame(
        {
            "capture_id": [f"g{i}" for i in range(12)],
            "vlm_disrepair": values,
            "vlm_neglect": values,
            "vlm_informality": values,
            "vlm_visual_disorder": values,
            "vlm_renewal_priority_average": values,
            "vlm_evidence": ["fixture evidence"] * 12,
            "vlm_status": "ok",
            "vlm_error": "",
            "vlm_model": "recorded-model",
            "prompt_hash": "recorded-hash",
        }
    )
    cv = pd.DataFrame(
        [{"capture_id": f"g{i}", "GVI": 0.3, "SVF": 0.2, "cv_status": "ok", "cv_error": ""} for i in range(12)]
    )
    result = run_task(
        manifest, renewal_priority_task_file, offline_cfg,
        cv_features=cv, vlm_scores=vlm,
        out_dir=tmp_path / "single-high-high",
    )
    table = result["indicators"]
    assert "candidate_high_cluster" in table.columns
    assert "candidate_low_cluster" in table.columns
    # With n=12 and Benjamini-Hochberg correction across 12 simultaneous local tests, the small
    # injected cluster is not guaranteed to survive FDR adjustment (the same reason
    # test_run_task_offline_with_validation_and_map does not assert on candidate_low_cluster
    # either) -- but the quadrant classification itself (High-High for the four hot captures) is
    # a deterministic, non-statistical fact this test can rely on.
    assert (table.loc[:3, "lisa_q"] == 1).any(), "at least one clustered high value should read High-High"
    assert (table.loc[4:, "lisa_q"] != 1).all(), "the low/flat captures should not read as High-High"

    stats = pd.read_csv(result["paths"]["spatial_stats"])
    assert stats.loc[0, "screening"] == "lisa_high_high"
    assert stats.loc[0, "candidate_col"] == "candidate_high_cluster"
    assert "n_candidate_high_clusters" in stats.columns
    assert "n_candidate_low_clusters" in stats.columns


def test_run_task_paired_end_to_end(tmp_path, paired_manifest, renewal_task_file, fake_cv_extractor, offline_cfg):
    client = FakeVLMClient(renewal_reply(detected="yes"))
    cfg = {**offline_cfg, "vlm": {**offline_cfg["vlm"], "offline": False}}
    result = run_task(
        paired_manifest, renewal_task_file, cfg,
        cv_extractor=fake_cv_extractor, vlm_client=client, make_map=False,
        out_dir=tmp_path / "paired-live",
    )
    table = result["indicators"]
    assert len(table) == 4
    assert (table["vlm_renewal_detected"] == "yes").all()
    assert {"GVI_before", "GVI_after", "GVI_delta"}.issubset(table.columns)
    assert (table["GVI_delta"] == 0.0).all()
    assert result["provenance"].iloc[0]["n_pairs_excluded"] == 0
    assert result["provenance"].iloc[0]["input_signature"] == "paired_scene"
    assert result["provenance"].iloc[0]["n_cv_success"] == 8


def test_run_task_paired_counts_excluded_candidates(
    tmp_path, paired_manifest, renewal_task_file, fake_cv_extractor, offline_cfg
):
    # One site loses its dates -> its candidate pair is excluded but remains a record.
    without_dates = paired_manifest.copy()
    without_dates.loc[without_dates["site_id"] == "site0", "captured_at"] = None
    client = FakeVLMClient(renewal_reply(detected="no"))
    cfg = {**offline_cfg, "vlm": {**offline_cfg["vlm"], "offline": False}}
    result = run_task(
        without_dates, renewal_task_file, cfg,
        cv_extractor=fake_cv_extractor, vlm_client=client, make_map=False,
        out_dir=tmp_path / "paired-excluded",
    )
    assert result["provenance"].iloc[0]["n_pairs_excluded"] == 1
    assert len(result["indicators"]) == 3
    assert len(result["observations"]) == 4  # excluded candidate kept as a record
    assert (result["observations"]["comparability_status"] == "excluded").sum() == 1


def test_run_task_paired_from_pairs_csv(tmp_path, paired_manifest, renewal_task_file, fake_cv_extractor, offline_cfg):
    from pyrestore.pairs import make_pairs

    pairs_path = tmp_path / "pairs.csv"
    make_pairs(paired_manifest).to_csv(pairs_path, index=False)
    client = FakeVLMClient(renewal_reply(detected="uncertain"))
    cfg = {**offline_cfg, "vlm": {**offline_cfg["vlm"], "offline": False}}
    result = run_task(
        pairs_path, renewal_task_file, cfg,
        cv_extractor=fake_cv_extractor, vlm_client=client, make_map=False,
        out_dir=tmp_path / "paired-csv",
    )
    assert (result["indicators"]["vlm_renewal_detected"] == "uncertain").all()
    assert result["provenance"].iloc[0]["vlm_execution"] == "injected"


def test_run_task_paired_empty_contract_error_is_actionable(
    tmp_path, paired_manifest, renewal_task_file, offline_cfg
):
    """When nothing passes the pair contract the error names the exclusion reasons."""
    no_dates = paired_manifest.copy()
    no_dates["captured_at"] = None
    with pytest.raises(ValueError, match="Exclusion reasons"):
        run_task(
            no_dates,
            renewal_task_file,
            offline_cfg,
            make_map=False,
            out_dir=tmp_path / "paired-empty",
        )


def test_run_task_provenance_marks_precomputed_execution(tmp_path, offline_cfg, prs11_task_file):
    manifest = _grid_manifest(tmp_path)
    cv = pd.DataFrame(
        [{"capture_id": f"g{i}", "GVI": 0.3, "cv_status": "ok", "cv_error": ""} for i in range(12)]
    )
    vlm = pd.DataFrame([
        {
            "capture_id": f"g{i}",
            "vlm_being_away": 0.4,
            "vlm_coherence": 0.5,
            "vlm_scope": 0.6,
            "vlm_fascination": 0.3,
            "vlm_restorative_average": 0.45,
            "vlm_status": "ok",
            "vlm_error": "",
            "vlm_model": "actual-source-model",
            "prompt_hash": "actual-source-hash",
        }
        for i in range(12)
    ])
    result = run_task(
        manifest, prs11_task_file, offline_cfg,
        cv_features=cv, vlm_scores=vlm, make_map=False,
        out_dir=tmp_path / "single-precomputed",
    )
    provenance = result["provenance"].iloc[0]
    assert provenance["cv_execution"] == "precomputed"
    assert provenance["vlm_execution"] == "precomputed"
    assert provenance["vlm_model"] == "actual-source-model"
    assert provenance["prompt_hash"] == "actual-source-hash"
    assert provenance["configured_vlm_model"] == "fake-model"
    assert not provenance["precomputed_contract_match"]


def test_run_task_rejects_incomplete_precomputed_vlm(
    tmp_path, offline_cfg, prs11_task_file
):
    manifest = _grid_manifest(tmp_path, n=2)
    cv = pd.DataFrame([
        {"capture_id": f"g{i}", "GVI": 0.3, "cv_status": "ok", "cv_error": ""}
        for i in range(2)
    ])
    incomplete = pd.DataFrame([
        {
            "capture_id": f"g{i}",
            "vlm_being_away": 0.4,
            "vlm_status": "ok",
            "vlm_error": "",
            "vlm_model": "source-model",
            "prompt_hash": "source-hash",
        }
        for i in range(2)
    ])
    output = tmp_path / "out"
    output.mkdir()
    old_provenance = output / "provenance.csv"
    old_provenance.write_text("previous successful run\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing task output columns"):
        run_task(
            manifest,
            prs11_task_file,
            offline_cfg,
            cv_features=cv,
            vlm_scores=incomplete,
            make_map=False,
            out_dir=output,
        )
    assert old_provenance.read_text(encoding="utf-8") == "previous successful run\n"


def test_successful_rerun_removes_stale_managed_map(
    tmp_path, offline_cfg, prs11_task_file
):
    manifest = _grid_manifest(tmp_path)
    cv = pd.DataFrame([
        {"capture_id": f"g{i}", "GVI": 0.3, "cv_status": "ok", "cv_error": ""}
        for i in range(12)
    ])
    values = [0.1 + i * 0.05 for i in range(12)]
    vlm = pd.DataFrame([
        {
            "capture_id": f"g{i}",
            "vlm_being_away": value,
            "vlm_coherence": 0.5,
            "vlm_scope": 0.6,
            "vlm_fascination": 0.3,
            "vlm_restorative_average": (value + 1.4) / 4,
            "vlm_status": "ok",
            "vlm_error": "",
            "vlm_model": "source-model",
            "prompt_hash": "source-hash",
        }
        for i, value in enumerate(values)
    ])
    output = tmp_path / "rerun"
    first = run_task(
        manifest,
        prs11_task_file,
        offline_cfg,
        cv_features=cv,
        vlm_scores=vlm,
        out_dir=output,
    )
    assert Path(first["paths"]["map"]).exists()
    second = run_task(
        manifest,
        prs11_task_file,
        offline_cfg,
        cv_features=cv,
        vlm_scores=vlm,
        make_map=False,
        out_dir=output,
    )
    assert not (output / "screening_map.html").exists()
    assert Path(second["paths"]["run_manifest"]).exists()
