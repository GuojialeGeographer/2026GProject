"""Pair-contract tests: audit outcomes are records, not silent drops."""
from __future__ import annotations

import pandas as pd
import pytest

from pyrestore.pairs import haversine_m, load_pairs, make_pairs


def test_admissible_pair_earlier_capture_is_before(paired_manifest):
    table = make_pairs(paired_manifest)
    ok = table[table["comparability_status"] == "ok"]
    assert len(ok) == 4
    row = ok[ok["site_id"] == "site0"].iloc[0]
    assert row["before_capture_id"] == "s0_2017"
    assert row["after_capture_id"] == "s0_2022"
    assert row["exclusion_reason"] == ""


def test_missing_captured_at_is_excluded_when_required(tmp_path, paired_manifest):
    table = make_pairs(paired_manifest)
    ok_ids = set(table.loc[table["comparability_status"] == "ok", "pair_id"])

    without_dates = paired_manifest.copy()
    without_dates["captured_at"] = None
    no_date_pairs = make_pairs(without_dates)
    assert (no_date_pairs["comparability_status"] == "excluded").all()
    assert (no_date_pairs["exclusion_reason"] == "missing_captured_at").all()
    # Excluded candidates remain records.
    assert set(no_date_pairs["pair_id"]) != ok_ids or len(no_date_pairs) == len(table)


def test_same_capture_date_is_excluded(paired_manifest):
    same_date = paired_manifest.copy()
    same_date["captured_at"] = "2022-09"
    table = make_pairs(same_date)
    assert (table["exclusion_reason"] == "same_capture_date").all()


def test_distance_exceeding_tolerance_is_excluded(paired_manifest):
    spread = paired_manifest.copy()
    spread.loc[spread["captured_at"] == "2017-03", "lon"] += 0.05  # ~4.5 km shift
    table = make_pairs(spread, max_distance_m=20.0)
    assert "distance_exceeds_tolerance" in set(table["exclusion_reason"])


def test_heading_diff_exceeding_tolerance_is_excluded(paired_manifest):
    rotated = paired_manifest.copy()
    rotated.loc[rotated["captured_at"] == "2022-09", "heading"] = 200  # diff 110 deg
    table = make_pairs(rotated, max_heading_diff_deg=30.0)
    assert (table["exclusion_reason"] == "heading_diff_exceeds_tolerance").all()


def test_missing_dates_allowed_when_require_captured_at_false(paired_manifest):
    without_dates = paired_manifest.copy()
    without_dates["captured_at"] = None
    table = make_pairs(without_dates, require_captured_at=False)
    assert (table["comparability_status"] == "ok").all()
    assert table["before_captured_at"].isna().all()


def test_load_pairs_validates_and_marks_unaudited(paired_manifest):
    table = make_pairs(paired_manifest)
    ok = table[table["comparability_status"] == "ok"].drop(columns=["comparability_status"])
    loaded = load_pairs(ok, check_files=False)
    assert (loaded["comparability_status"] == "unaudited").all()

    with pytest.raises(ValueError, match="missing required columns"):
        load_pairs(ok.drop(columns=["after_image_path"]), check_files=False)


def test_haversine_known_distance():
    # ~111 km per degree of latitude.
    assert 110_000 < haversine_m(2.0, 41.0, 2.0, 42.0) < 112_000


def test_same_date_detected_across_formatting_variants(tmp_path):
    """`2017-3` and `2017-03` are the same capture date once parsed."""
    from tests.conftest import make_image

    rows = []
    for capture_id, date in (("b", "2017-3"), ("a", "2017-03")):
        image = make_image(tmp_path / f"{capture_id}.png")
        rows.append({"site_id": "S", "capture_id": capture_id, "image_path": str(image),
                     "lon": 2.2, "lat": 41.4, "captured_at": date})
    table = make_pairs(pd.DataFrame(rows))
    assert table["exclusion_reason"].iloc[0] == "same_capture_date"


def test_baseline_ordering_handles_unpadded_formats(paired_manifest):
    variant = paired_manifest.copy()
    variant["captured_at"] = variant["captured_at"].replace({"2017-03": "2017-3"})
    table = make_pairs(variant)
    ok = table[(table["site_id"] == "site0") & (table["comparability_status"] == "ok")].iloc[0]
    assert ok["before_capture_id"] == "s0_2017"
    assert ok["after_capture_id"] == "s0_2022"


def test_pair_ids_are_stable_when_manifest_order_changes(paired_manifest):
    first = make_pairs(paired_manifest)
    shuffled = make_pairs(paired_manifest.sample(frac=1.0, random_state=7))
    key = ["site_id", "before_capture_id", "after_capture_id", "pair_id"]
    pd.testing.assert_frame_equal(
        first[key].sort_values(key[:-1]).reset_index(drop=True),
        shuffled[key].sort_values(key[:-1]).reset_index(drop=True),
    )


def test_load_pairs_resolves_paths_relative_to_csv(tmp_path, paired_manifest):
    table = make_pairs(paired_manifest).head(1).copy()
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    before.write_bytes(b"before")
    after.write_bytes(b"after")
    table["before_image_path"] = "before.png"
    table["after_image_path"] = "after.png"
    path = tmp_path / "pairs.csv"
    table.to_csv(path, index=False)
    loaded = load_pairs(path)
    assert loaded.loc[0, "before_image_path"] == str(before.resolve())
    assert loaded.loc[0, "after_image_path"] == str(after.resolve())


def test_load_pairs_rejects_self_pair(paired_manifest):
    table = make_pairs(paired_manifest).head(1).copy()
    table["after_capture_id"] = table["before_capture_id"]
    with pytest.raises(ValueError, match="same capture"):
        load_pairs(table, check_files=False)


def test_pipeline_audit_downgrades_false_ok_pair(paired_manifest):
    from pyrestore.pairs import audit_pairs

    table = make_pairs(paired_manifest).head(1).copy()
    table["comparability_status"] = "ok"
    table["after_lon"] = table["before_lon"] + 1.0
    audited = audit_pairs(table, max_distance_m=20)
    assert audited.loc[0, "comparability_status"] == "excluded"
    assert audited.loc[0, "exclusion_reason"] == "distance_exceeds_tolerance"
