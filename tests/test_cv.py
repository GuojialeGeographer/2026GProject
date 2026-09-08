"""CV extractor tests (injected fake extractor — no torch)."""
from __future__ import annotations

import pandas as pd
import pytest

from pyrestore.cv import derive_from_precomputed, extract_cv_batch, validate_against


def test_batch_with_fake_extractor(manifest, fake_cv_extractor, offline_cfg):
    table = extract_cv_batch(manifest, offline_cfg, extractor=fake_cv_extractor)
    assert (table["cv_status"] == "ok").all()
    assert (table["GVI"] == 0.30).all()
    assert {"seg_vegetation", "seg_sky", "Hue_Mean"}.issubset(table.columns)


def test_failed_image_recorded_not_fatal(manifest, offline_cfg):
    def failing_extractor(image_path, cfg):
        if image_path.endswith("img_2.png"):
            raise FileNotFoundError("unreadable")
        return {"GVI": 0.3, "SVF": 0.2, "enclosure": 0.1}

    table = extract_cv_batch(manifest, offline_cfg, extractor=failing_extractor)
    assert len(table) == 6
    assert table.loc[table["capture_id"] == "c2", "cv_status"].iloc[0] == "error"
    assert "unreadable" in table.loc[table["capture_id"] == "c2", "cv_error"].iloc[0]
    assert (table.loc[table["capture_id"] != "c2", "cv_status"] == "ok").all()


def test_missing_capture_id_raises(manifest, offline_cfg):
    with pytest.raises(ValueError, match="must contain columns"):
        extract_cv_batch(manifest.drop(columns=["capture_id"]), offline_cfg)


def test_precomputed_derivation_and_cross_model_validation(offline_cfg):
    row = pd.Series(
        {
            "seg_vegetation": 0.3,
            "seg_sky": 0.2,
            "seg_building": 0.1,
            "seg_wall": 0.02,
            "seg_fence": 0.01,
            "Canny_Edges": 20.0,
        }
    )
    derived = derive_from_precomputed(row, offline_cfg)
    assert derived["GVI"] == 0.3
    assert derived["enclosure"] == pytest.approx(0.13)

    extracted = pd.DataFrame(
        {"capture_id": ["a", "b"], "seg_vegetation": [0.2, 0.4]}
    )
    reference = pd.DataFrame(
        {"capture_id": ["a", "b"], "seg_vegetation": [0.1, 0.5]}
    )
    report = validate_against(extracted, reference)
    assert report.loc[0, "feature"] == "seg_vegetation"
    assert report.loc[0, "n"] == 2
