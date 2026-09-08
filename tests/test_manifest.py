"""Manifest contract tests."""
from __future__ import annotations

import pandas as pd
import pytest

from pyrestore.manifest import (
    index_images_by_coord,
    load_manifest,
    manifest_from_directory,
    parse_lonlat,
    resolve_image_path,
)
from tests.conftest import make_image


def test_load_valid_manifest_adds_optional_columns(manifest):
    table = load_manifest(manifest, check_files=False)
    assert len(table) == 6
    for column in ("captured_at", "heading", "source", "licence"):
        assert column in table.columns


def test_legacy_four_column_manifest_is_adapted(manifest):
    legacy = manifest.rename(columns={"capture_id": "image"}).drop(columns=["site_id"])
    table = load_manifest(legacy, check_files=False)
    assert "site_id" in table.columns and "capture_id" in table.columns
    assert (table["capture_id"] == table["site_id"]).all()
    assert set(table["capture_id"]) == set(manifest["capture_id"])


def test_missing_required_columns_raises(manifest):
    broken = manifest.drop(columns=["lon"])
    with pytest.raises(ValueError, match="missing required columns"):
        load_manifest(broken, check_files=False)


def test_duplicate_capture_id_raises(manifest):
    duplicated = pd.concat([manifest, manifest.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate capture_id"):
        load_manifest(duplicated, check_files=False)


def test_invalid_coordinates_raise(manifest):
    bad = manifest.copy()
    bad.loc[0, "lon"] = 999.0
    with pytest.raises(ValueError, match="outside valid longitude"):
        load_manifest(bad, check_files=False)


def test_missing_image_files_raise(manifest):
    import os

    os.remove(manifest.loc[0, "image_path"])
    with pytest.raises(FileNotFoundError, match="missing image"):
        load_manifest(manifest, check_files=True)


def test_parse_lonlat_roundtrip():
    lon, lat = parse_lonlat("201601_114.0055_22.6300.png")
    assert (lon, lat) == (114.0055, 22.63)
    with pytest.raises(ValueError):
        parse_lonlat("no-coordinates-here.png")


def test_manifest_directory_and_coordinate_index_helpers(tmp_path):
    image = make_image(tmp_path / "2022_2.12345_41.54321.png")
    table = manifest_from_directory(tmp_path)
    assert len(table) == 1
    assert table.loc[0, "lon"] == pytest.approx(2.12345)
    index = index_images_by_coord(str(tmp_path))
    assert resolve_image_path(image.name, index) == str(image)
    assert resolve_image_path("invalid.png", index) is None


def test_manifest_directory_rejects_unparseable_images(tmp_path):
    make_image(tmp_path / "arbitrary-name.png")
    with pytest.raises(ValueError, match="Could not parse coordinates"):
        manifest_from_directory(tmp_path)
