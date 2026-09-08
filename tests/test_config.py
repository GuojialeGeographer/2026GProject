"""Runtime configuration merge and validation tests."""
from __future__ import annotations

import pytest

from pyrestore.config import load_config


def test_nested_overrides_preserve_other_defaults():
    cfg = load_config({"vlm": {"model": "test-model", "max_image_px": 512}})
    assert cfg["vlm"]["model"] == "test-model"
    assert cfg["vlm"]["max_image_px"] == 512
    assert cfg["vlm"]["jpeg_quality"] == 85


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"vlm": {"max_retries": 0}}, "max_retries"),
        ({"vlm": {"max_image_px": -1}}, "max_image_px"),
        ({"vlm": {"jpeg_quality": 100}}, "jpeg_quality"),
        ({"vlm": {"allow_remote_images": "yes"}}, "allow_remote_images"),
    ],
)
def test_invalid_runtime_values_are_rejected(override, message):
    with pytest.raises(ValueError, match=message):
        load_config(override)
