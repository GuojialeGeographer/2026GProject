"""Cache tests: JSON responses, task-scoped keys, scene-level content addressing."""
from __future__ import annotations

import sqlite3

import pytest

from pyrestore.cache import Cache, scene_cache_key
from tests.conftest import make_image


def test_put_get_roundtrip(tmp_path):
    cache = Cache(str(tmp_path / "cache.sqlite"))
    response = {"being_away": 0.5, "renewal_types": ["greening_landscape"]}
    cache.put("key1", "task_a", "fake-model", "hash1", response, "raw text")
    assert cache.get("key1", "task_a", "fake-model", "hash1") == response
    cache.close()


def test_cache_context_manager_closes_connection(tmp_path):
    with Cache(str(tmp_path / "cache.sqlite")) as cache:
        cache.put("key", "task", "model", "hash", {"v": 1}, "raw")
    with pytest.raises(sqlite3.ProgrammingError):
        cache.get("key", "task", "model", "hash")


def test_task_and_prompt_scope_prevent_collisions(tmp_path):
    cache = Cache(str(tmp_path / "cache.sqlite"))
    cache.put("key1", "task_a", "fake-model", "hash1", {"v": 1}, "raw")
    assert cache.get("key1", "task_b", "fake-model", "hash1") is None
    assert cache.get("key1", "task_a", "other-model", "hash1") is None
    assert cache.get("key1", "task_a", "fake-model", "hash2") is None


def test_scene_cache_key_covers_every_image(tmp_path):
    """A pair's cache key must change when the follow-up image changes, not only the baseline."""
    baseline = make_image(tmp_path / "base.png", color=(10, 10, 10))
    followup_x = make_image(tmp_path / "x.png", color=(100, 10, 10))
    followup_y = make_image(tmp_path / "y.png", color=(200, 10, 10))
    baseline_twin = make_image(tmp_path / "twin.png", color=(10, 10, 10))

    key_bx = scene_cache_key([str(baseline), str(followup_x)])
    key_by = scene_cache_key([str(baseline), str(followup_y)])
    assert key_bx != key_by, "pairs sharing a baseline must not collide"

    # Same content + same explicit label → same key, regardless of file name.
    assert scene_cache_key([str(baseline)], label="k") == scene_cache_key(
        [str(baseline_twin)], label="k"
    )
    with pytest.raises(ValueError):
        scene_cache_key([])
