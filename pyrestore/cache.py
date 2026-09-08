"""Content-addressed SQLite cache of VLM responses.

One row per (image content, task, model, prompt). The response is stored as JSON so any task's
output shape fits the same table; the cache must never encode the semantics of one particular
task. An earlier, restorative-quality-only version of this cache hardcoded the four ART columns
as REALs, and storing JSON instead is what makes the pipeline task-agnostic.

Content addressing (hashing the image file itself) prevents two different files that happen to
share an id from returning each other's cached response.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3


class Cache:
    """SQLite-backed cache of VLM responses keyed by content, task, model and prompt."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        if os.path.dirname(db_path):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vlm_response ("
            "cache_key TEXT NOT NULL, task_id TEXT NOT NULL, model TEXT NOT NULL, "
            "prompt_hash TEXT NOT NULL, response_json TEXT NOT NULL, raw_text TEXT NOT NULL, "
            "created_at TEXT NOT NULL, "
            "PRIMARY KEY (cache_key, task_id, model, prompt_hash))"
        )
        self._conn.commit()

    def get(self, cache_key: str, task_id: str, model: str, prompt_hash: str) -> dict | None:
        """Return the cached response dict for this exact key combination, or None."""
        cursor = self._conn.execute(
            "SELECT response_json FROM vlm_response "
            "WHERE cache_key=? AND task_id=? AND model=? AND prompt_hash=?",
            (cache_key, task_id, model, prompt_hash),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put(
        self,
        cache_key: str,
        task_id: str,
        model: str,
        prompt_hash: str,
        response: dict,
        raw: str,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO vlm_response "
            "(cache_key, task_id, model, prompt_hash, response_json, raw_text, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
            (cache_key, task_id, model, prompt_hash, json.dumps(response), raw),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Cache:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def scene_cache_key(
    image_paths: list[str], *, content_addressed: bool = True, label: str | None = None
) -> str:
    """Return a stable cache key covering the content of EVERY image in a SceneSet.

    For a paired task, hashing only the baseline would let a different follow-up image reuse
    the baseline's cached judgement (cache poisoning: two pairs sharing a before image would
    collide). The key is a digest of per-file digests, so a change to *any* image — baseline or
    follow-up — changes the key.
    """
    paths = [str(p) for p in image_paths]
    if not paths:
        raise ValueError("scene_cache_key requires at least one image path")
    label = label or os.path.basename(paths[0])
    if not content_addressed:
        return label
    combined = hashlib.sha256()
    for path in paths:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        combined.update(digest.digest())
    return f"{label}:{combined.hexdigest()[:16]}"


def image_cache_key(
    image_id: str, image_path: str, *, content_addressed: bool = True
) -> str:
    """Single-image cache key (thin wrapper over :func:`scene_cache_key`)."""
    return scene_cache_key([image_path], content_addressed=content_addressed, label=image_id)
