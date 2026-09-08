"""Runtime configuration: provider settings, model defaults and output locations.

Task *semantics* (prompts, output fields, validation contracts) live in task-definition YAML
files loaded by :func:`pyrestore.vlm.load_task`; this module holds only environment-level settings
shared by every task. ``load_config`` merges, highest priority last:

1. built-in defaults, 2. a user YAML file, 3. explicit ``overrides``.

Credentials are read from the environment / ``.env`` at call time (see ``pyrestore.vlm``) and are
never stored in configuration files or source.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "vlm": {
        "model": "gpt-5.4-mini",
        "provider_id": "openai-compatible",
        "max_image_px": 1024,
        "jpeg_quality": 85,
        "max_completion_tokens": 800,
        "max_retries": 3,
        "offline": False,
        "allow_remote_images": False,
        "content_addressed_cache": True,
        "cache_db": "cache/vlm_cache.sqlite",
    },
    "cv": {
        "segmenter": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        "gvi_class": "seg_vegetation",
        "svf_class": "seg_sky",
        "enclosure_classes": ["seg_building", "seg_wall", "seg_fence"],
    },
    "output": {"dir": "outputs"},
}


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    path: str | Path | dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a merged configuration dict.

    ``path`` may be a YAML file path or an already-loaded dict; ``overrides`` is applied last.
    Unknown keys are preserved so deployments can attach their own sections without forking.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if isinstance(path, dict):
        _merge(cfg, copy.deepcopy(path))
    elif path is not None:
        resolved = Path(path).expanduser()
        with open(resolved, encoding="utf-8") as stream:
            user = yaml.safe_load(stream) or {}
        if not isinstance(user, dict):
            raise ValueError(f"Config file must contain a mapping: {resolved}")
        _merge(cfg, user)
    if overrides:
        _merge(cfg, copy.deepcopy(dict(overrides)))
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    for section in ("vlm", "cv", "output"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"Config section {section!r} must be a mapping.")
    vlm = cfg["vlm"]
    if not isinstance(vlm.get("model"), str) or not vlm["model"].strip():
        raise ValueError("vlm.model must be a non-empty string.")
    integer_bounds = {
        "max_image_px": (1, None),
        "jpeg_quality": (1, 95),
        "max_completion_tokens": (1, None),
        "max_retries": (1, None),
    }
    for key, (minimum, maximum) in integer_bounds.items():
        value = vlm.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"vlm.{key} must be an integer.")
        if value < minimum or (maximum is not None and value > maximum):
            interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
            raise ValueError(f"vlm.{key} must be {interval}.")
    for key in ("offline", "allow_remote_images", "content_addressed_cache"):
        if not isinstance(vlm.get(key), bool):
            raise ValueError(f"vlm.{key} must be true or false.")
    if not isinstance(cfg["cv"].get("enclosure_classes"), list):
        raise ValueError("cv.enclosure_classes must be a list.")
    if not isinstance(cfg["output"].get("dir"), (str, os.PathLike)):
        raise ValueError("output.dir must be a path string.")
