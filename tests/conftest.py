"""Shared fixtures: synthetic images, fake analyzers and table builders.

The whole suite runs offline — no API credentials, no torch, no network.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from PIL import Image


def make_image(path: Path, size: tuple[int, int] = (64, 48), color: tuple[int, int, int] = (60, 140, 80)) -> Path:
    """Write a tiny synthetic PNG so tests never need real street-view imagery."""
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture
def manifest(tmp_path: Path) -> pd.DataFrame:
    """Six single captures at distinct Barcelona-ish coordinates."""
    rows = []
    for i in range(6):
        image = make_image(tmp_path / f"img_{i}.png", color=(40 + 30 * i, 120, 90))
        rows.append(
            {
                "site_id": f"s{i}",
                "capture_id": f"c{i}",
                "image_path": str(image),
                "lon": 2.1630 + i * 0.0015,
                "lat": 41.3950 + i * 0.0012,
                "captured_at": "2022-11",
                "heading": 0,
                "source": "demo",
                "licence": "demo",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def paired_manifest(tmp_path: Path) -> pd.DataFrame:
    """Four sites × two capture dates = four admissible candidate pairs."""
    rows = []
    for site in range(4):
        for year, month in ((2017, "03"), (2022, "09")):
            capture_id = f"s{site}_{year}"
            image = make_image(tmp_path / f"{capture_id}.png", color=(90, 60 + 25 * (year == 2022), 70))
            rows.append(
                {
                    "site_id": f"site{site}",
                    "capture_id": capture_id,
                    "image_path": str(image),
                    "lon": 2.17 + site * 0.01,
                    "lat": 41.40 + site * 0.008,
                    "captured_at": f"{year}-{month}",
                    "heading": 90,
                    "source": "demo",
                    "licence": "demo",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def fake_cv_extractor():
    """Deterministic stand-in for the SegFormer extractor (no torch needed)."""

    def _extract(image_path: str, cfg: dict) -> dict[str, float]:
        name = Path(image_path).name
        if "broken" in name:
            raise FileNotFoundError(f"cannot read image: {image_path}")
        seed = float(sum(ord(ch) for ch in name) % 50)
        return {
            "seg_vegetation": 0.30,
            "seg_sky": 0.20,
            "seg_building": 0.10,
            "seg_wall": 0.02,
            "seg_fence": 0.01,
            "GVI": 0.30,
            "SVF": 0.20,
            "enclosure": 0.13,
            "Hue_Mean": seed,
        }

    return _extract


class FakeVLMClient:
    """Stand-in for an OpenAI-compatible client; records calls, can fail N times first."""

    def __init__(self, content: str, fail_first: int = 0, fail_when=None):
        self._content = content
        self._fail_first = fail_first
        self._fail_when = fail_when
        self.calls = 0

    def _create(self, **kwargs):
        self.calls += 1
        if self.calls <= self._fail_first:
            raise RuntimeError("connection reset by peer")
        if self._fail_when:
            sent = json.dumps(kwargs.get("messages", ""))
            if self._fail_when(sent):
                raise RuntimeError("synthetic permanent failure")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))


def prs11_reply(being_away: float = 0.4, coherence: float = 0.5, scope: float = 0.6, fascination: float = 0.3) -> str:
    return json.dumps(
        {"being_away": being_away, "coherence": coherence, "scope": scope, "fascination": fascination}
    )


def renewal_reply(detected: str = "yes") -> str:
    return json.dumps(
        {
            "renewal_detected": detected,
            "renewal_types": ["greening_landscape"] if detected == "yes" else [],
            "nuisance": ["season"],
            "confidence": 0.8,
            "evidence": "new trees appeared between the captures",
        }
    )


def write_task(
    tmp_path: Path,
    task_id: str,
    *,
    input_signature: str = "single_scene",
    fields: dict,
    template_text: str,
    extra: dict | None = None,
) -> str:
    """Write a minimal but valid task YAML + prompt template; return the YAML path."""
    task_dir = tmp_path / "tasks"
    task_dir.mkdir(exist_ok=True)
    template_path = task_dir / f"{task_id}.txt"
    template_path.write_text(template_text, encoding="utf-8")
    spec = {
        "id": task_id,
        "version": "1.0",
        "input_signature": input_signature,
        "prompt": {"template": f"{task_id}.txt", "temperature": None},
        "output_fields": fields,
    }
    if extra:
        spec.update(extra)
    path = task_dir / f"{task_id}.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(path)


@pytest.fixture
def prs11_task_file(tmp_path: Path) -> str:
    return write_task(
        tmp_path,
        "test_prs11",
        fields={"being_away": "score", "coherence": "score", "scope": "score", "fascination": "score"},
        template_text=prs11_reply(),
        extra={
            "derived_fields": {
                "restorative_average": {"func": "mean", "fields": ["being_away", "coherence", "scope", "fascination"]}
            },
            "validation": {
                "reference": None,
                "mapping": {
                    "vlm_being_away": "Being-away",
                    "vlm_coherence": "Coherence",
                    "vlm_scope": "Scope",
                    "vlm_fascination": "Fascination",
                    "vlm_restorative_average": "Average",
                },
            },
            "spatial": {
                "value_field": "vlm_restorative_average",
                "screening": "lisa_low_low",
                "k": 8,
                "p_threshold": 0.05,
            },
        },
    )


@pytest.fixture
def renewal_priority_task_file(tmp_path: Path) -> str:
    """A single-scene task using ``lisa_high_high`` screening (the opposite quadrant from
    prs11_task_file's ``lisa_low_low``), so pipeline tests can exercise the
    ``candidate_high_cluster`` wiring end to end."""
    return write_task(
        tmp_path,
        "test_renewal_priority",
        fields={
            "disrepair": "score",
            "neglect": "score",
            "informality": "score",
            "visual_disorder": "score",
            "evidence": "string",
        },
        template_text=json.dumps(
            {
                "disrepair": 0.5,
                "neglect": 0.5,
                "informality": 0.5,
                "visual_disorder": 0.5,
                "evidence": "test fixture reply, not a real image judgement",
            }
        ),
        extra={
            "derived_fields": {
                "renewal_priority_average": {
                    "func": "mean",
                    "fields": ["disrepair", "neglect", "informality", "visual_disorder"],
                }
            },
            "spatial": {
                "value_field": "vlm_renewal_priority_average",
                "screening": "lisa_high_high",
                "k": 3,
                "p_threshold": 0.05,
            },
        },
    )


@pytest.fixture
def renewal_task_file(tmp_path: Path) -> str:
    return write_task(
        tmp_path,
        "test_renewal",
        input_signature="paired_scene",
        fields={
            "renewal_detected": {"type": "category", "options": ["yes", "no", "uncertain"]},
            "renewal_types": {
                "type": "list",
                "items": {
                    "type": "category",
                    "options": ["greening_landscape", "road_traffic", "other"],
                },
                "unique_items": True,
            },
            "nuisance": "list",
            "confidence": "score",
            "evidence": "string",
        },
        template_text=renewal_reply(),
        extra={
            "pair_contract": {"max_distance_m": 20, "max_heading_diff_deg": 30, "require_captured_at": True},
            "cv_change": {"delta_columns": ["GVI", "SVF", "enclosure"]},
        },
    )


@pytest.fixture
def offline_cfg(tmp_path: Path):
    from pyrestore import load_config

    return load_config(
        {
            "vlm": {
                "cache_db": str(tmp_path / "cache" / "vlm_cache.sqlite"),
                "offline": True,
                "model": "fake-model",
            }
        }
    )
