"""Task-configurable VLM semantic interpretation with caching, retries and provenance.

The task definition (``tasks/*.yaml``) owns everything semantic: the prompt template, the
output-field contract, derived fields and how results are validated. This module owns everything
operational: message construction, caching, retries, offline mode and per-record failure records.
A failed record is returned with ``vlm_status='error'`` — batch failures never silently disappear.

Output columns are prefixed ``vlm_`` to keep the semantic-interpretation family distinct from CV
physical measurements (framework invariant: the two families are never merged into one type of
evidence).
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io as _io
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .cache import Cache, scene_cache_key

_TRANSIENT = ("timeout", "rate limit", "429", "500", "502", "503", "overloaded", "connection")
_INPUT_SIGNATURES = ("single_scene", "paired_scene")
_FIELD_TYPES = ("score", "number", "string", "category", "list")
_RESERVED_OUTPUT_NAMES = {"status", "error", "model", "task_id", "prompt_hash", "cache_hit"}


# ------------------------------------------------------------------ task definition
def load_task(task: str | os.PathLike[str] | dict[str, Any]) -> dict[str, Any]:
    """Load and validate a task definition from a YAML path (or accept an in-memory dict).

    Required keys: ``id``, ``input_signature`` (single_scene | paired_scene), ``prompt.template``
    and a non-empty ``output_fields`` mapping. Field specs accept the shorthand ``"score"`` or a
    mapping like ``{type: category, options: [yes, no]}``. The prompt template path is resolved
    relative to the YAML file's directory. Validation of *results* (validation section) is
    declared here but executed by ``pyrestore.validate`` / ``pyrestore.pipeline``.
    """
    if isinstance(task, dict):
        spec = copy.deepcopy(task)
        base_dir = Path.cwd()
    else:
        path = Path(task).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Task definition not found: {path}")
        with open(path, encoding="utf-8") as stream:
            spec = yaml.safe_load(stream)
        if not isinstance(spec, dict):
            raise ValueError(f"Task definition must be a mapping: {path}")
        base_dir = path.parent

    task_id = spec.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("Task definition is missing a non-empty 'id'.")
    signature = spec.get("input_signature")
    if signature not in _INPUT_SIGNATURES:
        raise ValueError(
            f"Task {task_id!r} has invalid input_signature {signature!r}; "
            f"expected one of {_INPUT_SIGNATURES}."
        )
    prompt = spec.get("prompt") or {}
    template = prompt.get("template")
    if not template:
        raise ValueError(f"Task {task_id!r} is missing prompt.template.")
    template_path = Path(str(template)).expanduser()
    if not template_path.is_absolute():
        template_path = base_dir / template_path
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found for task {task_id!r}: {template_path}")
    with open(template_path, encoding="utf-8") as stream:
        text = stream.read()

    raw_fields = spec.get("output_fields")
    if not isinstance(raw_fields, dict) or not raw_fields:
        raise ValueError(f"Task {task_id!r} must declare a non-empty output_fields mapping.")
    fields: dict[str, dict[str, Any]] = {}
    for name, raw_spec in raw_fields.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Task {task_id!r}: output field names must be non-empty strings.")
        if name in _RESERVED_OUTPUT_NAMES:
            raise ValueError(
                f"Task {task_id!r}: output field {name!r} is reserved by the VLM result contract."
            )
        if isinstance(raw_spec, str):
            field_spec: dict[str, Any] = {"type": raw_spec}
        elif isinstance(raw_spec, dict):
            field_spec = dict(raw_spec)
        else:
            raise ValueError(f"Task {task_id!r}: unsupported output_fields entry for {name!r}")
        field_type = field_spec.get("type")
        if field_type not in _FIELD_TYPES:
            raise ValueError(
                f"Task {task_id!r}: field {name!r} has unknown type {field_type!r}; "
                f"expected one of {_FIELD_TYPES}."
            )
        if field_type == "category":
            _validate_options(task_id, name, field_spec.get("options"))
        if field_type == "list" and field_spec.get("items") is not None:
            items = field_spec["items"]
            if not isinstance(items, dict) or items.get("type") not in ("string", "category"):
                raise ValueError(
                    f"Task {task_id!r}: list field {name!r} items must declare type "
                    "'string' or 'category'."
                )
            if items["type"] == "category":
                _validate_options(task_id, f"{name}[]", items.get("options"))
        for bound in ("minimum", "maximum"):
            if bound in field_spec and field_type not in ("score", "number"):
                raise ValueError(
                    f"Task {task_id!r}: field {name!r} uses {bound} but is not numeric."
                )
        if (
            "minimum" in field_spec
            and "maximum" in field_spec
            and float(field_spec["minimum"]) > float(field_spec["maximum"])
        ):
            raise ValueError(f"Task {task_id!r}: field {name!r} has minimum > maximum.")
        fields[str(name)] = field_spec
    spec["output_fields"] = fields
    spec["prompt"] = {**prompt, "template": str(template_path), "_text": text}
    spec["_base_dir"] = str(base_dir)

    for name, derived in (spec.get("derived_fields") or {}).items():
        if name in fields or name in _RESERVED_OUTPUT_NAMES:
            raise ValueError(f"Task {task_id!r}: derived field {name!r} conflicts with another field.")
        if not isinstance(derived, dict) or derived.get("func") not in ("mean", "sum"):
            raise ValueError(
                f"Task {task_id!r}: derived field {name!r} must declare func 'mean' or 'sum'."
            )
        if not derived.get("fields"):
            raise ValueError(f"Task {task_id!r}: derived field {name!r} needs a fields list.")
        unknown = [field for field in derived["fields"] if field not in fields]
        if unknown:
            raise ValueError(
                f"Task {task_id!r}: derived field {name!r} references unknown fields {unknown}."
            )
        non_numeric = [
            field for field in derived["fields"] if fields[field]["type"] not in ("score", "number")
        ]
        if non_numeric:
            raise ValueError(
                f"Task {task_id!r}: derived field {name!r} requires numeric inputs; got {non_numeric}."
            )
    return spec


def _validate_options(task_id: str, field_name: str, options: Any) -> None:
    if not isinstance(options, list) or not options:
        raise ValueError(f"Task {task_id!r}: category field {field_name!r} requires options.")
    if any(not isinstance(option, str) or not option.strip() for option in options):
        raise ValueError(
            f"Task {task_id!r}: category field {field_name!r} options must be non-empty strings."
        )
    if len(set(options)) != len(options):
        raise ValueError(f"Task {task_id!r}: category field {field_name!r} has duplicate options.")


def template_text(task: dict[str, Any]) -> str:
    """Return the task's prompt template text (loaded by :func:`load_task`)."""
    return task["prompt"]["_text"]


def task_prompt_hash(task: dict[str, Any]) -> str:
    """Stable hash of all task-level inputs that can change a VLM result."""
    contract = {
        "id": task["id"],
        "version": task.get("version"),
        "input_signature": task["input_signature"],
        "template": template_text(task),
        "temperature": task["prompt"].get("temperature"),
        "output_fields": task["output_fields"],
        "derived_fields": task.get("derived_fields") or {},
    }
    payload = json.dumps(contract, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def request_hash(task: dict[str, Any], cfg: dict[str, Any]) -> str:
    """Hash the task plus image/request preprocessing settings used for one VLM call."""
    vcfg = cfg["vlm"]
    request_contract = {
        "task_prompt_hash": task_prompt_hash(task),
        "model": vcfg.get("model", "gpt-4o"),
        "provider_id": vcfg.get("provider_id", "openai-compatible"),
        "max_image_px": int(vcfg.get("max_image_px", 1024)),
        "jpeg_quality": int(vcfg.get("jpeg_quality", 85)),
        "max_completion_tokens": int(vcfg.get("max_completion_tokens", 800)),
    }
    payload = json.dumps(request_contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ response contract
def parse_response(content: str | None, task: dict[str, Any]) -> dict[str, Any]:
    """Extract and validate the task's output fields from a model reply.

    Handles code fences and surrounding prose by taking the first JSON object in the reply, then
    enforces the field contract declared in the task definition. Raises ``ValueError`` with a
    precise message on any violation — the caller records it as a per-record failure.
    """
    if not content or not content.strip():
        raise ValueError("empty model reply")
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in reply: {content[:120]!r}")
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in reply: {exc}") from exc

    return validate_response_object(obj, task)


def validate_response_object(obj: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Validate an already-decoded response object against a task's output contract."""
    if not isinstance(obj, dict):
        raise ValueError("model reply must decode to a JSON object")
    response: dict[str, Any] = {}
    for name, spec in task["output_fields"].items():
        if name not in obj:
            raise ValueError(f"missing output field {name!r} in reply")
        value = obj[name]
        field_type = spec["type"]
        if field_type in ("score", "number"):
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"field {name!r} is not numeric: {value!r}") from exc
            if not math.isfinite(numeric):
                raise ValueError(f"field {name!r} must be a finite number")
            if field_type == "score" and not 0.0 <= numeric <= 1.0:
                raise ValueError(f"field {name!r} must be a score between 0 and 1")
            if "minimum" in spec and numeric < float(spec["minimum"]):
                raise ValueError(f"field {name!r} must be >= {spec['minimum']}")
            if "maximum" in spec and numeric > float(spec["maximum"]):
                raise ValueError(f"field {name!r} must be <= {spec['maximum']}")
            response[name] = numeric
        elif field_type == "string":
            if value is None or not str(value).strip():
                raise ValueError(f"field {name!r} must be a non-empty string")
            response[name] = str(value)
        elif field_type == "category":
            category = str(value)
            if category not in spec.get("options", []):
                raise ValueError(
                    f"field {name!r} value {category!r} not in options {spec.get('options')}"
                )
            response[name] = category
        elif field_type == "list":
            if not isinstance(value, list):
                raise ValueError(f"field {name!r} must be a list")
            items = [str(item) for item in value]
            item_spec = spec.get("items") or {"type": "string"}
            if item_spec.get("type") == "category":
                invalid = [item for item in items if item not in item_spec["options"]]
                if invalid:
                    raise ValueError(
                        f"field {name!r} contains values {invalid!r} not in options "
                        f"{item_spec['options']}"
                    )
            if spec.get("unique_items") and len(set(items)) != len(items):
                raise ValueError(f"field {name!r} must not contain duplicate values")
            response[name] = items
    return response


def apply_derived_fields(response: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Compute the task's derived fields (mean/sum over declared input fields)."""
    derived = task.get("derived_fields") or {}
    out: dict[str, Any] = {}
    for name, spec in derived.items():
        values = [float(response[field]) for field in spec["fields"] if field in response]
        if len(values) != len(spec["fields"]):
            raise ValueError(f"derived field {name!r}: missing inputs among {spec['fields']}")
        if spec["func"] == "mean":
            out[name] = sum(values) / len(values)
        else:
            out[name] = sum(values)
    return out


# ------------------------------------------------------------------ transport
def image_data_url(image_path: str, max_px: int = 1024, jpeg_quality: int = 85) -> str:
    """Encode an image as a base64 JPEG data URL, downscaling large images.

    Full street-view PNGs are ~2 MB; their base64 payloads cause proxy connection drops and
    needless cost. Downscaling the longest side to ``max_px`` and re-encoding as JPEG shrinks
    the payload ~10x with no meaningful loss for perceptual scoring.
    """
    if max_px <= 0:
        raise ValueError("max_px must be positive")
    if not 1 <= jpeg_quality <= 95:
        raise ValueError("jpeg_quality must be between 1 and 95")
    buf = _io.BytesIO()
    from PIL import Image

    with Image.open(image_path) as img:
        converted = img.convert("RGB")
        width, height = converted.size
        if max(width, height) > max_px:
            scale = max_px / max(width, height)
            converted = converted.resize((max(1, int(width * scale)), max(1, int(height * scale))))
        converted.save(buf, format="JPEG", quality=jpeg_quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _credentials() -> tuple[str, str]:
    try:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(os.getcwd(), ".env"))
    except ImportError:
        pass
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set (put it in .env or the environment).")
    return key, os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


def build_messages(
    task: dict[str, Any], image_paths: list[str], max_px: int, jpeg_quality: int = 85
) -> list[dict]:
    """Assemble one user message: the task prompt text followed by one image per capture."""
    content: list[dict[str, Any]] = [{"type": "text", "text": template_text(task)}]
    for path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(path, max_px, jpeg_quality)},
            }
        )
    return [{"role": "user", "content": content}]


def analyse_images(
    image_paths: list[str],
    task: dict[str, Any],
    cfg: dict[str, Any],
    *,
    cache: Cache | None = None,
    client: Any = None,
    cache_key: str | None = None,
) -> tuple[dict[str, Any], str, bool]:
    """One VLM call for one SceneSet (1 image = single scene, 2 = before/after pair).

    Cache-first; honours ``cfg['vlm']['offline']``. Returns ``(response, raw_text, cache_hit)``.
    ``client`` may be injected for testing; otherwise an OpenAI-compatible client is built from
    the environment. Raises ``RuntimeError`` on an offline cache miss or after exhausting retries.
    """
    vcfg = cfg["vlm"]
    model = vcfg.get("model", "gpt-4o")
    task_id = task["id"]
    prompt_hash = request_hash(task, cfg)
    paths = [str(p) for p in image_paths]
    if not paths:
        raise ValueError("analyse_images requires at least one image path")
    if cache is None and vcfg.get("cache_db"):
        with Cache(vcfg["cache_db"]) as owned_cache:
            return analyse_images(
                paths,
                task,
                cfg,
                cache=owned_cache,
                client=client,
                cache_key=cache_key,
            )
    key = cache_key or scene_cache_key(
        paths, content_addressed=bool(vcfg.get("content_addressed_cache", True))
    )

    if cache is not None:
        hit = cache.get(key, task_id, model, prompt_hash)
        if hit is not None:
            return hit, "", True

    if vcfg.get("offline", False):
        raise RuntimeError(f"offline mode: no cached VLM response for {key!r} (task {task_id!r})")

    if client is None:
        if not vcfg.get("allow_remote_images", False):
            raise RuntimeError(
                "Remote image transmission is disabled. Set vlm.allow_remote_images: true "
                "only after confirming image privacy, licence and provider terms."
            )
        from openai import OpenAI

        api_key, base_url = _credentials()
        client = OpenAI(api_key=api_key, base_url=base_url)

    messages = build_messages(
        task,
        paths,
        int(vcfg.get("max_image_px", 1024)),
        int(vcfg.get("jpeg_quality", 85)),
    )
    max_tokens = int(vcfg.get("max_completion_tokens", 800))
    retries = int(vcfg.get("max_retries", 3))
    temperature = task["prompt"].get("temperature")

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_completion_tokens": max_tokens}
            if temperature is not None:
                kwargs["temperature"] = float(temperature)
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            response = parse_response(content, task)
            if cache is not None:
                cache.put(key, task_id, model, prompt_hash, response, content)
            return response, content, False
        except Exception as exc:  # noqa: BLE001 — classify transient vs fatal, retry either way
            last_err = exc
            if any(t in str(exc).lower() for t in _TRANSIENT) and attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            if attempt < retries - 1:
                continue
    raise RuntimeError(f"VLM analysis failed for {key!r} after {retries} attempts: {last_err}")


# ------------------------------------------------------------------ batch
def score_vlm_batch(
    inputs: pd.DataFrame,
    task: dict[str, Any],
    cfg: dict[str, Any],
    *,
    cache: Cache | None = None,
    client: Any = None,
) -> pd.DataFrame:
    """Score a single-scene manifest or a paired table; one output row per input record.

    Single-scene inputs need ``capture_id`` and ``image_path``; paired inputs need ``pair_id``,
    ``before_image_path`` and ``after_image_path``. Every row records model, prompt hash, task id
    and cache use; failures are recorded with ``vlm_status='error'`` and never dropped.
    """
    signature = task["input_signature"]
    if signature == "single_scene":
        required = {"capture_id", "image_path"}
        id_col, path_cols = "capture_id", ["image_path"]
    else:
        required = {"pair_id", "before_image_path", "after_image_path"}
        id_col, path_cols = "pair_id", ["before_image_path", "after_image_path"]
    if not required.issubset(inputs.columns):
        raise ValueError(f"Input table is missing required columns: {sorted(required)}")

    vcfg = cfg["vlm"]
    model = vcfg.get("model", "gpt-4o")
    prompt_hash = request_hash(task, cfg)
    owns_cache = cache is None and bool(vcfg.get("cache_db"))
    if owns_cache:
        cache = Cache(vcfg["cache_db"])

    rows: list[dict[str, Any]] = []
    for record in inputs.to_dict("records"):
        record_id = str(record[id_col])
        paths = [str(record[column]) for column in path_cols]
        # The key covers every image of the SceneSet — for pairs, hashing only the baseline
        # would let a different follow-up image reuse the baseline's cached judgement.
        key = scene_cache_key(
            paths,
            content_addressed=bool(vcfg.get("content_addressed_cache", True)),
            label=record_id,
        )
        row: dict[str, Any] = {
            id_col: record_id,
            "task_id": task["id"],
            "vlm_model": model,
            "prompt_hash": prompt_hash,
        }
        try:
            response, _raw, cache_hit = analyse_images(
                paths, task, cfg, cache=cache, client=client, cache_key=key
            )
            row["cache_hit"] = cache_hit
            for name, value in response.items():
                row[f"vlm_{name}"] = value
            for name, value in apply_derived_fields(response, task).items():
                row[f"vlm_{name}"] = value
            row["vlm_status"] = "ok"
            row["vlm_error"] = ""
        except Exception as exc:  # noqa: BLE001 — preserve the batch, expose the failure
            row["cache_hit"] = False
            row["vlm_status"] = "error"
            row["vlm_error"] = str(exc)
        rows.append(row)
    result = pd.DataFrame(rows)
    if owns_cache and cache is not None:
        cache.close()
    return result
