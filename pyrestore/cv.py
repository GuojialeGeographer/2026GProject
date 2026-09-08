"""Computer-vision physical measurements.

Derived indicators: GVI = vegetation share, SVF = sky share, enclosure = building+wall+fence
share, plus colour (OpenCV HSV/HLS means) and Canny edge density. Segmentation uses SegFormer
fine-tuned on Cityscapes, whose 19 classes match the ``seg_*`` columns of the case-study
reference data — enabling cross-model validation of the live extractor.

Heavy dependencies (torch/transformers/cv2) are imported lazily so the package imports without
the optional ``cv`` extra; tests inject a fake extractor instead of loading SegFormer.

Framework invariant: these are *physical measurements* of the visible image — never merged with
VLM semantic interpretations into one type of evidence.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

PHYSICAL_FEATURES = [
    "GVI",
    "SVF",
    "enclosure",
    "Canny_Edges",
    "Hue_Mean",
    "Saturation_Mean",
    "Lightness_Mean",
]
DEFAULT_SEGMENTER = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
_MODEL_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------- precomputed path
def derive_from_precomputed(row: pd.Series, cfg: dict[str, Any]) -> dict[str, float]:
    """Derive GVI/SVF/enclosure (+ colour/edge) from precomputed ``seg_*`` columns of a row.

    Used when a case study ships reference segmentation values and the live extractor is not
    needed — the analysis then runs without torch installed.
    """
    c = cfg["cv"]
    out: dict[str, float] = {
        "GVI": float(row.get(c["gvi_class"], 0.0)),
        "SVF": float(row.get(c["svf_class"], 0.0)),
        "enclosure": float(sum(float(row.get(x, 0.0)) for x in c["enclosure_classes"])),
    }
    for col in ("Canny_Edges", "Hue_Mean", "Saturation_Mean", "Lightness_Mean"):
        if col in row:
            out[col] = float(row[col])
    return out


# ---------------------------------------------------------------- live extractor
def _load_segmenter(model_id: str):
    if model_id not in _MODEL_CACHE:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        proc = SegformerImageProcessor.from_pretrained(model_id)
        model = SegformerForSemanticSegmentation.from_pretrained(model_id).eval()
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        _MODEL_CACHE[model_id] = (proc, model, device)
    return _MODEL_CACHE[model_id]


def segment_shares(image_path: str, model_id: str = DEFAULT_SEGMENTER) -> dict[str, float]:
    """Return per-Cityscapes-class pixel share for one image (keys are class names)."""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    proc, model, device = _load_segmenter(model_id)
    img = Image.open(image_path).convert("RGB")
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    upsampled = F.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False)
    pred = upsampled.argmax(dim=1)[0].cpu().numpy()
    total = pred.size
    id2label = model.config.id2label
    return {name: float((pred == int(cid)).sum()) / total for cid, name in id2label.items()}


def _colour_edge(image_path: str) -> dict[str, float]:
    import cv2

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "Hue_Mean": float(hsv[:, :, 0].mean()),
        "Saturation_Mean": float(hsv[:, :, 1].mean()),
        "Lightness_Mean": float(hls[:, :, 1].mean()),
        "Canny_Edges": float(cv2.Canny(gray, 100, 200).mean()),
    }


def extract_cv_features(image_path: str, cfg: dict[str, Any]) -> dict[str, float]:
    """Extract physical features from one image.

    Returns derived indicators (GVI/SVF/enclosure), colour/edge stats and per-class ``seg_<name>``
    shares (useful for validation). Requires the optional ``cv`` extra.
    """
    c = cfg["cv"]
    model_id = c.get("segmenter", DEFAULT_SEGMENTER)
    shares = segment_shares(image_path, model_id)
    out: dict[str, float] = {f"seg_{name}": v for name, v in shares.items()}
    out["GVI"] = shares.get(_strip(c["gvi_class"]), 0.0)
    out["SVF"] = shares.get(_strip(c["svf_class"]), 0.0)
    out["enclosure"] = sum(shares.get(_strip(x), 0.0) for x in c["enclosure_classes"])
    out.update(_colour_edge(image_path))
    return out


def extract_cv_batch(
    manifest: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    extractor: Any = None,
) -> pd.DataFrame:
    """Extract CV indicators for every row in a manifest.

    A failed image is recorded with ``cv_status='error'`` rather than stopping the batch. The
    optional ``extractor`` argument keeps orchestration testable without loading SegFormer.
    """
    required = {"capture_id", "image_path"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest must contain columns {sorted(required)}")

    run_extractor = extractor or extract_cv_features
    rows: list[dict[str, Any]] = []
    for record in manifest[["capture_id", "image_path"]].to_dict("records"):
        row: dict[str, Any] = {"capture_id": record["capture_id"]}
        try:
            row.update(run_extractor(record["image_path"], cfg))
            row.update({"cv_status": "ok", "cv_error": ""})
        except Exception as exc:  # noqa: BLE001 — one unreadable image must not discard the area
            row.update({"cv_status": "error", "cv_error": str(exc)})
        rows.append(row)
    return pd.DataFrame(rows)


def _strip(seg_col: str) -> str:
    """Map a ``seg_<name>`` config key to the bare Cityscapes class name."""
    return seg_col[len("seg_"):] if seg_col.startswith("seg_") else seg_col


# ---------------------------------------------------------------- validation support
def validate_against(
    extracted: pd.DataFrame, reference: pd.DataFrame, id_col: str = "capture_id"
) -> pd.DataFrame:
    """Compare live-extracted ``seg_*`` features to a reference segmentation.

    Returns a per-class table with Pearson correlation and mean absolute error over shared images.
    This is *cross-model agreement*, not field-survey truth — reports must state the reference type.
    """
    merged = extracted.merge(reference, on=id_col, suffixes=("_pred", "_gt"))
    rows = []
    seg_cols = [
        c for c in reference.columns if c.startswith("seg_") and f"{c}_pred" in merged.columns
    ]
    for col in seg_cols:
        p, g = merged[f"{col}_pred"], merged[f"{col}_gt"]
        corr = float(p.corr(g)) if len(merged) > 1 else float("nan")
        mae = float((p - g).abs().mean())
        rows.append({"feature": col, "pearson_r": corr, "mae": mae, "n": len(merged)})
    return pd.DataFrame(rows)
