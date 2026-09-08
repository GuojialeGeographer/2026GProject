"""Build the submission-safe paired-scene mechanism demonstration.

Both analyzers are deterministic injected fixtures and provenance records that fact explicitly.
The product demonstrates contracts, deltas, abstention-ready fields and failure transparency; it
does not estimate real urban-renewal detection performance.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from pyrestore import load_config, run_task

ROOT = Path(__file__).resolve().parent.parent


class DemonstrationVLMClient:
    def __init__(self) -> None:
        self.reply = json.dumps(
            {
                "renewal_detected": "yes",
                "renewal_types": ["greening_landscape"],
                "nuisance": ["illumination"],
                "confidence": 0.7,
                "evidence": "vegetation share increases between the synthetic captures",
            }
        )

    def _create(self, **_kwargs: object) -> SimpleNamespace:
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    @property
    def chat(self) -> SimpleNamespace:
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))


def demonstration_cv_extractor(image_path: str, _cfg: dict) -> dict[str, float]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=float) / 255.0
    green_share = float(
        ((image[:, :, 1] > image[:, :, 0]) & (image[:, :, 1] > image[:, :, 2])).mean()
    )
    return {
        "seg_vegetation": green_share,
        "seg_sky": 0.2,
        "seg_building": 0.1,
        "seg_wall": 0.02,
        "seg_fence": 0.01,
        "GVI": green_share,
        "SVF": 0.2,
        "enclosure": 0.13,
    }


def main() -> int:
    cfg = load_config(
        {
            "vlm": {
                "model": "injected-demonstration-response",
                "cache_db": "cache/renewal_mechanism_demo.sqlite",
            },
            "provenance": {
                "data_status": "synthetic paired-scene mechanism demonstration",
                "run_note": (
                    "four solid-colour before/after pairs; no real detection-performance claim"
                ),
            },
        }
    )
    result = run_task(
        ROOT / "data" / "demo_pairs" / "manifest.csv",
        ROOT / "tasks" / "renewal.yaml",
        cfg,
        vlm_client=DemonstrationVLMClient(),
        cv_extractor=demonstration_cv_extractor,
        make_map=False,
        out_dir=ROOT / "outputs" / "submission" / "urban_renewal_pair",
    )
    print(result["quality"].to_string(index=False))
    print(result["provenance"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
