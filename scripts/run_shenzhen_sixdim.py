"""Run the real 465-record Shenzhen six-dimension task with resumable VLM batches.

The script has no command-line interface by design. It reads the prepared local package and the
submission configuration, writes a resumable work table after every batch, then hands the complete
table back to ``run_task`` for contract validation, harmonisation, validation and provenance.

This script performs live VLM calls and therefore needs API credentials in `.env`; it is the
one-time run that produced the frozen `data/shenzhen_sixdim/vlm_scores.csv` this repository ships.
It is not part of the reproducible pipeline a fresh clone needs to run -- notebook 03 replays the
already-frozen result offline instead.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from pyrestore import load_config, load_manifest, load_task, run_task
from pyrestore.vlm import score_vlm_batch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "shenzhen_sixdim"
CONFIG_PATH = ROOT / "config" / "submission.yaml"
TASK_PATH = ROOT / "tasks" / "sixdim.yaml"
WORK_DIR = ROOT / "outputs" / "submission" / "gpt-5.4-mini" / "_work"
PARTIAL_PATH = WORK_DIR / "sixdim_vlm_scores.csv"
FINAL_DIR = ROOT / "outputs" / "submission" / "gpt-5.4-mini" / "urban_perception_sixdim"
BATCH_SIZE = 20


def _write_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> int:
    manifest = load_manifest(DATA_DIR / "manifest.csv")
    cv_features = pd.read_csv(DATA_DIR / "cv_features.csv")
    reference = pd.read_csv(DATA_DIR / "reference.csv")
    task = load_task(TASK_PATH)
    cfg = load_config(CONFIG_PATH)

    if PARTIAL_PATH.exists():
        accumulated = pd.read_csv(PARTIAL_PATH)
        complete_ids = set(
            accumulated.loc[accumulated["vlm_status"].eq("ok"), "capture_id"].astype(str)
        )
    else:
        accumulated = pd.DataFrame()
        complete_ids = set()

    pending = manifest[~manifest["capture_id"].isin(complete_ids)].reset_index(drop=True)
    print(
        f"Shenzhen sixdim: total={len(manifest)}, cached/complete={len(complete_ids)}, "
        f"pending={len(pending)}",
        flush=True,
    )
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending.iloc[start : start + BATCH_SIZE]
        scored = score_vlm_batch(batch, task, cfg)
        accumulated = pd.concat([accumulated, scored], ignore_index=True, sort=False)
        accumulated = accumulated.drop_duplicates("capture_id", keep="last")
        accumulated = accumulated.sort_values("capture_id").reset_index(drop=True)
        _write_atomic(accumulated, PARTIAL_PATH)
        done = min(start + len(batch), len(pending))
        counts = accumulated["vlm_status"].value_counts(dropna=False).to_dict()
        print(
            f"batch_progress={done}/{len(pending)}; accumulated={len(accumulated)}/{len(manifest)}; "
            f"status={counts}",
            flush=True,
        )

    missing_ids = set(manifest["capture_id"].astype(str)) - set(
        accumulated["capture_id"].astype(str)
    )
    if missing_ids:
        raise RuntimeError(f"VLM work table is incomplete: {len(missing_ids)} capture ids missing")

    ordered_vlm = manifest[["capture_id"]].merge(
        accumulated, on="capture_id", how="left", validate="one_to_one"
    )
    result = run_task(
        manifest,
        task,
        cfg,
        cv_features=cv_features,
        vlm_scores=ordered_vlm,
        reference=reference,
        make_map=False,
        out_dir=FINAL_DIR,
    )
    print("final_vlm_status", result["vlm_scores"]["vlm_status"].value_counts().to_dict())
    print("validation_status", result["validation_status"])
    print("final_dir", FINAL_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
