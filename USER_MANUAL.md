# PyRestore User Manual

## 1. Installation

```bash
uv sync --extra dev --frozen
uv run pytest --cov
# Optional live SegFormer extraction:
uv sync --extra dev --extra cv --frozen
```

Requirements: Python ≥ 3.10. Core dependencies (pandas, GeoPandas, folium, esda, openai, …)
install automatically. Live VLM scoring needs an OpenAI-compatible endpoint.

## 2. Credentials

Copy `.env.example` to `.env` and fill in:

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1     # any OpenAI-compatible endpoint
```

Credentials are read at call time from the environment (`.env` in the working directory is
honoured). They must never be written into config files, notebooks or committed files.

Live transmission also requires an explicit runtime acknowledgement:

```yaml
vlm:
  allow_remote_images: true  # only after privacy, licence and provider-term review
```

## 3. Inputs

### 3.1 Image manifest (single-scene tasks)

One row per capture. Required columns: `site_id, capture_id, image_path, lon, lat`.
Optional provenance: `captured_at, heading, camera, source, licence` (never inferred from
pixels or filenames). Legacy PyRestore manifests (`image, image_path, lon, lat`) are adapted
automatically, and precomputed CV/VLM tables keyed by the legacy `image` column are also
accepted — `capture_id` is the contract column, `image` its recognised alias. Example:

```csv
site_id,capture_id,image_path,lon,lat,captured_at,heading,source,licence
SZ-01,SZ-01-2022,images/sz01_2022.png,114.05229,22.62006,2022-11,270,baidu,restricted
```

`image_path` may be absolute or relative. A relative path resolves against the **manifest CSV's
own directory**, not the current working directory — the recommended layout is one data folder
holding both the manifest and an `images/` subfolder, as in `data/demo/manifest.csv` +
`data/demo/images/`. Passing a DataFrame instead of a CSV path resolves relative paths against
the current working directory.

### 3.2 Pair table (paired-scene tasks)

Either derive pairs with `pyrestore.make_pairs("m.csv")` or supply a table with columns
`pair_id, site_id, before_capture_id, after_capture_id, before_image_path,
after_image_path, before_lon, before_lat, after_lon, after_lat`. Relative paths are resolved from
the pair CSV location. Rows that fail the recomputed comparability audit (distance / heading / capture-date
rules) remain in the table with `comparability_status='excluded'` and an `exclusion_reason`;
`run_task` processes only `ok` pairs and reports the excluded count in `provenance.csv`.

## 4. Task definitions

A task is a YAML file under `tasks/` plus a prompt template under `tasks/prompts/`. Fields:

| Key | Meaning |
|---|---|
| `id`, `version` | identity recorded in every provenance row |
| `input_signature` | `single_scene` or `paired_scene` |
| `prompt.template`, `prompt.temperature` | template path; `temperature: null` omits the sampling parameter (some reasoning models reject it) |
| `output_fields` | `score` (0–1), bounded `number`, `string`, `category`, or `list` with optional item category/options |
| `derived_fields` | `mean`/`sum` over declared fields |
| `validation` | `reference` CSV + `mapping` (continuous) or label columns (classification); `null` keeps runs `not_validated` |
| `pair_contract` | paired tasks: `max_distance_m`, `max_heading_diff_deg`, `require_captured_at` |
| `cv_change.delta_columns` | paired tasks: CV columns that get a `_delta` (after − before) |
| `spatial` | `value_field`, `screening`, `k`, `p_threshold`, `p_adjust: fdr_bh` |

Copy an existing task as a starting point; validate edits by re-running the demo.

## 5. Running

PyRestore intentionally exposes a Python API rather than a separate command-line interface. The
notebooks are the primary executable walkthroughs. A direct Python run is:

```python
from pyrestore import make_pairs, run_task

result = run_task(
    source,
    "tasks/prs11.yaml",
    "config.yaml",
    labels="labels.csv",
    reference="human_prs.csv",
)

pairs = make_pairs("manifest.csv")
pairs.to_csv("pairs.csv", index=False)
```

`run_task` parameters:

| Parameter | Required | Meaning |
|---|---|---|
| `source` | yes | manifest CSV/DataFrame (single-scene) or pair table CSV/DataFrame (paired-scene) |
| `task` | yes | task YAML path or loaded dict |
| `config` | no | config YAML path, dict, or `None` for defaults (§4 of `pyrestore/config.py`) |
| `cv_features`, `vlm_scores` | no | precomputed analyzer tables for an offline/reproduced run; omit both to call the live CV/VLM analyzers |
| `labels` | no | human category labels for classification validation (paired tasks) |
| `reference` | no | human continuous reference values for correlation validation (single-scene tasks) |
| `out_dir` | no | output directory; defaults to `<config output.dir>/<task_id>`, i.e. `outputs/<task_id>/` |

`reference`/`labels` are only consulted when the task YAML declares a `validation` block (§4);
otherwise the run is reported `not_validated` regardless of what is passed.

Offline/reproduced runs pass `cv_features=` and/or `vlm_scores=` tables carrying the analyzer
contract and provenance columns. Standalone validation and map functions remain importable from
`pyrestore.validate` and `pyrestore.map` for notebook use.

## 6. Outputs (identical for every task)

Every file below is written under one run directory: `out_dir` if you passed it, otherwise
`outputs/<task_id>/` relative to the current working directory (for example
`outputs/restorative_quality_prs11/`). A run first writes to a temporary staging directory next
to that path and only replaces the published one on success, so a failed run never corrupts the
previous result.

| File | Content |
|---|---|
| `observations.csv` | manifest, or all candidate pairs incl. excluded ones; portable image paths + SHA-256 |
| `cv_features.csv`, `vlm_scores.csv` | per-capture / per-pair analyzer outputs with `*_status` |
| `indicators.gpkg`, `indicators.csv` | analysis-ready dataset (EPSG:4326 points) |
| `quality_report.csv` | counts: inputs, valid coordinates, per-analyzer success/error, labelled rows |
| `validation_report.csv` | target status, reference type, overlap, continuous r/ρ or classification selective + overall metrics, missing/invalid/abstention counts |
| `provenance.csv` | actual source model/hash, current task/request hashes, contract-match flag, analyzer modes (`live`, `precomputed`, `injected`), data status, counts and validation status |
| `screening_map.html`, `spatial_stats.csv` | LISA screening map + global Moran's I with permutation p |
| `run_manifest.json` | file set from the latest successful staged run; used to remove stale products |

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `OPENAI_API_KEY not set` | create `.env` (see §2) or export the variable |
| `Remote image transmission is disabled` | review privacy/licence/provider terms, then explicitly set `vlm.allow_remote_images: true` |
| `precomputed_contract_match=false` | scores were produced by a different task/request hash; use source provenance or regenerate them |
| `offline mode: no cached VLM response` | the config sets `vlm.offline: true` but no cached response exists for (image, task, model, prompt) — run live once or supply precomputed tables |
| rows with `vlm_status='error'` | inspect `vlm_error` (schema violations, network); fix and re-run — cached successes are reused |
| `Task ... declares a reference but no validation.mapping` | add the `validation.mapping` (continuous) or label columns (classification) to the task YAML |
| `too few valid ... values for LISA screening` | fewer valid values than `spatial.k`; lower `k` or score more images |
| SegFormer import error | `pip install -e ".[cv]"`, or pass precomputed CV tables |
| slow VLM runs / proxy drops | images are auto-downscaled (`vlm.max_image_px`); the SQLite cache makes re-runs free |

## 8. Interpretation limits (read before publishing results)

- VLM outputs are schema-constrained **interpretations**; validate against a task-appropriate
  reference before quoting any accuracy.
- State the reference type and target status: `no_overlap`, `constant_input` and
  `partially_validated` are not equivalent to `validated`.
- LISA clusters are screening candidates for follow-up investigation.
- Renewal detection needs genuine temporal evidence; a single image can never support that claim.
