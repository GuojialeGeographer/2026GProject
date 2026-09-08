# PyRestore Architecture

## 1. Design claim (software-level)

> A shared, explicit data and execution contract can support heterogeneous street-view tasks
> (single-scene and paired-scene) and two evidence families — fixed-taxonomy CV physical
> measurements and schema-constrained VLM semantic interpretations — without rewriting the
> geospatial pipeline, while preserving validation, provenance and GIS interoperability.

The framework does not assume that CV+VLM outperforms either family alone; complementarity is a
case-level empirical question.

## 2. Layers and modules

```text
contract        manifest.py   image-manifest contract (site/capture identity, provenance cols)
                pairs.py      pair contract + comparability audit (make_pairs, load_pairs)
                vlm.load_task task-definition loading/validation (YAML + prompt template)
analyzers       cv.py         SegFormer shares + GVI/SVF/enclosure + colour/edge (lazy heavy deps)
                vlm.py        task-driven VLM calls: messages, cache, retries, offline, parsing
                cache.py      SQLite/WAL cache (JSON responses; SceneSet + request fingerprint)
harmonisation   harmonise.py  manifest-left joins; paired leg suffixes + cv deltas; quality; export
validation      validate.py   validate_continuous (r, ρ) · validate_classification (P/R/F1 + abstain)
products        map.py        projected-KNN Local Moran + two-sided/FDR screening + Folium map
orchestration   pipeline.py   run_task(): one entry point, two input signatures, same exports
```

## 3. Domain entities (as implemented)

| Entity | Where | Contract |
|---|---|---|
| `Site` | `manifest.site_id` | owns spatial identity; filenames never do |
| `Capture` | `manifest.capture_id` | one observation; time/direction come from data (`captured_at`, `heading`), never inferred from pixels |
| `SceneSet` | implicit: manifest row (single) / audited pair (paired) | declared by the task's `input_signature` |
| `TaskDefinition` | `tasks/*.yaml` | owns all semantics: prompt, output fields, validation, spatial screening |
| `AnalyzerRun` | rows in `cv_features.csv` / `vlm_scores.csv` | model, prompt hash, cache hit, per-record status |
| `AnalyzerResult` | `vlm_*` / CV feature columns | two families, never merged into one evidence type |
| `ValidationResult` | `validation_report.csv` | reference type always stated; optional to execute, mandatory for accuracy claims |
| `GeospatialProduct` | `indicators.gpkg/.csv`, maps | EPSG:4326 points; QGIS-readable |

## 4. Core invariants (enforced by code + tests)

1. A VLM task on a single scene cannot produce temporal-comparison fields (single harmonisation
   has no `*_delta` columns — tested).
2. CV and VLM values are never presented as the same type of evidence (family prefixes tested).
3. Failed analyzer runs remain attached to their record (`*_status='error'`, batch continues).
4. Excluded pair candidates remain records with `exclusion_reason` (tested).
5. Validation is optional for execution; target-level states prevent zero overlap, constant input
   or missing columns from being relabelled `validated`.
6. Every run exports provenance: model id + prompt hash + task id + software version + per-channel
   execution mode (`live` vs `precomputed`).
7. Cache is content-addressed, request-scoped and **SceneSet-complete**: the key covers the file
   content of *every* image in the SceneSet, so two pairs sharing a baseline but differing in
   the follow-up never collide. The request fingerprint also covers model, provider id, task
   contract, image size/JPEG quality and token limit.
8. Precomputed analyzer tables pass the same value/schema contract as live outputs. Provenance
   records their actual source model/hash and a current-contract match flag.
9. Outputs are staged; failed runs preserve the previous successful product, while a successful
   run manifest removes stale managed files.

## 5. Cache schema (ER)

```text
vlm_response ──< cache_key TEXT        (image content hash + capture id)
                task_id TEXT           ┐
                model TEXT             ├ PRIMARY KEY (cache_key, task_id, model, prompt_hash)
                prompt_hash TEXT       ┘  (full request fingerprint)
                response_json TEXT     (schema-shaped output, any task)
                raw_text TEXT, created_at TEXT
```

An earlier, restorative-quality-only version of this project hardcoded the four ART columns as
REALs in the cache schema; storing JSON instead is the single schema change that makes the
pipeline task-agnostic.

## 6. Data flow (paired path)

```text
manifest ──make_pairs──► candidates (ok + excluded) ──filter ok──► pair table
   │                                                        │
   └─► CV on both legs (capture_id keyed) ──► *_before/*_after (+ *_delta)
                                                            │
                          VLM pair prompt (before+after images) ──► vlm_* fields
                                                            │
                                  harmonise_paired ──► indicators.gpkg/csv
                                  validate_classification (optional) ──► validation_report
                                  provenance ──► provenance.csv
```

## 7. Extension points (deliberately minimal)

- **New task** — YAML + prompt only (demonstrated by notebook 03 and the sixdim task).
- **New CV analyzer** — any callable `(image_path, cfg) -> dict[str, float]` (the `extractor`
  injection is how the test suite runs without torch).
- **New VLM provider** — anything exposing `chat.completions.create(**kwargs)` (OpenAI SDK
  compatible; the client injection is how tests run offline).
- Future work: sequence (`n>=3`) SceneSets, full JSON-Schema interchange, pluggable validation
  adapters, privacy-preserving image preprocessing and package publication.
