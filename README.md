# PyRestore

**A Configurable Data Processing Framework for Street-View Perception
Analysis.**

Street-view imagery is a common source of geospatial data for urban analysis. Most existing work
applies image segmentation to measure physical composition: the visible share of sky, vegetation,
road and buildings in a scene. Segmentation does not capture how a place is perceived by someone
walking through it. Qualities such as disorder, neglect or restorative value are visible in a
street-view image but are not represented by a segmentation percentage.

PyRestore processes the same street-view images through two channels: a computer-vision
segmentation channel for physical composition, and a vision-language model (VLM) channel for
perception-based interpretation. The two outputs are joined by capture identity into one
georeferenced, GIS-ready dataset. Each task is a declarative YAML definition with a prompt, an
output schema and an optional validation reference, so a new task is added as a file rather than
a code change.

The case study is urban visual restorative quality, following the ART/PRS-11 construct, run on
465 Shenzhen street-view images and validated against human PRS-11 ratings (Ma and Kwan, 2026,
*Scientific Reports*). This case is the project's focus.

The same pipeline also runs three further tasks with no code changes: a six-dimension perception
audit, a paired before/after renewal-detection demonstration on synthetic images, and a
single-scene renewal-priority screen. They show that the task contract generalizes beyond
restorative quality, but only the restorative-quality case is validated against a human
reference.

PyRestore separates two evidence families:

- **CV physical measurements**: segmentation-based composition (green view index, sky view
  factor, enclosure, per-class shares, colour and edge statistics) from SegFormer.
- **VLM semantic interpretations**: schema-shaped structured outputs from a vision-language
  model, defined entirely by the task file.

> Jiale Guo, MSc Geoinformatics Engineering, Politecnico di Milano. The renewal-pair prompt design
> follows the expert-knowledge-guided scheme of Bai et al. (*Applied Geography*, accepted).

## One run, one product family

```text
manifest.csv (site_id, capture_id, lon/lat, captured_at, ...)
        │                          tasks/*.yaml  (prompt + output contract + validation)
        ├── CV channel ── physical measurements ──┐
        └── VLM channel ── semantic interpretations ─┤
                                                     ▼
              indicators.gpkg / .csv  +  quality_report.csv
              + validation_report.csv  +  provenance.csv
              + screening_map.html (LISA low-score clusters)
```

- `observations.csv`: manifest or audited pair table. Excluded pairs remain records with a reason.
  Exported image paths are relative where possible and full input SHA-256 columns identify bytes.
- `cv_features.csv` / `vlm_scores.csv`: per-capture analyzer outputs with per-record status.
- `indicators.gpkg` + `indicators.csv`: the analysis-ready dataset. QGIS reads the GeoPackage directly.
- `validation_report.csv`: present only what a declared reference supports, otherwise `not_validated`.
- `provenance.csv`: actual source model/hash, current task/request hashes, contract-match flag,
  analyzer modes, software version and quality counts.
- `run_manifest.json`: files owned by the latest successful run. Failed runs leave the previous
  product untouched, and successful re-runs remove stale managed products.

## Installation

```bash
uv sync --extra dev --frozen     # exact versions from uv.lock
uv run pytest --cov              # 75 offline tests; coverage gate >=75%
uv run ruff check pyrestore tests notebooks

# Optional live SegFormer extractor:
uv sync --extra dev --extra cv --frozen
```

Plain `pip install -e ".[dev]"` remains supported, but does not reproduce the exact lock.

## Quickstart

This runs as-is from a fresh clone (offline, no credentials, no network):

```python
from pyrestore import run_task

# Offline demonstration: 12 Barcelona panoramas + precomputed analyzer tables.
result = run_task(
    "data/demo/manifest.csv",
    "tasks/prs11.yaml",
    {"vlm": {"offline": True}},
    cv_features="data/demo/cv_features.csv",
    vlm_scores="data/demo/vlm_scores.csv",
    out_dir="outputs/demo_single",
)
```

Pair derivation follows the same pattern, but needs your own manifest (`my_captures.csv` below is
illustrative, not a file this repository ships):

```python
from pyrestore import make_pairs

pairs = make_pairs("my_captures.csv")   # your own capture manifest, not a shipped file
pairs.to_csv("my_pairs.csv", index=False)
```

The four notebooks walk the whole project: `01` single-scene demo (offline), `02` Shenzhen
PRS-11 replay (validated), `03` offline replay of the real 465-record six-dimension task, and `04`
paired-scene renewal mechanics (synthetic). The notebooks use the Python API directly; PyRestore has
no separate CLI.

## Built-in tasks

| Task | Signature | Purpose | Validation reference |
|---|---|---|---|
| `restorative_quality_prs11` | single_scene | ART/PRS-11-aligned restorative quality, four dimensions and mean (the case study) | human PRS-11 ratings (optional) |
| `urban_perception_sixdim` | single_scene | Place-Pulse-style safe/lively/beautiful/wealthy/boring/depressing audit | human six-dimension ratings (optional) |
| `urban_renewal_pair` | paired_scene | substantive before/after renewal detection with types, nuisance notes and abstention | human pair labels (optional) |
| `renewal_priority_screening` | single_scene | single-scene disrepair, neglect, informality and visual-disorder proxy, explored against restorative quality | none, `not_validated` by design |

## Authoring a new task

1. Write a prompt template (`tasks/prompts/my_task.txt`) ending in the JSON contract;
2. Declare it: `tasks/my_task.yaml` with `id`, `input_signature`, `prompt.template`,
   `output_fields` (`score | number | string | category | list`), optional `derived_fields`,
   `validation` (reference + mapping or label columns) and `spatial` (screening value column);
3. Run: `run_task("m.csv", "tasks/my_task.yaml")`.

No framework file changes. Precomputed tables pass the same output contract as live replies.
Their recorded source model/hash remains authoritative; if it differs from the current task,
`precomputed_contract_match=false` is written instead of silently relabelling the old scores.

## Data and reproducibility

1. **Offline, zero credentials, zero setup.** `data/demo/manifest.csv`, `cv_features.csv` and
   `vlm_scores.csv` (12 Barcelona panoramas, numeric tables only) ship in this repository.
   `pyrestore.load_manifest` checks that every referenced image file exists by design, so
   `data/demo/images/` ships too — as flat-colour placeholder JPEGs, since the original Barcelona
   photos are not redistributable. Notebook 01 never derives any value from their pixels (the CV
   and VLM channels both read the precomputed tables above), so the placeholders satisfy the
   existence check without affecting a single reported number. A fresh clone can run the notebook
   with no credentials and no network access. The test suite needs neither images nor credentials
   and is fully offline.
2. **Case studies.** Notebooks 02 and 03 reuse derived Shenzhen tables for 465 labelled captures
   (also shipped as precomputed tables, same reasoning: no image pixels are read). Notebook 03
   reproduces the authorized six-dimension VLM run without another remote request. Raw Shenzhen
   imagery was obtained from the dataset authors and is **not** redistributed.
3. **Live extraction.** `run_task` without precomputed tables calls the SegFormer extractor
   (optional `cv` extra) and an OpenAI-compatible VLM endpoint. Remote image transmission is off
   by default. Set `vlm.allow_remote_images: true` only after checking privacy, image licence and
   provider terms. Successful responses are request-fingerprint cached, so cached re-runs are
   stable, though the first uncached model call may still be nondeterministic.

## Limitations

- VLM scores are perceptual interpretations, not measurements of human experience.
- Cross-model agreement (CV vs a reference segmentation) is not field-survey truth.
- LISA low-score clusters are candidates for follow-up investigation, not measured health
  outcomes or confirmed intervention sites.
- Local weights are built in an estimated projected CRS. Zero-variance inputs are rejected, and
  local pseudo-p values are two-sided and FDR-adjusted before Low-Low candidates are flagged.
- A single-image condition score is not renewal detection. The renewal detector classifies
  substantive interventions and does not measure policy effectiveness or justice.

See `docs/TERMINOLOGY.md` for the full contract, `docs/ARCHITECTURE.md` for the design,
`docs/REQUIREMENTS.md` for the functional and non-functional requirements catalogue, and
`docs/CASE_B_PROTOCOL.md` for the paired-data protocol.

## License

MIT. See [`LICENSE`](LICENSE).
