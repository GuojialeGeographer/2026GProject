# data/ — case-study and demo data

Everything a fresh clone needs to run all four notebooks ships in this directory; none of it
requires an external workspace or credentials.

```
data/demo/manifest.csv          12 legacy-format rows (image, image_path, lon, lat)
data/demo/images/*.jpg          flat-colour placeholder JPEGs (real Barcelona photos are not
                                 redistributable; notebook 01 never reads their pixels, only the
                                 precomputed tables below, so the placeholders are functionally
                                 equivalent for the demo)
data/demo/cv_features.csv       precomputed CV physical measurements (capture_id keyed)
data/demo/vlm_scores.csv        precomputed VLM interpretations (capture_id keyed)
data/demo_pairs/                synthetic paired-scene fixtures used by notebook 04
data/shenzhen_prs11/            465-record restorative-quality case (manifest, precomputed CV/VLM
                                 tables, human reference) used by notebook 02
data/shenzhen_sixdim/           465-record six-dimension case (manifest, precomputed CV/VLM
                                 tables, human reference) used by notebook 03
```

- Notebook 01 needs only `data/demo/`.
- Notebook 02 needs only `data/shenzhen_prs11/`.
- Notebook 03 needs only `data/shenzhen_sixdim/`.
- Notebook 04 is self-contained (synthetic/injected analyzers, no external data).

Precomputed tables (CV shares, VLM scores) are frozen artifacts of the author's own runs and carry
their own provenance columns (`vlm_model`, `prompt_hash`). Raw street-view imagery for the two
Shenzhen cases was obtained from the dataset authors and is not redistributed; only the derived,
numeric tables ship. `scripts/prepare_demo.py` regenerates `data/demo/`'s tables from the original
Barcelona source workspace (not part of this repository) — a one-time provenance script, not
something a fresh clone needs to run.
