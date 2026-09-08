# PyRestore requirements and data contract

## 1. Purpose

PyRestore converts authorized, georeferenced street-view observations into traceable multimodal indicator products. It keeps computer-vision estimates of visible composition separate from VLM semantic interpretations, supports single-scene and audited paired-scene tasks, and exports quality, validation, provenance and GIS products through a Python API and executable notebooks.

## 2. Users and external systems

- Researcher or geospatial analyst: prepares the manifest, task and optional references and interprets quality and validation reports.
- GIS practitioner: inspects the GeoPackage and maps as evidence for follow-up, not as an automatic intervention decision.
- Image provider: supplies imagery with source and licence information.
- VLM provider: processes images only after explicit remote-transmission authorization.
- Reference-data provider: supplies task-appropriate human labels or reference model outputs.

## 3. Functional requirements

| ID | Requirement | Priority | Evidence |
|---|---|---|---|
| FR-1 | Load a manifest with stable site/capture identity, image path and WGS84 coordinates. | MUST | `manifest.py`, tests |
| FR-2 | Support `single_scene` and `paired_scene` SceneSets. | MUST | `pairs.py`, task files, tests |
| FR-3 | Re-audit external pairs for dates, distance, heading and before/after identity. | MUST | `audit_pairs`, tests |
| FR-4 | Load and validate declarative task prompts, output fields, derived fields, validation mappings and spatial settings. | MUST | `vlm.load_task`, tests |
| FR-5 | Produce CV scene-composition indicators and per-record status/error fields. | MUST | `cv.py`, tests |
| FR-6 | Produce schema-constrained VLM interpretations for one or two images, with retries and per-record failure preservation. | MUST | `vlm.py`, tests |
| FR-7 | Cache VLM results using every image in the SceneSet and a request fingerprint. | MUST | `cache.py`, cache regression tests |
| FR-8 | Validate live and precomputed analyzer tables against the same task contract. | MUST | `pipeline.py`, tests |
| FR-9 | Harmonise observations, both analyzer families and optional human labels without dropping failed inputs. | MUST | `harmonise.py`, tests |
| FR-10 | Report continuous agreement with confidence intervals and classification performance with coverage and abstention. | MUST | `validate.py`, tests |
| FR-11 | Compute projected-KNN global/local spatial statistics and control local false discovery rate. | SHOULD | `map.py`, tests |
| FR-12 | Export observations, analyzer tables, indicators, quality, validation, provenance and a managed run manifest. | MUST | `pipeline.py`, end-to-end tests |
| FR-13 | Export the analysis-ready indicator table as CSV and GeoPackage. | MUST | `harmonise.py`, round-trip test |
| FR-14 | Demonstrate the public Python API through four executed notebooks. | MUST | `notebooks/01-04` |

## 4. Non-functional requirements

- Reproducibility: pin the environment with `uv.lock`; record software/task/model/request identities and input SHA-256 values.
- Failure transparency: preserve errors and excluded pair candidates as records.
- Responsible interpretation: distinguish physical estimates, semantic interpretations, validation references and screening candidates.
- Privacy: disable remote image transmission by default and require explicit configuration to enable it.
- Portability: resolve input paths relative to their manifests and export relocatable paths where possible.
- Testability: keep the default suite offline and require at least 75% branch-aware coverage.
- Usability: expose Python functions and notebooks only; do not maintain a separate command-line interface.
- GIS interoperability: produce valid EPSG:4326 GeoPackages and use QGIS for submission cartography.

## 5. Use cases

### UC-1: Execute an unlabelled single-scene task

The analyst supplies a manifest and task definition. PyRestore extracts or reads CV/VLM results, preserves processing states and exports GIS-ready indicators with `not_validated` status.

### UC-2: Validate a single-scene interpretation task

The analyst supplies a task-appropriate human reference. PyRestore reports overlap, target status, Pearson/Spearman agreement, confidence intervals and calibration evidence.

### UC-3: Execute a paired-scene task

The analyst supplies a temporal manifest or pair table. PyRestore audits comparability, attaches before/after CV values and deltas, and produces structured pair interpretations without treating a single image as change evidence.

### UC-4: Screen spatial patterns

The analyst selects a valid numeric indicator. PyRestore constructs projected spatial weights, reports global autocorrelation and flags local candidates only after the declared multiple-testing correction.

## 6. Acceptance criteria

1. All 75 offline tests and the coverage/static-analysis gates pass in the locked environment.
2. Four notebooks execute from clean kernels without hidden exceptions.
3. A 465-record real six-dimension task produces complete analyzer, validation, provenance and GIS products.
4. Historical PRS results replay without losing their source model/hash or disguising a task-contract mismatch.
5. The paired mechanism demo records both analyzers as injected and the data as synthetic.
6. Zero-overlap, constant-input, missing-value and zero-variance cases cannot be reported as validated spatial findings.
