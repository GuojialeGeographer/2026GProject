# Case B Protocol — Paired-Scene Urban-Renewal Detection

Status: **mechanics implemented and demonstrated (notebook 04, synthetic); real panel data not
yet acquired.** This document is the protocol for turning the mechanics into a substantive case.

## 1. What counts as a pair

Each pair must satisfy the `pair_contract` (declared in `tasks/renewal.yaml`):

- one stable `site_id`;
- baseline and follow-up `captured_at` (known, different);
- spatial proximity within `max_distance_m` (default 20 m);
- comparable viewing direction within `max_heading_diff_deg` (default 30°) when headings exist;
- source and licence metadata;
- a `comparability_status` and — when excluded — an `exclusion_reason` (records, not drops);
- manual reference labels for at least a validation subset.

## 2. Data routes (decision order)

| Route | Coverage | Licence | Notes |
|---|---|---|---|
| **Mapillary (Milan/Barcelona pilot)** | multi-year sequences along streets | CC-BY-SA — redistributable | 1-day coverage probe first; aligns with an open-source framework and ships a demo subset |
| **Baidu historical panoramas (Shenzhen)** | best for Chinese cities; same-city dialogue with Ma & Kwan and Bai et al. | restricted | ship sampling points + derived outputs only (SAGAI precedent) |
| **Fallback** | — | — | deliver contract + taxonomy + validation design on a small public sample; software contribution stands via the add-a-task demonstration |

Scale: 200–500 audited pairs suffice for a methods demonstration (Bai et al. used ~105k pairs for
a city-scale study; that is a scaling question, not a methods question).

## 3. Taxonomy and prompt design (from Bai et al., Applied Geography, accepted)

- **7 non-overlapping taxonomy types (multi-label selection allowed)**: greening_landscape · road_traffic ·
  building_facade_material · public_facility_street_furniture · spatial_interface ·
  commercial_signage · other (multi-select allowed);
- **3 explicit negative rules** injected into the prompt (the F1 0.721 → 0.798 lever):
  natural evolution (season, weather, light), temporary use (hoarding, vehicles, stalls),
  routine maintenance (repaint-in-place, like-for-like replacement);
- structured output with evidence, nuisance list, confidence and an explicit
  `uncertain` abstention;
- model pre-experiment on ~50 pairs across ≥2 providers before locking one model.

## 4. Validation design

- Human pair labels: pilot ~50 pairs to calibrate the taxonomy, then 150–300 pairs;
  include **≥30% nuisance negatives**;
- single annotator: repeat-label ~30 pairs after two weeks and report self-consistency;
  two annotators: report inter-annotator agreement;
- metrics: selective Precision/Recall/F1, coverage/abstention, overall metrics that count abstention
  as unresolved/error, and the false-positive rate on nuisance negatives;
- repeatability: re-run ~30 pairs after two weeks; report output drift;
- sample-size rationale recorded in the run notes (expected positive rate → CI width), not
  chosen by convenience.

## 5. Prohibited claims (see TERMINOLOGY.md)

The detector classifies substantive visible interventions. It does not establish when an
element was installed, whether a condition is new, the causal impact of an intervention, or its
distributional justice — those require external socioeconomic/policy data and a separate design.
