"""Claim-specific validation: continuous agreement and pair classification.

Validation is optional to execute but mandatory for accuracy claims: when no reference is
available the pipeline records ``not_validated`` instead of inventing precision. The reference
type is always carried in the report so that, e.g., cross-model agreement is never presented as
ground truth (terminology contract).
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import pearsonr, spearmanr


def validate_continuous(
    pred: pd.DataFrame,
    reference: pd.DataFrame,
    mapping: dict[str, str],
    *,
    id_col: str = "capture_id",
    reference_type: str = "unspecified_reference",
) -> pd.DataFrame:
    """Compare predicted scores against reference values for declared column pairs.

    ``mapping`` is ``{predicted_column: reference_column}``. Returns one tidy row per pair with
    Pearson r, Spearman rho and n. Column pairs missing from either table are reported with
    NaN metrics and n=0 instead of raising, so partially available references stay usable.
    """
    if id_col not in pred.columns or id_col not in reference.columns:
        raise KeyError(f"Both prediction and reference tables must contain {id_col!r}")
    if pred[id_col].duplicated().any() or reference[id_col].duplicated().any():
        raise ValueError(f"Prediction/reference tables must have unique {id_col!r} values")

    reference_columns = {column: f"__reference__{column}" for column in reference.columns if column != id_col}
    reference_renamed = reference.rename(columns=reference_columns)
    merged = pred.merge(reference_renamed, on=id_col, how="inner", validate="one_to_one")
    rows = []
    for pred_col, ref_col in mapping.items():
        ref_merged_col = reference_columns.get(ref_col)
        if pred_col not in pred.columns or ref_col not in reference.columns:
            rows.append(
                {
                    "kind": "continuous",
                    "target": pred_col,
                    "reference_column": ref_col,
                    "reference_type": reference_type,
                    "pearson_r": float("nan"),
                    "pearson_p": float("nan"),
                    "pearson_ci_low": float("nan"),
                    "pearson_ci_high": float("nan"),
                    "spearman_rho": float("nan"),
                    "spearman_p": float("nan"),
                    "n": 0,
                    "n_overlap": int(len(merged)),
                    "status": "missing_column",
                }
            )
            continue
        p = pd.to_numeric(merged[pred_col], errors="coerce")
        r = pd.to_numeric(merged[ref_merged_col], errors="coerce")
        valid = p.notna() & r.notna()
        n = int(valid.sum())
        status = "validated"
        if n >= 2:
            if p[valid].nunique() < 2 or r[valid].nunique() < 2:
                pearson = spearman = float("nan")
                pearson_p = spearman_p = float("nan")
                pearson_ci_low = pearson_ci_high = float("nan")
                status = "constant_input"
            else:
                pearson_result = pearsonr(p[valid], r[valid])
                pearson = float(pearson_result.statistic)
                pearson_p = float(pearson_result.pvalue)
                pearson_ci = pearson_result.confidence_interval(confidence_level=0.95)
                pearson_ci_low = float(pearson_ci.low)
                pearson_ci_high = float(pearson_ci.high)
                spearman_result = spearmanr(p[valid], r[valid])
                spearman = float(spearman_result.statistic)
                spearman_p = float(spearman_result.pvalue)
        else:
            pearson = spearman = float("nan")
            pearson_p = spearman_p = float("nan")
            pearson_ci_low = pearson_ci_high = float("nan")
            status = "no_overlap" if len(merged) == 0 else "insufficient_sample"
        rows.append(
            {
                "kind": "continuous",
                "target": pred_col,
                "reference_column": ref_col,
                "reference_type": reference_type,
                "pearson_r": round(pearson, 4) if pearson == pearson else float("nan"),
                "pearson_p": pearson_p,
                "pearson_ci_low": (
                    round(pearson_ci_low, 4)
                    if pearson_ci_low == pearson_ci_low
                    else float("nan")
                ),
                "pearson_ci_high": (
                    round(pearson_ci_high, 4)
                    if pearson_ci_high == pearson_ci_high
                    else float("nan")
                ),
                "spearman_rho": round(spearman, 4) if spearman == spearman else float("nan"),
                "spearman_p": spearman_p,
                "n": n,
                "n_overlap": int(len(merged)),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def validate_classification(
    pred: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    label_column: str,
    prediction_column: str,
    positive_value: str,
    id_col: str = "pair_id",
    abstain_values: tuple[str, ...] = ("uncertain",),
    negative_values: tuple[str, ...] = ("no",),
    reference_type: str = "unspecified_reference",
) -> pd.DataFrame:
    """Compare a categorical prediction against pair-level reference labels.

    Returns one row with the confusion counts (tp/fp/fn/tn), the number of abstentions (predicted
    values in ``abstain_values`` are neither positive nor negative predictions), and
    precision/recall/F1/accuracy. Abstentions are excluded from the denominator of every rate and
    reported separately — hiding them would overstate confidence.
    """
    if id_col not in pred.columns or id_col not in reference.columns:
        raise KeyError(f"Both prediction and reference tables must contain {id_col!r}")
    if pred[id_col].duplicated().any() or reference[id_col].duplicated().any():
        raise ValueError(f"Prediction/reference tables must have unique {id_col!r} values")
    if label_column not in reference.columns:
        raise KeyError(f"Reference table is missing label column {label_column!r}")
    if prediction_column not in pred.columns:
        raise KeyError(f"Prediction table is missing prediction column {prediction_column!r}")

    reference_label = "__reference_label__"
    merged = pred.merge(
        reference[[id_col, label_column]].rename(columns={label_column: reference_label}),
        on=id_col,
        how="inner",
        validate="one_to_one",
    )

    positive = str(positive_value).strip().lower()
    abstain = tuple(str(v).strip().lower() for v in abstain_values)
    negatives = tuple(str(v).strip().lower() for v in negative_values)
    if positive in negatives or positive in abstain or set(negatives) & set(abstain):
        raise ValueError("positive, negative and abstain values must be disjoint")

    missing_true = merged[reference_label].isna()
    missing_pred = merged[prediction_column].isna()
    y_true = merged[reference_label].astype("string").str.strip().str.lower()
    y_pred = merged[prediction_column].astype("string").str.strip().str.lower()

    true_pos = (y_true == positive).fillna(False)
    true_neg = y_true.isin(list(negatives)).fillna(False)
    pred_pos = (y_pred == positive).fillna(False)
    pred_neg = y_pred.isin(list(negatives)).fillna(False)
    pred_abstain = y_pred.isin(list(abstain)).fillna(False)
    invalid_true = ~missing_true & ~true_pos & ~true_neg
    invalid_pred = ~missing_pred & ~pred_pos & ~pred_neg & ~pred_abstain
    evaluable = (true_pos | true_neg) & (pred_pos | pred_neg)

    tp = int((evaluable & true_pos & pred_pos).sum())
    fp = int((evaluable & true_neg & pred_pos).sum())
    fn = int((evaluable & true_pos & pred_neg).sum())
    tn = int((evaluable & true_neg & pred_neg).sum())
    valid_reference = true_pos | true_neg
    n_abstain = int((valid_reference & pred_abstain).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = float("nan")
    if precision == precision and recall == recall and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else float("nan")
    n_valid_reference = int(valid_reference.sum())
    n_evaluable = int(evaluable.sum())
    coverage = n_evaluable / n_valid_reference if n_valid_reference else float("nan")
    overall_tp = int((true_pos & pred_pos).sum())
    overall_fn = int((true_pos & ~pred_pos).sum())
    overall_tn = int((true_neg & pred_neg).sum())
    overall_recall = (
        overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) else float("nan")
    )
    overall_accuracy = (
        (overall_tp + overall_tn) / n_valid_reference if n_valid_reference else float("nan")
    )
    if len(merged) == 0:
        status = "no_overlap"
    elif n_evaluable == 0:
        status = "no_evaluable_rows"
    else:
        status = "validated"

    return pd.DataFrame(
        [
            {
                "kind": "classification",
                "n": int(len(merged)),
                "n_overlap": int(len(merged)),
                "n_evaluable": n_evaluable,
                "n_missing_reference": int(missing_true.sum()),
                "n_missing_prediction": int((valid_reference & missing_pred).sum()),
                "n_invalid_reference": int(invalid_true.sum()),
                "n_invalid_prediction": int((valid_reference & invalid_pred).sum()),
                "reference_type": reference_type,
                "positive_value": positive,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "n_abstain": n_abstain,
                "precision": round(precision, 4) if precision == precision else float("nan"),
                "recall": round(recall, 4) if recall == recall else float("nan"),
                "f1": round(f1, 4) if f1 == f1 else float("nan"),
                "accuracy": round(accuracy, 4) if accuracy == accuracy else float("nan"),
                "coverage": round(coverage, 4) if coverage == coverage else float("nan"),
                "overall_recall": (
                    round(overall_recall, 4) if overall_recall == overall_recall else float("nan")
                ),
                "overall_accuracy": (
                    round(overall_accuracy, 4)
                    if overall_accuracy == overall_accuracy
                    else float("nan")
                ),
                "status": status,
            }
        ]
    )


def validation_report(
    continuous: pd.DataFrame | None = None, classification: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Combine continuous and classification results into one tidy report table."""
    parts = []
    if continuous is not None and len(continuous):
        parts.append(continuous.assign(kind=continuous["kind"] if "kind" in continuous else "continuous"))
    if classification is not None and len(classification):
        parts.append(classification)
    if not parts:
        return pd.DataFrame([{"kind": "none", "status": "not_validated"}])
    report = pd.concat(parts, ignore_index=True, sort=False)
    if "status" not in report.columns:
        report["status"] = "validated"
    return report
