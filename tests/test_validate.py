"""Validation tests: continuous agreement and classification with abstentions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyrestore.validate import validate_classification, validate_continuous


def test_validate_continuous_perfect_and_missing_pairs():
    pred = pd.DataFrame(
        {"capture_id": [f"c{i}" for i in range(5)], "vlm_x": [0.1, 0.3, 0.5, 0.7, 0.9]}
    )
    reference = pd.DataFrame(
        {"capture_id": [f"c{i}" for i in range(5)], "x": [0.1, 0.3, 0.5, 0.7, 0.9], "y": [1, 2, 3, 4, 5]}
    )
    report = validate_continuous(pred, reference, {"vlm_x": "x", "vlm_missing": "y"})
    assert report.loc[0, "pearson_r"] == 1.0
    assert report.loc[0, "pearson_p"] == pytest.approx(0.0)
    assert report.loc[0, "pearson_ci_low"] == pytest.approx(1.0)
    assert report.loc[0, "pearson_ci_high"] == pytest.approx(1.0)
    assert report.loc[0, "spearman_p"] == pytest.approx(0.0)
    assert report.loc[0, "n"] == 5
    # Missing predicted column reported, not raised.
    missing = report[report["target"] == "vlm_missing"].iloc[0]
    assert np.isnan(missing["pearson_r"]) and missing["n"] == 0
    assert missing["status"] == "missing_column"


def test_validate_classification_counts_abstentions():
    pred = pd.DataFrame(
        {
            "pair_id": [f"p{i}" for i in range(5)],
            "vlm_renewal_detected": ["yes", "yes", "uncertain", "no", "yes"],
        }
    )
    reference = pd.DataFrame(
        {"pair_id": [f"p{i}" for i in range(5)], "renewal_label": ["yes", "yes", "yes", "no", "no"]}
    )
    report = validate_classification(
        pred,
        reference,
        label_column="renewal_label",
        prediction_column="vlm_renewal_detected",
        positive_value="yes",
    )
    row = report.iloc[0]
    assert row["tp"] == 2 and row["fp"] == 1 and row["fn"] == 0 and row["tn"] == 1
    assert row["n_abstain"] == 1
    assert row["precision"] == pytest.approx(2 / 3, abs=1e-3)
    assert row["recall"] == 1.0
    assert row["f1"] == pytest.approx(0.8)
    assert row["accuracy"] == pytest.approx(0.75)
    assert row["coverage"] == pytest.approx(0.8)
    assert row["overall_recall"] == pytest.approx(2 / 3, abs=1e-3)


def test_validate_classification_requires_columns():
    pred = pd.DataFrame({"pair_id": ["p0"], "other": ["yes"]})
    reference = pd.DataFrame({"pair_id": ["p0"], "renewal_label": ["yes"]})
    with pytest.raises(KeyError, match="prediction column"):
        validate_classification(
            pred, reference, label_column="renewal_label", prediction_column="vlm_renewal_detected",
            positive_value="yes",
        )


def test_continuous_zero_overlap_is_not_validated():
    pred = pd.DataFrame({"capture_id": ["p"], "vlm_x": [0.5]})
    reference = pd.DataFrame({"capture_id": ["r"], "x": [0.5]})
    report = validate_continuous(pred, reference, {"vlm_x": "x"})
    assert report.iloc[0]["status"] == "no_overlap"
    assert report.iloc[0]["n_overlap"] == 0


def test_missing_classification_values_are_not_true_negatives():
    pred = pd.DataFrame({"pair_id": ["p1", "p2"], "pred": ["yes", None]})
    reference = pd.DataFrame({"pair_id": ["p1", "p2"], "truth": ["yes", None]})
    row = validate_classification(
        pred,
        reference,
        label_column="truth",
        prediction_column="pred",
        positive_value="yes",
    ).iloc[0]
    assert row["tn"] == 0
    assert row["n_missing_reference"] == 1
    assert row["n_evaluable"] == 1
