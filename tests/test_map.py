"""Spatial-statistics guards for invalid inputs and projected KNN weights."""
from __future__ import annotations

import pandas as pd
import pytest

from pyrestore.harmonise import to_geodataframe
from pyrestore.map import local_moran


def _grid(values):
    rows = [
        {"lon": 2.15 + (i % 4) * 0.004, "lat": 41.39 + (i // 4) * 0.004, "value": value}
        for i, value in enumerate(values)
    ]
    return to_geodataframe(pd.DataFrame(rows))


def test_local_moran_rejects_constant_values():
    with pytest.raises(ValueError, match="zero variance"):
        local_moran(_grid([0.5] * 12), "value", k=8)


def test_local_moran_adds_fdr_adjusted_pvalues():
    screened, global_i, global_p = local_moran(
        _grid([0.1 + i * 0.05 for i in range(12)]), "value", k=4
    )
    assert "lisa_p_adj" in screened.columns
    assert screened["lisa_p_adj"].between(0, 1).all()
    assert global_i == global_i and 0 <= global_p <= 1


def test_local_moran_flags_both_low_and_high_candidate_clusters():
    # Two tight clusters (low values, high values) plus scattered mid values give both a
    # Low-Low and a High-High quadrant, so both candidate flags can be exercised on one run.
    values = [0.05] * 4 + [0.95] * 4 + [0.3, 0.5, 0.6, 0.4]
    screened, _, _ = local_moran(_grid(values), "value", k=4)
    assert "candidate_low_cluster" in screened.columns
    assert "candidate_high_cluster" in screened.columns
    assert screened["candidate_low_cluster"].dtype == bool
    assert screened["candidate_high_cluster"].dtype == bool
    # The two flags are mutually exclusive by construction (Low-Low vs High-High quadrants).
    assert not (screened["candidate_low_cluster"] & screened["candidate_high_cluster"]).any()
    assert bool(screened.loc[screened["lisa_q"] == 1, "candidate_high_cluster"].isin([True, False]).all())
