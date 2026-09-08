"""Spatial analysis and interactive screening maps.

Local Moran's I (LISA) identifies statistically significant clusters of an indicator value.
Significant Low-Low clusters are presented as *candidates for follow-up investigation* — never as
measured health outcomes or confirmed intervention sites. The global statistic is always reported
with its permutation p-value; "significant" without the p-value is prohibited wording.
"""
from __future__ import annotations

import warnings
from typing import Any

_LISA_LABELS = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}


def local_moran(
    gdf: Any,
    value_col: str,
    k: int = 8,
    seed: int = 42,
    p_threshold: float = 0.05,
    p_adjust: str = "fdr_bh",
) -> tuple[Any, float, float]:
    """Compute Local Moran's I for ``value_col``.

    Adds columns: ``local_I``, ``lisa_q`` (1=HH,2=LH,3=LL,4=HL), ``lisa_p``, ``lisa_label``, the
    ``candidate_low_cluster`` flag (significant Low-Low) and the ``candidate_high_cluster`` flag
    (significant High-High). Both flags use the same adjusted p-value and threshold; which one is
    the relevant "candidate" for a given task depends on that task's own value_col — e.g. a low
    restorative-quality score or a high renewal-priority score. Returns
    ``(gdf, global_moran_I, global_moran_p)`` — the global p-value is a permutation-based pseudo
    p-value (999 permutations, esda).
    """
    import numpy as np
    from esda.moran import Moran, Moran_Local
    from libpysal.weights import KNN

    gdf = gdf.copy()
    if value_col not in gdf.columns:
        raise KeyError(f"value column {value_col!r} not found")
    if not 0.0 < p_threshold < 1.0:
        raise ValueError("p_threshold must be between 0 and 1")
    if p_adjust not in {"none", "fdr_bh"}:
        raise ValueError("p_adjust must be 'none' or 'fdr_bh'")
    if k < 1 or len(gdf) <= k:
        raise ValueError(f"local Moran requires at least k+1 rows (n={len(gdf)}, k={k})")
    if gdf.crs is None:
        raise ValueError("local Moran requires a declared CRS")
    y = np.asarray(gdf[value_col], dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f"value column {value_col!r} contains non-finite values")
    if np.unique(y).size < 2 or float(np.var(y)) <= np.finfo(float).eps:
        raise ValueError(f"value column {value_col!r} has zero variance; Moran statistics are undefined")
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ValueError("local Moran requires non-empty geometry for every row")

    weights_gdf = gdf
    if gdf.crs.is_geographic:
        projected_crs = gdf.estimate_utm_crs()
        if projected_crs is None:
            raise ValueError("could not estimate a projected CRS for spatial weights")
        weights_gdf = gdf.to_crs(projected_crs)
    unique_locations = len(
        set(zip(weights_gdf.geometry.x, weights_gdf.geometry.y, strict=True))
    )
    if unique_locations <= k:
        raise ValueError(
            f"local Moran needs more than k={k} unique locations; found {unique_locations}"
        )

    w = KNN.from_dataframe(weights_gdf, k=k)
    w.transform = "r"

    # esda's permutation pseudo-p is combined with the quadrant below; only significant Low-Low
    # observations become candidates. Avoid version-specific ``alternative`` parameters so the
    # locked environment and supported esda releases share the same call contract.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The alternative hypothesis for conditional randomization is changing.*",
            category=DeprecationWarning,
        )
        ml = Moran_Local(y, w, permutations=999, seed=seed)
    # Supported esda versions expose the historical directed pseudo-p but do not all accept an
    # ``alternative`` constructor argument. Doubling gives a conservative two-sided pseudo-p and
    # keeps behavior stable across those versions.
    raw_local_p = np.minimum(1.0, 2.0 * ml.p_sim)
    adjusted_p = _adjust_pvalues(raw_local_p, method=p_adjust)
    sig = adjusted_p < p_threshold
    gdf["local_I"] = ml.Is
    gdf["lisa_q"] = ml.q
    gdf["lisa_p"] = raw_local_p
    gdf["lisa_p_adj"] = adjusted_p
    gdf["lisa_label"] = [
        (_LISA_LABELS[q] if s else "Not significant")
        for q, s in zip(ml.q, sig, strict=True)
    ]
    gdf["candidate_low_cluster"] = (ml.q == 3) & sig
    gdf["candidate_high_cluster"] = (ml.q == 1) & sig

    # ``Moran`` uses NumPy's global random state for permutations.
    random_state = np.random.get_state()
    try:
        np.random.seed(seed)
        global_moran = Moran(y, w, permutations=999)
    finally:
        np.random.set_state(random_state)
    return gdf, float(global_moran.I), float(global_moran.p_sim)


def _adjust_pvalues(values: Any, method: str) -> Any:
    """Return raw or Benjamini-Hochberg adjusted local permutation p-values."""
    import numpy as np

    p = np.asarray(values, dtype=float)
    if method == "none":
        return p
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def make_map(
    gdf: Any,
    value_col: str,
    out_html: str,
    candidate_col: str = "candidate_low_cluster",
    boundary: Any | None = None,
    boundary_name: str = "Administrative boundary",
) -> str:
    """Render an interactive screening map.

    ``boundary`` is optional because a new study area may use any administrative geography. When
    supplied, it is reprojected to WGS 84 and drawn beneath the observations so the map retains a
    recognisable spatial frame without depending on a commercial basemap.
    """
    import branca.colormap as cm
    import folium

    points_wgs84 = gdf.to_crs("EPSG:4326")
    center = [float(points_wgs84.geometry.y.mean()), float(points_wgs84.geometry.x.mean())]
    m = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")
    extent_frames = [points_wgs84.total_bounds]
    if boundary is not None:
        boundary_wgs84 = boundary.to_crs("EPSG:4326")
        extent_frames.append(boundary_wgs84.total_bounds)
        folium.GeoJson(
            boundary_wgs84.__geo_interface__,
            name=boundary_name,
            style_function=lambda _feature: {
                "color": "#4d4d4d",
                "weight": 1.5,
                "fillColor": "#f2f2f2",
                "fillOpacity": 0.08,
            },
        ).add_to(m)

    # A fixed 0–1 scale gives the same colour the same meaning in every city.
    cmap = cm.LinearColormap(
        ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"],
        vmin=0.0,
        vmax=1.0,
        caption=f"{value_col} screening score (fixed 0-1 scale)",
    )

    pts = folium.FeatureGroup(name=f"Screening score ({value_col})")
    for _, r in points_wgs84.iterrows():
        folium.CircleMarker(
            [r.geometry.y, r.geometry.x],
            radius=3,
            weight=0,
            fill=True,
            fill_color=cmap(r[value_col]),
            fill_opacity=0.7,
            popup=f"{value_col}={r[value_col]:.2f}",
        ).add_to(pts)
    pts.add_to(m)

    if candidate_col in gdf.columns:
        high = candidate_col == "candidate_high_cluster"
        label = "Candidate high-score cluster" if high else "Candidate low-score cluster"
        quadrant = "High-High" if high else "Low-Low"
        candidates = folium.FeatureGroup(name=f"{label}s (significant {quadrant})")
        for _, r in points_wgs84[points_wgs84[candidate_col]].iterrows():
            folium.CircleMarker(
                [r.geometry.y, r.geometry.x],
                radius=5,
                color="black",
                weight=1,
                fill=False,
                popup=label,
            ).add_to(candidates)
        candidates.add_to(m)

    west = min(bounds[0] for bounds in extent_frames)
    south = min(bounds[1] for bounds in extent_frames)
    east = max(bounds[2] for bounds in extent_frames)
    north = max(bounds[3] for bounds in extent_frames)
    m.fit_bounds([[south, west], [north, east]], padding=(18, 18))
    cmap.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(out_html)
    return out_html
