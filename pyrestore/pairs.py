"""Paired-scene contract: comparable before/after captures at the same site.

``make_pairs`` derives candidate pairs from a manifest and audits every candidate against an
explicit comparability contract: spatial tolerance, viewing-direction tolerance and known
capture dates. Pairs that fail the audit **remain in the output** with
``comparability_status='excluded'`` and an ``exclusion_reason`` — exclusions are records, not
silent drops (framework invariant).

A pair is a claim about change; a single capture can never produce one. ``load_pairs`` reads an
externally supplied pair table carrying the same contract columns, for example produced by
``pyrestore.make_pairs`` or a data provider.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .manifest import load_manifest

PAIR_REQUIRED_COLUMNS = [
    "pair_id",
    "site_id",
    "before_capture_id",
    "after_capture_id",
    "before_image_path",
    "after_image_path",
    "before_lon",
    "before_lat",
    "after_lon",
    "after_lat",
]
PAIR_OPTIONAL_COLUMNS = [
    "before_captured_at",
    "after_captured_at",
    "heading_diff_deg",
    "distance_m",
    "comparability_status",
    "exclusion_reason",
]


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two WGS 84 points, in metres."""
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _heading_diff(a: float, b: float) -> float:
    """Absolute angular difference in degrees, folded into [0, 180]."""
    diff = abs(float(a) - float(b)) % 360.0
    return min(diff, 360.0 - diff)


def _known_date(value: object) -> bool:
    """A capture date is known when the manifest supplied a non-empty value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    text = str(value).strip()
    return text != "" and text.lower() != "nan"


def _date_key(value: object) -> tuple[str, Any]:
    """Classify a capture-date value as ``("missing"|"timestamp"|"text", comparable_key)``.

    Dates that pandas can parse (``2017-03``, ``2017/3/5``, ISO strings, …) are compared as
    timestamps, so formatting variants such as ``2017-3`` vs ``2017-03`` are recognised as the
    same capture date; anything else known is compared as plain text.
    """
    if not _known_date(value):
        return "missing", None
    text = str(value).strip()
    try:
        ts = pd.to_datetime(text, errors="coerce")
    except (TypeError, ValueError):  # pragma: no cover — errors="coerce" rarely raises
        ts = pd.NaT
    if pd.notna(ts):
        return "timestamp", ts
    return "text", text.lower()


def _is_number(value: object) -> bool:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed)


def make_pairs(
    manifest: str | os.PathLike[str] | pd.DataFrame,
    *,
    max_distance_m: float = 20.0,
    max_heading_diff_deg: float = 30.0,
    require_captured_at: bool = True,
) -> pd.DataFrame:
    """Derive audited before/after candidate pairs from an image manifest.

    Within each ``site_id``, every combination of two captures becomes one candidate pair. Each
    candidate carries the audit outcome: ``comparability_status='ok'`` or ``'excluded'`` with the
    first failing rule as ``exclusion_reason``. When both capture dates are known, the earlier
    capture is the baseline ("before"). Deterministic given a deterministic manifest order.
    """
    manifest = load_manifest(manifest, check_files=False)
    manifest = manifest.copy()
    manifest["heading"] = pd.to_numeric(manifest["heading"], errors="coerce")

    rows: list[dict[str, object]] = []
    for site_id, group in manifest.groupby("site_id", sort=True):
        records = group.to_dict("records")
        for first, second in itertools.combinations(records, 2):
            distance = haversine_m(
                float(first["lon"]), float(first["lat"]), float(second["lon"]), float(second["lat"])
            )
            if _is_number(first["heading"]) and _is_number(second["heading"]):
                heading_diff: float | None = _heading_diff(first["heading"], second["heading"])
            else:
                heading_diff = None

            first_kind, first_key = _date_key(first["captured_at"])
            second_kind, second_key = _date_key(second["captured_at"])

            reason = ""
            if require_captured_at and (first_kind == "missing" or second_kind == "missing"):
                reason = "missing_captured_at"
            elif first_kind == second_kind != "missing" and first_key == second_key:
                reason = "same_capture_date"
            elif distance > max_distance_m:
                reason = "distance_exceeds_tolerance"
            elif heading_diff is not None and heading_diff > max_heading_diff_deg:
                reason = "heading_diff_exceeds_tolerance"

            before, after = first, second
            if first_kind == second_kind != "missing" and second_key < first_key:
                before, after = second, first

            rows.append(
                {
                    "pair_id": _pair_id(
                        str(site_id), str(before["capture_id"]), str(after["capture_id"])
                    ),
                    "site_id": site_id,
                    "before_capture_id": before["capture_id"],
                    "after_capture_id": after["capture_id"],
                    "before_image_path": before["image_path"],
                    "after_image_path": after["image_path"],
                    "before_captured_at": before["captured_at"],
                    "after_captured_at": after["captured_at"],
                    "before_lon": before["lon"],
                    "before_lat": before["lat"],
                    "after_lon": after["lon"],
                    "after_lat": after["lat"],
                    "heading_diff_deg": heading_diff,
                    "distance_m": round(distance, 3),
                    "comparability_status": "ok" if reason == "" else "excluded",
                    "exclusion_reason": reason,
                }
            )

    columns = PAIR_REQUIRED_COLUMNS + PAIR_OPTIONAL_COLUMNS
    if not rows:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(rows)
    return table[columns]


def _pair_id(site_id: str, before_capture_id: str, after_capture_id: str) -> str:
    """Stable pair identity independent of manifest row order and unrelated candidates."""
    payload = "\x1f".join((site_id, before_capture_id, after_capture_id)).encode("utf-8")
    return f"pair_{hashlib.sha256(payload).hexdigest()[:12]}"


def load_pairs(
    pairs: str | os.PathLike[str] | pd.DataFrame,
    *,
    check_files: bool = True,
) -> pd.DataFrame:
    """Load and validate an externally supplied pair table.

    Required columns are ``PAIR_REQUIRED_COLUMNS``. Rows lacking an audit outcome are marked
    ``comparability_status='unaudited'`` — the caller decides whether unaudited pairs may be
    processed; ``run_task`` only processes ``'ok'`` pairs.
    """
    if isinstance(pairs, pd.DataFrame):
        table = pairs.copy()
        base_dir = Path.cwd()
    else:
        pairs_path = Path(pairs).expanduser().resolve()
        if not pairs_path.exists():
            raise FileNotFoundError(f"Pairs table not found: {pairs_path}")
        table = pd.read_csv(pairs_path)
        base_dir = pairs_path.parent

    missing = [column for column in PAIR_REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"Pairs table is missing required columns: {missing}")
    if table.empty:
        raise ValueError("Pairs table is empty.")
    for column in ("pair_id", "site_id", "before_capture_id", "after_capture_id"):
        if table[column].isna().any():
            raise ValueError(f"Pairs table contains missing {column} values.")
        table[column] = table[column].astype(str).str.strip()
        if table[column].eq("").any():
            raise ValueError(f"Pairs table contains empty {column} values.")
    if table["pair_id"].duplicated().any():
        duplicates = table.loc[table["pair_id"].duplicated(), "pair_id"].tolist()[:5]
        raise ValueError(f"Pairs table contains duplicate pair_id values: {duplicates}")

    for column in PAIR_OPTIONAL_COLUMNS:
        if column not in table.columns:
            table[column] = pd.NA
    table["comparability_status"] = table["comparability_status"].fillna("unaudited")
    table["comparability_status"] = table["comparability_status"].astype(str).str.strip().str.lower()
    invalid_status = sorted(
        set(table["comparability_status"]) - {"ok", "excluded", "unaudited"}
    )
    if invalid_status:
        raise ValueError(f"Pairs table contains invalid comparability_status values: {invalid_status}")
    if (table["before_capture_id"] == table["after_capture_id"]).any():
        raise ValueError("Pairs table cannot use the same capture as both before and after.")

    def _resolve_image_path(value: object) -> str:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return str(path.resolve())

    for column in ("before_image_path", "after_image_path"):
        if table[column].isna().any() or table[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Pairs table contains missing or empty {column} values.")
        table[column] = table[column].map(_resolve_image_path)
    if (table["before_image_path"] == table["after_image_path"]).any():
        raise ValueError("Pairs table cannot use the same image file as both before and after.")

    for lon_column, lat_column in (("before_lon", "before_lat"), ("after_lon", "after_lat")):
        table[lon_column] = pd.to_numeric(table[lon_column], errors="coerce")
        table[lat_column] = pd.to_numeric(table[lat_column], errors="coerce")
        if table[[lon_column, lat_column]].isna().any().any():
            raise ValueError("Pairs table contains missing or non-numeric coordinates.")
        if not table[lon_column].between(-180, 180).all() or not table[lat_column].between(-90, 90).all():
            raise ValueError("Pairs table contains coordinates outside longitude/latitude ranges.")

    if check_files:
        missing_files = [
            str(p)
            for p in list(table["before_image_path"]) + list(table["after_image_path"])
            if not Path(str(p)).is_file()
        ]
        if missing_files:
            sample = missing_files[:3]
            raise FileNotFoundError(
                f"Pairs table references {len(missing_files)} missing image(s): {sample}"
            )
    return table.reset_index(drop=True)


def audit_pairs(
    table: pd.DataFrame,
    *,
    max_distance_m: float = 20.0,
    max_heading_diff_deg: float = 30.0,
    require_captured_at: bool = True,
) -> pd.DataFrame:
    """Recompute the machine-checkable pair contract for ok/unaudited external rows.

    Explicitly excluded rows remain excluded with their supplied reason. Rows marked ok or
    unaudited are never trusted blindly: distance, dates and heading difference are checked again.
    """
    out = table.copy()
    out["comparability_status"] = out["comparability_status"].astype("string")
    out["exclusion_reason"] = out["exclusion_reason"].astype("string").fillna("")
    out["distance_m"] = pd.to_numeric(out["distance_m"], errors="coerce")
    for index, row in out.iterrows():
        if row.get("comparability_status") == "excluded":
            supplied_reason = row.get("exclusion_reason")
            if pd.isna(supplied_reason) or not str(supplied_reason).strip():
                out.at[index, "exclusion_reason"] = "externally_excluded"
            continue
        before_kind, before_key = _date_key(row.get("before_captured_at"))
        after_kind, after_key = _date_key(row.get("after_captured_at"))
        distance = haversine_m(
            float(row["before_lon"]),
            float(row["before_lat"]),
            float(row["after_lon"]),
            float(row["after_lat"]),
        )
        out.at[index, "distance_m"] = round(distance, 3)
        heading = row.get("heading_diff_deg")
        reason = ""
        if require_captured_at and (before_kind == "missing" or after_kind == "missing"):
            reason = "missing_captured_at"
        elif before_kind != after_kind:
            reason = "incomparable_capture_dates"
        elif before_kind == after_kind != "missing" and before_key == after_key:
            reason = "same_capture_date"
        elif before_kind == after_kind != "missing" and after_key < before_key:
            reason = "capture_order_invalid"
        elif distance > max_distance_m:
            reason = "distance_exceeds_tolerance"
        elif _is_number(heading):
            if not 0.0 <= float(heading) <= 180.0:
                reason = "invalid_heading_diff"
            elif float(heading) > max_heading_diff_deg:
                reason = "heading_diff_exceeds_tolerance"
        out.at[index, "comparability_status"] = "ok" if not reason else "excluded"
        out.at[index, "exclusion_reason"] = reason
    return out
