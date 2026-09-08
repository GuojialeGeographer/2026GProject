"""Image-manifest contract: the spatial-identity layer every task shares.

A manifest is one row per **capture** — one image observation at a site, time and direction.
Required columns are ``site_id, capture_id, image_path, lon, lat``; optional provenance columns
are ``captured_at, heading, camera, source, licence``. ``site_id`` owns spatial identity; file
names do not. Legacy four-column manifests (``image, image_path, lon, lat``, the restorative-
quality-only contract this project started from) are adapted automatically so existing
case-study tables load unchanged.

Optional columns are never inferred from pixels or file names: when absent they are added empty
and stay empty. In particular ``captured_at`` must be supplied by the data provider when a task
needs time information (see ``pyrestore.pairs``).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["site_id", "capture_id", "image_path", "lon", "lat"]
LEGACY_ID_COLUMNS = ["image", "image_path", "lon", "lat"]
OPTIONAL_COLUMNS = ["captured_at", "heading", "camera", "source", "licence"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _adapt_legacy(table: pd.DataFrame) -> pd.DataFrame:
    """Map the legacy four-column manifest (image, image_path, lon, lat) onto the current contract."""
    if "site_id" in table.columns:
        return table
    missing = [column for column in LEGACY_ID_COLUMNS if column not in table.columns]
    if missing:
        return table  # let load_manifest report the missing required columns in full
    table = table.copy()
    table["site_id"] = table["image"].astype(str)
    if "capture_id" not in table.columns:
        table["capture_id"] = table["site_id"]
    return table


def load_manifest(
    manifest: str | os.PathLike[str] | pd.DataFrame,
    *,
    check_files: bool = True,
) -> pd.DataFrame:
    """Load and validate an image manifest; return one row per capture.

    Relative image paths are resolved against the manifest CSV's parent directory (or the working
    directory for in-memory frames). Raises ``ValueError`` on contract violations and
    ``FileNotFoundError`` when ``check_files`` is set and referenced images are missing.
    """
    if isinstance(manifest, pd.DataFrame):
        table = manifest.copy()
        base_dir = Path.cwd()
    else:
        manifest_path = Path(manifest).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Image manifest not found: {manifest_path}")
        table = pd.read_csv(manifest_path)
        base_dir = manifest_path.parent

    table = _adapt_legacy(table)
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(
            f"Image manifest is missing required columns: {missing}. "
            "A legacy manifest with an 'image' column (PyRestore format) is adapted automatically."
        )
    if table.empty:
        raise ValueError("Image manifest is empty.")

    for column in ("site_id", "capture_id"):
        if table[column].isna().any():
            raise ValueError(f"Image manifest contains missing {column} values.")
        table[column] = table[column].astype(str).str.strip()
        if table[column].eq("").any():
            raise ValueError(f"Image manifest contains empty {column} values.")
    if table["capture_id"].duplicated().any():
        duplicates = table.loc[table["capture_id"].duplicated(), "capture_id"].tolist()[:5]
        raise ValueError(f"Image manifest contains duplicate capture_id values: {duplicates}")

    for column in ("lon", "lat"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table[["lon", "lat"]].isna().any().any():
        raise ValueError("Image manifest contains missing or non-numeric coordinates.")
    if not table["lon"].between(-180, 180).all() or not table["lat"].between(-90, 90).all():
        raise ValueError("Image manifest contains coordinates outside valid longitude/latitude ranges.")

    for column in OPTIONAL_COLUMNS:
        if column not in table.columns:
            table[column] = pd.NA

    def _resolve_image_path(value: object) -> str:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return str(path.resolve())

    table["image_path"] = table["image_path"].map(_resolve_image_path)
    if check_files:
        missing_files = [p for p in table["image_path"] if not Path(p).is_file()]
        if missing_files:
            sample = missing_files[:3]
            raise FileNotFoundError(
                f"Manifest references {len(missing_files)} missing image(s): {sample}"
            )
    return table.reset_index(drop=True)


def parse_lonlat(image_id: str) -> tuple[float, float]:
    """Parse ``(lon, lat)`` from an id like ``201601_114.0055_22.6300.png``.

    Case-study convenience for the Shenzhen/Wuxi naming convention; manifests for new study
    areas should carry explicit coordinates instead of relying on this helper.
    """
    stem = os.path.basename(image_id)
    for ext in (".png", ".jpg", ".jpeg"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse lon/lat from image id {image_id!r}")
    try:
        lon, lat = float(parts[-2]), float(parts[-1])
    except ValueError as exc:
        raise ValueError(f"Cannot parse lon/lat from image id {image_id!r}") from exc
    return lon, lat


def manifest_from_directory(images_dir: str | os.PathLike[str]) -> pd.DataFrame:
    """Create a manifest for image files whose names encode ``..._<lon>_<lat>``.

    Arbitrary filenames require an explicit manifest CSV instead.
    """
    root = Path(images_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")

    rows: list[dict[str, object]] = []
    invalid: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
        try:
            lon, lat = parse_lonlat(path.name)
        except ValueError:
            invalid.append(path.name)
            continue
        rows.append(
            {
                "site_id": path.name,
                "capture_id": path.name,
                "image_path": str(path),
                "lon": lon,
                "lat": lat,
            }
        )

    if invalid:
        sample = invalid[:5]
        raise ValueError(
            f"Could not parse coordinates from {len(invalid)} filename(s), for example {sample}. "
            "Provide an explicit manifest CSV for arbitrary filenames."
        )
    if not rows:
        raise ValueError(f"No supported images found under {root}")
    return load_manifest(pd.DataFrame(rows), check_files=False)


def index_images_by_coord(
    images_dir: str, precision: int = 5
) -> dict[tuple[float, float], str]:
    """Index every image under ``images_dir`` by its (lon, lat) rounded to ``precision`` decimals."""
    index: dict[tuple[float, float], str] = {}
    for path in glob.glob(os.path.join(images_dir, "**", "*.png"), recursive=True):
        try:
            lon, lat = parse_lonlat(os.path.basename(path))
        except ValueError:
            continue
        index[(round(lon, precision), round(lat, precision))] = path
    return index


def resolve_image_path(
    image_id: str, coord_index: dict[tuple[float, float], str], precision: int = 5
) -> str | None:
    """Find the on-disk file for a labelled ``image_id`` via its rounded coordinates, or None."""
    try:
        lon, lat = parse_lonlat(image_id)
    except ValueError:
        return None
    return coord_index.get((round(lon, precision), round(lat, precision)))
