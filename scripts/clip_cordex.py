#!/usr/bin/env python3
"""
clip_cordex.py — Spatial subsetting of CORDEX NetCDF files to a country bounding box.

Clips full EUR-11 European domain files (~450,000 grid points, tens of GB) down
to a country-scale subset (~1,600 grid points for Estonia, ~200 MB total) before
any metric computation begins. This is the mandatory first step after downloading
raw CORDEX data.

CORDEX EUR-11 uses a rotated-pole coordinate system. The dimension coordinates
are rlat/rlon (rotated latitude/longitude), while lat/lon are 2D auxiliary
coordinates that give the real-world geographic location of each grid point.
Subsetting is done by finding the rectangular rlat/rlon index range that covers
the target bounding box.

Output files:
    Clipped NetCDFs are written to data/raw/cordex/{country_code}/ with their
    original CORDEX filenames preserved. This follows the same country-subdirectory
    convention as data/raw/era5land/ee/. Files are gitignored (*.nc rule) and must
    be reproduced locally from the source data — they are not committed to the repo.

Extending to other Baltic countries:
    Pass --country LV or --country LT. The bounding boxes are read from
    config/config.yaml and no code changes are needed.

Usage:
    conda activate climate-risk
    python scripts/clip_cordex.py --source /path/to/cordex/files
    python scripts/clip_cordex.py --source /path/to/cordex/files --country LV
    python scripts/clip_cordex.py --source /path/to/cordex/files --country EE --pattern "tasmax_*.nc"
"""

from __future__ import annotations

import logging
import argparse
import sys
import numpy as np
import xarray as xr
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from load_data import load_config  # noqa: E402

logger = logging.getLogger(__name__)

CORDEX_RAW_DIR = ROOT / "data" / "raw" / "cordex"


# ── Coordinate detection ──────────────────────────────────────────────────────

def _find_coord(ds: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate name present in ds.coords, or None."""
    return next((n for n in candidates if n in ds.coords), None)


# ── Clipping ──────────────────────────────────────────────────────────────────

def clip_to_bbox(
    ds: xr.Dataset,
    north: float, west: float, south: float, east: float,
) -> xr.Dataset:
    """
    Clip an xarray Dataset to a geographic bounding box.

    Handles two coordinate layouts:
      1. Regular grid (1-D lat/lon as dimension coordinates):
         uses simple .sel(lat=slice(...), lon=slice(...)).

      2. Rotated-pole grid (standard for EUR-11 CORDEX):
         lat and lon are 2-D auxiliary coordinates on rlat/rlon dimensions.
         Finds the rectangular rlat/rlon index range that covers the bbox,
         then clips with .isel().

    The result may include a small margin of grid cells just outside the exact
    bbox boundary due to the rectangular index selection on the rotated grid.
    This is expected and harmless.
    """
    lat_name = _find_coord(ds, ("lat", "latitude"))
    lon_name = _find_coord(ds, ("lon", "longitude"))

    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Cannot find lat/lon coordinates. "
            f"Available coords: {list(ds.coords)}"
        )

    lat = ds[lat_name]
    lon = ds[lon_name]

    if lat.ndim == 1:
        # Regular grid: direct slice selection
        return ds.sel({lat_name: slice(south, north), lon_name: slice(west, east)})

    # Rotated-pole grid: lat/lon are 2-D with dims (rlat, rlon) or similar.
    # Build a boolean mask over the 2-D grid, then find the minimal rectangular
    # index range that covers all True cells.
    mask = (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)

    dim0, dim1 = lat.dims  # e.g. ('rlat', 'rlon')

    # Indices along dim0 where at least one dim1 cell is inside the bbox
    rows = np.where(mask.any(dim=dim1).values)[0]
    # Indices along dim1 where at least one dim0 cell is inside the bbox
    cols = np.where(mask.any(dim=dim0).values)[0]

    if rows.size == 0 or cols.size == 0:
        raise ValueError(
            f"Bounding box [N={north}, W={west}, S={south}, E={east}] "
            f"does not overlap with this dataset's spatial extent "
            f"(lat {float(lat.min()):.1f}–{float(lat.max()):.1f}, "
            f"lon {float(lon.min()):.1f}–{float(lon.max()):.1f})."
        )

    return ds.isel(
        {dim0: slice(int(rows[0]), int(rows[-1]) + 1),
         dim1: slice(int(cols[0]), int(cols[-1]) + 1)}
    )


# ── File-level processing ─────────────────────────────────────────────────────

def clip_file(
    nc_path: Path,
    out_dir: Path,
    north: float, west: float, south: float, east: float,
) -> Path:
    """
    Clip one CORDEX NetCDF to the bounding box and write to out_dir.

    Skips the file if the output already exists (idempotent — safe to re-run).
    Preserves the original CORDEX filename so the full provenance chain
    (variable, domain, GCM, scenario, ensemble, RCM, version, dates) remains
    readable from the filename alone.

    Returns the output path.
    """
    out_path = out_dir / nc_path.name
    if out_path.exists():
        size_mb = out_path.stat().st_size / 1e6
        logger.info("skip  %s  (%.1f MB, already clipped)", nc_path.name, size_mb)
        return out_path

    logger.info("clip  %s", nc_path.name)
    raw_mb = nc_path.stat().st_size / 1e6

    ds = xr.open_dataset(nc_path)
    ds_clipped = clip_to_bbox(ds, north, west, south, east)
    ds.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    ds_clipped.to_netcdf(out_path)

    clipped_mb = out_path.stat().st_size / 1e6
    reduction  = (1 - clipped_mb / raw_mb) * 100 if raw_mb > 0 else 0
    logger.info(
        "      %.0f MB → %.1f MB  (%.0f%% reduction)",
        raw_mb, clipped_mb, reduction,
    )
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Clip CORDEX EUR-11 NetCDF files to a country bounding box",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", required=True,
        help="Directory containing the full-domain CORDEX NetCDF source files.",
    )
    parser.add_argument(
        "--country", default="EE",
        help="ISO 3166-1 alpha-2 country code. Must match a key in config.yaml → countries.",
    )
    parser.add_argument(
        "--pattern", default="*.nc",
        help="Glob pattern to filter source files.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Output directory. Defaults to data/raw/cordex/{country_code}/ "
            "inside the project root."
        ),
    )
    args = parser.parse_args()

    cfg         = load_config()
    country     = args.country.upper()

    if country not in cfg.get("countries", {}):
        logger.error(
            "Country '%s' not found in config.yaml. Available: %s",
            country, list(cfg["countries"]),
        )
        sys.exit(1)

    # Bounding box from config — stored as [N, W, S, E]
    area               = cfg["countries"][country]["area"]
    north, west, south, east = area
    country_name       = cfg["countries"][country]["name"]

    source_dir = Path(args.source)
    if not source_dir.is_dir():
        logger.error("Source directory does not exist: %s", source_dir)
        sys.exit(1)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else CORDEX_RAW_DIR / country.lower()
    )

    files = sorted(source_dir.glob(args.pattern))
    if not files:
        logger.error("No files matching '%s' in %s", args.pattern, source_dir)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Country  : %s  (%s)", country, country_name)
    logger.info("BBox     : N=%.2f  W=%.2f  S=%.2f  E=%.2f", north, west, south, east)
    logger.info("Source   : %s", source_dir)
    logger.info("Output   : %s", out_dir)
    logger.info("Files    : %d matching '%s'", len(files), args.pattern)
    logger.info("=" * 60)

    succeeded, failed = [], []

    for nc_path in files:
        try:
            clip_file(nc_path, out_dir, north, west, south, east)
            succeeded.append(nc_path.name)
        except Exception as exc:
            logger.error("FAILED  %s — %s", nc_path.name, exc)
            failed.append(nc_path.name)

    logger.info("=" * 60)
    logger.info(
        "Done: %d clipped/skipped, %d failed",
        len(succeeded), len(failed),
    )
    if failed:
        for name in failed:
            logger.error("  FAILED: %s", name)
        sys.exit(1)


if __name__ == "__main__":
    main()
