#!/usr/bin/env python3
"""
load_data.py — ERA5-Land data acquisition for the Baltic climate risk pipeline.

Responsibilities:
  - Load pipeline configuration from config/config.yaml
  - Connect to the Copernicus Climate Data Store (CDS) via cdsapi
  - Download ERA5-Land hourly 2m_temperature for a given country, one month at a time
  - Cache downloaded NetCDF files so re-runs skip completed months

CDS request strategy:
  Requests are split by month rather than by year because the CDS imposes a
  per-request field-count limit. One month of hourly data (~744 time steps)
  for a Baltic bounding box stays well within that limit; a full year (~8 760
  steps) does not.

Data selected:
  Dataset   : reanalysis-era5-land (ECMWF ERA5-Land reanalysis, 0.1° grid)
  Variable  : 2m_temperature (t2m) — instantaneous hourly values in Kelvin
  Time      : all 24 hours (00:00–23:00) of every day in the requested month.
              All hours are required so the transform step can derive the true
              daily maximum temperature (TX).
  Area      : country bounding box in [N, W, S, E] decimal degrees, read from
              config/config.yaml → countries.<code>.area
  Format    : NetCDF4, unarchived (single file, not a zip archive)
"""

import sys
import argparse
import yaml
import cdsapi
from pathlib import Path

# ── Shared paths ──────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config/config.yaml"
RAW_DIR     = ROOT / "data/raw/era5land"
OUT_CSV     = ROOT / "data/processed/estonia_extreme_heat_days.csv"

# ── CDS request parameters ────────────────────────────────────────────────────

# WMO standard 30-year reference period for current-climate baselines.
REFERENCE_YEARS = list(range(1991, 2021))

# All calendar days 01–31 are always requested. CDS silently ignores day
# numbers that do not exist in a given month (e.g. day 31 in April), so
# listing all 31 days is safe and avoids per-month conditional logic.
DAYS = [f"{d:02d}" for d in range(1, 32)]

# All 24 hours are requested. Downloading fewer hours (e.g. every 3 h) would
# miss the true daily maximum in months with sharp diurnal temperature cycles.
HOURS = [f"{h:02d}:00" for h in range(24)]


# ── Functions ─────────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load the project configuration file (config/config.yaml)."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_country_area(country_code: str) -> list:
    """
    Return the CDS bounding box [N, W, S, E] for the given ISO 3166-1 alpha-2
    country code, as defined in config/config.yaml → countries.<code>.area.
    """
    cfg = load_config()
    try:
        return cfg["countries"][country_code]["area"]
    except KeyError:
        raise KeyError(
            f"Country code '{country_code}' not found in config.yaml. "
            f"Available codes: {list(cfg['countries'].keys())}"
        )


def download_month(client: cdsapi.Client, year: int, month: int,
                   area: list, raw_dir: Path) -> Path:
    """
    Download one calendar month of ERA5-Land hourly 2m_temperature for the
    given bounding box.

    Files are cached by filename: if the expected NetCDF already exists on disk
    the download is skipped and the existing path is returned immediately.

    Parameters
    ----------
    area    : [N, W, S, E] in decimal degrees — use get_country_area() or
              supply directly from config.
    raw_dir : destination directory for this country's NetCDF files.

    Returns the path to the downloaded (or cached) NetCDF file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    nc_path = raw_dir / f"era5land_t2m_{year}_{month:02d}.nc"

    if nc_path.exists():
        size_mb = nc_path.stat().st_size / 1e6
        print(f"    [{year}-{month:02d}] cached ({size_mb:.0f} MB)")
        return nc_path

    print(f"    [{year}-{month:02d}] submitting CDS request …")
    client.retrieve(
        "reanalysis-era5-land",
        {
            # Only "reanalysis" is available for ERA5-Land (no ensemble members).
            "product_type": "reanalysis",

            # "2m_temperature" is the CDS long name; the NetCDF variable will be
            # stored under the short name "t2m" in Kelvin.
            "variable": "2m_temperature",

            "year":  str(year),
            "month": f"{month:02d}",
            "day":   DAYS,
            "time":  HOURS,

            # Subset to the country bounding box to avoid downloading the full
            # European domain (reduces file size by ~99 %).
            "area": area,

            # "data_format" is the parameter name used by the new CDS API
            # (cds.climate.copernicus.eu). Older API used "format".
            "data_format": "netcdf",

            # "unarchived" returns a single NetCDF file. Without this the new
            # CDS API may return a zip archive containing the NetCDF.
            "download_format": "unarchived",
        },
        str(nc_path),
    )
    size_mb = nc_path.stat().st_size / 1e6
    print(f"    [{year}-{month:02d}] downloaded ({size_mb:.0f} MB)")
    return nc_path


def download_year(client: cdsapi.Client, year: int,
                  area: list, raw_dir: Path) -> list:
    """Download all 12 months for a given year; return the list of NetCDF paths."""
    return [
        download_month(client, year, month, area, raw_dir)
        for month in range(1, 13)
    ]


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Download all reference-period months for a country that are not yet cached."""
    parser = argparse.ArgumentParser(description="Download ERA5-Land data for a Baltic country")
    parser.add_argument("--country", default="EE",
                        help="ISO 3166-1 alpha-2 country code (default: EE)")
    parser.add_argument("--year", type=int, default=None,
                        help="Download a single year only")
    args    = parser.parse_args()
    years   = [args.year] if args.year else REFERENCE_YEARS
    area    = get_country_area(args.country)
    raw_dir = RAW_DIR / args.country.lower()
    client  = cdsapi.Client()

    for year in years:
        print(f"  [{year}] downloading …")
        try:
            download_year(client, year, area, raw_dir)
        except Exception as exc:
            print(f"  [{year}] ERROR: {exc}", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()
