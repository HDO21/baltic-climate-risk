#!/usr/bin/env python3
"""
download_cordex_rcp85.py — Download CORDEX EUR-11 RCP8.5 daily data via CDS API.

Downloads the same model chain used for RCP4.5 (MPI-M-MPI-ESM-LR / MPI-CSC-REMO2009)
but for the RCP8.5 high-emissions scenario, covering 2021–2100. Files are written to
a user-specified directory as full EUR-11 European-domain NetCDFs. Clip them to the
Estonia bounding box afterwards with:

    python scripts/clip_cordex.py \\
        --source /path/to/raw_temp \\
        --country EE

CDS notes:
  - Dataset: projections-cordex-domains-single-levels
  - CORDEX does NOT support the 'area' subsetting parameter — full EUR-11 domain only.
  - Files arrive in 5-year chunks matching the existing rcp45 naming convention:
      {var}_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r1i1p1_MPI-CSC-REMO2009_v1_day_{YYYYMMDD}-{YYYYMMDD}.nc
  - Requests are one variable × one 5-year period at a time to stay within CDS
    field-count limits and to allow resuming interrupted downloads.

Credentials:
  Set up ~/.cdsapirc with your CDS API key before running.
  See: https://cds.climate.copernicus.eu/how-to-api

Usage:
    conda activate climate-risk
    python scripts/download_cordex_rcp85.py
    python scripts/download_cordex_rcp85.py --out-dir /custom/path
    python scripts/download_cordex_rcp85.py --variable tasmax --start-year 2021 --end-year 2040
"""

from __future__ import annotations

import time
import logging
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from load_data import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# ── Default output directory ───────────────────────────────────────────────────
# Matches the location the user requested. Override with --out-dir.
DEFAULT_OUT_DIR = Path(
    "/Users/hando-laurhabicht/Library/CloudStorage"
    "/OneDrive-DataSky/Koolitused/Agent/raw_temp"
)

# ── CORDEX configuration ───────────────────────────────────────────────────────
# These values are read from config/config.yaml; the constants below are the
# corresponding CDS API parameter strings for the same model chain.

# CDS dataset identifier for Euro-CORDEX projections
CDS_DATASET = "projections-cordex-domains-single-levels"

# CDS parameter strings — kept as constants because the CDS API uses lowercase
# with underscores, while config.yaml stores the original CORDEX identifiers.
CDS_DOMAIN              = "europe"
CDS_EXPERIMENT          = "rcp_8_5"
CDS_HORIZONTAL_RES      = "0_11_degree_x_0_11_degree"   # dots → underscores in CDS API
CDS_TEMPORAL_RES        = "daily_mean"
CDS_GCM_MODEL           = "mpi_m_mpi_esm_lr"            # MPI-M-MPI-ESM-LR
CDS_RCM_MODEL           = "mpi_csc_remo2009"             # MPI-CSC-REMO2009
CDS_ENSEMBLE_MEMBER     = "r1i1p1"

# CDS variable long-name → CORDEX short-name used in filenames.
# Verified against the live CDS form at:
#   /api/catalogue/v1/collections/projections-cordex-domains-single-levels
# (The CDS API variable names differ from CORDEX CF standard names.)
VARIABLES: dict[str, str] = {
    "maximum_2m_temperature_in_the_last_24_hours": "tasmax",
    "minimum_2m_temperature_in_the_last_24_hours": "tasmin",
    "2m_air_temperature":                          "tas",
    "mean_precipitation_flux":                     "pr",
}

# Projection period: RCP8.5 runs 2006–2100; we download 2021–2100 because
# 2006–2020 would be covered by the calibration period (historical + rcp45).
PROJECTION_START = 2021
PROJECTION_END   = 2100

# CORDEX files arrive in 5-year chunks (same as the existing rcp45 downloads).
CHUNK_YEARS = 5


# ── Helpers ────────────────────────────────────────────────────────────────────

def _year_chunks(start: int, end: int, chunk: int) -> list[tuple[int, int]]:
    """Return [(start_yr, end_yr), ...] in non-overlapping chunks of size `chunk`."""
    periods = []
    y = start
    while y <= end:
        y_end = min(y + chunk - 1, end)
        periods.append((y, y_end))
        y = y_end + 1
    return periods


def _expected_filename(short_var: str, start_yr: int, end_yr: int) -> str:
    """
    Reconstruct the standard CORDEX filename for one variable × period.
    Matches the naming used by the existing rcp45 files, e.g.:
        tasmax_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r1i1p1_MPI-CSC-REMO2009_v1_day_20210101-20251231.nc
    """
    return (
        f"{short_var}_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r1i1p1"
        f"_MPI-CSC-REMO2009_v1_day"
        f"_{start_yr}0101-{end_yr}1231.nc"
    )


def _retrieve_with_retry(
    client,
    params: dict,
    output_path: Path,
    max_retries: int = 3,
    initial_wait: float = 10.0,
) -> None:
    """Submit a CDS request with exponential back-off on transient failures."""
    for attempt in range(1, max_retries + 1):
        try:
            client.retrieve(CDS_DATASET, params, str(output_path))
            return
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait = initial_wait * (2 ** (attempt - 1))
            logger.warning(
                "CDS request failed (attempt %d/%d): %s. Retrying in %.0f s ...",
                attempt, max_retries, exc, wait,
            )
            time.sleep(wait)


def download_chunk(
    client,
    cds_var: str,
    short_var: str,
    start_yr: int,
    end_yr: int,
    out_dir: Path,
) -> Path:
    """
    Download one variable × 5-year period. Skips if the output file already exists.

    Returns the output path (whether freshly downloaded or cached).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = _expected_filename(short_var, start_yr, end_yr)
    out_path = out_dir / filename

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1e6
        logger.info(
            "SKIP  %s  (%.0f MB — already downloaded)", filename, size_mb
        )
        return out_path

    logger.info(
        "REQUEST  %s  %d–%d  →  %s",
        short_var, start_yr, end_yr, filename,
    )

    params = {
        "domain":               CDS_DOMAIN,
        "experiment":           CDS_EXPERIMENT,
        "horizontal_resolution": CDS_HORIZONTAL_RES,
        "temporal_resolution":  CDS_TEMPORAL_RES,
        "variable":             cds_var,
        "gcm_model":            CDS_GCM_MODEL,
        "rcm_model":            CDS_RCM_MODEL,
        "ensemble_member":      CDS_ENSEMBLE_MEMBER,
        # CDS CORDEX API uses start_year / end_year as strings
        "start_year":           str(start_yr),
        "end_year":             str(end_yr),
        "data_format":          "netcdf",
    }

    _retrieve_with_retry(client, params, out_path)

    size_mb = out_path.stat().st_size / 1e6
    logger.info("DONE   %s  (%.0f MB)", filename, size_mb)
    return out_path


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Download CORDEX EUR-11 RCP8.5 daily data "
            "(MPI-M-MPI-ESM-LR / MPI-CSC-REMO2009) from CDS."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to write full EUR-11 domain NetCDF files.",
    )
    parser.add_argument(
        "--variable",
        choices=list(VARIABLES.values()) + ["all"],
        default="all",
        help=(
            "CORDEX variable short name to download, or 'all' for all four variables "
            "(tasmax, tasmin, tas, pr)."
        ),
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=PROJECTION_START,
        help="First year of the download range (inclusive).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=PROJECTION_END,
        help="Last year of the download range (inclusive).",
    )
    args = parser.parse_args()

    # Resolve which CDS long-name / short-name pairs to download
    if args.variable == "all":
        var_pairs = list(VARIABLES.items())
    else:
        # Invert to find the CDS long name for the requested short name
        reverse = {v: k for k, v in VARIABLES.items()}
        if args.variable not in reverse:
            logger.error("Unknown variable '%s'", args.variable)
            sys.exit(1)
        var_pairs = [(reverse[args.variable], args.variable)]

    periods = _year_chunks(args.start_year, args.end_year, CHUNK_YEARS)

    logger.info(
        "Download plan: %d variable(s) × %d periods = %d requests",
        len(var_pairs), len(periods), len(var_pairs) * len(periods),
    )
    logger.info("Output directory: %s", args.out_dir)

    # Lazy import so the script can be used for inspection without credentials
    import cdsapi  # noqa: PLC0415
    client = cdsapi.Client()

    errors: list[str] = []

    for cds_var, short_var in var_pairs:
        for start_yr, end_yr in periods:
            try:
                download_chunk(client, cds_var, short_var, start_yr, end_yr, args.out_dir)
            except Exception as exc:
                msg = f"{short_var} {start_yr}–{end_yr}: {exc}"
                logger.error("FAIL   %s", msg)
                errors.append(msg)

    if errors:
        logger.warning("\n%d request(s) failed:", len(errors))
        for e in errors:
            logger.warning("  %s", e)
    else:
        logger.info("All downloads complete.")

    logger.info(
        "\nNext step — clip to Estonia bounding box:\n"
        "  python scripts/clip_cordex.py --source '%s' --country EE",
        args.out_dir,
    )


if __name__ == "__main__":
    main()
