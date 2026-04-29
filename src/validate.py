#!/usr/bin/env python3
from __future__ import annotations
"""
validate.py — data quality checks for the Baltic climate risk pipeline.

Two validation stages are defined:

  Stage 1 — Raw file validation (after load_data, before transform):
    Checks that a downloaded NetCDF is structurally sound and that raw Kelvin
    values are within the physical plausibility bounds defined in config.yaml.

  Stage 2 — Result validation (after transform):
    Sanity-checks the derived TX DataArray and the final annual metric value.

All check functions return {"passed": bool, "issues": list[str]} so the
caller can decide whether to abort or continue with a warning.

Plausibility bounds are read from config/config.yaml → validation.*
rather than being hardcoded, so they can be adjusted per region without
touching source code.
"""

import sys
import logging
import argparse
import calendar
import xarray as xr
from pathlib import Path

from load_data import RAW_DIR, REFERENCE_YEARS, load_config

logger = logging.getLogger(__name__)


def _load_bounds() -> dict:
    """Return the validation bounds block from config.yaml."""
    return load_config()["validation"]


# ── Stage 1: raw file checks ──────────────────────────────────────────────────

def validate_raw_file(nc_path: Path, year: int, month: int,
                      bounds: dict = None) -> dict:
    """
    Validate a downloaded monthly ERA5-Land NetCDF file.

    Checks (in order):
      1. File opens without error (basic NetCDF integrity)
      2. Variable "t2m" is present (ERA5-Land short name for 2m_temperature)
      3. A recognised time coordinate is present ("valid_time" or "time")
      4. Number of hourly timesteps matches the calendar for the given month:
           expected = days_in_month × 24
           e.g. February 2020 (leap year) → 29 × 24 = 696 steps
      5. Raw Kelvin values are within the plausibility range from config.yaml
         (validation.t2m_min_k / validation.t2m_max_k). Values outside this
         range indicate a unit mismatch or corrupted file, not a real extreme.
      6. No intermittent NaN values in t2m. ERA5-Land stores sea/water cells as
         NaN in every timestep (land-sea mask) — these are expected and ignored.
         Only NaN values that appear at some timesteps but not others (temporal
         gaps in land cells) are flagged, as they would corrupt heat-day counts.

    Parameters
    ----------
    bounds : validation block from config.yaml (loaded automatically if None).
    """
    if bounds is None:
        bounds = _load_bounds()

    t2m_min_k = bounds["t2m_min_k"]
    t2m_max_k = bounds["t2m_max_k"]
    issues    = []

    # ── Check 1: file is readable ─────────────────────────────────────────────
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as exc:
        return {"passed": False, "issues": [f"Cannot open NetCDF: {exc}"]}

    # ── Check 2: expected variable is present ─────────────────────────────────
    if "t2m" not in ds.data_vars:
        issues.append(f"Variable 't2m' missing; found: {list(ds.data_vars)}")

    # ── Check 3: time coordinate is present ───────────────────────────────────
    time_dim = "valid_time" if "valid_time" in ds.coords else "time"
    if time_dim not in ds.coords:
        issues.append("No time coordinate found (checked 'valid_time' and 'time')")
        ds.close()
        return {"passed": False, "issues": issues}

    # ── Check 4: timestep count matches the calendar ──────────────────────────
    # calendar.monthrange returns (weekday_of_first_day, number_of_days).
    days_in_month  = calendar.monthrange(year, month)[1]
    expected_steps = days_in_month * 24   # 24 hourly readings per day
    actual_steps   = ds.sizes[time_dim]
    if actual_steps != expected_steps:
        issues.append(
            f"Timestep count mismatch: expected {expected_steps} "
            f"({days_in_month} days × 24 h), got {actual_steps}"
        )

    if "t2m" in ds.data_vars:
        t2m   = ds["t2m"]
        t_min = float(t2m.min().values)
        t_max = float(t2m.max().values)

        # ── Check 5: physical plausibility of raw Kelvin values ───────────────
        if t_min < t2m_min_k:
            issues.append(
                f"t2m below configured minimum: {t_min:.1f} K < {t2m_min_k} K "
                f"({t_min - 273.15:.1f} °C)"
            )
        if t_max > t2m_max_k:
            issues.append(
                f"t2m above configured maximum: {t_max:.1f} K > {t2m_max_k} K "
                f"({t_max - 273.15:.1f} °C)"
            )

        # ── Check 6: intermittent NaN in land cells ───────────────────────────
        # ERA5-Land is a land-only product: sea and inland-water grid cells are
        # stored as NaN in every timestep (the land-sea mask). These are expected
        # and must not be flagged as errors.
        #
        # What we actually care about is *intermittent* NaN — a grid point that
        # has valid data in some timesteps but NaN in others. That indicates a
        # genuine temporal gap in a land cell, which would silently corrupt the
        # daily-maximum resampling and the heat-day count.
        nan_spatial  = t2m.isnull().sum(dim=time_dim)   # NaN timestep count per cell
        n_steps      = ds.sizes[time_dim]
        intermittent = int(((nan_spatial > 0) & (nan_spatial < n_steps)).sum())

        if intermittent > 0:
            issues.append(
                f"{intermittent} land grid point(s) have intermittent NaN values "
                f"(temporal gaps) — heat-day counts would be unreliable"
            )

    ds.close()

    if issues:
        logger.warning("[%d-%02d] validation failed: %s", year, month, "; ".join(issues))
    else:
        logger.info("[%d-%02d] raw validation passed", year, month)

    return {"passed": len(issues) == 0, "issues": issues}


# ── Stage 2: transformed data and result checks ───────────────────────────────

def validate_tx(tx_c: xr.DataArray, year: int, month: int,
                bounds: dict = None) -> dict:
    """
    Validate the derived daily maximum temperature (TX) DataArray in °C.

    Called after hourly_to_daily_tx() in transform.py to catch unit errors
    or resampling artefacts before the threshold is applied. For example:
      - If the K→°C subtraction was skipped, TX values would be ~270–300,
        registering as thousands of false heat days.
      - If resampling produced NaNs (e.g. from timestamp gaps), counts would
        be silently wrong downstream.

    Checks TX against validation.tx_min_c / validation.tx_max_c from config.

    Parameters
    ----------
    bounds : validation block from config.yaml (loaded automatically if None).
    """
    if bounds is None:
        bounds = _load_bounds()

    tx_min_c = bounds["tx_min_c"]
    tx_max_c = bounds["tx_max_c"]
    issues   = []

    tx_min = float(tx_c.min().values)
    tx_max = float(tx_c.max().values)

    if tx_min < tx_min_c:
        issues.append(
            f"TX below configured minimum: {tx_min:.1f} °C < {tx_min_c} °C "
            f"— possible missing K→°C conversion"
        )
    if tx_max > tx_max_c:
        issues.append(
            f"TX above configured maximum: {tx_max:.1f} °C > {tx_max_c} °C "
            f"— possible missing K→°C conversion"
        )

    nan_count = int(tx_c.isnull().sum().values)
    if nan_count > 0:
        issues.append(
            f"{nan_count} NaN values in derived TX for {year}-{month:02d}"
        )

    if issues:
        logger.warning("[%d-%02d] TX validation failed: %s", year, month, "; ".join(issues))

    return {"passed": len(issues) == 0, "issues": issues}


# Physical plausibility bounds per metric column.
# (lo, hi) — values outside this range indicate a processing error.
_RESULT_BOUNDS: dict[str, tuple] = {
    "extreme_heat_days": (0,    366),
    "frost_days":        (0,    366),
    "id0":               (0,    366),
    "tr15":              (0,    366),
    "txx":               (-50,   50),   # °C, annual max TX for Baltic
    "tnn":               (-60,   30),   # °C, annual min TN for Baltic
    "cdd":               (0,    366),
    "r20mm":             (0,    366),
    "sdii":              (0,     80),   # mm/day — Baltic realistic upper bound
    "prcptot":           (0,   5000),   # mm/year
}


def validate_annual_result(row: dict, metric_col: str = "extreme_heat_days") -> dict:
    """
    Sanity-check an annual result dict produced by transform.process_year().

    Checks:
      - metric_col key is present in row
      - Value is within the physically plausible range for the metric
    """
    issues = []
    val    = row.get(metric_col)
    lo, hi = _RESULT_BOUNDS.get(metric_col, (0, 366))

    if val is None:
        issues.append(f"'{metric_col}' key missing from result dict")
    else:
        if val < lo:
            issues.append(f"{metric_col} below plausible minimum ({val:.2f} < {lo})")
        if val > hi:
            issues.append(f"{metric_col} above plausible maximum ({val:.2f} > {hi})")

    if issues:
        logger.warning("Result validation failed for year %s: %s", row.get("year"), issues)

    return {"passed": len(issues) == 0, "issues": issues}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Validate all cached raw NetCDF files for a country and print a report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Validate cached ERA5-Land files for the Baltic pipeline"
    )
    parser.add_argument("--country", default="EE",
                        help="ISO 3166-1 alpha-2 country code (default: EE)")
    parser.add_argument("--year", type=int, default=None,
                        help="Validate a single year only")
    args    = parser.parse_args()
    years   = [args.year] if args.year else REFERENCE_YEARS
    raw_dir = RAW_DIR / args.country.lower()
    bounds  = _load_bounds()
    failed  = []

    for year in years:
        for month in range(1, 13):
            nc_path = raw_dir / f"era5land_t2m_{year}_{month:02d}.nc"
            if not nc_path.exists():
                logger.warning("[%d-%02d] MISSING — not yet downloaded", year, month)
                continue

            result = validate_raw_file(nc_path, year, month, bounds)
            if result["issues"]:
                failed.append((year, month, result["issues"]))

    if failed:
        logger.error("FAILED: %d file(s) have issues — re-run with DEBUG for details", len(failed))
        sys.exit(1)
    else:
        logger.info("All checked files passed validation.")


if __name__ == "__main__":
    main()
