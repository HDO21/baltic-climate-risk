#!/usr/bin/env python3
"""
transform.py — ERA5-Land data transformation for the Baltic climate risk pipeline.

Responsibilities:
  - Open downloaded ERA5-Land NetCDF files produced by load_data.py
  - Convert temperature from Kelvin to Celsius
  - Derive daily maximum temperature (TX) by resampling 24 hourly values
  - Count extreme heat days using the ETCCDI per-grid-point approach
  - Aggregate per-grid-point annual counts to a single spatial mean for the country
  - Write annual results to the processed output CSV

ETCCDI metric — Extreme Heat Days (TX >= threshold):
  The Expert Team on Climate Change Detection and Indices (ETCCDI) prescribes
  computing threshold exceedance at each grid point independently, then reporting
  the spatial mean. This avoids the smoothing bias that arises from averaging
  temperatures first: spatial averaging reduces peak temperatures, which
  systematically underestimates the number of exceedance days.

  Processing chain per month:
    Step 1  t2m[K]  →  t2m[°C]           subtract 273.15 (unit conversion only)
    Step 2  t2m[°C, hour, lat, lon]
              →  TX[°C, day, lat, lon]    resample to daily maximum
    Step 3  TX >= threshold               boolean heat-day mask per grid point
    Step 4  sum over days                 heat day count per grid point this month

  End of year:
    Step 5  sum over 12 months            annual count per grid point  (lat, lon)
    Step 6  mean over (lat, lon)          scalar: mean annual heat days for country
"""

import logging
import argparse
import xarray as xr
import pandas as pd
from pathlib import Path

from load_data import RAW_DIR, OUT_CSV, REFERENCE_YEARS, load_config

logger = logging.getLogger(__name__)

# Maps metric key → output CSV column name.
METRIC_COL = {
    "heat_days":  "extreme_heat_days",
    "frost_days": "frost_days",
}


# ── Functions ─────────────────────────────────────────────────────────────────

def open_monthly_nc(nc_path: Path) -> tuple:
    """
    Open a monthly ERA5-Land NetCDF and return (dataset, time_coordinate_name).

    The time coordinate is called 'valid_time' in files retrieved from the new
    CDS API (cds.climate.copernicus.eu) and 'time' in older ERA5 files.
    Detecting the name here keeps all downstream functions API-version agnostic.
    """
    ds       = xr.open_dataset(nc_path)
    time_dim = "valid_time" if "valid_time" in ds.coords else "time"
    return ds, time_dim


def hourly_to_daily_tx(t2m_k: xr.DataArray, time_dim: str) -> xr.DataArray:
    """
    Convert hourly 2m_temperature from Kelvin to Celsius, then resample to
    daily maximum temperature (TX).

    Unit conversion:
        ERA5-Land always stores temperature in Kelvin. The conversion to Celsius
        is a simple offset with no scaling: T[°C] = T[K] − 273.15.

    Resampling to TX:
        .resample({time_dim: "1D"}) groups all timestamps that share the same
        calendar date (in UTC), then .max() selects the highest value across all
        24 hourly readings in that day. This gives TX: the daily maximum 2m
        temperature as defined by ETCCDI.

    Returns a DataArray of shape (n_days, n_lat, n_lon) in degrees Celsius.
    """
    # Subtract the 0 °C point in Kelvin. No other factor is needed.
    t2m_c = t2m_k - 273.15

    # Group by calendar day (UTC) and take the maximum across the 24 hours.
    tx_c = t2m_c.resample({time_dim: "1D"}).max()

    return tx_c


def hourly_to_daily_tn(t2m_k: xr.DataArray, time_dim: str) -> xr.DataArray:
    """
    Convert hourly 2m_temperature from Kelvin to Celsius, then resample to
    daily minimum temperature (TN).

    Parallel to hourly_to_daily_tx but uses .min() instead of .max().
    Returns a DataArray of shape (n_days, n_lat, n_lon) in degrees Celsius.
    """
    t2m_c = t2m_k - 273.15
    return t2m_c.resample({time_dim: "1D"}).min()


def count_frost_days_per_gridpoint(tn_c: xr.DataArray, threshold_c: float,
                                    time_dim: str) -> xr.DataArray:
    """
    Count the number of days where TN < threshold_c at each grid point.

    Returns a DataArray of shape (n_lat, n_lon) — one integer count per grid cell.
    Uses strict less-than (<) matching the ETCCDI FD0 definition (TN < 0 °C).
    """
    frost_mask = tn_c < threshold_c
    return frost_mask.sum(dim=time_dim)


def count_heat_days_per_gridpoint(tx_c: xr.DataArray, threshold_c: float,
                                   time_dim: str) -> xr.DataArray:
    """
    Count the number of days where TX >= threshold_c at each grid point.

    Returns a DataArray of shape (n_lat, n_lon) — one integer count per grid cell.

    Why per-grid-point rather than spatial-mean-first:
        Averaging TX over the country before applying the threshold would smooth
        out local hot spots. A day where one grid cell reaches 31 °C but the
        rest stays at 27 °C would be missed, even though real heat stress occurred.
        Counting per grid point first, then averaging the counts, correctly
        captures those localised extremes.
    """
    # Boolean mask: True for every (day, lat, lon) where TX meets the threshold.
    heat_mask = tx_c >= threshold_c

    # Sum along the time axis to collapse days → count per grid point.
    # Result shape: (n_lat, n_lon).
    return heat_mask.sum(dim=time_dim)


def compute_annual_grid(year: int, raw_dir: Path, threshold_c: float,
                        metric: str = "heat_days") -> xr.DataArray:
    """
    Process all 12 monthly NetCDF files for a given year and return a
    DataArray of shape (n_lat, n_lon) with the annual day-count per grid point
    for the requested metric. Sea cells that are always NaN in ERA5-Land remain
    NaN in the output.

    metric : "heat_days" — TX >= threshold_c (daily maximum temperature)
             "frost_days" — TN < threshold_c  (daily minimum temperature)

    Processing one month at a time caps peak memory to one month of hourly
    data rather than requiring a full year to be held in RAM simultaneously.
    """
    if metric not in METRIC_COL:
        raise ValueError(f"Unknown metric '{metric}'. Known: {list(METRIC_COL)}")

    logger.info("[%d] computing annual grid (%s)", year, metric)
    annual_count = None

    for month in range(1, 13):
        nc_path = raw_dir / f"era5land_t2m_{year}_{month:02d}.nc"
        ds, time_dim = open_monthly_nc(nc_path)

        if metric == "heat_days":
            daily_extreme = hourly_to_daily_tx(ds["t2m"], time_dim)
            monthly_count = count_heat_days_per_gridpoint(daily_extreme, threshold_c, time_dim)
        else:  # frost_days
            daily_extreme = hourly_to_daily_tn(ds["t2m"], time_dim)
            monthly_count = count_frost_days_per_gridpoint(daily_extreme, threshold_c, time_dim)

        annual_count = (
            monthly_count if annual_count is None
            else annual_count + monthly_count
        )
        ds.close()

    return annual_count


def process_year(year: int, raw_dir: Path, threshold_c: float,
                 metric: str = "heat_days") -> dict:
    """
    Process all 12 monthly NetCDF files for a given year and return a dict
    with the annual day-count (spatial mean over the country grid).

    Monthly files are processed sequentially and the per-grid-point counts are
    accumulated before the final spatial average. This keeps peak memory use
    to one month of hourly data at a time rather than loading a full year.
    """
    logger.info("[%d] starting transform (%s)", year, metric)
    annual_count = compute_annual_grid(year, raw_dir, threshold_c, metric)

    # Reduce (n_lat, n_lon) → scalar by averaging over all country grid points.
    # The bounding box includes sea cells (stored as NaN in ERA5-Land), which
    # are naturally excluded from the mean by xarray's default skipna=True.
    mean_val = float(annual_count.mean().values)
    col      = METRIC_COL[metric]
    result   = {"year": year, col: round(mean_val, 2)}

    logger.info("[%d] transform complete — %s: %.2f days", year, col, mean_val)
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Process all downloaded years and write results to the output CSV."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _METRIC_CSV = {
        "heat_days":  "estonia_extreme_heat_days.csv",
        "frost_days": "estonia_frost_days.csv",
    }
    _METRIC_CFG_KEY = {
        "heat_days":  ("heat_days",  "threshold_tx_degC"),
        "frost_days": ("frost_days", "threshold_tn_degC"),
    }

    parser = argparse.ArgumentParser(
        description="Compute climate day-count metrics from cached ERA5-Land files"
    )
    parser.add_argument("--country", default="EE",
                        help="ISO 3166-1 alpha-2 country code (default: EE)")
    parser.add_argument("--year", type=int, default=None,
                        help="Process a single year only")
    parser.add_argument("--metric", default="heat_days",
                        choices=list(METRIC_COL),
                        help="Metric to compute (default: heat_days)")
    args     = parser.parse_args()
    cfg      = load_config()
    metric   = args.metric
    col      = METRIC_COL[metric]
    cfg_sect, cfg_key = _METRIC_CFG_KEY[metric]
    threshold_c = float(cfg["metrics"][cfg_sect][cfg_key])
    years       = [args.year] if args.year else REFERENCE_YEARS
    raw_dir     = RAW_DIR / args.country.lower()
    out_csv     = OUT_CSV.parent / _METRIC_CSV[metric]

    existing = {}
    if out_csv.exists():
        existing = pd.read_csv(out_csv).set_index("year")[col].to_dict()

    rows = []
    for year in years:
        if year in existing:
            logger.info("[%d] already processed — skipping", year)
            rows.append({"year": year, col: existing[year]})
            continue
        try:
            row = process_year(year, raw_dir, threshold_c, metric)
            rows.append(row)
        except Exception as exc:
            logger.error("[%d] transform failed: %s", year, exc)

    if rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
        df.to_csv(out_csv, index=False)
        logger.info("Saved → %s", out_csv)


if __name__ == "__main__":
    main()
