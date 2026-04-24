#!/usr/bin/env python3
"""
transform.py — ERA5-Land data transformation for the Baltic climate risk pipeline.

Responsibilities:
  - Open downloaded ERA5-Land NetCDF files produced by load_data.py
  - Convert temperature from Kelvin to Celsius
  - Derive daily maximum temperature (TX) by resampling 24 hourly values
  - Count extreme heat days using the ETCCDI per-grid-point approach
  - Aggregate per-grid-point annual counts to a single spatial mean for Estonia
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
    Step 6  mean over (lat, lon)          scalar: mean annual heat days for Estonia
"""

import sys
import argparse
import xarray as xr
import pandas as pd
from pathlib import Path

from load_data import RAW_DIR, OUT_CSV, REFERENCE_YEARS, load_config


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
        .resample({"time_dim": "1D"}) groups all timestamps that share the same
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


def count_heat_days_per_gridpoint(tx_c: xr.DataArray, threshold_c: float,
                                   time_dim: str) -> xr.DataArray:
    """
    Count the number of days where TX >= threshold_c at each grid point.

    Returns a DataArray of shape (n_lat, n_lon) — one integer count per grid cell.

    Why per-grid-point rather than spatial-mean-first:
        Averaging TX over Estonia before applying the threshold would smooth out
        local hot spots. A day where one corner of Estonia reaches 31 °C but the
        rest stays at 27 °C would be missed, even though real heat stress occurred.
        Counting per grid point first, then averaging the counts, correctly
        captures those localised extremes.
    """
    # Boolean mask: True for every (day, lat, lon) where TX meets the threshold.
    heat_mask = tx_c >= threshold_c

    # Sum along the time axis to collapse days → count per grid point.
    # Result shape: (n_lat, n_lon).
    return heat_mask.sum(dim=time_dim)


def process_year(year: int, raw_dir: Path, threshold_c: float) -> dict:
    """
    Process all 12 monthly NetCDF files for a given year and return a dict
    with the annual extreme heat day count (spatial mean over Estonia grid).

    Monthly files are processed sequentially and the per-grid-point counts are
    accumulated before the final spatial average. This keeps peak memory use
    to one month of hourly data at a time rather than loading a full year.
    """
    # Running total of heat days per grid point; shape (n_lat, n_lon).
    # Initialised to None so the first month's DataArray sets the coordinates.
    annual_count = None

    for month in range(1, 13):
        nc_path = raw_dir / f"era5land_t2m_ee_{year}_{month:02d}.nc"
        ds, time_dim = open_monthly_nc(nc_path)

        # "t2m" is the ERA5-Land short name for 2m_temperature (in Kelvin).
        tx_c = hourly_to_daily_tx(ds["t2m"], time_dim)

        monthly_count = count_heat_days_per_gridpoint(tx_c, threshold_c, time_dim)

        # Accumulate: add this month's counts to the running annual total.
        annual_count = (
            monthly_count if annual_count is None
            else annual_count + monthly_count
        )
        ds.close()

    # Reduce (n_lat, n_lon) → scalar by averaging over all Estonia grid points.
    # The bounding box includes a few sea cells, but their heat-day counts are
    # indistinguishable from adjacent land cells at this metric level, so no
    # land-sea mask is applied at this stage.
    mean_heat_days = float(annual_count.mean().values)

    return {"year": year, "extreme_heat_days": round(mean_heat_days, 2)}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Process all downloaded years and write results to the output CSV."""
    parser = argparse.ArgumentParser(
        description="Compute extreme heat days from cached ERA5-Land files"
    )
    parser.add_argument("--year", type=int, default=None,
                        help="Process a single year only")
    args        = parser.parse_args()
    cfg         = load_config()
    threshold_c = float(cfg["metrics"]["heat_days"]["threshold_tx_degC"])
    years       = [args.year] if args.year else REFERENCE_YEARS

    # Load any previously computed years to support incremental runs.
    existing = {}
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV).set_index("year")["extreme_heat_days"].to_dict()

    rows = []
    for year in years:
        if year in existing:
            print(f"  [{year}] already processed — skipping")
            rows.append({"year": year, "extreme_heat_days": existing[year]})
            continue
        try:
            print(f"  [{year}] transforming …")
            row = process_year(year, RAW_DIR, threshold_c)
            rows.append(row)
            print(f"  [{year}] extreme heat days: {row['extreme_heat_days']:.2f}")
        except Exception as exc:
            print(f"  [{year}] ERROR: {exc}", file=sys.stderr)

    if rows:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
        df.to_csv(OUT_CSV, index=False)
        print(f"\nSaved → {OUT_CSV}")


if __name__ == "__main__":
    main()
