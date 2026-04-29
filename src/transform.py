#!/usr/bin/env python3
from __future__ import annotations
"""
transform.py — ERA5-Land data transformation for the Baltic climate risk pipeline.

Supports two source variables:
  t2m (2m_temperature) — temperature-based ETCCDI metrics
  tp  (total_precipitation) — precipitation-based ETCCDI metrics

Temperature metrics (from t2m, files era5land_t2m_YYYY_MM.nc):
  heat_days  TX30  days with TX >= 30 °C
  frost_days FD0   days with TN <   0 °C
  id0        ID0   days with TX <   0 °C
  tr15       TR15  days with TN >  15 °C
  txx        TXx   annual maximum of daily TX (°C)
  tnn        TNn   annual minimum of daily TN (°C)

Precipitation metrics (from tp, files era5land_tp_YYYY_MM.nc):
  cdd        CDD   maximum consecutive dry days (pr < 1 mm/day)
  r20mm      R20mm days with pr > 20 mm/day
  sdii       SDII  mean daily pr on wet days (pr >= 1 mm/day), mm/day
  prcptot    PRCPTOT annual total pr on wet days (pr >= 1 mm/day), mm

ETCCDI convention: compute threshold exceedance per grid point, then
report the spatial mean over the country bounding box. Sea cells (always
NaN for t2m) are excluded automatically via xarray's skipna=True default.
"""

import logging
import argparse
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

from load_data import RAW_DIR, OUT_CSV, REFERENCE_YEARS, load_config

logger = logging.getLogger(__name__)

# ── Metric registry ───────────────────────────────────────────────────────────

METRIC_COL = {
    "heat_days":  "extreme_heat_days",
    "frost_days": "frost_days",
    "hard_frost": "hard_frost_days",
    "id0":        "id0",
    "tr15":       "tr15",
    "txx":        "txx",
    "tnn":        "tnn",
    "cdd":        "cdd",
    "r20mm":      "r20mm",
    "sdii":       "sdii",
    "prcptot":    "prcptot",
}

# Metrics that read tp files instead of t2m files.
PRECIP_METRICS = frozenset({"cdd", "r20mm", "sdii", "prcptot"})


# ── Shared helper ─────────────────────────────────────────────────────────────

def open_monthly_nc(nc_path: Path) -> tuple:
    """
    Open a monthly ERA5-Land NetCDF and return (dataset, time_coordinate_name).

    The time coordinate is called 'valid_time' in files retrieved from the new
    CDS API (cds.climate.copernicus.eu) and 'time' in older ERA5 files.
    """
    ds       = xr.open_dataset(nc_path)
    time_dim = "valid_time" if "valid_time" in ds.coords else "time"
    return ds, time_dim


# ── Temperature helpers ───────────────────────────────────────────────────────

def hourly_to_daily_tx(t2m_k: xr.DataArray, time_dim: str) -> xr.DataArray:
    """K → °C then resample to daily maximum (TX). Shape: (n_days, lat, lon)."""
    return (t2m_k - 273.15).resample({time_dim: "1D"}).max()


def hourly_to_daily_tn(t2m_k: xr.DataArray, time_dim: str) -> xr.DataArray:
    """K → °C then resample to daily minimum (TN). Shape: (n_days, lat, lon)."""
    return (t2m_k - 273.15).resample({time_dim: "1D"}).min()


def count_heat_days_per_gridpoint(tx_c: xr.DataArray, threshold_c: float,
                                   time_dim: str) -> xr.DataArray:
    """Count days where TX >= threshold_c. Shape: (lat, lon)."""
    return (tx_c >= threshold_c).sum(dim=time_dim)


def count_frost_days_per_gridpoint(tn_c: xr.DataArray, threshold_c: float,
                                    time_dim: str) -> xr.DataArray:
    """Count days where TN < threshold_c. Shape: (lat, lon)."""
    return (tn_c < threshold_c).sum(dim=time_dim)


# ── Precipitation helpers ─────────────────────────────────────────────────────

def hourly_to_daily_pr(tp_m: xr.DataArray, time_dim: str) -> xr.DataArray:
    """
    Sum 24 per-hour tp values (metres) per calendar day and convert to mm.

    ERA5-Land total_precipitation stores the accumulation within each 1-hour
    period only (not cumulative from 00:00). Summing all 24 hourly values and
    multiplying by 1000 gives the daily total in mm.

    Returns shape (n_days, lat, lon) in mm/day.
    """
    return tp_m.resample({time_dim: "1D"}).sum() * 1000.0


def _max_cdd_per_gridpoint(pr_mm: xr.DataArray, time_dim: str) -> xr.DataArray:
    """
    Maximum consecutive dry day run (pr < 1 mm) across the full year per grid point.
    Requires the full annual daily pr array — do not call on a single month.
    Shape: (lat, lon).
    """
    def _max_run(arr: np.ndarray) -> float:
        count, max_run = 0, 0
        for v in arr:
            count = count + 1 if v < 1.0 else 0
            if count > max_run:
                max_run = count
        return float(max_run)

    return xr.apply_ufunc(
        _max_run,
        pr_mm,
        input_core_dims=[[time_dim]],
        vectorize=True,
        output_dtypes=[float],
    )


# ── Annual grid computation ───────────────────────────────────────────────────

def compute_annual_grid(year: int, raw_dir: Path, threshold_c: float,
                        metric: str = "heat_days") -> xr.DataArray:
    """
    Process all 12 monthly t2m NetCDF files for a given year and return a
    DataArray of shape (lat, lon) for the requested temperature metric.

    For count metrics (heat_days, frost_days, id0, tr15) the result is
    accumulated by summing monthly counts. For extremes (txx, tnn) the result
    is accumulated by taking the running max/min across months.
    """
    if metric not in METRIC_COL or metric in PRECIP_METRICS:
        raise ValueError(
            f"'{metric}' is not a t2m metric. Use compute_annual_precip_grid "
            f"for precipitation metrics."
        )

    logger.info("[%d] computing annual grid (%s)", year, metric)
    annual = None

    for month in range(1, 13):
        nc_path = raw_dir / f"era5land_t2m_{year}_{month:02d}.nc"
        ds, time_dim = open_monthly_nc(nc_path)
        t2m = ds["t2m"]

        if metric == "heat_days":
            monthly = count_heat_days_per_gridpoint(
                hourly_to_daily_tx(t2m, time_dim), threshold_c, time_dim)
            annual = monthly if annual is None else annual + monthly

        elif metric == "frost_days":
            monthly = count_frost_days_per_gridpoint(
                hourly_to_daily_tn(t2m, time_dim), threshold_c, time_dim)
            annual = monthly if annual is None else annual + monthly

        elif metric == "id0":
            tx = hourly_to_daily_tx(t2m, time_dim)
            monthly = (tx < threshold_c).sum(dim=time_dim)
            annual = monthly if annual is None else annual + monthly

        elif metric == "hard_frost":
            tn = hourly_to_daily_tn(t2m, time_dim)
            monthly = (tn < threshold_c).sum(dim=time_dim)
            annual = monthly if annual is None else annual + monthly

        elif metric == "tr15":
            tn = hourly_to_daily_tn(t2m, time_dim)
            monthly = (tn >= threshold_c).sum(dim=time_dim)   # ETCCDI warm-side uses >=
            annual = monthly if annual is None else annual + monthly

        elif metric == "txx":
            monthly_max = hourly_to_daily_tx(t2m, time_dim).max(dim=time_dim)
            if annual is None:
                annual = monthly_max
            else:
                # Use fillna(-inf) so that a NaN in one month never wins over
                # a valid value in another month (np.fmax semantics).
                annual = xr.where(
                    monthly_max.notnull() & (monthly_max > annual.fillna(-np.inf)),
                    monthly_max, annual,
                )

        elif metric == "tnn":
            monthly_min = hourly_to_daily_tn(t2m, time_dim).min(dim=time_dim)
            if annual is None:
                annual = monthly_min
            else:
                annual = xr.where(
                    monthly_min.notnull() & (monthly_min < annual.fillna(np.inf)),
                    monthly_min, annual,
                )

        ds.close()

    return annual


def compute_annual_precip_grid(year: int, raw_dir: Path, threshold_mm: float,
                                metric: str) -> xr.DataArray:
    """
    Process all 12 monthly tp NetCDF files for a given year and return a
    DataArray of shape (lat, lon) for the requested precipitation metric.

    CDD requires concatenating the full year before computing run lengths.
    PRCPTOT, R20mm, SDII are accumulated month-by-month.
    """
    if metric not in PRECIP_METRICS:
        raise ValueError(f"'{metric}' is not a precipitation metric.")

    logger.info("[%d] computing annual precip grid (%s)", year, metric)

    def _open_daily_pr(month: int):
        nc_path = raw_dir / f"era5land_tp_{year}_{month:02d}.nc"
        ds, time_dim = open_monthly_nc(nc_path)
        tp_var = "tp" if "tp" in ds.data_vars else list(ds.data_vars)[0]
        daily = hourly_to_daily_pr(ds[tp_var], time_dim)
        # Normalise the time dimension name so xr.concat never sees a mismatch
        # between files using "valid_time" and files using "time".
        if time_dim != "time":
            daily = daily.rename({time_dim: "time"})
        ds.close()
        return daily

    if metric == "cdd":
        # CDD spans month boundaries — concatenate the full year first.
        days = [_open_daily_pr(m) for m in range(1, 13)]
        pr_year = xr.concat(days, dim="time")
        return _max_cdd_per_gridpoint(pr_year, "time")

    # Month-by-month accumulation for PRCPTOT, R20mm, SDII.
    annual_wet_pr   = None  # sum of pr on wet days (mm)
    annual_wet_days = None  # count of wet days
    annual_r20      = None  # count of days > 20 mm

    for month in range(1, 13):
        daily = _open_daily_pr(month)

        wet_mask     = daily >= 1.0           # wet day: pr >= 1 mm
        wet_pr       = (daily * wet_mask).sum(dim=daily.dims[0])
        wet_day_cnt  = wet_mask.sum(dim=daily.dims[0])
        heavy_day_cnt = (daily >= 20.0).sum(dim=daily.dims[0])  # ETCCDI R20mm: >= 20 mm

        annual_wet_pr   = wet_pr       if annual_wet_pr   is None else annual_wet_pr   + wet_pr
        annual_wet_days = wet_day_cnt  if annual_wet_days is None else annual_wet_days + wet_day_cnt
        annual_r20      = heavy_day_cnt if annual_r20     is None else annual_r20      + heavy_day_cnt

    if metric == "prcptot":
        return annual_wet_pr
    if metric == "r20mm":
        return annual_r20
    # sdii
    return xr.where(annual_wet_days > 0, annual_wet_pr / annual_wet_days, np.nan)


# ── Scalar result ─────────────────────────────────────────────────────────────

def process_year(year: int, raw_dir: Path, threshold_c: float,
                 metric: str = "heat_days") -> dict:
    """
    Compute the annual metric for one year and return a result dict.
    Dispatches to compute_annual_grid or compute_annual_precip_grid based on metric.
    """
    logger.info("[%d] starting transform (%s)", year, metric)

    if metric in PRECIP_METRICS:
        annual = compute_annual_precip_grid(year, raw_dir, threshold_c, metric)
    else:
        annual = compute_annual_grid(year, raw_dir, threshold_c, metric)

    mean_val = float(annual.mean().values)
    col      = METRIC_COL[metric]
    result   = {"year": year, col: round(mean_val, 2)}
    logger.info("[%d] transform complete — %s: %.2f", year, col, mean_val)
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _METRIC_CSV = {
        "heat_days":  "estonia_extreme_heat_days.csv",
        "frost_days": "estonia_frost_days.csv",
        "hard_frost": "estonia_hard_frost.csv",
        "id0":        "estonia_id0.csv",
        "tr15":       "estonia_tr15.csv",
        "txx":        "estonia_txx.csv",
        "tnn":        "estonia_tnn.csv",
        "cdd":        "estonia_cdd.csv",
        "r20mm":      "estonia_r20mm.csv",
        "sdii":       "estonia_sdii.csv",
        "prcptot":    "estonia_prcptot.csv",
    }
    _METRIC_CFG = {
        "heat_days":  ("heat_days",  "threshold_tx_degC"),
        "frost_days": ("frost_days", "threshold_tn_degC"),
        "hard_frost": ("hard_frost", "threshold_tn_degC"),
        "id0":        ("id0",        "threshold_tx_degC"),
        "tr15":       ("tr15",       "threshold_tn_degC"),
        "txx":        None,
        "tnn":        None,
        "cdd":        ("cdd",        "threshold_pr_mm"),
        "r20mm":      ("r20mm",      "threshold_pr_mm"),
        "sdii":       ("sdii",       "threshold_pr_mm"),
        "prcptot":    ("prcptot",    "threshold_pr_mm"),
    }

    parser = argparse.ArgumentParser(
        description="Compute ETCCDI metrics from cached ERA5-Land files"
    )
    parser.add_argument("--country", default="EE")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--metric", default="heat_days", choices=list(METRIC_COL))
    args    = parser.parse_args()
    cfg     = load_config()
    metric  = args.metric
    col     = METRIC_COL[metric]
    cfg_key = _METRIC_CFG[metric]
    threshold_c = (
        float(cfg["metrics"][cfg_key[0]][cfg_key[1]]) if cfg_key else 0.0
    )
    years   = [args.year] if args.year else REFERENCE_YEARS
    raw_dir = RAW_DIR / args.country.lower()
    out_csv = OUT_CSV.parent / _METRIC_CSV[metric]

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
        pd.DataFrame(rows).sort_values("year").reset_index(drop=True).to_csv(
            out_csv, index=False)
        logger.info("Saved → %s", out_csv)


if __name__ == "__main__":
    main()
