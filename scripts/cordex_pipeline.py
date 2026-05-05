#!/usr/bin/env python3
"""
cordex_pipeline.py — Compute ETCCDI metrics from bias-corrected CORDEX projections.

Reads bias-corrected daily CORDEX NetCDF files (produced by run_bias_correction.py)
and computes the same ETCCDI metrics as the ERA5-Land pipeline, writing results to
data/processed/cordex/{scenario}/ as CSVs and grid Parquets compatible with the
existing Streamlit dashboard.

Key differences from the ERA5-Land pipeline (run_pipeline.py):
  - CORDEX tasmax/tasmin are native daily max/min — no hourly resampling needed.
  - CORDEX pr after bias correction is in mm/day — no per-hour accumulation logic.
  - CORDEX files are 5-year chunks, not monthly; opening is done with open_mfdataset.
  - No CDS download stage; no raw-file validation stage.
  - CORDEX uses a rotated-pole grid (rlat/rlon dims, 2-D lat/lon aux coords).
    A geographic bounding-box mask is applied before the spatial mean to exclude
    grid cells outside the target country but inside the clipped rectangle.
  - CORDEX has no explicit land-sea mask; coastal sea cells within the bbox
    contribute to the spatial mean (known PoC limitation).

Reuses from transform.py (unchanged):
  count_heat_days_per_gridpoint, count_frost_days_per_gridpoint,
  _max_cdd_per_gridpoint, METRIC_COL, PRECIP_METRICS

Usage:
    conda activate climate-risk
    python scripts/cordex_pipeline.py --country EE --scenario rcp45 --metric heat_days

    # Process all metrics:
    for metric in heat_days frost_days hard_frost id0 tr15 txx tnn cdd r20mm sdii prcptot; do
        python scripts/cordex_pipeline.py --country EE --scenario rcp45 --metric $metric
    done
"""

from __future__ import annotations

import logging
import argparse
import sys
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from load_data import load_config  # noqa: E402
from transform import (  # noqa: E402
    METRIC_COL, PRECIP_METRICS,
    count_heat_days_per_gridpoint,
    count_frost_days_per_gridpoint,
    _max_cdd_per_gridpoint,
)
from validate import validate_annual_result  # noqa: E402

logger = logging.getLogger(__name__)

CORDEX_BC_DIR  = ROOT / "data" / "raw" / "cordex_bc"
CORDEX_OUT_DIR = ROOT / "data" / "processed" / "cordex"

# BC files store temperature in Kelvin (bias correction runs in K to avoid
# pint offset-unit issues with degC). Convert to °C here before metric computation.
_BC_TEMP_VARS = frozenset({"tasmax", "tasmin", "tas"})

# Which CORDEX variable each metric reads from.
_METRIC_CORDEX_VAR = {
    "heat_days":  "tasmax",
    "id0":        "tasmax",
    "txx":        "tasmax",
    "frost_days": "tasmin",
    "hard_frost": "tasmin",   # TN < -10 °C — uses tasmin
    "tr15":       "tasmin",
    "tnn":        "tasmin",
    "cdd":        "pr",
    "r20mm":      "pr",
    "sdii":       "pr",
    "prcptot":    "pr",
}

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


# ── CORDEX file access ────────────────────────────────────────────────────────

def open_cordex_year(
    bc_dir: Path, cordex_var: str, year: int
) -> xr.DataArray:
    """
    Lazily open all bias-corrected CORDEX files for `cordex_var`, select the
    given calendar year, and return the data loaded into memory.

    Returns a DataArray of shape (days, rlat, rlon). Days is 365 or 366
    depending on whether `year` is a leap year in the CORDEX calendar.

    Data is already in pipeline units: °C for temperature, mm/day for pr.
    """
    files = sorted(bc_dir.glob(f"{cordex_var}_EUR-11_*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No bias-corrected CORDEX files for '{cordex_var}' in {bc_dir}"
        )

    _time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="time",
                           decode_times=_time_coder, data_vars="minimal",
                           coords="minimal")
    mask = ds.time.dt.year == year
    if not bool(mask.any()):
        raise ValueError(
            f"Year {year} not found in CORDEX files for {cordex_var}."
        )
    da = ds[cordex_var].sel(time=mask).load()

    # Ensure time is the first dimension. combine="nested" with concat_dim="time"
    # can produce (rlat, time, rlon) instead of (time, rlat, rlon), which causes
    # downstream sum(dim=da.dims[0]) to sum over space instead of time.
    if da.dims[0] != "time" and "time" in da.dims:
        other_dims = [d for d in da.dims if d != "time"]
        da = da.transpose("time", *other_dims)

    # coords="minimal" drops 2-D lat/lon auxiliary coordinates because they are
    # not dimension coords. Re-attach from the first file so mask_to_bbox can work.
    if "lat" not in da.coords and "lon" not in da.coords:
        try:
            _ds0 = xr.open_dataset(files[0], decode_times=False)
            if "lat" in _ds0.coords:
                da = da.assign_coords(lat=_ds0["lat"], lon=_ds0["lon"])
            _ds0.close()
        except Exception:
            pass  # mask_to_bbox will warn and skip if coords are still absent

    ds.close()
    if cordex_var in _BC_TEMP_VARS:
        da = da - 273.15   # K → °C (bc files keep temperature in K)
    return da


# ── Geographic masking ────────────────────────────────────────────────────────

def mask_to_bbox(
    da: xr.DataArray, north: float, west: float, south: float, east: float
) -> xr.DataArray:
    """
    Set CORDEX grid cells outside the geographic bounding box to NaN.

    CORDEX EUR-11 is on a rotated-pole grid. After spatial clipping, the
    rectangular rlat/rlon subset includes cells slightly outside the target
    country boundary. This mask ensures the spatial mean covers only the
    target area.

    Requires 2-D `lat` and `lon` auxiliary coordinates in `da`.
    """
    if "lat" not in da.coords or "lon" not in da.coords:
        logger.warning("No lat/lon aux coords found — skipping bbox mask")
        return da
    mask = (
        (da["lat"] >= south) & (da["lat"] <= north) &
        (da["lon"] >= west)  & (da["lon"] <= east)
    )
    return da.where(mask)


# ── Annual metric computation ─────────────────────────────────────────────────

def compute_annual_cordex_temp_grid(
    year: int,
    bc_dir: Path,
    metric: str,
    threshold_c: float,
) -> xr.DataArray:
    """
    Compute one year's temperature-based ETCCDI metric from bias-corrected
    CORDEX daily tasmax or tasmin.

    Returns a DataArray of shape (rlat, rlon) in metric units:
      - day-count metrics: integer count
      - TXx / TNn: temperature in °C (the annual extreme value)
    """
    cordex_var = _METRIC_CORDEX_VAR[metric]
    da = open_cordex_year(bc_dir, cordex_var, year)
    time_dim = da.dims[0]  # typically "time"

    if metric == "heat_days":
        result = count_heat_days_per_gridpoint(da, threshold_c, time_dim)

    elif metric == "frost_days":
        result = count_frost_days_per_gridpoint(da, threshold_c, time_dim)

    elif metric == "hard_frost":
        result = count_frost_days_per_gridpoint(da, threshold_c, time_dim)

    elif metric == "id0":
        result = (da < threshold_c).sum(dim=time_dim)

    elif metric == "tr15":
        result = (da >= threshold_c).sum(dim=time_dim)

    elif metric == "txx":
        result = da.max(dim=time_dim)

    elif metric == "tnn":
        result = da.min(dim=time_dim)

    else:
        raise ValueError(f"'{metric}' is not a temperature-based metric")

    return result


def compute_annual_cordex_precip_grid(
    year: int,
    bc_dir: Path,
    metric: str,
    threshold_mm: float,
) -> xr.DataArray:
    """
    Compute one year's precipitation-based ETCCDI metric from bias-corrected
    CORDEX daily pr (already in mm/day after bias correction).

    CDD requires the full 365-day annual series to correctly detect dry spells
    that span month boundaries — all days are loaded at once per year.

    Returns a DataArray of shape (rlat, rlon).
    """
    da = open_cordex_year(bc_dir, "pr", year)
    da = da.clip(min=0)           # guard against residual negative values
    time_dim = da.dims[0]

    if metric == "cdd":
        return _max_cdd_per_gridpoint(da, time_dim)

    wet_mask      = da >= 1.0
    wet_pr        = (da * wet_mask).sum(dim=time_dim)
    wet_day_count = wet_mask.sum(dim=time_dim)
    heavy_count   = (da >= 20.0).sum(dim=time_dim)

    if metric == "prcptot":
        return wet_pr
    if metric == "r20mm":
        return heavy_count
    # sdii
    return xr.where(wet_day_count > 0, wet_pr / wet_day_count, np.nan)


# ── Parquet writer (mirrors run_pipeline._write_grid_parquet) ─────────────────

def _write_grid_parquet(
    annual_grid: xr.DataArray, year: int, path: Path, col: str
) -> None:
    """Upsert one year's per-grid-point values into the grid Parquet."""
    # Drop any scalar cftime coordinates (e.g. residual 'time') that pyarrow
    # cannot serialise; they are not needed in the Parquet output.
    _drop = [c for c in annual_grid.coords
             if c not in annual_grid.dims and annual_grid[c].ndim == 0]
    if _drop:
        annual_grid = annual_grid.drop_vars(_drop)
    df_year = annual_grid.to_dataframe(name=col).reset_index()
    df_year = df_year.assign(year=year).dropna(subset=[col])

    if path.exists():
        df_existing = pd.read_parquet(path)
        df_existing = df_existing[df_existing["year"] != year]
        df_year = pd.concat([df_existing, df_year], ignore_index=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    df_year.to_parquet(path, index=False)


# ── Orchestration ─────────────────────────────────────────────────────────────

def process_cordex_year(
    year: int,
    bc_dir: Path,
    metric: str,
    threshold_c: float,
    area: list,
) -> dict:
    """
    Compute one year's metric, apply bbox mask, take spatial mean.
    Returns {year: ..., col: ...} ready to append to the output CSV.
    """
    north, west, south, east = area

    if metric in PRECIP_METRICS:
        grid = compute_annual_cordex_precip_grid(year, bc_dir, metric, threshold_c)
    else:
        grid = compute_annual_cordex_temp_grid(year, bc_dir, metric, threshold_c)

    # Apply geographic mask before spatial mean so cells outside the country
    # boundary (but inside the rectangular clip) do not bias the result.
    grid = mask_to_bbox(grid, north, west, south, east)

    mean_val = float(grid.mean().values)
    col      = METRIC_COL[metric]
    return {"year": year, col: round(mean_val, 2), "_grid": grid}


def run_cordex_pipeline(
    country: str,
    scenario: str,
    metric: str,
    bc_dir:  Path | None = None,
    out_dir: Path | None = None,
    no_bias_correction: bool = False,
) -> None:
    """
    Process all projection years for one country, scenario, and metric.

    Output files:
      {out_dir}/{scenario}/estonia_{metric}.csv
      {out_dir}/{scenario}/{metric_key}_grid_ee.parquet
    """
    cfg       = load_config()
    area      = cfg["countries"][country]["area"]   # [N, W, S, E]
    ccfg      = cfg["cordex"]
    scen_cfg  = ccfg["scenarios"][scenario]

    _default_bc  = CORDEX_BC_DIR / country.lower() / scenario
    _clipped_raw = ROOT / "data" / "raw" / "cordex" / country.lower()

    if no_bias_correction:
        bc_dir = bc_dir or _clipped_raw
        logger.warning(
            "Running WITHOUT bias correction — using raw clipped files from %s. "
            "Results are indicative only; run run_bias_correction.py for "
            "scientifically valid projections.",
            bc_dir,
        )
    else:
        bc_dir = bc_dir or _default_bc
        # If bias-corrected directory is empty, offer a helpful message.
        if not bc_dir.exists() or not any(bc_dir.glob("*.nc")):
            logger.error(
                "No bias-corrected files found in %s.\n"
                "  Option 1 — Run bias correction first (recommended):\n"
                "    python scripts/run_bias_correction.py "
                "--country %s --scenario %s\n"
                "  Option 2 — Use raw clipped files without bias correction:\n"
                "    python scripts/cordex_pipeline.py --metric %s "
                "--no-bias-correction",
                bc_dir, country, scenario, metric,
            )
            return

    out_dir = out_dir or (CORDEX_OUT_DIR / scenario)
    out_dir.mkdir(parents=True, exist_ok=True)

    col          = METRIC_COL[metric]
    out_csv      = out_dir / _METRIC_CSV[metric]
    grid_parquet = out_dir / f"{metric}_grid_ee.parquet"

    # Threshold value from config (None for TXx / TNn).
    cfg_entry = _METRIC_CFG[metric]
    threshold_c = (
        float(cfg["metrics"][cfg_entry[0]][cfg_entry[1]])
        if cfg_entry else 0.0
    )

    proj_start = scen_cfg["projection_start"]
    proj_end   = scen_cfg["projection_end"]
    years      = list(range(proj_start, proj_end + 1))

    # Load any previously computed years so re-runs are incremental.
    existing: dict[int, float] = {}
    if out_csv.exists():
        existing = pd.read_csv(out_csv).set_index("year")[col].to_dict()

    logger.info("=" * 55)
    logger.info("%s | %s | %s", country, scenario, metric)
    logger.info("Threshold : %.1f | Period : %d–%d", threshold_c, proj_start, proj_end)
    logger.info("Source    : %s", bc_dir)
    logger.info("=" * 55)

    rows = []
    for year in years:
        # Check Parquet too so grid is regenerated if missing.
        parquet_has_year = False
        if grid_parquet.exists():
            try:
                _yrs = pd.read_parquet(grid_parquet, columns=["year"])["year"].values
                parquet_has_year = int(year) in _yrs
            except Exception:
                pass

        if year in existing and parquet_has_year:
            logger.info("[%d] already done — skipping", year)
            rows.append({"year": year, col: existing[year]})
            continue

        try:
            result = process_cordex_year(year, bc_dir, metric, threshold_c, area)
            grid   = result.pop("_grid")
            rows.append(result)
        except Exception as exc:
            logger.error("[%d] failed: %s", year, exc)
            continue

        # Validate result before writing.
        check = validate_annual_result(result, metric_col=col)
        if not check["passed"]:
            logger.warning("[%d] result validation issues: %s", year, check["issues"])

        # Write grid Parquet.
        try:
            _write_grid_parquet(grid, year, grid_parquet, col)
        except Exception as exc:
            logger.warning("[%d] grid Parquet write failed: %s", year, exc)

        logger.info("[%d] %s = %.2f", year, col, result[col])

    if rows:
        df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
        df.to_csv(out_csv, index=False)
        logger.info("Saved → %s  (%d years)", out_csv, len(df))
        if len(df) > 1:
            logger.info("Mean: %.2f | Range: %.2f–%.2f",
                        df[col].mean(), df[col].min(), df[col].max())


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Compute ETCCDI metrics from bias-corrected CORDEX projections",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--country",  default="EE",
                        choices=list(cfg["countries"]),
                        help="ISO 3166-1 alpha-2 country code.")
    parser.add_argument("--scenario", default="rcp45",
                        choices=list(cfg["cordex"]["scenarios"]),
                        help="CORDEX scenario.")
    parser.add_argument("--metric",   default="heat_days",
                        choices=list(METRIC_COL),
                        help="ETCCDI metric to compute.")
    parser.add_argument("--bc-dir",   default=None,
                        help="Bias-corrected CORDEX file directory (overrides default).")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for CSVs and Parquets (overrides default).")
    parser.add_argument("--no-bias-correction", action="store_true",
                        help=(
                            "Use raw clipped CORDEX files instead of bias-corrected ones. "
                            "Results are indicative only — run run_bias_correction.py first "
                            "for scientifically valid projections."
                        ))
    args = parser.parse_args()

    run_cordex_pipeline(
        country=args.country.upper(),
        scenario=args.scenario,
        metric=args.metric,
        bc_dir=Path(args.bc_dir) if args.bc_dir else None,
        out_dir=Path(args.output_dir) if args.output_dir else None,
        no_bias_correction=args.no_bias_correction,
    )


if __name__ == "__main__":
    main()
