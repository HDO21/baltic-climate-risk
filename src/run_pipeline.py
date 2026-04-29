#!/usr/bin/env python3
from __future__ import annotations
"""
run_pipeline.py — ERA5-Land ETCCDI metrics pipeline orchestrator.

Runs the full pipeline for one or all years of the 1991–2020 WMO reference
period for a given Baltic country and metric:

  1. LOAD      — download monthly ERA5-Land NetCDF files from CDS (cached)
  2. VALIDATE  — check raw t2m files (temperature metrics only)
  3. TRANSFORM — compute the ETCCDI metric per grid point, average over country
  4. VALIDATE  — sanity-check the annual scalar result

Usage:
    conda activate climate-risk
    python src/run_pipeline.py [--country EE] [--year YYYY] [--metric heat_days]
"""

import sys
import logging
import argparse
import pandas as pd

from load_data import (
    REFERENCE_YEARS, RAW_DIR, OUT_CSV,
    load_config, download_year,
)
from validate  import validate_raw_file, validate_raw_tp_file, validate_annual_result
from transform import (
    compute_annual_grid, compute_annual_precip_grid,
    METRIC_COL, PRECIP_METRICS,
)

logger = logging.getLogger(__name__)

# ── Output path mappings ──────────────────────────────────────────────────────

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

# (config section, config key) — None for metrics without a threshold.
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


def _write_grid_parquet(annual_grid, year: int, path, col: str) -> None:
    """Upsert one year's per-grid-point values into the grid Parquet."""
    df_year = annual_grid.to_dataframe(name=col).reset_index()
    df_year = df_year.assign(year=year).dropna(subset=[col])

    if path.exists():
        df_existing = pd.read_parquet(path)
        df_existing = df_existing[df_existing["year"] != year]
        df_year = pd.concat([df_existing, df_year], ignore_index=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    df_year.to_parquet(path, index=False)


def _run_transform(year, raw_dir, threshold_c, metric_key):
    """Dispatch to the correct transform function based on metric type."""
    if metric_key in PRECIP_METRICS:
        return compute_annual_precip_grid(year, raw_dir, threshold_c, metric_key)
    return compute_annual_grid(year, raw_dir, threshold_c, metric_key)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="ERA5-Land ETCCDI metrics pipeline")
    parser.add_argument("--country", default="EE",
                        help="ISO 3166-1 alpha-2 country code (default: EE)")
    parser.add_argument("--year", type=int, default=None,
                        help="Run for a single year only")
    parser.add_argument("--metric", default="heat_days", choices=list(METRIC_COL),
                        help="ETCCDI metric to compute (default: heat_days)")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip the CDS download stage and transform from cached files only")
    args = parser.parse_args()
    cfg  = load_config()

    country_code = args.country.upper()
    metric_key   = args.metric
    col          = METRIC_COL[metric_key]
    country_name = cfg["countries"][country_code]["name"]
    area         = cfg["countries"][country_code]["area"]
    raw_dir      = RAW_DIR / country_code.lower()
    bounds       = cfg["validation"]
    years        = [args.year] if args.year else REFERENCE_YEARS
    is_precip    = metric_key in PRECIP_METRICS
    no_download  = args.no_download

    cfg_entry   = _METRIC_CFG[metric_key]
    threshold_c = (
        float(cfg["metrics"][cfg_entry[0]][cfg_entry[1]]) if cfg_entry else 0.0
    )

    out_csv      = OUT_CSV.parent / _METRIC_CSV[metric_key]
    grid_parquet = OUT_CSV.parent / f"{metric_key}_grid_{country_code.lower()}.parquet"

    logger.info("=" * 55)
    logger.info("%s — %s | ERA5-Land", col, country_name)
    logger.info("Years  : %d–%d", years[0], years[-1])
    logger.info("Source : %s", "total_precipitation (tp)" if is_precip else "2m_temperature (t2m)")
    logger.info("Area   : %s  (N, W, S, E)", area)
    logger.info("=" * 55)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if out_csv.exists():
        existing = pd.read_csv(out_csv).set_index("year")[col].to_dict()

    client = None   # created on first download; skipped entirely with --no-download
    rows   = []

    for year in years:
        # Check whether this year already exists in both CSV and Parquet.
        parquet_has_year = False
        if grid_parquet.exists():
            try:
                _yrs = pd.read_parquet(grid_parquet, columns=["year"])["year"].values
                parquet_has_year = int(year) in _yrs
            except Exception:
                pass

        if year in existing and parquet_has_year:
            logger.info("[%d] already in output CSV and grid — skipping", year)
            rows.append({"year": year, col: existing[year]})
            continue

        if year in existing and not parquet_has_year:
            logger.info("[%d] in CSV but grid missing — recomputing grid", year)
            try:
                annual_grid = _run_transform(year, raw_dir, threshold_c, metric_key)
                _write_grid_parquet(annual_grid, year, grid_parquet, col)
                logger.info("[%d] grid saved → %s", year, grid_parquet)
            except Exception as exc:
                logger.error("[%d] grid recompute failed: %s", year, exc)
            rows.append({"year": year, col: existing[year]})
            continue

        # ── Stage 1: LOAD ──────────────────────────────────────────────────────
        if no_download:
            logger.info("[%d] LOAD skipped (--no-download)", year)
        else:
            logger.info("[%d] LOAD", year)
            cds_variable = "total_precipitation" if is_precip else "2m_temperature"
            try:
                if client is None:
                    import cdsapi          # deferred so --no-download needs no credentials
                    client = cdsapi.Client()
                download_year(client, year, area, raw_dir, variable=cds_variable)
            except Exception as exc:
                logger.error("[%d] download failed — skipping year: %s", year, exc)
                continue

        # ── Stage 2: VALIDATE raw files ────────────────────────────────────────
        logger.info("[%d] VALIDATE (raw %s)", year, "tp" if is_precip else "t2m")
        raw_ok = True
        for month in range(1, 13):
            if is_precip:
                nc_path = raw_dir / f"era5land_tp_{year}_{month:02d}.nc"
                result  = validate_raw_tp_file(nc_path, year, month)
            else:
                nc_path = raw_dir / f"era5land_t2m_{year}_{month:02d}.nc"
                result  = validate_raw_file(nc_path, year, month, bounds)
            if not result["passed"]:
                raw_ok = False
        if not raw_ok:
            logger.error("[%d] raw validation failed — skipping transform", year)
            continue

        # ── Stage 3: TRANSFORM ─────────────────────────────────────────────────
        logger.info("[%d] TRANSFORM", year)
        try:
            annual_grid = _run_transform(year, raw_dir, threshold_c, metric_key)
            mean_val    = float(annual_grid.mean().values)
            row         = {"year": year, col: round(mean_val, 2)}
        except Exception as exc:
            logger.error("[%d] transform failed — skipping year: %s", year, exc)
            continue

        # ── Stage 4: VALIDATE result ───────────────────────────────────────────
        logger.info("[%d] VALIDATE (result)", year)
        result_check = validate_annual_result(row, metric_col=col)
        if not result_check["passed"]:
            logger.error("[%d] result validation failed — skipping: %s",
                         year, result_check["issues"])
            continue

        # ── Write grid Parquet ─────────────────────────────────────────────────
        try:
            _write_grid_parquet(annual_grid, year, grid_parquet, col)
            logger.info("[%d] grid saved → %s", year, grid_parquet)
        except Exception as exc:
            logger.warning("[%d] grid Parquet write failed (non-fatal): %s", year, exc)

        rows.append(row)

    if not rows:
        logger.error("No data processed — check errors above.")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    logger.info("=" * 55)
    logger.info("Output : %s", out_csv)
    logger.info("=" * 55)
    for _, r in df.iterrows():
        logger.info("  %d  %.2f", r["year"], r[col])
    if len(df) > 1:
        logger.info("Mean : %.2f", df[col].mean())
        logger.info("Range: %.2f – %.2f", df[col].min(), df[col].max())


if __name__ == "__main__":
    main()
