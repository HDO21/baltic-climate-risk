#!/usr/bin/env python3
"""
run_pipeline.py — ERA5-Land extreme heat days pipeline orchestrator.

Runs the full three-stage pipeline for one or all years of the 1991–2020
WMO reference period, for a given Baltic country:

  1. LOAD      — download monthly ERA5-Land NetCDF files from CDS (cached on disk)
  2. VALIDATE  — check each raw file for structural integrity and physical plausibility
  3. TRANSFORM — derive daily TX, count extreme heat days, average over country grid
  4. VALIDATE  — sanity-check the annual scalar result before writing to CSV

Each stage is implemented in its own module:
  load_data.py  → CDS API download, caching, config helpers
  validate.py   → quality checks at raw-file and result level
  transform.py  → unit conversion, resampling, metric computation

Usage:
    conda activate climate-risk
    python src/run_pipeline.py [--country EE] [--year YYYY]
"""

import sys
import logging
import argparse
import pandas as pd
import cdsapi

from load_data import (
    REFERENCE_YEARS, RAW_DIR, OUT_CSV,
    load_config, download_year,
)
from validate  import validate_raw_file, validate_annual_result
from transform import process_year

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="ERA5-Land extreme heat days pipeline")
    parser.add_argument(
        "--country", default="EE",
        help="ISO 3166-1 alpha-2 country code (default: EE). "
             "Must be defined in config/config.yaml → countries."
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Run for a single year only (default: full 1991–2020 reference period)"
    )
    args = parser.parse_args()
    cfg  = load_config()

    country_code = args.country.upper()
    country_name = cfg["countries"][country_code]["name"]
    area         = cfg["countries"][country_code]["area"]    # [N, W, S, E]
    raw_dir      = RAW_DIR / country_code.lower()
    threshold_c  = float(cfg["metrics"]["heat_days"]["threshold_tx_degC"])
    bounds       = cfg["validation"]
    years        = [args.year] if args.year else REFERENCE_YEARS

    logger.info("=" * 55)
    logger.info("Extreme Heat Days — %s | ERA5-Land", country_name)
    logger.info("Years  : %d–%d", years[0], years[-1])
    logger.info("Metric : mean annual days with TX >= %.0f °C", threshold_c)
    logger.info("Area   : %s  (N, W, S, E)", area)
    logger.info("=" * 55)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Load any previously saved results so partial runs accumulate correctly
    # without re-downloading or re-processing already-completed years.
    existing = {}
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV).set_index("year")["extreme_heat_days"].to_dict()

    client = cdsapi.Client()
    rows   = []

    for year in years:
        if year in existing:
            logger.info("[%d] already in output CSV — skipping", year)
            rows.append({"year": year, "extreme_heat_days": existing[year]})
            continue

        # ── Stage 1: LOAD ──────────────────────────────────────────────────────
        logger.info("[%d] LOAD", year)
        try:
            download_year(client, year, area, raw_dir)
        except Exception as exc:
            logger.error("[%d] download failed — skipping year: %s", year, exc)
            continue

        # ── Stage 2: VALIDATE raw files ────────────────────────────────────────
        logger.info("[%d] VALIDATE (raw)", year)
        raw_ok = True
        for month in range(1, 13):
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
            row = process_year(year, raw_dir, threshold_c)
        except Exception as exc:
            logger.error("[%d] transform failed — skipping year: %s", year, exc)
            continue

        # ── Stage 4: VALIDATE result ───────────────────────────────────────────
        logger.info("[%d] VALIDATE (result)", year)
        result_check = validate_annual_result(row)
        if not result_check["passed"]:
            logger.error("[%d] result validation failed — skipping: %s",
                         year, result_check["issues"])
            continue

        rows.append(row)

    if not rows:
        logger.error("No data processed — check errors above.")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    logger.info("=" * 55)
    logger.info("Output : %s", OUT_CSV)
    logger.info("=" * 55)
    for _, r in df.iterrows():
        logger.info("  %d  %.2f days", r["year"], r["extreme_heat_days"])
    if len(df) > 1:
        logger.info("Mean : %.2f days/year", df["extreme_heat_days"].mean())
        logger.info("Range: %.0f – %.0f days/year",
                    df["extreme_heat_days"].min(), df["extreme_heat_days"].max())


if __name__ == "__main__":
    main()
