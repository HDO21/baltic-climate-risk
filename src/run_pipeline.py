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
import argparse
import pandas as pd
import cdsapi

from load_data import (
    REFERENCE_YEARS, RAW_DIR, OUT_CSV,
    load_config, get_country_area, download_year,
)
from validate  import validate_raw_file, validate_annual_result
from transform import process_year


def main():
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
    args    = parser.parse_args()
    cfg     = load_config()

    country_code = args.country.upper()
    country_name = cfg["countries"][country_code]["name"]
    area         = cfg["countries"][country_code]["area"]    # [N, W, S, E]
    raw_dir      = RAW_DIR / country_code.lower()
    threshold_c  = float(cfg["metrics"]["heat_days"]["threshold_tx_degC"])
    years        = [args.year] if args.year else REFERENCE_YEARS

    print("=" * 60)
    print(f"  Extreme Heat Days — {country_name}  |  ERA5-Land")
    if len(years) > 1:
        print(f"  Years  : {years[0]}–{years[-1]}")
    else:
        print(f"  Year   : {years[0]}")
    print(f"  Metric : mean annual days with TX ≥ {threshold_c} °C")
    print(f"  Area   : {area}  (N, W, S, E)")
    print("=" * 60)

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
            print(f"\n  [{year}] already in output CSV — skipping")
            rows.append({"year": year, "extreme_heat_days": existing[year]})
            continue

        # ── Stage 1: LOAD ──────────────────────────────────────────────────────
        print(f"\n  [{year}] ── LOAD ──────────────────────────────────────────")
        try:
            download_year(client, year, area, raw_dir)
        except Exception as exc:
            print(f"  [{year}] download failed — skipping year: {exc}", file=sys.stderr)
            continue

        # ── Stage 2: VALIDATE raw files ────────────────────────────────────────
        print(f"  [{year}] ── VALIDATE (raw) ───────────────────────────────")
        bounds = cfg["validation"]
        raw_ok = True
        for month in range(1, 13):
            nc_path = raw_dir / f"era5land_t2m_{year}_{month:02d}.nc"
            result  = validate_raw_file(nc_path, year, month, bounds)
            status  = "OK  " if result["passed"] else "FAIL"
            print(f"    [{year}-{month:02d}] {status}", end="")
            if result["issues"]:
                print(f"  →  {'; '.join(result['issues'])}", end="")
                raw_ok = False
            print()

        if not raw_ok:
            print(f"  [{year}] raw validation failed — skipping transform", file=sys.stderr)
            continue

        # ── Stage 3: TRANSFORM ─────────────────────────────────────────────────
        print(f"  [{year}] ── TRANSFORM ────────────────────────────────────")
        try:
            row = process_year(year, raw_dir, threshold_c)
        except Exception as exc:
            print(f"  [{year}] transform failed — skipping year: {exc}", file=sys.stderr)
            continue

        # ── Stage 4: VALIDATE result ───────────────────────────────────────────
        print(f"  [{year}] ── VALIDATE (result) ─────────────────────────────")
        result_check = validate_annual_result(row)
        if not result_check["passed"]:
            print(
                f"  [{year}] result validation failed: {result_check['issues']}",
                file=sys.stderr,
            )
            continue

        rows.append(row)
        print(f"  [{year}] extreme heat days: {row['extreme_heat_days']:.2f}")

    if not rows:
        print("No data processed — check errors above.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 60)
    print(f"  Output : {OUT_CSV}")
    print("=" * 60)
    print(df.to_string(index=False))
    if len(df) > 1:
        print(f"\n  Mean : {df['extreme_heat_days'].mean():.2f} days/year")
        print(f"  Range: {df['extreme_heat_days'].min():.0f} – "
              f"{df['extreme_heat_days'].max():.0f} days/year")


if __name__ == "__main__":
    main()
