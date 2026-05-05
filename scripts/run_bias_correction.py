#!/usr/bin/env python3
"""
run_bias_correction.py — Entry point for CORDEX QDM bias correction.

Reads all configuration from config/config.yaml. No thresholds, paths, or
model identifiers are hardcoded here.

Usage:
    conda activate climate-risk
    python scripts/run_bias_correction.py --country EE --scenario rcp45

    # Adjust number of quantile bins (default 50 is fine for PoC)
    python scripts/run_bias_correction.py --country EE --scenario rcp45 --nquantiles 100

    # Override default directories (useful for testing)
    python scripts/run_bias_correction.py --country EE --scenario rcp45 \\
        --cordex-dir /path/to/clipped/files \\
        --output-dir /path/to/corrected/output

Runtime:
    For Estonia (~575 land cells, 30-year calibration, 50 quantiles):
    Expect ~5–20 minutes per variable on a modern laptop CPU.
    Total for all 4 variables: ~30–60 minutes.
"""

from __future__ import annotations

import logging
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from load_data import load_config  # noqa: E402
from bias_correct import run_bias_correction  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="CORDEX Quantile Delta Mapping bias correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--country", default="EE",
        help="ISO 3166-1 alpha-2 country code (must be in config.yaml → countries).",
    )
    parser.add_argument(
        "--scenario", default="rcp45",
        choices=list(cfg["cordex"]["scenarios"]),
        help="CORDEX scenario to bias-correct.",
    )
    parser.add_argument(
        "--nquantiles", type=int, default=50,
        help="Number of quantile bins for QDM (50 for PoC, 100 for production).",
    )
    parser.add_argument(
        "--era5-dir", default=None,
        help=(
            "Directory containing ERA5-Land NetCDF files for the country. "
            "Defaults to data/raw/era5land/{country}/ inside the project root."
        ),
    )
    parser.add_argument(
        "--cordex-dir", default=None,
        help=(
            "Directory containing clipped CORDEX NetCDF files. "
            "Defaults to data/raw/cordex/{country}/ inside the project root."
        ),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Directory for bias-corrected output files. "
            "Defaults to data/raw/cordex_bc/{country}/{scenario}/."
        ),
    )
    args = parser.parse_args()

    country  = args.country.upper()
    scenario = args.scenario

    if country not in cfg.get("countries", {}):
        logging.error(
            "Country '%s' not found in config.yaml. Available: %s",
            country, list(cfg["countries"]),
        )
        sys.exit(1)

    cordex_dir = Path(args.cordex_dir) if args.cordex_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    run_bias_correction(
        country=country,
        scenario=scenario,
        era5_dir=Path(args.era5_dir) if args.era5_dir else None,
        cordex_dir=cordex_dir,
        bc_dir=output_dir,
        nquantiles=args.nquantiles,
    )


if __name__ == "__main__":
    main()
