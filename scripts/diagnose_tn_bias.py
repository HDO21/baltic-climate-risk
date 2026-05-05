#!/usr/bin/env python3
"""
diagnose_tn_bias.py — Compare ERA5-Land vs raw CORDEX TN to assess bias direction.

The key diagnostic: if raw CORDEX TN is colder than ERA5-Land (cold bias),
QDM will warm it up during correction. A large cold bias explains why
bias-corrected future projections show fewer frost days and more warm nights
than expected — the QDM warm shift is baked into the transfer function.

Usage:
    conda activate climate-risk
    python scripts/diagnose_tn_bias.py
"""

import sys
import numpy as np
import xarray as xr
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bias_correct import _era5_daily_year, load_cordex_period, ERA5_DIR

ERA5_DIR_EE = ERA5_DIR / "ee"
CORDEX_DIR  = ROOT / "data" / "raw" / "cordex" / "ee"


def mean_c(da: xr.DataArray) -> float:
    """Spatial+temporal mean in °C (converts from K if needed)."""
    m = float(np.nanmean(da.values))
    return m - 273.15 if m > 100 else m


def percentile_c(da: xr.DataArray, q: float) -> float:
    v = da.values.ravel()
    v = v[~np.isnan(v)]
    p = float(np.percentile(v, q))
    return p - 273.15 if p > 100 else p


def main():
    print("=" * 60)
    print("TN warm-bias diagnostic — ERA5-Land vs raw CORDEX")
    print("Calibration period: 1991–2005 (CORDEX historical)")
    print("=" * 60)

    # ── ERA5-Land TN (1991–2005) ──────────────────────────────────
    print("\nLoading ERA5-Land TN 1991–2005 …")
    era5_years = [_era5_daily_year(yr, ERA5_DIR_EE, "tasmin")
                  for yr in range(1991, 2006)]
    era5 = xr.concat(era5_years, dim="time")
    m_era5 = mean_c(era5)
    p5_era5  = percentile_c(era5,  5)
    p95_era5 = percentile_c(era5, 95)
    print(f"  Mean: {m_era5:+.2f} °C  |  5th pct: {p5_era5:+.1f} °C  |  95th pct: {p95_era5:+.1f} °C")

    # ── Raw CORDEX historical TN (1991–2005) ──────────────────────
    print("\nLoading raw CORDEX historical tasmin 1991–2005 …")
    raw = load_cordex_period(CORDEX_DIR, "tasmin", 1991, 2005)
    m_raw  = mean_c(raw)
    p5_raw  = percentile_c(raw,  5)
    p95_raw = percentile_c(raw, 95)
    print(f"  Mean: {m_raw:+.2f} °C  |  5th pct: {p5_raw:+.1f} °C  |  95th pct: {p95_raw:+.1f} °C")

    # ── Bias ──────────────────────────────────────────────────────
    bias_mean = m_raw - m_era5
    bias_cold = p5_raw - p5_era5    # cold tail (winter nights)
    bias_warm = p95_raw - p95_era5  # warm tail (summer nights)

    print(f"\n  Bias (CORDEX − ERA5):")
    print(f"    Mean:       {bias_mean:+.2f} °C")
    print(f"    Cold tail (5th pct):  {bias_cold:+.1f} °C  ← drives frost days")
    print(f"    Warm tail (95th pct): {bias_warm:+.1f} °C  ← drives tropical nights")

    # ── Estimate corrected FD0 ────────────────────────────────────
    # ERA5-Land FD0 baseline is ~128 days/yr (known).
    # Raw CORDEX FD0 vs ERA5 can be inferred from cold-tail bias.
    # After QDM: corrected FD0 ≈ ERA5-Land FD0 baseline.
    # Future projected FD0 = ERA5 baseline + model climate change signal.
    print()

    # ── Verdict ───────────────────────────────────────────────────
    print("=" * 60)
    print("Interpretation")
    print("=" * 60)

    if bias_mean < -1.5:
        verdict = "CORDEX IS COLD-BIASED — QDM warm correction is expected and correct."
        detail = (
            f"REMO2009 TN is {abs(bias_mean):.1f}°C colder than ERA5-Land on average.\n"
            f"Cold-tail bias: {bias_cold:+.1f}°C — this is why QDM adds warmth at cold quantiles.\n"
            f"After correction the historical period should match ERA5-Land.\n"
            f"The future projections (fewer frost days, more warm nights) reflect\n"
            f"the genuine REMO2009 climate-change signal PLUS the warm correction.\n\n"
            f"ACTION: Results are not erroneous. Add this single-model caveat:\n"
            f'  "REMO2009 has a {abs(bias_mean):.0f}°C cold bias in TN; QDM corrects this.\n'
            f'   Projected frost-day reduction and tropical-night increase reflect\n'
            f'   both bias correction and genuine warming — treat as upper bound."'
        )
    elif -1.5 <= bias_mean <= 1.5:
        verdict = "CORDEX TN IS CLOSE TO ERA5-LAND — warm projections are genuine model signal."
        detail = (
            f"Raw CORDEX TN is within {abs(bias_mean):.1f}°C of ERA5-Land.\n"
            f"QDM makes only a small adjustment. The projected warm signal\n"
            f"(fewer frost days, more tropical nights) is genuine REMO2009 physics.\n\n"
            f"ACTION: Results are valid. Add caveat that REMO2009 sits at the\n"
            f"warm end of EURO-CORDEX ensemble for Baltic TN under RCP4.5."
        )
    else:
        verdict = "CORDEX TN IS WARM-BIASED — QDM would cool it, not warm it."
        detail = (
            f"Raw CORDEX TN is {bias_mean:+.1f}°C warmer than ERA5-Land.\n"
            f"QDM should be cooling the projection, yet we see more warm nights.\n"
            f"This may indicate a QDM error. Re-examine the correction pipeline."
        )

    print(f"\nVERDICT: {verdict}")
    print(f"\n{detail}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
