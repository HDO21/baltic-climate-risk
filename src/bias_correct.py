#!/usr/bin/env python3
"""
bias_correct.py — Quantile Delta Mapping bias correction for CORDEX projections.

Corrects systematic biases in CORDEX RCM output by comparing it against
ERA5-Land reanalysis over the 1991–2020 calibration period and applying
the derived quantile-wise transfer functions to the full projection period.

Processing chain:
  1. Load ERA5-Land reference data for 1991–2020, compute daily statistics.
  2. Regrid ERA5-Land (regular 0.1° lat/lon) → CORDEX rotated-pole grid
     using nearest-neighbour index mapping.
  3. Join CORDEX historical (1991–2005) + CORDEX RCP (2006–2020) to form
     the 30-year model calibration series.
  4. Train QuantileDeltaMapping (xclim) on ERA5-Land vs CORDEX calibration.
  5. Apply the trained correction to each CORDEX future file (2021–2100)
     and write bias-corrected NetCDF to data/raw/cordex_bc/{country}/{scenario}/.

Variable mapping:
  CORDEX tasmax (K)           → daily TX in °C   (QDM kind="+", monthly groups)
  CORDEX tasmin (K)           → daily TN in °C   (QDM kind="+", monthly groups)
  CORDEX tas    (K)           → daily Tmean in °C (QDM kind="+", monthly groups)
  CORDEX pr     (kg/m²/s)     → daily pr in mm/d (QDM kind="*", monthly groups)

Known limitation: the forcing discontinuity at 2005/2006 (historical→RCP) in the
calibration series introduces a minor mixing of constrained and free-running model
climate. This is standard practice in EURO-CORDEX bias-correction studies.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from xsdba import QuantileDeltaMapping  # xclim >=0.60 split sdba into xsdba

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from load_data import RAW_DIR, load_config  # noqa: E402

logger = logging.getLogger(__name__)

# xarray emits FutureWarnings about upcoming kwarg defaults that don't affect
# correctness for our usage (nested concat along a single time dimension).
warnings.filterwarnings(
    "ignore",
    message=".*compat.*no_conflicts.*override.*",
    category=FutureWarning,
    module="xarray",
)

ERA5_DIR    = RAW_DIR              # RAW_DIR = data/raw/era5land already
CORDEX_DIR  = ROOT / "data" / "raw" / "cordex"
CORDEX_BC_DIR = ROOT / "data" / "raw" / "cordex_bc"

# Units passed to xsdba for QDM training and adjustment.
# Temperature bias correction is performed in Kelvin to avoid pint's offset-unit
# handling for Celsius (degC), which causes xsdba to misplace quantile lookups
# and produces values ~273°C too low. The K→°C conversion is applied AFTER
# bias correction in cordex_pipeline.py, not here.
_PIPELINE_UNITS = {
    "tasmax": "K",
    "tasmin": "K",
    "tas":    "K",
    "pr":     "mm d-1",
}

# Maps CORDEX variable name → ERA5-Land source file tag and daily-stat function.
# stat: "max" | "min" | "mean" | "pr"
_VAR_ERA5 = {
    "tasmax": ("t2m", "max"),
    "tasmin": ("t2m", "min"),
    "tas":    ("t2m", "mean"),
    "pr":     ("tp",  "pr"),
}

# CORDEX units → bias-correction-ready conversion.
# Temperature variables are kept in K (no conversion) so xsdba works with
# non-offset units. Precipitation is converted to mm/day.
_CORDEX_CONVERT = {
    "tasmax": ("K",       lambda da: da),                            # keep K
    "tasmin": ("K",       lambda da: da),
    "tas":    ("K",       lambda da: da),
    "pr":     ("kg/m²/s", lambda da: (da * 86400).clip(min=0)),     # → mm/day
}

# QDM correction kind per variable.
_QDM_KIND = {"tasmax": "+", "tasmin": "+", "tas": "+", "pr": "*"}


# ── ERA5-Land daily derivation ────────────────────────────────────────────────

def _era5_daily_year(
    year: int, era5_dir: Path, cordex_var: str
) -> xr.DataArray:
    """
    Load one year of ERA5-Land monthly files, compute the daily statistic
    corresponding to `cordex_var`, and return a (time, latitude, longitude)
    DataArray in pipeline units (°C for temperature, mm/day for precipitation).
    """
    era5_tag, stat = _VAR_ERA5[cordex_var]
    monthly = []

    for month in range(1, 13):
        nc = era5_dir / f"era5land_{era5_tag}_{year}_{month:02d}.nc"
        ds = xr.open_dataset(nc)
        td = "valid_time" if "valid_time" in ds.coords else "time"

        if stat == "max":
            da = ds["t2m"].resample({td: "1D"}).max()    # stays in K
        elif stat == "min":
            da = ds["t2m"].resample({td: "1D"}).min()
        elif stat == "mean":
            da = ds["t2m"].resample({td: "1D"}).mean()
        else:  # pr — cumulative daily total
            da = (ds["tp"].resample({td: "1D"}).last() * 1000.0).clip(min=0)

        # Normalise time dim name so concat works across all months.
        if td != "time":
            da = da.rename({td: "time"})
        da = da.load()   # force computation before ds is closed
        ds.close()
        monthly.append(da)

    return xr.concat(monthly, dim="time")


def load_era5_daily(
    era5_dir: Path,
    cordex_var: str,
    start_year: int,
    end_year: int,
) -> xr.DataArray:
    """
    Load ERA5-Land daily statistics for `cordex_var` over [start_year, end_year].
    Returns a (time, latitude, longitude) DataArray in pipeline units.
    """
    logger.info(
        "Loading ERA5-Land %s daily %d–%d", cordex_var, start_year, end_year
    )
    years = []
    for yr in range(start_year, end_year + 1):
        years.append(_era5_daily_year(yr, era5_dir, cordex_var))
    da = xr.concat(years, dim="time")
    # Convert ERA5 numpy datetime64 → cftime proleptic_gregorian to match CORDEX.
    da = da.convert_calendar("proleptic_gregorian", use_cftime=True)
    return da


# ── CORDEX file loading ───────────────────────────────────────────────────────

def load_cordex_period(
    cordex_dir: Path,
    cordex_var: str,
    start_year: int,
    end_year: int,
) -> xr.DataArray:
    """
    Open all clipped CORDEX files for `cordex_var` that overlap [start_year, end_year],
    concatenate along time, slice to the exact requested period, and convert units.

    Returns a (time, rlat, rlon) DataArray in pipeline units.
    """
    files = sorted(cordex_dir.glob(f"{cordex_var}_EUR-11_*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No CORDEX files for variable '{cordex_var}' in {cordex_dir}"
        )

    _time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_mfdataset(
        files, combine="nested", concat_dim="time",
        decode_times=_time_coder, data_vars="minimal", coords="minimal",
    )
    da = ds[cordex_var].sel(time=slice(str(start_year), str(end_year)))
    _, convert_fn = _CORDEX_CONVERT[cordex_var]
    da = convert_fn(da).load()   # load into memory — xsdba QDM rejects dask chunks
    da.attrs["units"] = _PIPELINE_UNITS[cordex_var]   # required by xsdba
    ds.close()
    return da


# ── Nearest-neighbour regridding ──────────────────────────────────────────────

def build_regrid_indices(
    era5_lat: np.ndarray,
    era5_lon: np.ndarray,
    cordex_lat_2d: np.ndarray,
    cordex_lon_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute nearest-neighbour index arrays mapping every CORDEX rotated-pole
    grid cell to the closest ERA5-Land regular-grid cell.

    Parameters
    ----------
    era5_lat      : 1-D array of ERA5-Land latitude values, shape (nlat,)
    era5_lon      : 1-D array of ERA5-Land longitude values, shape (nlon,)
    cordex_lat_2d : 2-D array of CORDEX geographic latitudes, shape (rlat, rlon)
    cordex_lon_2d : 2-D array of CORDEX geographic longitudes, shape (rlat, rlon)

    Returns
    -------
    lat_idx, lon_idx : integer arrays of shape (rlat, rlon) containing the
                       ERA5-Land row and column index for each CORDEX cell.
    """
    # Joint 2-D distance minimisation. Independent 1-D argmin on latitude and
    # longitude separately is incorrect on a rotated-pole grid at high latitudes
    # where 1° longitude ≈ 55 km but 1° latitude ≈ 111 km — the two axes are
    # not interchangeable in distance terms.
    n_lat, n_lon = len(era5_lat), len(era5_lon)
    era5_lat_2d, era5_lon_2d = np.meshgrid(era5_lat, era5_lon, indexing="ij")
    # shape: (n_lat, n_lon, rlat, rlon)
    dist_sq = (
        (era5_lat_2d[:, :, None, None] - cordex_lat_2d[None, None, :, :]) ** 2 +
        (era5_lon_2d[:, :, None, None] - cordex_lon_2d[None, None, :, :]) ** 2
    )
    flat_idx = dist_sq.reshape(n_lat * n_lon, *cordex_lat_2d.shape).argmin(axis=0)
    lat_idx = flat_idx // n_lon
    lon_idx = flat_idx %  n_lon
    return lat_idx, lon_idx


def regrid_era5_to_cordex(
    da_era5: xr.DataArray,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    cordex_template: xr.DataArray,
) -> xr.DataArray:
    """
    Regrid an ERA5-Land DataArray (time, latitude, longitude) onto the CORDEX
    rotated-pole grid (time, rlat, rlon) using pre-computed nearest-neighbour
    index arrays.

    The output shares CORDEX spatial coordinates (rlat, rlon, lat, lon) so it
    can be passed directly to xclim's QDM alongside CORDEX model data.
    """
    # era5 data: (time, nlat, nlon); fancy-index → (time, rlat, rlon)
    data = da_era5.values[:, lat_idx, lon_idx]

    dim0, dim1 = cordex_template.dims[-2], cordex_template.dims[-1]

    coords = {
        "time": da_era5["time"],
        dim0:   cordex_template[dim0],
        dim1:   cordex_template[dim1],
    }
    # Carry 2-D lat/lon aux coords from CORDEX template for spatial reference.
    if "lat" in cordex_template.coords:
        coords["lat"] = cordex_template["lat"]
    if "lon" in cordex_template.coords:
        coords["lon"] = cordex_template["lon"]

    return xr.DataArray(data, dims=["time", dim0, dim1], coords=coords)


# ── Spatial masking for national mean ─────────────────────────────────────────

def apply_bbox_mask(da: xr.DataArray, north: float, west: float,
                    south: float, east: float) -> xr.DataArray:
    """
    Apply a geographic bounding-box mask to a CORDEX DataArray with 2-D lat/lon
    auxiliary coordinates. Cells outside the bbox are set to NaN.

    This prevents the spatial mean from being inflated by non-target-country cells
    that sit inside the rectangular rlat/rlon clip but outside the actual bbox.
    """
    if "lat" not in da.coords or "lon" not in da.coords:
        return da
    mask = (
        (da["lat"] >= south) & (da["lat"] <= north) &
        (da["lon"] >= west)  & (da["lon"] <= east)
    )
    return da.where(mask)


# ── QDM training and application ─────────────────────────────────────────────

def train_qdm(
    ref: xr.DataArray,
    hist: xr.DataArray,
    kind: str,
    nquantiles: int = 50,
    group: str = "time.month",
    adapt_freq_thresh: float | None = None,
) -> QuantileDeltaMapping:
    """
    Train a Quantile Delta Mapping corrector.

    Parameters
    ----------
    ref               : ERA5-Land reference DataArray (time, rlat, rlon), pipeline units.
    hist              : CORDEX historical DataArray (same grid, calibration period).
    kind              : "+" additive (temperature), "*" multiplicative (precipitation).
    nquantiles        : quantile bins (50 adequate for PoC, 100 for production).
    group             : xclim grouping — "time.month" applies monthly correction.
    adapt_freq_thresh : for precipitation ("*" kind), threshold below which a day
                        is considered dry (mm/day). Adjusts the dry-day frequency
                        independently from intensity, preventing near-zero CORDEX
                        drizzle values from inflating multiplicative correction factors.
    """
    logger.info(
        "Training QDM (kind=%s, nquantiles=%d, group=%s, adapt_freq=%s)",
        kind, nquantiles, group, adapt_freq_thresh,
    )
    kwargs = dict(ref=ref, hist=hist, nquantiles=nquantiles, kind=kind, group=group)
    if adapt_freq_thresh is not None:
        kwargs["adapt_freq_thresh"] = adapt_freq_thresh
    return QuantileDeltaMapping.train(**kwargs)


def apply_and_save(
    qdm: QuantileDeltaMapping,
    cordex_var: str,
    sim_path: Path,
    out_path: Path,
) -> None:
    """
    Apply a trained QDM corrector to one CORDEX simulation file and write the
    bias-corrected result to `out_path`.

    Skips the file if `out_path` already exists (idempotent).
    """
    if out_path.exists():
        logger.info("skip  %s (already corrected)", sim_path.name)
        return

    logger.info("apply %s", sim_path.name)
    _time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(sim_path, decode_times=_time_coder)
    _, convert_fn = _CORDEX_CONVERT[cordex_var]
    sim = convert_fn(ds[cordex_var]).load()   # must be in memory for xsdba
    sim.attrs["units"] = _PIPELINE_UNITS[cordex_var]  # must match training units

    corrected = qdm.adjust(sim=sim, interp="linear")

    # Precipitation: multiplicative QDM can produce extreme values where the
    # historical CORDEX distribution has near-zero drizzle but ERA5-Land does not.
    # Clip to a physical ceiling (500 mm/day >> any plausible Baltic extreme event).
    if _QDM_KIND[cordex_var] == "*":
        corrected = corrected.clip(min=0, max=150)  # Estonia daily record ~130 mm

    # Write back: preserve original file structure, replace variable values.
    ds_out = ds.copy(deep=False)
    ds_out[cordex_var] = corrected
    ds_out[cordex_var].attrs.update({
        "units":           _PIPELINE_UNITS[cordex_var],   # overwrite original CORDEX units
        "bias_correction": "Quantile Delta Mapping (xclim)",
        "reference":       "ERA5-Land 1991–2020",
        "bc_units":        "K" if _QDM_KIND[cordex_var] == "+" else "mm d-1",
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.load()          # materialise before closing the source dataset
    ds.close()
    ds_out.to_netcdf(out_path)
    logger.info("      → %s", out_path.name)


# ── Top-level orchestration ───────────────────────────────────────────────────

def run_bias_correction(
    country: str,
    scenario: str,
    era5_dir: Path | None = None,
    cordex_dir: Path | None = None,
    bc_dir: Path | None = None,
    nquantiles: int = 50,
) -> None:
    """
    Run the full bias correction pipeline for all variables for one country
    and one CORDEX scenario.

    Steps for each variable:
      1. Load ERA5-Land daily reference for the calibration period.
      2. Load CORDEX calibration data (historical + early RCP joined).
      3. Regrid ERA5-Land to the CORDEX grid.
      4. Train QDM on matched calibration period.
      5. Apply to each CORDEX projection file and save.
    """
    cfg      = load_config()
    ccfg     = cfg["cordex"]
    cal      = ccfg["calibration"]
    scenarios_cfg = ccfg["scenarios"]

    if scenario not in scenarios_cfg:
        raise ValueError(
            f"Scenario '{scenario}' not in config. Available: {list(scenarios_cfg)}"
        )

    era5_dir   = era5_dir   or (ERA5_DIR   / country.lower())
    cordex_dir = cordex_dir or (CORDEX_DIR / country.lower())
    bc_dir     = bc_dir     or (CORDEX_BC_DIR / country.lower() / scenario)

    cal_start  = cal["start"]
    cal_end    = cal["end"]

    logger.info("=" * 60)
    logger.info("Bias correction: %s | %s | %d–%d calibration",
                country, scenario, cal_start, cal_end)
    logger.info("CORDEX source : %s", cordex_dir)
    logger.info("Output        : %s", bc_dir)
    logger.info("=" * 60)

    # Projection files are those belonging to the target scenario
    # and falling after the calibration window.
    scen_cfg = scenarios_cfg[scenario]
    proj_start = scen_cfg["projection_start"]

    for cordex_var in ccfg["variables"].values():
        logger.info("── Variable: %s ──", cordex_var)

        # ── Step 1: ERA5-Land reference ────────────────────────────────────
        da_era5 = load_era5_daily(era5_dir, cordex_var, cal_start, cal_end)

        # ── Step 2: CORDEX calibration series (historical + early RCP) ─────
        # Load the full CORDEX calibration window from the clipped files.
        # Both historical and scenario files live in the same cordex_dir.
        da_cordex_cal = load_cordex_period(
            cordex_dir, cordex_var, cal_start, cal_end
        )

        # ── Step 3: Regrid ERA5-Land → CORDEX rotated-pole grid ───────────
        # Build regrid weights once per variable using the first CORDEX time slice.
        cordex_template = da_cordex_cal.isel(time=0)
        era5_lat = da_era5["latitude"].values
        era5_lon = da_era5["longitude"].values
        cordex_lat = cordex_template["lat"].values
        cordex_lon = cordex_template["lon"].values

        lat_idx, lon_idx = build_regrid_indices(
            era5_lat, era5_lon, cordex_lat, cordex_lon
        )
        da_ref = regrid_era5_to_cordex(da_era5, lat_idx, lon_idx, cordex_template)
        da_ref.attrs["units"] = _PIPELINE_UNITS[cordex_var]   # required by xsdba

        # Align calibration time axes.
        # ERA5 timestamps are at midnight after convert_calendar; CORDEX are at
        # noon. Normalise both to date-only (00:00) before aligning so the inner
        # join matches on calendar date rather than exact timestamp.
        import cftime as _cft
        def _norm_time(da: xr.DataArray) -> xr.DataArray:
            new_t = [_cft.DatetimeProlepticGregorian(t.year, t.month, t.day)
                     for t in da.time.values]
            return da.assign_coords(time=new_t)

        da_ref        = _norm_time(da_ref)
        da_cordex_cal = _norm_time(da_cordex_cal)
        da_ref, da_cordex_cal = xr.align(
            da_ref, da_cordex_cal, join="inner",
            exclude=[d for d in da_ref.dims if d != "time"],
        )

        logger.info(
            "  Calibration: ERA5 %d days | CORDEX %d days",
            da_ref.sizes["time"], da_cordex_cal.sizes["time"],
        )

        # ── Step 4: Train QDM ──────────────────────────────────────────────
        # For precipitation use adapt_freq_thresh to handle the CORDEX drizzle
        # bias: CORDEX models simulate too many near-zero rain days which inflate
        # the multiplicative correction ratio to physically impossible values.
        # adapt_freq_thresh must be a pint-compatible unit string, not a bare float.
        _thresh_mm = ccfg.get("pr_wet_day_threshold_mm", 1.0)
        _adapt = (
            f"{_thresh_mm} mm d-1"
            if _QDM_KIND[cordex_var] == "*" else None
        )
        qdm = train_qdm(
            ref=da_ref,
            hist=da_cordex_cal,
            kind=_QDM_KIND[cordex_var],
            nquantiles=nquantiles,
            group="time.month",
            adapt_freq_thresh=_adapt,
        )

        # ── Step 5: Apply to projection files ─────────────────────────────
        proj_files = sorted(
            cordex_dir.glob(f"{cordex_var}_EUR-11_*_{scenario}*.nc")
        )
        # Keep only files that start at or after the projection start year.
        proj_files = [
            f for f in proj_files
            if int(f.stem.split("_day_")[-1][:4]) >= proj_start
        ]

        if not proj_files:
            logger.warning(
                "  No projection files found for %s %s after %d",
                cordex_var, scenario, proj_start,
            )
            continue

        logger.info("  Applying to %d projection files", len(proj_files))
        for src in proj_files:
            apply_and_save(qdm, cordex_var, src, bc_dir / src.name)

        logger.info("  Done: %s", cordex_var)

    logger.info("=" * 60)
    logger.info("Bias correction complete → %s", bc_dir)
