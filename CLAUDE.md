# Baltic States Climate Risk Analysis — PoC

## Overview
PoC pipeline for analysing climate risks in Estonia, Latvia and Lithuania. Downloads ERA5-Land (current climate 1991-2020) and EURO-CORDEX (projections to 2100) via the Copernicus CDS API. Computes ETCCDI climate risk metrics and serves them via a Streamlit dashboard.
Status: PoC, ~10 users, all open-source.

Data sources:
- ERA5-Land for the historical/reference baseline (1991–2020)
- EURO-CORDEX for future climate projections (2031–2100)

Goals:
- download and cache climate data
- compute climate risk metrics
- expose results through a lightweight user-facing dashboard

This project should stay beginner-friendly. Code should be heavily commented so that users with different skill levels can follow the logic.

## Quick-start commands

```bash
# Run the pipeline for one metric (all 30 years, downloads if needed)
python src/run_pipeline.py --metric heat_days --country EE

# Run pipeline without triggering CDS downloads (transform from cached files only)
python src/run_pipeline.py --metric heat_days --country EE --no-download

# Download a specific ERA5-Land variable for all years
python src/load_data.py --variable total_precipitation --country EE

# Run the Streamlit dashboard locally
streamlit run app/streamlit_app.py

# Reprocess all metrics from scratch (delete outputs first)
rm data/processed/estonia_*.csv data/processed/*_grid_ee.parquet
for metric in heat_days frost_days id0 tr15 txx tnn; do
    python src/run_pipeline.py --metric $metric --country EE --no-download
done
for metric in cdd r20mm sdii prcptot; do
    python src/run_pipeline.py --metric $metric --country EE
done
```

## Architecture
Three-layer pipeline — keep layers strictly separate:

1. **Data acquisition** (`src/load_data.py`)
   - download raw climate data from CDS API
   - save raw files to `data/raw/`
   - restrict requests to the country bounding box

2. **Processing and metric computation** (`src/transform.py`, `src/validate.py`, `src/run_pipeline.py`)
   - validate raw data
   - compute ETCCDI climate risk metrics per grid point, then spatial mean
   - save processed outputs to `data/processed/`

3. **Interface** (`app/streamlit_app.py`)
   - read processed outputs only
   - no heavy recomputation from the UI layer

## Repository Map
- `config/config.yaml` — single source of truth for paths, countries, metrics, thresholds
- `src/load_data.py` — CDS API download with retry and caching
- `src/transform.py` — metric computation (temperature and precipitation metrics)
- `src/validate.py` — raw file and result quality checks
- `src/run_pipeline.py` — orchestrator: LOAD → VALIDATE → TRANSFORM → VALIDATE
- `app/streamlit_app.py` — Streamlit dashboard
- `agents/` — custom Claude agents (code-reviewer, climate-consultant)
- `tests/` — pytest tests (to be populated)
- `data/raw/` — downloaded NetCDF cache (gitignored)
- `data/processed/` — computed CSVs and Parquets (committed for Streamlit Cloud)
- `CLAUDE.md` — project instructions for Claude Code

## Key code structures

### Metric registry (`src/transform.py`)
```python
METRIC_COL    = { "heat_days": "extreme_heat_days", "frost_days": "frost_days", ... }
PRECIP_METRICS = frozenset({"cdd", "r20mm", "sdii", "prcptot"})  # use tp files
```
Adding a new metric requires: entry in `METRIC_COL`, dispatch branch in `compute_annual_grid` or `compute_annual_precip_grid`, entry in `_RESULT_BOUNDS` (validate.py), entry in `_METRIC_CSV` and `_METRIC_CFG` (run_pipeline.py), and entry in `METRICS` dict (streamlit_app.py).

### Dashboard metric registration (`app/streamlit_app.py`)
Each metric entry in `METRICS` requires these keys:
`col`, `csv`, `parquet`, `threshold_label`, `bar_color`, `y_label`, `pipeline_flag`, `csv_header`
Bar colours must be from the Okabe-Ito colourblind-safe palette.

## Geographic Scope
Countries: Estonia (EE), Latvia (LV), Lithuania (LT)

Spatial reference for analysis: ETRS89 / LAEA Europe (EPSG:3035)

**Bounding boxes — two conventions in use:**
- `config/config.yaml` stores country areas as `[N, W, S, E]` (CDS API order)
- CLAUDE.md initial values below are in `(W, S, E, N)` order — do not mix these up

Initial country bounding box values (W, S, E, N):
- EE: (23.3397953631, 57.4745283067, 28.1316992531, 59.6110903998)
- LT: (21.0558004086, 53.9057022162, 26.5882792498, 56.3725283881)
- LV: (21.0558004086, 55.61510692,   28.1767094256, 57.9701569688)

API requests must be limited to the country bounding box — do not request larger areas.

## Data Sources

### ERA5-Land (`reanalysis-era5-land`)
- Resolution: 0.1° × 0.1° (~9 km), 1950–present
- Temperature (`t2m`): instantaneous hourly values in **Kelvin** — subtract 273.15 for Celsius
  - Daily TX = `.resample("1D").max()`; daily TN = `.resample("1D").min()`
- Precipitation (`tp`): **per-hour accumulation in metres** (each timestep = that hour only, NOT cumulative from 00:00)
  - Daily total (mm) = sum of all 24 hourly values × 1000
- Sea/water cells are always NaN in `t2m` (ERA5-Land land mask) — expected, not a data error
- Do NOT bias-correct ERA5-Land — it already incorporates observational constraints
- Raw file naming: `era5land_t2m_{year}_{month:02d}.nc`, `era5land_tp_{year}_{month:02d}.nc`

### EURO-CORDEX (`projections-cordex-domains-single-levels`)
- Resolution: EUR-11 (~12.5 km), CMIP5 generation, RCP4.5 and RCP8.5
- Recommended chains: MPI-M-MPI-ESM-LR/MPI-CSC-REMO2009 (primary), MPI-M-MPI-ESM-LR/SMHI-RCA4 (secondary)
- CRITICAL: Always request `"temporal_resolution": "daily"` — monthly data cannot produce ETCCDI threshold counts
- CORDEX is NOT bias-corrected — use `xclim.sdba.QuantileDeltaMapping` calibrated on 1991–2020 CORDEX historical vs ERA5-Land before computing absolute-threshold metrics
- Do NOT mix CMIP5 (RCP) and CMIP6 (SSP) ensemble members in the same analysis

## Climate Risk Metrics

### Implemented (t2m source — fully available 1991–2020)
| Pipeline flag  | ETCCDI | Definition              | Column name         |
|----------------|--------|-------------------------|---------------------|
| `heat_days`    | TX30   | TX ≥ 30 °C days/year    | extreme_heat_days   |
| `frost_days`   | FD0    | TN < 0 °C days/year     | frost_days          |
| `id0`          | ID0    | TX < 0 °C days/year     | id0                 |
| `tr15`         | TR15   | TN > 15 °C days/year    | tr15                |
| `txx`          | TXx    | Annual max daily TX (°C)| txx                 |
| `tnn`          | TNn    | Annual min daily TN (°C)| tnn                 |

### Implemented (tp source — available 1991–2010, downloading)
| Pipeline flag  | ETCCDI  | Definition                        | Column name |
|----------------|---------|-----------------------------------|-------------|
| `cdd`          | CDD     | Max consecutive dry days          | cdd         |
| `r20mm`        | R20mm   | Days with pr > 20 mm              | r20mm       |
| `sdii`         | SDII    | Mean pr on wet days (mm/day)      | sdii        |
| `prcptot`      | PRCPTOT | Annual total pr on wet days (mm)  | prcptot     |

### Planned
- `hard_frost` — days with TN < −10 °C (t2m source, can run now)
- `drought_severity` — see climate-consultant for definition

All metric thresholds are defined in `config/config.yaml` → `metrics:`. Never hardcode them.

## Coding Conventions
- Language: Python 3.11
- Comments: English
- Type hints: required on all public functions
- Line length: 100 characters (ruff)
- Linter: ruff — run before every commit
- File paths: always `pathlib.Path`
- Config access: `load_config()` from `src/load_data.py`
- Logging: `logging` module only — never `print()` in `src/`
- Exceptions: never bare `except` — always catch specific types
- Imports: stdlib → third-party → local (ruff enforces)

## Data Commit Rules
- **Do NOT commit** `data/raw/` (large NetCDF files, reproduced by the pipeline)
- **DO commit** `data/processed/*.csv` and `data/processed/*.parquet` (pre-computed outputs served directly by Streamlit Cloud)
- Never commit `*.nc`, `*.grib`, secrets, or credentials

## Idempotency and Caching
- Every pipeline stage checks whether output already exists before running
- `--no-download` flag skips the CDS download stage entirely (transform from cached files)
- Raw files are never overwritten once downloaded
- Log whether each step was executed or skipped

## Data Quality Control
Checks are in `src/validate.py`. At minimum:
- missing values / temporal gaps in land cells
- physically implausible raw Kelvin values
- derived TX/TN within Baltic plausibility bounds
- annual scalar result within metric-specific bounds (see `_RESULT_BOUNDS`)

QC failures are logged; blocking failures abort that year's processing.

## Process for Adding a Metric
1. Add threshold to `config/config.yaml` → `metrics:`
2. Add entry to `METRIC_COL` in `src/transform.py`
3. Add dispatch branch in `compute_annual_grid` (t2m) or `compute_annual_precip_grid` (tp)
4. Add bounds to `_RESULT_BOUNDS` in `src/validate.py`
5. Add entries to `_METRIC_CSV` and `_METRIC_CFG` in `src/run_pipeline.py`
6. Add entry to `METRICS` dict in `app/streamlit_app.py` (with all 8 required keys; use Okabe-Ito colour)

## Testing
Write tests for metric calculations, config loading, QC checks, and idempotent behaviour.
Use pytest with synthetic data — never real downloaded files.
*Tests are deferred for the current PoC phase.*

## Do NOT
- Hardcode credentials, API keys, thresholds, paths, periods, or BBOX values
- Silently overwrite cached downloads
- Bypass QC checks
- Mix UI logic with heavy processing logic
- Apply bias correction to ERA5-Land (it is already a reanalysis)
- Use `print()` in `src/` — use logging
- Install packages with `pip install` without updating `requirements.txt` and `environment_local.yml`
- Mix CORDEX CMIP5 (RCP) and CMIP6 (SSP) ensemble members in the same analysis
- Request `"temporal_resolution": "monthly"` for ETCCDI metrics — always use `"daily"`
- Run Nominatim geocoding without a 1-second delay between requests

## Agents
**code-reviewer**
  Purpose : Review code for correctness, xarray pitfalls, dashboard consistency, and architecture fit
  Trigger : After finishing a feature, before opening a pull request
  Invoke  : 'Review [file] using the code-reviewer agent'

**climate-consultant**
  Purpose : Validate metric definitions, data units, CORDEX configuration, bias correction approach
  Trigger : Before implementing a new metric; when output values look unexpected; before any CORDEX work
  Invoke  : 'I need advice on [metric/topic] — use the climate-consultant agent'
