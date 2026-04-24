# Skill: run-pipeline
# Invoke with: 'Run the pipeline using the run-pipeline skill'
# Or for a specific stage: 'Run the download stage using the run-pipeline skill'

## Purpose
Safely execute one or all pipeline stages, with validation at each step.
Prevent accidental re-downloads of large files.
Catch and explain errors before they propagate to downstream stages.

## Stages (in order)
1. download   — fetch raw NetCDF from CDS API to data/raw/
2. process    — compute annual metrics to data/processed/
3. validate   — run QC checks on the processed output
4. dashboard  — launch the Streamlit app

## Procedure — follow this exactly

### For each stage to be run:

STEP A — Idempotency check
  Check whether the expected output file(s) already exist.
  If yes: tell the user what was found and ask: 'Output exists.
  Re-run anyway? (yes/no)'
  If no: proceed to Step B.

STEP B — Pre-run check
  For 'download' stage: verify that config/config.yaml is readable
  and that the CDS API key environment variable is set.
  For 'process' stage: verify that data/raw/ contains .nc files.
  For 'validate' stage: verify that data/processed/ contains .parquet files.

STEP C — Run the stage
  Execute the appropriate command:
    download : python -m climate_risk.download
    process  : python -m climate_risk.process
    validate : python -m climate_risk.quality --check-output
    dashboard: streamlit run app/dashboard.py

STEP D — Evaluate output
  Check the return code. If non-zero:
    - Print the last 30 lines of output
    - Suggest the most likely cause based on the error message
    - STOP and wait for user instructions — do not proceed to the next stage
  If success:
    - Report: files written, row counts, any warnings in logs
    - Proceed to the next stage (if running full pipeline)

### Final report
After all stages complete, provide a one-paragraph summary:
  - Which stages ran vs skipped
  - Any warnings or QC flags raised
  - Recommended next action (e.g., 'launch dashboard to inspect results')
