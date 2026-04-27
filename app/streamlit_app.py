"""
Streamlit dashboard — Baltic Climate Risk: Extreme Heat Days.

Displays annual extreme heat day counts for the 1991-2020 WMO reference
period, derived from ERA5-Land data. Run with:

    conda activate climate-risk
    streamlit run app/streamlit_app.py
"""

import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
CSV         = ROOT / "data/processed/estonia_extreme_heat_days.csv"
CONFIG_PATH = ROOT / "config/config.yaml"

FULL_PERIOD = list(range(1991, 2021))

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def _load_grid_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _parse_coord(text: str):
    """Accept both '.' and ',' as decimal separator; return float or None."""
    try:
        return float(text.strip().replace(",", "."))
    except (ValueError, AttributeError):
        return None

# ── Config ────────────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)
THRESHOLD_C = float(_cfg["metrics"]["heat_days"]["threshold_tx_degC"])

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Baltic Climate Risk — Heat Days",
    page_icon="🌡",
    layout="centered",
)

st.title("Extreme Heat Days — Estonia")
st.caption(
    f"ERA5-Land reanalysis  ·  1991–2020 WMO reference period  ·  TX ≥ {THRESHOLD_C:.0f} °C"
)

# ── Load data ─────────────────────────────────────────────────────────────────
if not CSV.exists():
    st.error(f"No processed data found at `{CSV.relative_to(ROOT)}`.\n\n"
             "Run the pipeline first:\n```\npython src/run_pipeline.py --year YYYY\n```")
    st.stop()

processed = pd.read_csv(CSV)

# Build a complete 1991–2020 frame; years not yet processed appear as None.
df = (
    pd.DataFrame({"year": FULL_PERIOD})
    .merge(processed, on="year", how="left")
)

available = df["extreme_heat_days"].notna().sum()
missing   = len(FULL_PERIOD) - available

# ── Status banner ─────────────────────────────────────────────────────────────
if missing > 0:
    st.info(
        f"Pipeline in progress: **{available} of 30 years** processed. "
        f"Run `python src/run_pipeline.py` to compute the remaining {missing} year(s)."
    )

# ── Summary metric ────────────────────────────────────────────────────────────
period_mean = df["extreme_heat_days"].mean()   # NaN years excluded automatically

col1, col2 = st.columns(2)
col1.metric(
    label=f"Mean extreme heat days per year ({df['year'].min()}–{df['year'].max()})",
    value=f"{period_mean:.2f}" if pd.notna(period_mean) else "—",
    help=f"Average number of days per year where the daily maximum 2 m temperature "
         f"(TX) reached or exceeded {THRESHOLD_C} °C, computed as the spatial mean "
         f"across all ERA5-Land grid points covering Estonia.",
)
col2.metric(
    label="Years processed",
    value=f"{available} / 30",
)

# ── Bar chart ─────────────────────────────────────────────────────────────────
st.subheader("Annual extreme heat days")

# Separate processed and pending years for visual distinction.
df_done    = df[df["extreme_heat_days"].notna()]
df_pending = df[df["extreme_heat_days"].isna()]

fig = go.Figure()

fig.add_trace(go.Bar(
    x=df_done["year"],
    y=df_done["extreme_heat_days"],
    name="Processed",
    marker_color="#d62728",
    hovertemplate="%{x}: <b>%{y:.2f}</b> days<extra></extra>",
))

if not df_pending.empty:
    fig.add_trace(go.Bar(
        x=df_pending["year"],
        y=[0] * len(df_pending),
        name="Pending",
        marker_color="#cccccc",
        hovertemplate="%{x}: not yet processed<extra></extra>",
    ))

# Period-mean reference line (only when at least one year is processed).
if pd.notna(period_mean):
    fig.add_hline(
        y=period_mean,
        line_dash="dash",
        line_color="#555555",
        annotation_text=f"Period mean: {period_mean:.2f} days",
        annotation_position="top left",
    )

fig.update_layout(
    xaxis=dict(title="Year", tickmode="linear", tick0=1991, dtick=2),
    yaxis=dict(title="Extreme heat days", rangemode="tozero"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40, b=40),
    height=420,
    barmode="overlay",
)

st.plotly_chart(fig, use_container_width=True)

# ── Coordinate input ──────────────────────────────────────────────────────────
# Architecture note: country_code and metric_key are plain variables here so
# they can be replaced with st.selectbox widgets in a future multi-country build.
country_code = "EE"
metric_key   = "heat_days"  # noqa: F841 — placeholder for metric selector

_area             = _cfg["countries"][country_code]["area"]   # [N, W, S, E]
_N, _W, _S, _E   = _area
_grid_parquet     = ROOT / f"data/processed/heat_days_grid_{country_code.lower()}.parquet"

st.divider()
st.subheader("Location-specific analysis")
st.caption(
    "Enter a location to see the heat-day record at the nearest "
    "ERA5-Land grid point (0.1° resolution)."
)

col_lon, col_lat = st.columns(2)
with col_lon:
    lon_raw = st.text_input("X — Longitude", placeholder="25.8")
    st.caption(f"Sample: 25.8  ·  Valid range: {_W}°–{_E}°E")
with col_lat:
    lat_raw = st.text_input("Y — Latitude", placeholder="58.7")
    st.caption(f"Sample: 58.7  ·  Valid range: {_S}°–{_N}°N")

if lon_raw.strip() or lat_raw.strip():
    lon = _parse_coord(lon_raw)
    lat = _parse_coord(lat_raw)

    if lon is None or lat is None:
        st.warning("Enter a valid number for both X and Y. Decimals may use '.' or ','.")
    elif not (_S <= lat <= _N and _W <= lon <= _E):
        st.info(
            f"Coordinates ({lat_raw}, {lon_raw}) are not in the risk analysis area. "
            f"Valid area: {_S}°–{_N}°N, {_W}°–{_E}°E."
        )
    elif not _grid_parquet.exists():
        st.info(
            "Grid data not yet available. Re-run the pipeline to generate it:\n"
            "```\npython src/run_pipeline.py --country EE\n```"
        )
    else:
        df_grid = _load_grid_parquet(str(_grid_parquet))

        # ERA5-Land uses 'latitude'/'longitude'; guard against alt naming.
        lat_col = "latitude" if "latitude" in df_grid.columns else "lat"
        lon_col = "longitude" if "longitude" in df_grid.columns else "lon"

        lats_land = np.sort(df_grid[lat_col].unique())
        lons_land = np.sort(df_grid[lon_col].unique())
        nearest_lat = float(lats_land[np.argmin(np.abs(lats_land - lat))])
        nearest_lon = float(lons_land[np.argmin(np.abs(lons_land - lon))])

        df_point = df_grid[
            (df_grid[lat_col] == nearest_lat) &
            (df_grid[lon_col] == nearest_lon)
        ][["year", "extreme_heat_days"]]

        if df_point.empty:
            st.info(
                f"The nearest grid point ({nearest_lat:.1f}°N, {nearest_lon:.1f}°E) "
                "is a sea cell with no land-temperature data."
            )
        else:
            df_point_full = (
                pd.DataFrame({"year": FULL_PERIOD})
                .merge(df_point, on="year", how="left")
            )
            point_mean = df_point_full["extreme_heat_days"].mean()

            st.write(
                f"Nearest land grid cell: **{nearest_lat:.1f}°N, {nearest_lon:.1f}°E** "
                f"(input: {lat:.2f}°N, {lon:.2f}°E)"
            )

            col1p, col2p = st.columns(2)
            col1p.metric(
                label="Mean heat days/year — grid cell",
                value=f"{point_mean:.2f}" if pd.notna(point_mean) else "—",
                delta=(
                    f"{point_mean - period_mean:+.2f} vs national mean"
                    if pd.notna(point_mean) and pd.notna(period_mean) else None
                ),
            )
            col2p.metric(
                label="Years with data",
                value=f"{df_point_full['extreme_heat_days'].notna().sum()} / 30",
            )

            st.subheader(
                f"Annual heat days — {nearest_lat:.1f}°N, {nearest_lon:.1f}°E"
            )

            df_pt_done    = df_point_full[df_point_full["extreme_heat_days"].notna()]
            df_pt_pending = df_point_full[df_point_full["extreme_heat_days"].isna()]

            fig_pt = go.Figure()
            fig_pt.add_trace(go.Bar(
                x=df_pt_done["year"],
                y=df_pt_done["extreme_heat_days"],
                name="Processed",
                marker_color="#1f77b4",
                hovertemplate="%{x}: <b>%{y:.2f}</b> days<extra></extra>",
            ))
            if not df_pt_pending.empty:
                fig_pt.add_trace(go.Bar(
                    x=df_pt_pending["year"],
                    y=[0] * len(df_pt_pending),
                    name="Pending",
                    marker_color="#cccccc",
                    hovertemplate="%{x}: not yet processed<extra></extra>",
                ))
            if pd.notna(point_mean):
                fig_pt.add_hline(
                    y=point_mean,
                    line_dash="dash",
                    line_color="#555555",
                    annotation_text=f"Cell mean: {point_mean:.2f} days",
                    annotation_position="top left",
                )
            fig_pt.update_layout(
                xaxis=dict(title="Year", tickmode="linear", tick0=1991, dtick=2),
                yaxis=dict(title="Extreme heat days", rangemode="tozero"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=40, b=40),
                height=380,
                barmode="overlay",
            )
            st.plotly_chart(fig_pt, use_container_width=True)

# ── Notes ─────────────────────────────────────────────────────────────────────
with st.expander("Methodology"):
    st.markdown(f"""
**Metric:** number of days per year where the daily maximum 2 m temperature
(TX) ≥ {THRESHOLD_C} °C, following the ETCCDI extreme heat day definition.

**Computation:**
1. ERA5-Land hourly 2 m temperature downloaded from Copernicus CDS (0.1° grid).
2. Daily TX derived as the maximum over all 24 hourly values per calendar day.
3. Heat-day count computed independently at each ERA5-Land grid point.
4. Result reported as the spatial mean over all grid points within the Estonia
   bounding box (sea cells excluded by the ERA5-Land land mask).

**Data:** ERA5-Land reanalysis, ECMWF / Copernicus Climate Change Service.
**Period:** 1991–2020 (WMO standard reference period for current climate).
""")
