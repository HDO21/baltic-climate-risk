"""
Streamlit dashboard — Baltic Climate Risk.

Displays annual climate risk metrics for the 1991-2020 WMO reference period,
derived from ERA5-Land data. Run with:

    conda activate climate-risk
    streamlit run app/streamlit_app.py
"""

import folium
import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config/config.yaml"

FULL_PERIOD = list(range(1991, 2021))

# ── Config ────────────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

_THRESHOLD_HEAT_C  = float(_cfg["metrics"]["heat_days"]["threshold_tx_degC"])
_THRESHOLD_FROST_C = float(_cfg["metrics"]["frost_days"]["threshold_tn_degC"])

# ── Metric definitions ─────────────────────────────────────────────────────────
# Add an entry here to expose a new metric in the selector.
METRICS = {
    "Extreme Heat Days": {
        "col":             "extreme_heat_days",
        "csv":             ROOT / "data/processed/estonia_extreme_heat_days.csv",
        "parquet":         ROOT / "data/processed/heat_days_grid_ee.parquet",
        "threshold_label": f"TX ≥ {_THRESHOLD_HEAT_C:.0f} °C",
        "bar_color":       "#d62728",
        "y_label":         "Extreme heat days",
        "pipeline_flag":   "heat_days",
    },
    "Frost Days": {
        "col":             "frost_days",
        "csv":             ROOT / "data/processed/estonia_frost_days.csv",
        "parquet":         ROOT / "data/processed/frost_days_grid_ee.parquet",
        "threshold_label": f"TN < {_THRESHOLD_FROST_C:.0f} °C",
        "bar_color":       "rgb(24, 56, 245)",
        "y_label":         "Frost days",
        "pipeline_flag":   "frost_days",
    },
}

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


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Baltic Climate Risk",
    page_icon="🌡",
    layout="wide",
)

st.title("Baltic Climate Risk — Estonia")

# ── Metric selector ────────────────────────────────────────────────────────────
# Architecture note: country_code is a plain variable below so it can be
# replaced with a st.selectbox when multi-country support is added.
metric_name = st.selectbox("Metric", options=list(METRICS.keys()), index=0)
m           = METRICS[metric_name]
col         = m["col"]

st.caption(
    f"ERA5-Land reanalysis  ·  1991–2020 WMO reference period  ·  {m['threshold_label']}"
)

# ── Load data ─────────────────────────────────────────────────────────────────
if not m["csv"].exists():
    st.error(
        f"No processed data found at `{m['csv'].relative_to(ROOT)}`.\n\n"
        "Run the pipeline first:\n"
        f"```\npython src/run_pipeline.py --metric {m['pipeline_flag']}\n```"
    )
    st.stop()

processed = pd.read_csv(m["csv"])

# Build a complete 1991–2020 frame; years not yet processed appear as None.
df = (
    pd.DataFrame({"year": FULL_PERIOD})
    .merge(processed, on="year", how="left")
)

available = df[col].notna().sum()
missing   = len(FULL_PERIOD) - available

# ── Status banner ─────────────────────────────────────────────────────────────
if missing > 0:
    st.info(
        f"Pipeline in progress: **{available} of 30 years** processed. "
        f"Run `python src/run_pipeline.py --metric {m['pipeline_flag']}` "
        f"to compute the remaining {missing} year(s)."
    )

# ── Summary metrics ────────────────────────────────────────────────────────────
period_mean = df[col].mean()

col1, col2 = st.columns(2)
col1.metric(
    label=f"Mean {metric_name.lower()} per year ({df['year'].min()}–{df['year'].max()})",
    value=f"{period_mean:.2f}" if pd.notna(period_mean) else "—",
    help=(
        f"Average number of days per year where {m['threshold_label']}, "
        "computed as the spatial mean across all ERA5-Land grid points covering Estonia."
    ),
)
col2.metric(
    label="Years processed",
    value=f"{available} / 30",
)

# ── Bar chart ─────────────────────────────────────────────────────────────────
st.subheader(f"Annual {metric_name.lower()}")

df_done    = df[df[col].notna()]
df_pending = df[df[col].isna()]

fig = go.Figure()

fig.add_trace(go.Bar(
    x=df_done["year"],
    y=df_done[col],
    name="Processed",
    marker_color=m["bar_color"],
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
    yaxis=dict(title=m["y_label"], rangemode="tozero"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40, b=40),
    height=420,
    barmode="overlay",
)

st.plotly_chart(fig, use_container_width=True)

# ── Coordinate input ──────────────────────────────────────────────────────────
country_code    = "EE"
_area           = _cfg["countries"][country_code]["area"]   # [N, W, S, E]
_N, _W, _S, _E = _area
_grid_parquet   = m["parquet"]

st.divider()
st.subheader("Location-specific analysis")
st.caption(
    f"Enter a location to see the {metric_name.lower()} record at the nearest "
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
        st.warning("Enter a valid number for both X and Y.")
    elif not (_S <= lat <= _N and _W <= lon <= _E):
        st.info(
            f"Coordinates ({lat_raw}, {lon_raw}) are not in the risk analysis area. "
            f"Valid area: {_S}°–{_N}°N, {_W}°–{_E}°E."
        )
    elif not _grid_parquet.exists():
        st.info(
            f"Grid data for {metric_name} is not yet available. Re-run the pipeline:\n"
            f"```\npython src/run_pipeline.py --metric {m['pipeline_flag']}\n```"
        )
    else:
        df_grid = _load_grid_parquet(str(_grid_parquet))

        # ERA5-Land uses 'latitude'/'longitude'; guard against alt naming.
        lat_col_g = "latitude" if "latitude" in df_grid.columns else "lat"
        lon_col_g = "longitude" if "longitude" in df_grid.columns else "lon"

        lats_land   = np.sort(df_grid[lat_col_g].unique())
        lons_land   = np.sort(df_grid[lon_col_g].unique())
        nearest_lat = float(lats_land[np.argmin(np.abs(lats_land - lat))])
        nearest_lon = float(lons_land[np.argmin(np.abs(lons_land - lon))])

        df_point = df_grid[
            (df_grid[lat_col_g] == nearest_lat) &
            (df_grid[lon_col_g] == nearest_lon)
        ][["year", col]]

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
            point_mean = df_point_full[col].mean()

            st.write(
                f"Nearest land grid cell: **{nearest_lat:.1f}°N, {nearest_lon:.1f}°E** "
                f"(input: {lat:.2f}°N, {lon:.2f}°E)"
            )

            _fmap = folium.Map(
                location=[nearest_lat, nearest_lon],
                zoom_start=15,
                zoom_control=False,
                scrollWheelZoom=False,
                dragging=False,
                doubleClickZoom=False,
                touchZoom=False,
                keyboard=False,
            )
            folium.Marker([nearest_lat, nearest_lon]).add_to(_fmap)
            components.html(_fmap._repr_html_(), height=300)

            col1p, col2p = st.columns(2)
            col1p.metric(
                label=f"Mean {metric_name.lower()}/year — grid cell",
                value=f"{point_mean:.2f}" if pd.notna(point_mean) else "—",
                delta=(
                    f"{point_mean - period_mean:+.2f} vs national mean"
                    if pd.notna(point_mean) and pd.notna(period_mean) else None
                ),
            )
            col2p.metric(
                label="Years with data",
                value=f"{df_point_full[col].notna().sum()} / 30",
            )

            st.subheader(
                f"Annual {metric_name.lower()} — {nearest_lat:.1f}°N, {nearest_lon:.1f}°E"
            )

            df_pt_done    = df_point_full[df_point_full[col].notna()]
            df_pt_pending = df_point_full[df_point_full[col].isna()]

            fig_pt = go.Figure()
            fig_pt.add_trace(go.Bar(
                x=df_pt_done["year"],
                y=df_pt_done[col],
                name="Processed",
                marker_color=m["bar_color"],
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
                yaxis=dict(title=m["y_label"], rangemode="tozero"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=40, b=40),
                height=380,
                barmode="overlay",
            )
            st.plotly_chart(fig_pt, use_container_width=True)

# ── Notes ─────────────────────────────────────────────────────────────────────
with st.expander("Methodology"):
    if metric_name == "Extreme Heat Days":
        st.markdown(f"""
**Metric:** number of days per year where the daily maximum 2 m temperature
(TX) ≥ {_THRESHOLD_HEAT_C:.0f} °C, following the ETCCDI TX30 definition.

**Computation:**
1. ERA5-Land hourly 2 m temperature downloaded from Copernicus CDS (0.1° grid).
2. Daily TX derived as the maximum over all 24 hourly values per calendar day.
3. Heat-day count computed independently at each ERA5-Land grid point.
4. Result reported as the spatial mean over all grid points within the Estonia
   bounding box (sea cells excluded by the ERA5-Land land mask).

**Data:** ERA5-Land reanalysis, ECMWF / Copernicus Climate Change Service.
**Period:** 1991–2020 (WMO standard reference period for current climate).
""")
    elif metric_name == "Frost Days":
        st.markdown(f"""
**Metric:** number of days per year where the daily minimum 2 m temperature
(TN) < {_THRESHOLD_FROST_C:.0f} °C, following the ETCCDI FD0 definition.

**Computation:**
1. ERA5-Land hourly 2 m temperature downloaded from Copernicus CDS (0.1° grid).
2. Daily TN derived as the minimum over all 24 hourly values per calendar day.
3. Frost-day count computed independently at each ERA5-Land grid point.
4. Result reported as the spatial mean over all grid points within the Estonia
   bounding box (sea cells excluded by the ERA5-Land land mask).

**Data:** ERA5-Land reanalysis, ECMWF / Copernicus Climate Change Service.
**Period:** 1991–2020 (WMO standard reference period for current climate).
""")
