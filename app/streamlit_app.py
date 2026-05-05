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
_ccfg              = _cfg.get("cordex", {})
_scenarios_cfg     = _ccfg.get("scenarios", {})

# ── Metric definitions ─────────────────────────────────────────────────────────
# All colours are from the Okabe-Ito palette, which is safe for the most common
# forms of colour blindness (deuteranopia and protanopia).
# Add an entry here to expose a new metric in the selector.
_P = ROOT / "data/processed"
METRICS = {
    # ── Temperature: hot ──────────────────────────────────────────────────────
    "Extreme Heat Days": {
        "col":             "extreme_heat_days",
        "csv":             _P / "estonia_extreme_heat_days.csv",
        "parquet":         _P / "heat_days_grid_ee.parquet",
        "threshold_label": f"TX ≥ {_THRESHOLD_HEAT_C:.0f} °C",
        "bar_color":       "#D55E00",   # vermillion
        "y_label":         "Days per year",
        "pipeline_flag":   "heat_days",
        "csv_header":      f"Extreme Heat Days – TX≥{_THRESHOLD_HEAT_C:.0f}°C [days/year]",
    },
    "Annual Maximum Temperature": {
        "col":             "txx",
        "csv":             _P / "estonia_txx.csv",
        "parquet":         _P / "txx_grid_ee.parquet",
        "threshold_label": "Annual maximum daily TX",
        "bar_color":       "#E69F00",   # orange
        "y_label":         "Temperature (°C)",
        "pipeline_flag":   "txx",
        "csv_header":      "Annual Maximum Temperature – TXx [°C]",
    },
    "Tropical Nights": {
        "col":             "tr15",
        "csv":             _P / "estonia_tr15.csv",
        "parquet":         _P / "tr15_grid_ee.parquet",
        "threshold_label": "TN ≥ 17 °C",
        "bar_color":       "#CC79A7",   # reddish purple
        "y_label":         "Days per year",
        "pipeline_flag":   "tr15",
        "csv_header":      "Tropical Nights – TN≥17°C [days/year]",
    },
    # ── Temperature: cold ─────────────────────────────────────────────────────
    "Hard Frost Days": {
        "col":             "hard_frost_days",
        "csv":             _P / "estonia_hard_frost.csv",
        "parquet":         _P / "hard_frost_grid_ee.parquet",
        "threshold_label": "TN < −10 °C",
        "bar_color":       "#882255",   # wine — Paul Tol muted palette
        "y_label":         "Days per year",
        "pipeline_flag":   "hard_frost",
        "csv_header":      "Hard Frost Days – TN<−10°C [days/year]",
    },
    "Frost Days": {
        "col":             "frost_days",
        "csv":             _P / "estonia_frost_days.csv",
        "parquet":         _P / "frost_days_grid_ee.parquet",
        "threshold_label": f"TN < {_THRESHOLD_FROST_C:.0f} °C",
        "bar_color":       "#0072B2",   # blue
        "y_label":         "Days per year",
        "pipeline_flag":   "frost_days",
        "csv_header":      f"Frost Days – TN<{_THRESHOLD_FROST_C:.0f}°C [days/year]",
    },
    "Ice Days": {
        "col":             "id0",
        "csv":             _P / "estonia_id0.csv",
        "parquet":         _P / "id0_grid_ee.parquet",
        "threshold_label": "TX < 0 °C",
        "bar_color":       "#56B4E9",   # sky blue
        "y_label":         "Days per year",
        "pipeline_flag":   "id0",
        "csv_header":      "Ice Days – TX<0°C [days/year]",
    },
    "Annual Minimum Temperature": {
        "col":             "tnn",
        "csv":             _P / "estonia_tnn.csv",
        "parquet":         _P / "tnn_grid_ee.parquet",
        "threshold_label": "Annual minimum daily TN",
        "bar_color":       "#009E73",   # bluish green
        "y_label":         "Temperature (°C)",
        "pipeline_flag":   "tnn",
        "csv_header":      "Annual Minimum Temperature – TNn [°C]",
    },
    # ── Precipitation ─────────────────────────────────────────────────────────
    "Consecutive Dry Days": {
        "col":             "cdd",
        "csv":             _P / "estonia_cdd.csv",
        "parquet":         _P / "cdd_grid_ee.parquet",
        "threshold_label": "pr < 1 mm/day",
        "bar_color":       "#DDAA33",   # golden yellow
        "y_label":         "Days",
        "pipeline_flag":   "cdd",
        "csv_header":      "Consecutive Dry Days – CDD [days]",
    },
    "Heavy Precipitation Days": {
        "col":             "r20mm",
        "csv":             _P / "estonia_r20mm.csv",
        "parquet":         _P / "r20mm_grid_ee.parquet",
        "threshold_label": "pr > 20 mm/day",
        "bar_color":       "#332288",   # indigo
        "y_label":         "Days per year",
        "pipeline_flag":   "r20mm",
        "csv_header":      "Heavy Precipitation Days – R20mm [days/year]",
    },
    "Precipitation Intensity": {
        "col":             "sdii",
        "csv":             _P / "estonia_sdii.csv",
        "parquet":         _P / "sdii_grid_ee.parquet",
        "threshold_label": "Mean pr on wet days (≥ 1 mm)",
        "bar_color":       "#44BB99",   # mint
        "y_label":         "mm per day",
        "pipeline_flag":   "sdii",
        "csv_header":      "Precipitation Intensity – SDII [mm/day]",
    },
    "Annual Total Precipitation": {
        "col":             "prcptot",
        "csv":             _P / "estonia_prcptot.csv",
        "parquet":         _P / "prcptot_grid_ee.parquet",
        "threshold_label": "Annual total pr on wet days (≥ 1 mm)",
        "bar_color":       "#117733",   # dark green
        "y_label":         "mm per year",
        "pipeline_flag":   "prcptot",
        "csv_header":      "Annual Total Precipitation – PRCPTOT [mm/year]",
    },
}

# ── CORDEX projection paths (injected into each METRICS entry) ────────────────
# Filenames mirror those written by scripts/cordex_pipeline.py.
_CORDEX_P        = ROOT / "data" / "processed" / "cordex"
_CORDEX_CSV_NAME = {
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
for _meta in METRICS.values():
    _flag = _meta["pipeline_flag"]
    for _scen in _scenarios_cfg:          # adds rcp45, rcp85, … from config
        _meta[f"cordex_{_scen}_csv"]     = _CORDEX_P / _scen / _CORDEX_CSV_NAME[_flag]
        _meta[f"cordex_{_scen}_parquet"] = _CORDEX_P / _scen / f"{_flag}_grid_ee.parquet"

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def _load_grid_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def _build_location_csv(nearest_lat: float, nearest_lon: float,
                         parquet_paths: tuple, year_end: int = 2020) -> str:
    """
    Load every available metric Parquet, extract the nearest grid cell, join
    on a year spine (1991–year_end), append a period-mean row, and return a
    semicolon-delimited CSV string.

    parquet_paths : tuple of (source_col, csv_header, path_str)
                    Each entry is renamed to csv_header immediately on load so
                    ERA5 and CORDEX columns with the same source_col do not collide.
    year_end      : last year in the output (2020 for ERA5 only; 2100 with projections).
    """
    df_all = pd.DataFrame({"Year": list(range(1991, year_end + 1))})

    for source_col, csv_header, path_str in parquet_paths:
        if not Path(path_str).exists():
            continue
        try:
            df_grid = _load_grid_parquet(path_str)
            lat_col = "latitude" if "latitude" in df_grid.columns else "lat"
            lon_col = "longitude" if "longitude" in df_grid.columns else "lon"
            df_point = (
                df_grid[
                    (df_grid[lat_col] == nearest_lat) &
                    (df_grid[lon_col] == nearest_lon)
                ][["year", source_col]]
                .rename(columns={"year": "Year", source_col: csv_header})
            )
            if not df_point.empty:
                df_all = df_all.merge(df_point, on="Year", how="left")
        except Exception:
            continue

    mean_row = {"Year": "Period mean"}
    for c in df_all.columns:
        if c != "Year":
            mean_row[c] = round(df_all[c].mean(), 2)
    df_all = pd.concat([df_all, pd.DataFrame([mean_row])], ignore_index=True)
    return df_all.to_csv(sep=";", index=False)


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

# ── Sidebar — projection toggles ──────────────────────────────────────────────
with st.sidebar:
    st.subheader("Climate projections")
    _proj_enabled: dict[str, bool] = {}
    for _scen_key, _scen_meta in _scenarios_cfg.items():
        _scen_label = _scen_meta.get("label", _scen_key)
        _proj_enabled[_scen_key] = st.checkbox(_scen_label, value=False)
    if any(_proj_enabled.values()):
        st.caption(
            "Shows the 20-year centred running mean of annual projected values. "
            "Single scenario — no uncertainty range."
        )

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

# ── Load projection data ───────────────────────────────────────────────────────
_smoothing_w  = _ccfg.get("display_smoothing_years", 20)
_smoothing_mp = _ccfg.get("display_smoothing_min_periods", 10)
_df_proj: dict[str, pd.DataFrame] = {}      # scenario_key → DataFrame

for _scen_key, _show in _proj_enabled.items():
    if not _show:
        continue
    _csv_path = m.get(f"cordex_{_scen_key}_csv")
    if _csv_path and Path(_csv_path).exists():
        _dp = pd.read_csv(_csv_path)
        _dp["mean_20yr"] = _dp[col].rolling(
            window=_smoothing_w, center=True, min_periods=_smoothing_mp
        ).mean()
        _df_proj[_scen_key] = _dp
    else:
        st.sidebar.caption(
            f"⚠ {_scenarios_cfg[_scen_key]['label']}: "
            f"no data yet for {metric_name}."
        )

_has_proj = bool(_df_proj)

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
    hovertemplate=f"%{{x}}: <b>%{{y:.2f}}</b> {m['y_label']}<extra></extra>",
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
        annotation_text=f"Period mean: {period_mean:.2f} {m['y_label'].lower()}",
        annotation_position="top left",
    )

# ── Projection overlays ────────────────────────────────────────────────────────
for _scen_key, _dp in _df_proj.items():
    _sm = _scenarios_cfg[_scen_key]
    _color, _dash = _sm.get("line_color", "#888"), _sm.get("line_dash", "dot")
    _label = _sm.get("label", _scen_key)
    fig.add_trace(go.Scatter(           # annual values — faint background
        x=_dp["year"], y=_dp[col],
        name=f"{_label} (annual)",
        mode="lines",
        line=dict(color=_color, width=1),
        opacity=0.2,
        showlegend=False,
        hovertemplate=f"{_label} %{{x}}: <b>%{{y:.2f}}</b> {m['y_label']}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(           # 20-year running mean — primary display
        x=_dp["year"], y=_dp["mean_20yr"],
        name=_label,
        mode="lines",
        line=dict(color=_color, width=2.5, dash=_dash),
        hovertemplate=(
            f"{_label} (20yr mean) %{{x}}: <b>%{{y:.2f}}</b> {m['y_label']}<extra></extra>"
        ),
    ))

if _has_proj:
    fig.add_vline(x=2020.5, line_dash="dot", line_color="#aaaaaa", line_width=1)
    fig.add_annotation(
        x=2021, y=1, xref="x", yref="paper", text="Projection →",
        showarrow=False, xanchor="left", font=dict(size=10, color="#aaaaaa"),
    )

_x_end  = 2101 if _has_proj else 2022
_dtick  = 10   if _has_proj else 2

fig.update_layout(
    xaxis=dict(title="Year", range=[1990, _x_end], tickmode="linear",
               tick0=1990, dtick=_dtick),
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

_input_col, _map_col = st.columns([1, 2])

with _input_col:
    lon_raw = st.text_input("X — Longitude", placeholder="25.8")
    st.caption(f"Valid range: {_W}°–{_E}°E")
    lat_raw = st.text_input("Y — Latitude", placeholder="58.7")
    st.caption(f"Valid range: {_S}°–{_N}°N")

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

            _fig = folium.Figure(width="100%", height=350)
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
            _fig.add_child(_fmap)
            folium.Marker([nearest_lat, nearest_lon]).add_to(_fmap)
            with _map_col:
                st.write(
                    f"Nearest land grid cell: **{nearest_lat:.1f}°N, {nearest_lon:.1f}°E** "
                    f"(input: {lat:.2f}°N, {lon:.2f}°E)"
                )
                components.html(_fig._repr_html_(), height=350)

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
                hovertemplate=f"%{{x}}: <b>%{{y:.2f}}</b> {m['y_label']}<extra></extra>",
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
                    annotation_text=f"Cell mean: {point_mean:.2f} {m['y_label'].lower()}",
                    annotation_position="top left",
                )
            # ── Projection overlays on point chart ────────────────────────────
            for _scen_key, _dp in _df_proj.items():
                _pq_path = m.get(f"cordex_{_scen_key}_parquet")
                if not (_pq_path and Path(_pq_path).exists()):
                    continue
                try:
                    _dg = _load_grid_parquet(str(_pq_path))
                    _lc = "latitude" if "latitude" in _dg.columns else "lat"
                    _lnc = "longitude" if "longitude" in _dg.columns else "lon"
                    _lu = np.sort(_dg[_lc].unique())
                    _lou = np.sort(_dg[_lnc].unique())
                    _nlat = float(_lu[np.argmin(np.abs(_lu - lat))])
                    _nlon = float(_lou[np.argmin(np.abs(_lou - lon))])
                    _dp_pt = _dg[
                        (_dg[_lc] == _nlat) & (_dg[_lnc] == _nlon)
                    ][["year", col]].copy()
                    if _dp_pt.empty:
                        continue
                    _dp_pt["mean_20yr"] = _dp_pt[col].rolling(
                        window=_smoothing_w, center=True,
                        min_periods=_smoothing_mp,
                    ).mean()
                    _sm = _scenarios_cfg[_scen_key]
                    _color = _sm.get("line_color", "#888")
                    _dash  = _sm.get("line_dash", "dot")
                    _plabel = _sm.get("label", _scen_key)
                    fig_pt.add_trace(go.Scatter(
                        x=_dp_pt["year"], y=_dp_pt[col],
                        name=f"{_plabel} (annual)", mode="lines",
                        line=dict(color=_color, width=1), opacity=0.2,
                        showlegend=False,
                        hovertemplate=(
                            f"{_plabel} %{{x}}: <b>%{{y:.2f}}</b> "
                            f"{m['y_label']}<extra></extra>"
                        ),
                    ))
                    fig_pt.add_trace(go.Scatter(
                        x=_dp_pt["year"], y=_dp_pt["mean_20yr"],
                        name=_plabel, mode="lines",
                        line=dict(color=_color, width=2.5, dash=_dash),
                        hovertemplate=(
                            f"{_plabel} (20yr) %{{x}}: <b>%{{y:.2f}}</b> "
                            f"{m['y_label']}<extra></extra>"
                        ),
                    ))
                except Exception:
                    pass

            if _has_proj:
                fig_pt.add_vline(
                    x=2020.5, line_dash="dot", line_color="#aaaaaa", line_width=1
                )

            fig_pt.update_layout(
                xaxis=dict(title="Year", range=[1990, _x_end],
                           tickmode="linear", tick0=1990, dtick=_dtick),
                yaxis=dict(title=m["y_label"], rangemode="tozero"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=40, b=40),
                height=380,
                barmode="overlay",
            )
            st.plotly_chart(fig_pt, use_container_width=True)

            # ── CSV download ───────────────────────────────────────────────────
            _parquet_index = tuple(
                (meta["col"], meta["csv_header"], str(meta["parquet"]))
                for meta in METRICS.values()
            )
            # Extend with CORDEX projection Parquets for selected scenarios.
            for _scen_key in _df_proj:
                _scen_label = _scenarios_cfg[_scen_key]["label"]
                for meta in METRICS.values():
                    _pq = meta.get(f"cordex_{_scen_key}_parquet")
                    if _pq and Path(_pq).exists():
                        _hdr = f"{meta['csv_header']} [{_scen_label}]"
                        _parquet_index += ((meta["col"], _hdr, str(_pq)),)

            _csv_year_end = 2100 if _has_proj else 2020
            csv_bytes = _build_location_csv(
                nearest_lat, nearest_lon, _parquet_index, year_end=_csv_year_end
            ).encode("utf-8")
            st.download_button(
                label="Download all metrics as CSV",
                data=csv_bytes,
                file_name=(
                    f"baltic_climate_risk_{nearest_lat:.2f}N"
                    f"_{nearest_lon:.2f}E.csv"
                ),
                mime="text/csv",
            )

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
