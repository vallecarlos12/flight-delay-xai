import streamlit as st
import duckdb
import pandas as pd
import pydeck as pdk
import plotly.express as px
import os
import warnings
import datetime
from pandas.errors import SettingWithCopyWarning
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import shap
import plotly.graph_objects as go
import sys
import subprocess
#https://drive.google.com/file/d/1ajcEMH5Dc3WAu1XG71IOwFZakaxy8ScF/view?usp=sharing
#https://drive.google.com/uc?id=1ajcEMH5Dc3WAu1XG71IOwFZakaxy8ScF


##############
from pathlib import Path
import subprocess

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
st.set_page_config(page_title="Flight Route Visualizer", layout="wide")
st.markdown(
    """<style>.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {font-size: 20px;}</style>""",
    unsafe_allow_html=True)
# where to store large artifacts in Streamlit Cloud
ARTIFACT_DIR = Path("./artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

def _gdown(file_id: str, out_path: Path):
    """Download a Google Drive file id into out_path if not present."""
    if out_path.exists():
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        import gdown  # ensure module is importable in this env
        # Use the same Python interpreter that Streamlit is using
        subprocess.check_call([
            sys.executable,
            "-m", "gdown",
            "--fuzzy", url,
            "-O", str(out_path),
        ])
    except Exception as e:
        st.error(f"Failed to download artifact {out_path.name} from Drive: {e}")
        st.stop()

# ==== declare file locations (use artifacts/ instead of repo root) ====
FLIGHT_DATA_PATH = str(ARTIFACT_DIR / "all_data.parquet")
INFERENCE_LOOKUP_PATH = str(ARTIFACT_DIR / "inference_lookup.parquet")
#XGB_P10_PATH = str(ARTIFACT_DIR / "xgb_flight_delay_model_p10.json")
#XGB_P50_PATH = str(ARTIFACT_DIR / "xgb_flight_delay_model_p50.json")
#XGB_P90_PATH = str(ARTIFACT_DIR / "xgb_flight_delay_model_p90.json")

# ==== download if missing ====
with st.spinner("Downloading data/models..."):
    # replace with your real Google Drive file ids
    _gdown("1Y4IOD1SUtQf7U0dxd6rYcTkjy2XNgKBs", Path(FLIGHT_DATA_PATH))
    _gdown("1fyepCZFt_nHHAunMTPXInt__eKJhqgEB", Path(INFERENCE_LOOKUP_PATH))


##############

#FLIGHT_DATA_PATH = "all_data.parquet"
GEO_DATA_PATH = "flight_geo/L_AIRPORT_ID_with_Coordinates.csv"
AIRLINE_DATA_PATH = "flight_geo/L_AIRLINE_ID.csv"


@st.cache_resource
def get_data_conn():
    conn = duckdb.connect(database=':memory:')
    if os.path.exists(FLIGHT_DATA_PATH):
        conn.execute(f"CREATE VIEW flights AS SELECT * FROM read_parquet('{FLIGHT_DATA_PATH}')")
    else:
        st.error(f"Missing flight data file at {FLIGHT_DATA_PATH}")
        st.stop()

    if os.path.exists(GEO_DATA_PATH):
        conn.execute(f"CREATE VIEW geo AS SELECT * FROM read_csv_auto('{GEO_DATA_PATH}')")
    else:
        st.error(f"Missing location data file at {GEO_DATA_PATH}")
        st.stop()

    if os.path.exists(AIRLINE_DATA_PATH):
        conn.execute(f"CREATE VIEW airlines AS SELECT * FROM read_csv_auto('{AIRLINE_DATA_PATH}')")
    else:
        conn.execute("CREATE VIEW airlines AS SELECT NULL AS Code, NULL AS Description LIMIT 0")
    return conn


conn = get_data_conn()

DOW_SQL = """((CAST(strftime('%w', make_date(Year, Month, DayofMonth)) AS INTEGER) + 6) % 7) + 1"""

delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']


@st.cache_data(show_spinner="Loading airline data...")
def load_airline_data():
    df = conn.execute("SELECT Code, Description FROM airlines").fetchdf()
    if "Code" in df.columns:
        df["Code"] = df["Code"].astype(str)
        return dict(zip(df["Code"], df["Description"]))
    return {}


@st.cache_data
def load_geo_data():
    df = conn.execute("SELECT Code, \"Airport Name\" AS airport_name, Latitude, Longitude FROM geo").fetchdf()
    df["Code"] = df["Code"].astype(str)
    return df

df_geo = load_geo_data()
airline_lookup = load_airline_data()

@st.cache_data(show_spinner="Loading min/max dates...")
def get_min_max_dates():
    q = f"""
    SELECT 
      MIN(Year) as min_y, MIN(Month) as min_m, MIN(DayofMonth) as min_d,
      MAX(Year) as max_y, MAX(Month) as max_m, MAX(DayofMonth) as max_d
    FROM flights
    """
    r = conn.execute(q).fetchone()
    try:
        min_date = datetime.date(int(r[0]), int(r[1]), int(r[2]))
        max_date = datetime.date(int(r[3]), int(r[4]), int(r[5]))
    except Exception:
        min_date = datetime.date(2024, 1, 1)
        max_date = datetime.date(2024, 12, 31)
    return min_date, max_date


@st.cache_data(show_spinner="Finding airports for date...")
def get_airports_for_date(selected_date):
    q = f"""
    SELECT DISTINCT CAST(OriginAirportID AS VARCHAR) AS Code
    FROM flights
    WHERE Year = {selected_date.year}
      AND Month = {selected_date.month}
      AND DayofMonth = {selected_date.day}
    """
    origin_ids = [str(x[0]) for x in conn.execute(q).fetchall()]
    if not origin_ids:
        return {}
    dff = df_geo[df_geo["Code"].isin(origin_ids)].copy()
    dff["display"] = dff["Code"].astype(str) + " - " + dff["airport_name"].fillna("Unknown")
    return dict(zip(dff["Code"], dff["display"]))


@st.cache_data(show_spinner="Getting destinations for origin...")
def get_destinations_for_origin(origin_id, selected_date):
    q = f"""
    SELECT DISTINCT CAST(DestAirportID AS VARCHAR) AS Code
    FROM flights
    WHERE Year = {selected_date.year}
      AND Month = {selected_date.month}
      AND DayofMonth = {selected_date.day}
      AND CAST(OriginAirportID AS VARCHAR) = '{origin_id}'
    """
    dest_ids = [str(x[0]) for x in conn.execute(q).fetchall()]
    if not dest_ids:
        return {}
    # df_geo = load_geo_data()
    dfd = df_geo[df_geo["Code"].isin(dest_ids)].copy()
    dfd["display"] = dfd["Code"].astype(str) + " - " + dfd["airport_name"].fillna("Unknown")
    return dict(zip(dfd["Code"], dfd["display"]))


@st.cache_data(show_spinner="Getting departure times...")
def get_available_dep_times(origin_id, dest_ids, selected_date):
    where_origin = f"AND CAST(OriginAirportID AS VARCHAR) = '{origin_id}'"
    where_date = f"Year = {selected_date.year} AND Month = {selected_date.month} AND DayofMonth = {selected_date.day}"
    dest_clause = ""
    if dest_ids and dest_ids != ["ALL"]:
        dest_list = ",".join([f"'{d}'" for d in dest_ids])
        dest_clause = f"AND CAST(DestAirportID AS VARCHAR) IN ({dest_list})"
    q = f"""
    SELECT DISTINCT CRSDepTime FROM flights
    WHERE {where_date} {where_origin} {dest_clause}
      AND CRSDepTime IS NOT NULL
    """
    rows = conn.execute(q).fetchdf()
    times = sorted([int(x) for x in rows["CRSDepTime"].unique() if pd.notna(x)])
    return times


def format_time(time_int):
    if pd.isna(time_int) or time_int is None:
        return "N/A"
    try:
        t = int(time_int)
        if t < 0 or t > 2400:
            return "N/A"
        s = str(t).zfill(4)
        return f"{s[:2]}:{s[2:]}"
    except Exception:
        return "N/A"


def delay_ratio_to_color(ratio, alpha=200):
    ratio = max(0, min(1, ratio))
    if ratio < 0.5:
        r = int(510 * ratio)
        g = 255
    else:
        r = 255
        g = int(510 * (1 - ratio))
    return [r, g, 0, alpha]


@st.cache_data(show_spinner="Fetching routes...")
def get_routes_for_selection(origin_id, dest_ids, selected_date, delay_filter, time_filter):
    where_clauses = [
        f"Year = {selected_date.year}",
        f"Month = {selected_date.month}",
        f"DayofMonth = {selected_date.day}",
        f"CAST(OriginAirportID AS VARCHAR) = '{origin_id}'"
    ]
    if dest_ids and dest_ids != ["ALL"]:
        dest_list = ",".join([f"'{d}'" for d in dest_ids])
        where_clauses.append(f"CAST(DestAirportID AS VARCHAR) IN ({dest_list})")
    if time_filter != "ALL":
        where_clauses.append(f"CRSDepTime = {int(time_filter)}")
    if delay_filter == "Delayed":
        where_clauses.append("(ArrDel15 = 1 OR DepDel15 = 1)")
    elif delay_filter == "On-Time":
        where_clauses.append("(COALESCE(ArrDel15,0) != 1 AND COALESCE(DepDel15,0) != 1)")

    where_sql = " AND ".join(where_clauses)

    q_agg = f"""
    SELECT
      CAST(OriginAirportID AS VARCHAR) as origin,
      CAST(DestAirportID AS VARCHAR) as dest,
      COUNT(*) AS total_flights,
      SUM(CASE WHEN COALESCE(ArrDel15,0)=1 OR COALESCE(DepDel15,0)=1 THEN 1 ELSE 0 END) AS delayed_count
    FROM flights
    WHERE {where_sql}
    GROUP BY origin, dest
    """
    agg = conn.execute(q_agg).fetchdf()
    if agg.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    geo = load_geo_data()
    geo_src = geo.rename(
        columns={"Code": "origin", "airport_name": "orig_name", "Latitude": "src_lat", "Longitude": "src_lon"})
    geo_dst = geo.rename(
        columns={"Code": "dest", "airport_name": "dest_name", "Latitude": "dest_lat", "Longitude": "dest_lon"})

    merged = agg.merge(geo_src, on="origin", how="left").merge(geo_dst, on="dest", how="left")
    merged["delay_ratio"] = merged["delayed_count"] / merged["total_flights"]
    merged["pct_delayed"] = (merged["delay_ratio"] * 100).round(1).astype(str) + "%"
    merged["color"] = merged["delay_ratio"].apply(delay_ratio_to_color)

    arcs = merged[[
        "src_lon", "src_lat", "dest_lon", "dest_lat",
        "orig_name", "dest_name", "total_flights", "delayed_count", "pct_delayed", "color"
    ]].copy()

    origins = merged[["orig_name", "src_lat", "src_lon", "delay_ratio"]].drop_duplicates().rename(
        columns={"orig_name": "airport", "src_lat": "lat", "src_lon": "lon"}
    )
    origins["type"] = "Origin"
    destinations = merged[["dest_name", "dest_lat", "dest_lon", "delay_ratio"]].drop_duplicates().rename(
        columns={"dest_name": "airport", "dest_lat": "lat", "dest_lon": "lon"}
    )
    destinations["type"] = "Destination"
    points = pd.concat([origins, destinations], ignore_index=True)
    points["color"] = points["delay_ratio"].apply(delay_ratio_to_color)
    points["status_label"] = points["delay_ratio"].apply(
        lambda x: "On-Time"
        if x == 0
        else ("Delayed" if x == 1 else "Mixed (%.0f%% delayed)" % (x * 100))
    )
    points["total_flights"] = None
    points["delayed_count"] = None
    points["pct_delayed"] = None

    q_rows = f"""
    SELECT
      Flight_Number_Reporting_Airline as fl_num,
      CAST(OriginAirportID AS VARCHAR) as origin,
      CAST(DestAirportID AS VARCHAR) as dest,
      CRSDepTime as crs_dep_time_raw,
      COALESCE(ArrDel15,0) as ArrDel15,
      COALESCE(DepDel15,0) as DepDel15,
      COALESCE(ArrDelayMinutes,0) as ArrDelayMinutes,
      COALESCE(DepDelayMinutes,0) as DepDelayMinutes,
      COALESCE(CarrierDelay,0) as CarrierDelay,
      COALESCE(WeatherDelay,0) as WeatherDelay,
      COALESCE(NASDelay,0) as NASDelay,
      COALESCE(SecurityDelay,0) as SecurityDelay,
      COALESCE(LateAircraftDelay,0) as LateAircraftDelay
    FROM flights
    WHERE {where_sql}
    """
    rows_df = conn.execute(q_rows).fetchdf()
    if not rows_df.empty:
        rows_df["is_delay"] = ((rows_df["ArrDel15"] == 1) | (rows_df["DepDel15"] == 1)).astype(int)
        rows_df["crs_dep_time"] = rows_df["crs_dep_time_raw"].apply(format_time)
        rows_df["crs_dep_time_sort"] = rows_df["crs_dep_time_raw"].fillna(0).astype(int)

        rows_df = rows_df.merge(geo.rename(
            columns={"Code": "origin", "airport_name": "orig_name", "Latitude": "src_lat", "Longitude": "src_lon"}),
            on="origin", how="left")
        rows_df = rows_df.merge(geo.rename(
            columns={"Code": "dest", "airport_name": "dest_name", "Latitude": "dest_lat", "Longitude": "dest_lon"}),
            on="dest", how="left")

        rows_df = rows_df.sort_values("crs_dep_time_sort").reset_index(drop=True)

    return rows_df, arcs, points


@st.cache_data(show_spinner="Finding available airlines for route...")
def get_airlines_for_route(origin_id, dest_id):
    q = f"""
    SELECT DISTINCT CAST(DOT_ID_Reporting_Airline AS VARCHAR) as code 
    FROM flights 
    WHERE CAST(OriginAirportID AS VARCHAR) = '{origin_id}' 
      AND CAST(DestAirportID AS VARCHAR) = '{dest_id}'
      AND DOT_ID_Reporting_Airline IS NOT NULL
    """
    df_airlines_available = conn.execute(q).fetchdf()
    return sorted([str(x) for x in df_airlines_available['code'].unique() if pd.notna(x)])


@st.cache_data(show_spinner="Preparing training data...")
def prepare_training_data_for_route(origin_id, dest_id, selected_airlines):
    airline_filter = ""
    if selected_airlines:
        sel_airlines_csv = ",".join([f"'{a}'" for a in selected_airlines])
        airline_filter = f"AND CAST(DOT_ID_Reporting_Airline AS VARCHAR) IN ({sel_airlines_csv})"

    q_train = f"""
    SELECT
      Year,
      DayofMonth,
      {DOW_SQL} AS DayOfWeek,
      Month,
      CAST(COALESCE(CRSDepTime,0) AS INTEGER) / 100 AS DepHour,
      (COALESCE(ArrDelayMinutes,0) + COALESCE(DepDelayMinutes,0)) AS TotalDelay,
      CRSDepTime
    FROM flights
    WHERE CAST(OriginAirportID AS VARCHAR) = '{origin_id}'
      AND CAST(DestAirportID AS VARCHAR) = '{dest_id}'
      AND DayofMonth IS NOT NULL
      AND Month IS NOT NULL
      AND CRSDepTime IS NOT NULL
      {airline_filter}
    """
    df_train = conn.execute(q_train).fetchdf()
    if not df_train.empty:
        df_train["Year"] = df_train["Year"].astype(int)
        df_train["DayofMonth"] = df_train["DayofMonth"].astype(int)
        df_train["DayOfWeek"] = df_train["DayOfWeek"].astype(int)
        df_train["Month"] = df_train["Month"].astype(int)
        df_train["DepHour"] = df_train["DepHour"].astype(int)
    return df_train


@st.cache_data(show_spinner="Training prediction model...")
def train_knn_model_from_df(df_train, n_neighbors=5):
    if df_train.empty:
        return None
    X = df_train[["DayofMonth", "DayOfWeek", "Month", "DepHour"]].values.astype(float)
    y = df_train["TotalDelay"].values.astype(float)
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X, y)
    return model


def predict_delay_with_neighbors(model, day_of_month, day_of_week, month, dep_hour, df_train):
    if model is None or df_train.empty:
        return 0, pd.DataFrame()
    features = np.array([[day_of_month, day_of_week, month, dep_hour]], dtype=float)
    distances, indices = model.kneighbors(features)
    prediction = model.predict(features)[0]
    neighbors_data = df_train.iloc[indices[0]].copy()
    return max(0, prediction), neighbors_data


st.title("✈️ RouteLens")
st.markdown("Data-driven Flight Delay Visualization and Prediction")

min_date, max_date = get_min_max_dates()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Past Flight Activity",
    "📈 Historical Averages",
    "⚡ Forecast Lite",
    "🚀 Forecast Pro"
])

with tab1:
    st.markdown("### Filters")
    filter_cols = st.columns([1.5, 1.5, 1.5, 1.5, 1.5, 0.5])

    with filter_cols[0]:
        selected_date = st.date_input(
            "Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            format="MM/DD/YYYY",
        )

    airport_id_to_display = get_airports_for_date(selected_date)
    if not airport_id_to_display:
        st.warning(f"No flights available on {selected_date.strftime('%Y-%m-%d')}")
        st.stop()

    origin_display_names = list(airport_id_to_display.values())
    with filter_cols[1]:
        selected_display_origin = st.selectbox("Origin Airport", origin_display_names, label_visibility="visible")

    # Fix: safely extract origin_id with error handling
    matching_origins = [k for k, v in airport_id_to_display.items() if v == selected_display_origin]
    if not matching_origins:
        st.error("Selected origin airport not found. Please refresh and try again.")
        st.stop()
    selected_origin_id = matching_origins[0]

    valid_dest_options = get_destinations_for_origin(selected_origin_id, selected_date)
    destination_display_names = list(valid_dest_options.values())

    with filter_cols[2]:
        selected_display_dests = st.multiselect(
            "Destination(s)",
            options=destination_display_names,
            default=None,
            label_visibility="visible"
        )

    if selected_display_dests:
        selected_dest_ids = [k for k, v in valid_dest_options.items() if v in selected_display_dests]
    else:
        selected_dest_ids = ["ALL"]

    available_times = get_available_dep_times(selected_origin_id, selected_dest_ids, selected_date)
    time_options = ["ALL"] + [format_time(t) for t in available_times]
    time_values = ["ALL"] + [str(t) for t in available_times]

    with filter_cols[3]:
        time_display = st.selectbox(
            "Scheduled Departure Time",
            options=range(len(time_options)),
            format_func=lambda i: time_options[i],
            label_visibility="visible"
        )

    time_filter = time_values[time_display]

    with filter_cols[4]:
        delay_filter = st.selectbox("Flight Status", ["ALL", "On-Time", "Delayed"], label_visibility="visible")

    df_raw, arcs, points = get_routes_for_selection(
        selected_origin_id, selected_dest_ids, selected_date, delay_filter, time_filter
    )

    filter_applied = delay_filter != "ALL" or time_filter != "ALL" or selected_dest_ids != ["ALL"]

    if df_raw is None or (isinstance(df_raw, pd.DataFrame) and df_raw.empty):
        st.warning("No data for this selection.")
        st.stop()

    st.markdown("---")
    st.info(
        f"Showing **{len(df_raw):,}** flights from **{selected_display_origin}** on **{selected_date.strftime('%Y-%m-%d')}**"
    )
    st.markdown("🟢 Green = On-Time   🟡 Mixed   🔴 Delayed")

    tooltip = {
        "html": """
            <div style="font-size:13px; line-height:1.2;">
                <b>Airport:</b> {airport} ({type})<br/>
                <b>Total Flights:</b> {total_flights}<br/>
                <b>Delayed Flights:</b> {delayed_count}<br/>
                <b>Status:</b> {status_label}
            </div>
        """,
        "style": {
            "color": "black",
            "backgroundColor": "white",
            "fontSize": "13px",
            "padding": "10px",
            "borderRadius": "6px",
            "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
        },
    }

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arcs,
        get_source_position=["src_lon", "src_lat"],
        get_target_position=["dest_lon", "dest_lat"],
        get_source_color="color",
        get_target_color="color",
        get_width=5,
        pickable=True,
        auto_highlight=True,
    )

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position=["lon", "lat"],
        get_radius=25000,
        get_fill_color="color",
        get_line_color=[0, 0, 0, 150],
        get_line_width=150,
        pickable=True,
        auto_highlight=True,
    )

    center_lat = df_raw["src_lat"].mean() if not df_raw["src_lat"].empty else 40.7
    center_lon = df_raw["src_lon"].mean() if not df_raw["src_lon"].empty else -100.0

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=5,
        pitch=45,
    )

    r = pdk.Deck(
        layers=[arc_layer, scatter_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        height=700,
    )

    st.pydeck_chart(r)
    st.markdown("---")

    st.subheader("📊 Flight Statistics")
    col1, col2, col3, col4 = st.columns(4)
    delayed_count = int(df_raw["is_delay"].sum())
    ontime_count = len(df_raw) - delayed_count
    delayed_pct = (delayed_count / len(df_raw) * 100) if len(df_raw) > 0 else 0

    col1.metric("Total Flights", len(df_raw))
    col2.metric("On-Time", ontime_count)
    col3.metric("Delayed", delayed_count)
    col4.metric("Delayed %", f"{delayed_pct:.1f}%")

    st.subheader("🗓️ Route Delay Heatmap")
    st.markdown(f"**Origin Airport:** {selected_display_origin}")

    heatmap_data = df_raw.groupby(['dest_name', 'crs_dep_time']).agg({'is_delay': ['sum', 'count']}).reset_index()
    heatmap_data.columns = ['Destination', 'Time', 'Delayed', 'Total']
    heatmap_data['Delay_Ratio'] = heatmap_data['Delayed'] / heatmap_data['Total']
    pivot_data = heatmap_data.pivot(index='Destination', columns='Time', values='Delay_Ratio')

    num_destinations = len(pivot_data.index)
    cell_height = 55 if num_destinations > 4 else 80
    chart_height = max(500, num_destinations * cell_height + 250)

    fig = px.imshow(
        pivot_data,
        labels=dict(
            x="Scheduled Departure Time",
            y="Destination Airport",
            color="Delay Ratio"
        ),
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale=['green', 'yellow', 'red'],
        zmin=0,
        zmax=1,
        aspect="auto"
    )

    fig.update_traces(
        hovertemplate="<b>Destination:</b> %{y}<br>"
                      "<b>Time:</b> %{x}<br>"
                      "<b>Delay Ratio:</b> %{z:.1%}<extra></extra>"
    )

    fig.update_layout(
        height=chart_height,
        xaxis_title="Scheduled Departure Time",
        yaxis_title="Destination Airport",
        coloraxis_colorbar=dict(
            title="Delay<br>Ratio",
            tickvals=[0, 0.5, 1],
            ticktext=['On-Time', 'Mixed', 'Delayed'],
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            tickangle=0,
            showgrid=True,
            gridcolor='lightgray',
            tickmode="array",
            tickvals=pivot_data.columns[::max(1, len(pivot_data.columns) // 8)],
            ticktext=[t for i, t in enumerate(pivot_data.columns) if i % max(1, len(pivot_data.columns) // 8) == 0]
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=15)
        ),
        margin=dict(l=180, r=100, t=80, b=100),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=16)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    if st.checkbox("Show raw data table (First 500 rows)", value=False):
        display_cols = ["fl_num", "orig_name", "dest_name", "crs_dep_time", "is_delay"] + delay_cols
        df_display = df_raw[display_cols].copy()
        df_display["is_delay"] = df_display["is_delay"].map({0: "On-Time", 1: "Delayed"})
        st.dataframe(df_display.rename(columns={
            "fl_num": "Flight #", "orig_name": "Origin Airport", "dest_name": "Destination Airport",
            "crs_dep_time": "Scheduled Dep Time", "is_delay": "Status"
        }).head(500), use_container_width=True)

    delayed_flights = df_raw[(df_raw["ArrDel15"] == 1) | (df_raw["DepDel15"] == 1)].copy()

    if not delayed_flights.empty:
        delay_averages = {}
        for col in delay_cols:
            delay_averages[col] = delayed_flights[col].mean()

        delay_averages = {k: v for k, v in delay_averages.items() if v > 0}

        if delay_averages:
            st.markdown("---")
            st.markdown("### Average Breakdown of Delays (in Minutes)")
            st.info("Based on the **delayed** flights shown above.")

            chart_df = pd.DataFrame({
                'Delay Type': [name.replace('Delay', '').replace('Aircraft', ' Late Aircraft').replace('NAS',
                                                                                                       'National Aviation System')
                               for name in delay_averages.keys()],
                'Average Minutes': list(delay_averages.values())
            }).sort_values('Average Minutes', ascending=False)

            fig_delay = px.bar(
                chart_df,
                y='Delay Type',
                x='Average Minutes',
                orientation='h',
                text='Average Minutes',
                height=400,
                title="Average Delay by Category"
            )

            fig_delay.update_traces(
                texttemplate='%{text:.1f} min',
                textposition='outside',
                marker=dict(
                    color=chart_df['Average Minutes'],
                    colorscale='Viridis',
                    line=dict(color='white', width=1),
                )
            )

            fig_delay.update_layout(
                showlegend=False,
                xaxis_title="Average Delay (Minutes)",
                yaxis_title="",
                margin=dict(l=150, r=100, t=80, b=60),
                plot_bgcolor='rgba(240,240,240,0.5)',
                paper_bgcolor='white',
                font=dict(size=16),
                title_font_size=18,
                xaxis=dict(gridcolor='lightgray', showgrid=True),
                hovermode='closest'
            )

            st.plotly_chart(fig_delay, use_container_width=True)
        else:
            st.info("No individual delay categories found for the selected flights.")

with tab2:
    st.markdown("### Filters")
    filter_cols = st.columns([2, 2, 1.5])

    available_months = list(range(1, 13))
    month_names = [datetime.date(2024, m, 1).strftime('%B') for m in available_months]

    with filter_cols[0]:
        selected_months = st.multiselect(
            "Months",
            options=available_months,
            format_func=lambda m: datetime.date(2024, m, 1).strftime('%B'),
            default=available_months,
            label_visibility="visible"
        )

    df_airlines_available = conn.execute(
        "SELECT DISTINCT CAST(DOT_ID_Reporting_Airline AS VARCHAR) as code FROM flights WHERE DOT_ID_Reporting_Airline IS NOT NULL"
    ).fetchdf()
    available_airlines = sorted([str(x) for x in df_airlines_available['code'].unique() if pd.notna(x)])

    airline_display_dict = {}
    for code in available_airlines:
        display_name = f"{code} - {airline_lookup.get(code, 'Unknown')}"
        airline_display_dict[display_name] = code
    airline_display_names = list(airline_display_dict.keys())
    default_display_names = airline_display_names[:5] if len(airline_display_names) > 0 else []

    with filter_cols[1]:
        selected_display_airlines = st.multiselect(
            "Airlines",
            options=airline_display_names,
            default=default_display_names,
            label_visibility="visible"
        )
    selected_airlines = [airline_display_dict[d] for d in
                         selected_display_airlines] if selected_display_airlines else []

    with filter_cols[2]:
        delay_threshold = st.slider(
            "Delay Threshold (min)",
            min_value=0,
            max_value=120,
            value=30,
            step=5,
            label_visibility="visible"
        )

    if not selected_months:
        st.warning("Please select at least one month.")
        st.stop()

    if not selected_airlines:
        st.warning("Please select at least one airline.")
        st.stop()

    st.markdown("---")

    st.markdown("### Overall Average Delay by Airline")
    sel_airlines_csv = ",".join([f"'{a}'" for a in selected_airlines])
    sel_months_csv = ",".join([str(m) for m in selected_months])

    q_airline_avg = f"""
    SELECT 
        CAST(DOT_ID_Reporting_Airline AS VARCHAR) as AirlineCode,
        AVG(COALESCE(ArrDelayMinutes, 0) + COALESCE(DepDelayMinutes, 0)) AS AvgTotalDelay,
        COUNT(*) AS FlightCount
    FROM flights
    WHERE DOT_ID_Reporting_Airline IS NOT NULL
        AND CAST(DOT_ID_Reporting_Airline AS VARCHAR) IN ({sel_airlines_csv})
        AND Month IN ({sel_months_csv})
    GROUP BY AirlineCode
    ORDER BY AvgTotalDelay DESC
    """
    airline_avg_df = conn.execute(q_airline_avg).fetchdf()

    if not airline_avg_df.empty:
        airline_avg_df["AirlineName"] = airline_avg_df["AirlineCode"].apply(
            lambda code: f"{code} - {airline_lookup.get(code, 'Unknown')}"
        )

        fig_airline = px.bar(
            airline_avg_df,
            x="AvgTotalDelay",
            y="AirlineName",
            orientation='h',
            title="Overall Average Total Delay (Arrival + Departure) by Airline",
            labels={"AvgTotalDelay": "Avg Total Delay (Minutes)", "AirlineName": "Airline"},
            height=600,
            color="AvgTotalDelay",
            color_continuous_scale="RdYlGn_r"
        )
        fig_airline.add_vline(x=delay_threshold, line_dash="dash", line_color="red",
                              annotation_text=f"Threshold: {delay_threshold} min", annotation_position="top right")
        fig_airline.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Average Total Delay (Minutes)",
            yaxis_title="",
            font=dict(size=14),
            title_font_size=18,
            margin=dict(l=200, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_airline, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

    st.markdown("---")

    st.markdown("### Average Delay by Month")
    q_monthly = f"""
    SELECT 
        Month, 
        AVG(COALESCE(ArrDelayMinutes, 0) + COALESCE(DepDelayMinutes, 0)) AS AvgTotalDelay, 
        COUNT(*) AS FlightCount
    FROM flights
    WHERE CAST(DOT_ID_Reporting_Airline AS VARCHAR) IN ({sel_airlines_csv})
        AND Month IN ({sel_months_csv})
    GROUP BY Month
    ORDER BY Month
    """
    monthly_avg = conn.execute(q_monthly).fetchdf()

    if not monthly_avg.empty:
        monthly_avg["MonthName"] = monthly_avg["Month"].apply(lambda m: datetime.date(2024, int(m), 1).strftime('%B'))

        fig_monthly = px.bar(
            monthly_avg,
            x="MonthName",
            y="AvgTotalDelay",
            title="Average Total Delay by Month",
            labels={"AvgTotalDelay": "Avg Total Delay (Minutes)", "MonthName": "Month"},
            height=400,
            color="AvgTotalDelay",
            color_continuous_scale="RdYlGn_r"
        )
        fig_monthly.add_hline(y=delay_threshold, line_dash="dash", line_color="red",
                              annotation_text=f"Threshold: {delay_threshold} min", annotation_position="top right")
        fig_monthly.update_layout(
            xaxis_title="Month",
            yaxis_title="Avg Total Delay (Minutes)",
            font=dict(size=14),
            title_font_size=18,
            margin=dict(l=80, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

    st.markdown("---")

    st.markdown("### Daily Trend Within Selected Months by Airline")
    q_daily = f"""
    SELECT 
        CAST(DOT_ID_Reporting_Airline AS VARCHAR) as AirlineCode,
        DayofMonth,
        AVG(COALESCE(ArrDelayMinutes, 0) + COALESCE(DepDelayMinutes, 0)) AS AvgTotalDelay,
        COUNT(*) AS FlightCount
    FROM flights
    WHERE CAST(DOT_ID_Reporting_Airline AS VARCHAR) IN ({sel_airlines_csv})
        AND Month IN ({sel_months_csv})
    GROUP BY AirlineCode, DayofMonth
    ORDER BY AirlineCode, DayofMonth
    """
    daily_avg = conn.execute(q_daily).fetchdf()

    if not daily_avg.empty:
        daily_avg["AirlineName"] = daily_avg["AirlineCode"].apply(
            lambda code: f"{code} - {airline_lookup.get(code, 'Unknown')}"
        )

        fig_daily = px.line(
            daily_avg,
            x="DayofMonth",
            y="AvgTotalDelay",
            color="AirlineName",
            title="Average Daily Delay Trend by Airline",
            labels={"DayofMonth": "Day of Month", "AvgTotalDelay": "Avg Total Delay (Minutes)",
                    "AirlineName": "Airline"},
            height=500,
            markers=True
        )
        fig_daily.add_hline(y=delay_threshold, line_dash="dash", line_color="red",
                            annotation_text=f"Threshold: {delay_threshold} min", annotation_position="top right")
        fig_daily.update_layout(
            xaxis=dict(tickmode="linear", tick0=1, dtick=1),
            yaxis_title="Avg Total Delay (Minutes)",
            font=dict(size=14),
            title_font_size=18,
            margin=dict(l=80, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_daily, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

    st.markdown("---")

    st.markdown("### Delay Heatmap: Month vs Airline")
    q_heatmap = f"""
    SELECT 
        Month,
        CAST(DOT_ID_Reporting_Airline AS VARCHAR) as AirlineCode,
        AVG(COALESCE(ArrDelayMinutes, 0) + COALESCE(DepDelayMinutes, 0)) AS AvgTotalDelay
    FROM flights
    WHERE CAST(DOT_ID_Reporting_Airline AS VARCHAR) IN ({sel_airlines_csv})
        AND Month IN ({sel_months_csv})
    GROUP BY Month, AirlineCode
    ORDER BY Month, AirlineCode
    """
    heatmap_data = conn.execute(q_heatmap).fetchdf()

    if not heatmap_data.empty:
        heatmap_data["MonthName"] = heatmap_data["Month"].apply(lambda m: datetime.date(2024, int(m), 1).strftime('%B'))
        heatmap_data["AirlineName"] = heatmap_data["AirlineCode"].apply(
            lambda code: f"{code} - {airline_lookup.get(code, 'Unknown')}"
        )

        pivot_heatmap = heatmap_data.pivot(index="AirlineName", columns="MonthName", values="AvgTotalDelay")

        fig_heatmap = px.imshow(
            pivot_heatmap,
            labels=dict(color="Avg Delay (Min)"),
            title="Average Delay Heatmap: Airline vs Month",
            height=500,
            color_continuous_scale="RdYlGn_r"
        )
        fig_heatmap.update_layout(
            xaxis_title="Month",
            yaxis_title="Airline",
            font=dict(size=12),
            title_font_size=18,
            margin=dict(l=200, r=80, t=60, b=40)
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

with tab3:
    st.info("An Analog forecaster trained on historical patterns using K-Nearest neighbor model")
    st.markdown("### Filters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        forecast_date = st.date_input(
            "Date",
            value=datetime.date.today() + datetime.timedelta(days=7),
            format="MM/DD/YYYY",
            key="forecast_date"
        )

    forecast_month = forecast_date.month
    forecast_day = forecast_date.day
    forecast_dow = forecast_date.isoweekday()


    @st.cache_data(ttl=600)
    def get_airports_for_forecast(month, dow):
        q_orig = f"""
            SELECT DISTINCT CAST(OriginAirportID AS VARCHAR) as code 
            FROM flights 
            WHERE OriginAirportID IS NOT NULL
              AND Month = {month}
              AND {DOW_SQL} = {dow}
        """
        df_orig = conn.execute(q_orig).fetchdf()
        return sorted(df_orig["code"].astype(str).unique().tolist())


    @st.cache_data(ttl=600)
    def get_dests_for_forecast(origin_id, month, dow):
        q_dests = f"""
            SELECT DISTINCT CAST(DestAirportID AS VARCHAR) as code 
            FROM flights 
            WHERE CAST(OriginAirportID AS VARCHAR) = '{origin_id}'
              AND DestAirportID IS NOT NULL
              AND Month = {month}
              AND {DOW_SQL} = {dow}
        """
        df_dests = conn.execute(q_dests).fetchdf()
        return sorted(df_dests["code"].astype(str).unique().tolist())


    origins_all = get_airports_for_forecast(forecast_month, forecast_dow)
    origin_map = {c: f"{c} - {df_geo.loc[df_geo['Code'] == c, 'airport_name'].iloc[0]}"
    if c in df_geo["Code"].values else c for c in origins_all}

    if not origin_map:
        st.error(
            f"No historical flights found that operate on a {forecast_date.strftime('%A')} in {forecast_date.strftime('%B')}. Please try a different date.")
        st.stop()

    with col2:
        sel_origin_disp = st.selectbox("Origin Airport", list(origin_map.values()), key="forecast_origin")
    forecast_origin = [k for k, v in origin_map.items() if v == sel_origin_disp][0]

    dests_all = get_dests_for_forecast(forecast_origin, forecast_month, forecast_dow)
    dest_map = {c: f"{c} - {df_geo.loc[df_geo['Code'] == c, 'airport_name'].iloc[0]}"
    if c in df_geo["Code"].values else c for c in dests_all}

    if not dest_map:
        st.error(
            f"No destinations found from {sel_origin_disp} that operate on a {forecast_date.strftime('%A')} in {forecast_date.strftime('%B')}.")
        st.stop()

    with col3:
        sel_dest_disp = st.selectbox("Destination Airport", list(dest_map.values()), key="forecast_dest")
        forecast_dest = [k for k, v in dest_map.items() if v == sel_dest_disp][0]


    @st.cache_data(ttl=600)
    def get_airlines_for_forecast_route(origin_id, dest_id, month, dow):
        avail_air_q = f"""
            SELECT DISTINCT CAST(DOT_ID_Reporting_Airline AS VARCHAR) as code
            FROM flights
            WHERE CAST(OriginAirportID AS VARCHAR) = '{origin_id}'
              AND CAST(DestAirportID AS VARCHAR) = '{dest_id}'
              AND Month = {month}
              AND {DOW_SQL} = {dow}
            ORDER BY code
        """
        return conn.execute(avail_air_q).fetchdf()["code"].astype(str).unique().tolist()


    airlines_avail = get_airlines_for_forecast_route(forecast_origin, forecast_dest, forecast_month, forecast_dow)
    air_display = {f"{a} - {airline_lookup.get(a, 'Unknown')}": a for a in airlines_avail}

    if not air_display:
        st.error(
            f"No historical airlines found for this route that operate on a {forecast_date.strftime('%A')} in {forecast_date.strftime('%B')}.")
        st.stop()

    with col4:
        sel_display_airline = st.selectbox("Airline",
                                           list(air_display.keys()),
                                           key="forecast_airline")
    forecast_airline = air_display[sel_display_airline]

    q_avail_hours = f"""
        SELECT DISTINCT CAST(COALESCE(CRSDepTime,0) AS INTEGER) / 100 AS DepHour
        FROM flights
        WHERE CAST(OriginAirportID AS VARCHAR) = '{forecast_origin}'
          AND CAST(DestAirportID AS VARCHAR) = '{forecast_dest}'
          AND CAST(DOT_ID_Reporting_Airline AS VARCHAR) = '{forecast_airline}'
          AND Month = {forecast_month}
          AND {DOW_SQL} = {forecast_dow}
          AND CRSDepTime IS NOT NULL
        ORDER BY DepHour
    """
    df_avail_hours = conn.execute(q_avail_hours).fetchdf()

    if df_avail_hours.empty:
        st.error(
            f"No historical flights found for {sel_display_airline} on this route for {forecast_date.strftime('%A')}s in {forecast_date.strftime('%B')}. Cannot determine available departure times.")
        st.stop()

    available_hours = sorted(list(set([int(h) for h in df_avail_hours["DepHour"].unique() if pd.notna(h)])))
    hour_displays = [f"{h:02d}:00" for h in available_hours]

    col1t, col2t, col3t = st.columns([1.5, 1.5, 1])
    with col1t:
        hour_idx = st.selectbox(
            "Planned Departure Time (Hour)",
            range(len(hour_displays)),
            format_func=lambda i: hour_displays[i],
            key="forecast_time"
        )
        forecast_dep_hour = available_hours[hour_idx]
        forecast_time_int = forecast_dep_hour * 100

    with col2t:
        n_neighbors = st.slider(
            "Neighbors (k)",
            min_value=3,
            max_value=20,
            value=10,
            step=1,
            key="n_neighbors"
        )

    st.markdown(f"**Route:** {sel_origin_disp} → {sel_dest_disp}  ")
    st.markdown(f"**Date/Time:** {forecast_date.strftime('%A, %B %d, %Y')} at {format_time(forecast_time_int)}")
    st.markdown("---")

    airline_filter = f"AND CAST(DOT_ID_Reporting_Airline AS VARCHAR) = '{forecast_airline}'"
    q_train = f"""
    SELECT
      Year,
      DayofMonth,
      {DOW_SQL} AS DayOfWeek,
      Month,
      CAST(COALESCE(CRSDepTime,0) AS INTEGER) / 100 AS DepHour,
      CAST(COALESCE(CRSDepTime,0) AS INTEGER) AS DepTime,
      (COALESCE(ArrDelayMinutes,0) + COALESCE(DepDelayMinutes,0)) AS TotalDelay
    FROM flights
    WHERE CAST(OriginAirportID AS VARCHAR) = '{forecast_origin}'
      AND CAST(DestAirportID AS VARCHAR) = '{forecast_dest}'
      AND DayofMonth IS NOT NULL
      AND Month IS NOT NULL
      AND CRSDepTime IS NOT NULL
      {airline_filter}
    """
    df_train = conn.execute(q_train).fetchdf()
    if df_train.empty:
        st.error(f"No historical data found for {sel_display_airline} on this route.")
        st.stop()

    df_train["Year"] = df_train["Year"].astype(int)
    df_train["DayofMonth"] = df_train["DayofMonth"].astype(int)
    df_train["DayOfWeek"] = df_train["DayOfWeek"].astype(int)
    df_train["Month"] = df_train["Month"].astype(int)
    df_train["DepHour"] = df_train["DepHour"].astype(int)
    df_train["DepTime"] = df_train["DepTime"].astype(int)

    data1 = df_train[(df_train["Month"] == forecast_month) & (df_train["DepHour"] == forecast_dep_hour)]
    data2 = data1[data1["DayOfWeek"] == forecast_dow].copy()
    data3 = data1[data1["DayofMonth"] == forecast_day].copy()
    data2["Group"] = "DOW"
    data3["Group"] = "DOM"
    cand = pd.concat([data2, data3]).drop_duplicates().reset_index(drop=True)

    if cand.empty:
        st.error(
            f"No analog data available for the chosen Departure Time ({format_time(forecast_time_int)}) on this specific route/airline/month/day-of-week combination in the historical record. Try selecting a different hour.")
        st.stop()

    X = cand[["DayofMonth", "DayOfWeek", "Month", "DepHour"]].values
    y = cand["TotalDelay"].values
    k_eff = min(n_neighbors, len(cand))

    knn = train_knn_model_from_df(cand, k_eff)
    x_t = np.array([[forecast_day, forecast_dow, forecast_month, forecast_dep_hour]])

    distances, indices = knn.kneighbors(x_t)
    nbrs = cand.iloc[indices[0]].copy()
    nbrs["distance"] = distances[0]

    delay_values = nbrs["TotalDelay"].replace(np.nan, 0).apply(lambda x: max(0, x))
    # --- Compute Distance-Weighted Percentile Delays ---
    delay_values = nbrs["TotalDelay"].clip(lower=0)
    distances_safe = nbrs["distance"].replace(0, 1e-6)  # avoid div-by-zero
    weights = 1 / distances_safe
    weights = weights / weights.sum()  # normalize to 1

    def weighted_percentile(values, weights, percentiles):
        """
        Compute weighted percentiles using cumulative weighted distribution.
        Closer neighbors (higher weights) contribute more.
        """
        sorter = np.argsort(values)
        values_sorted = np.array(values.iloc[sorter])
        weights_sorted = np.array(weights.iloc[sorter])
        cum_w = np.cumsum(weights_sorted)
        cum_w /= cum_w[-1]
        return np.interp(percentiles, cum_w, values_sorted)

    if len(delay_values) < 3:
        # Fallback for very small samples
        p10 = p50 = p90 = delay_values.mean() if len(delay_values) > 0 else 0
    else:
        p10, p50, p90 = weighted_percentile(delay_values, weights, [0.1, 0.5, 0.9])


    st.markdown("### 🎯 Predicted Delay Range (Minutes)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "P10 (Best Case)",
            f"{p10:.1f} min",
            help="10% of historical flights had a delay less than or equal to this value."
        )

    with col2:
        status_median = "✅ On-Time Expected" if p50 < 10 else (
            "⚠️ Minor Delay Expected" if p50 < 30 else "🔴 Significant Delay Expected")
        st.metric(
            "P50 (Median)",
            f"{p50:.1f} min",
            delta=status_median,
            delta_color="off"
        )
        st.markdown(f"**Status:** {status_median}")

    with col3:
        st.metric(
            "P90 (Worst Case)",
            f"{p90:.1f} min",
            help="90% of historical flights had a delay less than or equal to this value."
        )

    st.markdown("---")


    st.markdown(f"### 📋 Closest Historical Analogs ({k_eff} Events Used for Analysis)")
    nbrs_disp = nbrs[["Year", "Month", "DayofMonth", "DayOfWeek", "DepTime", "TotalDelay", "Group", "distance"]].copy()
    nbrs_disp["Date"] = nbrs_disp.apply(lambda r: f"{int(r['Year'])}-{int(r['Month']):02d}-{int(r['DayofMonth']):02d}",
                                        axis=1)

    nbrs_disp["DayOfWeek"] = nbrs_disp["DayOfWeek"].map({1: "Monday", 2: "Tuesday", 3: "Wednesday",
                                                         4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"})
    nbrs_disp["Scheduled Dep Time"] = nbrs_disp["DepTime"].apply(lambda t: format_time(int(t)))
    st.dataframe(nbrs_disp[["Date", "DayOfWeek", "Scheduled Dep Time", "TotalDelay", "Group"]],
                 use_container_width=True, hide_index=True)

    try:
        o = df_geo[df_geo["Code"] == forecast_origin].iloc[0]
        d = df_geo[df_geo["Code"] == forecast_dest].iloc[0]
        arc_df = pd.DataFrame([{"src_lat": o["Latitude"], "src_lon": o["Longitude"],
                                "dest_lat": d["Latitude"], "dest_lon": d["Longitude"], "color": [0, 128, 255, 200]}])
        arc_layer = pdk.Layer("ArcLayer", data=arc_df,
                              get_source_position=["src_lon", "src_lat"],
                              get_target_position=["dest_lon", "dest_lat"],
                              get_source_color="color", get_target_color="color",
                              get_width=6)
        scatter_df = pd.DataFrame([
            {"lat": o["Latitude"], "lon": o["Longitude"], "label": "Origin"},
            {"lat": d["Latitude"], "lon": d["Longitude"], "label": "Destination"}
        ])
        scatter_layer = pdk.Layer("ScatterplotLayer", data=scatter_df,
                                  get_position=["lon", "lat"],
                                  get_radius=25000, get_fill_color=[0, 0, 0, 120])
        view_state = pdk.ViewState(latitude=(o["Latitude"] + d["Latitude"]) / 2,
                                   longitude=(o["Longitude"] + d["Longitude"]) / 2, zoom=4, pitch=40)
        st.markdown("### 🗺️ Route Map")
        st.pydeck_chart(pdk.Deck(layers=[arc_layer, scatter_layer],
                                 initial_view_state=view_state,
                                 map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                                 height=500))
    except Exception as e:
        st.warning(f"Map unavailable: {e}")

    st.markdown("---")
    st.markdown("### How Prediction Works")
    st.markdown(f"""
    - **Data Preparation**: Flight data is aggregated to create a training dataset with features like day-of-month, day-of-week, month, and scheduled departure hour.
    - **Model Training**: A K-Nearest Neighbors (KNN) regression model is used to find the **k most similar historical flights** (the "neighbors", k={k_eff}) based on the date and time features you selected.
    - **Prediction**: Instead of just the average, we analyze the delays of these {k_eff} neighbors to determine a **range of potential outcomes (P10 to P90)**.
        - **P50 (Median)**: Half of the similar historical flights had a delay less than this value. This is the most likely outcome.
        - **P10 (Best Case)**: 10% of the flights had a delay less than or equal to this.
        - **P90 (Worst Case)**: 90% of the flights had a delay less than or equal to this, providing a high-confidence worst-case estimate.
    - **Why No Flights on Some Routes?** Some routes may have limited historical data for specific date/time combinations. If you select a very specific date or a rare departure time, there might not be enough historical analogs for that combination.
    """)

# ========== TAB 4: XGBoost Multi-Percentile Forecast Pro (Replace existing tab4 code) ==========
# ========== TAB 4: XGBoost Multi-Percentile Forecast Pro (Replace existing tab4 code) ==========
with tab4:
    st.info(
        "XGBoost models trained on compressed route/time aggregates with delay driver features. P10, P50, P90 predictions with feature importance analysis.")

    import xgboost as xgb
    from datetime import datetime, timedelta


    # ---- Load models once (P10, P50, P90) ----
    @st.cache_resource
    def load_xgb_models_multipercentile():
        models = {}
        percentiles = ['p10', 'p50', 'p90']
        for percentile in percentiles:
            try:
                model = xgb.XGBRegressor()
                model.load_model(f"xgb_flight_delay_model_{percentile}.json")
                models[percentile] = model
            except Exception as e:
                st.error(f"Could not load XGBoost model for {percentile}: {e}")
                return None
        return models


    models = load_xgb_models_multipercentile()
    if models is None:
        st.stop()


    # ---- Load precomputed inference lookup (Parquet) ----
    @st.cache_data(show_spinner="Loading precomputed XGBoost features...")
    def load_inference_lookup_v2():
        try:
            df = pd.read_parquet(INFERENCE_LOOKUP_PATH)
        except Exception as e:
            st.error(
                "Could not load 'inference_lookup.parquet'. "
                "Make sure it is in the same folder as st.py. "
                f"Error: {e}"
            )
            return None

        # Normalize key columns to string for safe matching
        for col in ["carrier_code", "origin_id", "dest_id"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df


    lookup_df = load_inference_lookup_v2()
    if lookup_df is None or lookup_df.empty:
        st.stop()

    # ---- Filters (mirroring tab3 structure) ----
    st.markdown("### Filters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        xgb_date = st.date_input(
            "Date",
            value=datetime.now().date() + timedelta(days=7),
            format="MM/DD/YYYY",
            key="xgb_date_multi",
        )

    xgb_month = xgb_date.month
    xgb_day = xgb_date.day
    xgb_dow = xgb_date.isoweekday()


    @st.cache_data(ttl=600)
    def get_airports_for_xgb_multi(month, dow):
        q_orig = f"""
            SELECT DISTINCT CAST(OriginAirportID AS VARCHAR) as code 
            FROM flights 
            WHERE OriginAirportID IS NOT NULL
              AND Month = {month}
              AND {DOW_SQL} = {dow}
        """
        df_orig = conn.execute(q_orig).fetchdf()
        return sorted(df_orig["code"].astype(str).unique().tolist())


    @st.cache_data(ttl=600)
    def get_dests_for_xgb_multi(origin_id, month, dow):
        q_dests = f"""
            SELECT DISTINCT CAST(DestAirportID AS VARCHAR) as code 
            FROM flights 
            WHERE CAST(OriginAirportID AS VARCHAR) = '{origin_id}'
              AND DestAirportID IS NOT NULL
              AND Month = {month}
              AND {DOW_SQL} = {dow}
        """
        df_dests = conn.execute(q_dests).fetchdf()
        return sorted(df_dests["code"].astype(str).unique().tolist())


    origins_all = get_airports_for_xgb_multi(xgb_month, xgb_dow)
    origin_map = {
        c: f"{c} - {df_geo.loc[df_geo['Code'] == c, 'airport_name'].iloc[0] if c in df_geo['Code'].values else 'Unknown'}"
        for c in origins_all
    }

    if not origin_map:
        st.error(
            f"No historical flights found that operate on a {xgb_date.strftime('%A')} in {xgb_date.strftime('%B')}. "
            "Please try a different date."
        )
        st.stop()

    with col2:
        sel_origin_disp_xgb = st.selectbox("Origin Airport", list(origin_map.values()), key="xgb_origin_multi")
    xgb_origin = [k for k, v in origin_map.items() if v == sel_origin_disp_xgb][0]

    dests_all = get_dests_for_xgb_multi(xgb_origin, xgb_month, xgb_dow)
    dest_map = {
        c: f"{c} - {df_geo.loc[df_geo['Code'] == c, 'airport_name'].iloc[0] if c in df_geo['Code'].values else 'Unknown'}"
        for c in dests_all
    }

    if not dest_map:
        st.error(
            f"No destinations found from {sel_origin_disp_xgb} that operate on a "
            f"{xgb_date.strftime('%A')} in {xgb_date.strftime('%B')}."
        )
        st.stop()

    with col3:
        sel_dest_disp_xgb = st.selectbox("Destination Airport", list(dest_map.values()), key="xgb_dest_multi")
    xgb_dest = [k for k, v in dest_map.items() if v == sel_dest_disp_xgb][0]


    @st.cache_data(ttl=600)
    def get_airlines_for_xgb_route_multi(origin_id, dest_id, month, dow):
        avail_air_q = f"""
            SELECT DISTINCT CAST(DOT_ID_Reporting_Airline AS VARCHAR) as code
            FROM flights
            WHERE CAST(OriginAirportID AS VARCHAR) = '{origin_id}'
              AND CAST(DestAirportID AS VARCHAR) = '{dest_id}'
              AND Month = {month}
              AND {DOW_SQL} = {dow}
            ORDER BY code
        """
        return conn.execute(avail_air_q).fetchdf()["code"].astype(str).unique().tolist()


    airlines_avail_xgb = get_airlines_for_xgb_route_multi(xgb_origin, xgb_dest, xgb_month, xgb_dow)
    air_display_xgb = {f"{a} - {airline_lookup.get(a, 'Unknown')}": a for a in airlines_avail_xgb}

    if not air_display_xgb:
        st.error(
            f"No historical airlines found for this route that operate on a "
            f"{xgb_date.strftime('%A')} in {xgb_date.strftime('%B')}."
        )
        st.stop()

    with col4:
        sel_display_airline_xgb = st.selectbox("Airline", list(air_display_xgb.keys()), key="xgb_airline_multi")
    xgb_airline = air_display_xgb[sel_display_airline_xgb]

    q_avail_hours_xgb = f"""
        SELECT DISTINCT CAST(COALESCE(CRSDepTime,0) AS INTEGER) / 100 AS DepHour
        FROM flights
        WHERE CAST(OriginAirportID AS VARCHAR) = '{xgb_origin}'
          AND CAST(DestAirportID AS VARCHAR) = '{xgb_dest}'
          AND CAST(DOT_ID_Reporting_Airline AS VARCHAR) = '{xgb_airline}'
          AND Month = {xgb_month}
          AND {DOW_SQL} = {xgb_dow}
          AND CRSDepTime IS NOT NULL
        ORDER BY DepHour
    """
    df_avail_hours_xgb = conn.execute(q_avail_hours_xgb).fetchdf()

    if df_avail_hours_xgb.empty:
        st.error(
            f"No historical flights found for {sel_display_airline_xgb} on this route. "
            "Try a different date/airline."
        )
        st.stop()

    available_hours_xgb = sorted(list(set([int(h) for h in df_avail_hours_xgb["DepHour"].unique() if pd.notna(h)])))
    hour_displays_xgb = [f"{h:02d}:00" for h in available_hours_xgb]

    col1t, col2t = st.columns([2, 1])
    with col1t:
        hour_idx_xgb = st.selectbox(
            "Planned Departure Time (Hour)",
            range(len(hour_displays_xgb)),
            format_func=lambda i: hour_displays_xgb[i],
            key="xgb_time_multi",
        )
        xgb_dep_hour = available_hours_xgb[hour_idx_xgb]
        xgb_time_int = xgb_dep_hour * 100

    st.markdown(f"**Route:** {sel_origin_disp_xgb} → {sel_dest_disp_xgb}")
    st.markdown(f"**Date/Time:** {xgb_date.strftime('%A, %B %d, %Y')} at {format_time(xgb_time_int)}")
    st.markdown("---")

    # ========== GET FEATURES FOR INFERENCE (PRE-COMPUTED LOOKUP) ==========

    # Calculate time_of_day_category (1-4) - MUST MATCH TRAINING LOGIC
    time_of_day_cat = 1 if xgb_dep_hour < 6 else (2 if xgb_dep_hour < 12 else (3 if xgb_dep_hour < 17 else 4))


    def get_inference_features_from_lookup_multi(origin_id, dest_id, carrier_code, month, dow, time_of_day_cat):
        """
        Fetch precomputed aggregated features for a single future flight
        from inference_lookup.parquet using:
        (carrier, origin, dest, month, dow, time_of_day_cat).
        Includes P10, P50, P90 delay driver percentiles.
        """
        if lookup_df is None or lookup_df.empty:
            return None

        origin_str = str(origin_id)
        dest_str = str(dest_id)
        carrier_str = str(carrier_code)

        # First try full key: carrier + route + month + dow + time_of_day
        subset = lookup_df[
            (lookup_df["carrier_code"] == carrier_str)
            & (lookup_df["origin_id"] == origin_str)
            & (lookup_df["dest_id"] == dest_str)
            & (lookup_df["month"] == month)
            & (lookup_df["dow"] == dow)
            & (lookup_df["time_of_day_category"] == time_of_day_cat)
            ]

        # Fallback 1: relax DOW, keep carrier/route/month/time_of_day
        if subset.empty:
            subset = lookup_df[
                (lookup_df["carrier_code"] == carrier_str)
                & (lookup_df["origin_id"] == origin_str)
                & (lookup_df["dest_id"] == dest_str)
                & (lookup_df["month"] == month)
                & (lookup_df["time_of_day_category"] == time_of_day_cat)
                ]

        # Fallback 2: just carrier + route + time_of_day
        if subset.empty:
            subset = lookup_df[
                (lookup_df["carrier_code"] == carrier_str)
                & (lookup_df["origin_id"] == origin_str)
                & (lookup_df["dest_id"] == dest_str)
                & (lookup_df["time_of_day_category"] == time_of_day_cat)
                ]

        if subset.empty:
            return None

        row = subset.iloc[0]

        def safe_get(col, default):
            return float(row[col]) if col in row and pd.notna(row[col]) else float(default)

        return {
            "avg_crs_elapsed_time": safe_get("avg_crs_elapsed_time", 180.0),
            "avg_distance": safe_get("avg_distance", 1000.0),
            # P10 delay drivers
            "p10_carrier_delay": safe_get("hist_p10_carrier_delay", 0.0),
            "p10_weather_delay": safe_get("hist_p10_weather_delay", 0.0),
            "p10_nas_delay": safe_get("hist_p10_nas_delay", 0.0),
            "p10_security_delay": safe_get("hist_p10_security_delay", 0.0),
            "p10_late_aircraft_delay": safe_get("hist_p10_late_aircraft_delay", 0.0),
            # P50 delay drivers
            "p50_carrier_delay": safe_get("hist_p50_carrier_delay", 0.0),
            "p50_weather_delay": safe_get("hist_p50_weather_delay", 0.0),
            "p50_nas_delay": safe_get("hist_p50_nas_delay", 0.0),
            "p50_security_delay": safe_get("hist_p50_security_delay", 0.0),
            "p50_late_aircraft_delay": safe_get("hist_p50_late_aircraft_delay", 0.0),
            # P90 delay drivers
            "p90_carrier_delay": safe_get("hist_p90_carrier_delay", 0.0),
            "p90_weather_delay": safe_get("hist_p90_weather_delay", 0.0),
            "p90_nas_delay": safe_get("hist_p90_nas_delay", 0.0),
            "p90_security_delay": safe_get("hist_p90_security_delay", 0.0),
            "p90_late_aircraft_delay": safe_get("hist_p90_late_aircraft_delay", 0.0),
        }


    stats = get_inference_features_from_lookup_multi(
        xgb_origin, xgb_dest, xgb_airline, xgb_month, xgb_dow, time_of_day_cat
    )

    # Fallback values if no historical data in lookup
    if stats is None:
        stats = {
            "avg_crs_elapsed_time": 180.0,
            "avg_distance": 1000.0,
            "p10_carrier_delay": 0.0,
            "p10_weather_delay": 0.0,
            "p10_nas_delay": 0.0,
            "p10_security_delay": 0.0,
            "p10_late_aircraft_delay": 0.0,
            "p50_carrier_delay": 0.0,
            "p50_weather_delay": 0.0,
            "p50_nas_delay": 0.0,
            "p50_security_delay": 0.0,
            "p50_late_aircraft_delay": 0.0,
            "p90_carrier_delay": 0.0,
            "p90_weather_delay": 0.0,
            "p90_nas_delay": 0.0,
            "p90_security_delay": 0.0,
            "p90_late_aircraft_delay": 0.0,
        }


    # ---- Create feature vector matching trained model (25 features total) ----
    def create_features_xgb_multi(date, dep_hour):
        """Create feature vector with ALL percentiles' delay driver features.
        All three models (P10, P50, P90) were trained on the same 25-feature set.
        """
        m = date.month
        dow = date.isoweekday()

        # Derived features (same logic as training script)
        pct_weekday = 1.0 if 1 <= dow <= 5 else 0.0
        has_holiday_effect = 1.0 if (m, date.day) in [
            (1, 1), (7, 4), (12, 25), (12, 26),
        ] else 0.0
        has_peak_hour = 1.0 if (6 <= dep_hour < 9) or (16 <= dep_hour < 19) else 0.0

        distance = float(stats["avg_distance"])
        if distance < 250:
            distance_category = 1.0
        elif distance < 750:
            distance_category = 2.0
        elif distance < 1500:
            distance_category = 3.0
        else:
            distance_category = 4.0

        if m in [12, 1, 2]:
            season = 1.0  # Winter
        elif m in [3, 4, 5]:
            season = 2.0  # Spring
        elif m in [6, 7, 8]:
            season = 3.0  # Summer
        else:
            season = 4.0  # Fall

        # Include ALL three percentiles of each delay driver (P10, P50, P90)
        return pd.DataFrame(
            [
                {
                    "time_of_day_category": float(time_of_day_cat),
                    "Month": float(m),
                    "DayOfWeek": float(dow),
                    "avg_crs_elapsed_time": float(stats["avg_crs_elapsed_time"]),
                    "avg_distance": distance,
                    "pct_weekday": pct_weekday,
                    "has_holiday_effect": has_holiday_effect,
                    "has_peak_hour": has_peak_hour,
                    "distance_category": distance_category,
                    "season": season,
                    # P10, P50, P90 for carrier delay
                    "hist_p10_carrier_delay": float(stats["p10_carrier_delay"]),
                    "hist_p50_carrier_delay": float(stats["p50_carrier_delay"]),
                    "hist_p90_carrier_delay": float(stats["p90_carrier_delay"]),
                    # P10, P50, P90 for weather delay
                    "hist_p10_weather_delay": float(stats["p10_weather_delay"]),
                    "hist_p50_weather_delay": float(stats["p50_weather_delay"]),
                    "hist_p90_weather_delay": float(stats["p90_weather_delay"]),
                    # P10, P50, P90 for NAS delay
                    "hist_p10_nas_delay": float(stats["p10_nas_delay"]),
                    "hist_p50_nas_delay": float(stats["p50_nas_delay"]),
                    "hist_p90_nas_delay": float(stats["p90_nas_delay"]),
                    # P10, P50, P90 for security delay
                    "hist_p10_security_delay": float(stats["p10_security_delay"]),
                    "hist_p50_security_delay": float(stats["p50_security_delay"]),
                    "hist_p90_security_delay": float(stats["p90_security_delay"]),
                    # P10, P50, P90 for late aircraft delay
                    "hist_p10_late_aircraft_delay": float(stats["p10_late_aircraft_delay"]),
                    "hist_p50_late_aircraft_delay": float(stats["p50_late_aircraft_delay"]),
                    "hist_p90_late_aircraft_delay": float(stats["p90_late_aircraft_delay"]),
                }
            ]
        )


    # Make predictions for all three percentiles
    predictions = {}
    feature_importances = {}
    feature_dfs = {}
    shap_values = {}
    top_features_shap = []

    data_dict = {
            "time_of_day_category": 'Time of day category (1=early morning, 4=evening)',
            "Month": 'Month of flight (1-12)',
            "DayOfWeek": 'Day of the Week (1=Mon, 7=Sun)',
            "avg_crs_elapsed_time": 'Average scheduled elapsed time (local time: hhmm) ',
            "avg_distance": 'Average distance between airports (miles)',
            "pct_weekday": 'Proportion of historical flights on weekdays (Mon-Fri)',
            "has_holiday_effect": 'Indicator for major holiday dates (e.g., Jan 1, Jul 4, Dec 25)',
            "has_peak_hour": 'Indicator for peak travel hours (6-9am, 4-7pm)',
            "distance_category": 'Distance category (1:<250mi, 2:250-750mi, 3:750-1500mi, 4:>1500mi)',
            "season": 'Season of the year (1=Winter, 2=Spring, 3=Summer, 4=Fall)',
            # P10, P50, P90 for carrier delay
            "hist_p10_carrier_delay": '10th percentile of historical carrier delays (minutes)',
            "hist_p50_carrier_delay": '50th percentile (median) of historical carrier delays (minutes)',
            "hist_p90_carrier_delay": '90th percentile of historical carrier delays (minutes)',
            # P10, P50, P90 for weather delay
            "hist_p10_weather_delay": '10th percentile of historical weather delays (minutes)',
            "hist_p50_weather_delay": '50th percentile (median) of historical weather delays (minutes)',
            "hist_p90_weather_delay": '90th percentile of historical weather delays (minutes)',
            # P10, P50, P90 for NAS delay
            "hist_p10_nas_delay": '10th percentile of historical NAS delays (minutes)',
            "hist_p50_nas_delay": '50th percentile (median) of historical NAS delays (minutes)',
            "hist_p90_nas_delay": '90th percentile of historical NAS delays (minutes)',
            # P10, P50, P90 for security delay
            "hist_p10_security_delay": '10th percentile of historical security delays (minutes)',
            "hist_p50_security_delay": '50th percentile (median) of historical security delays (minutes)',
            "hist_p90_security_delay": '90th percentile of historical security delays (minutes)',
            # P10, P50, P90 for late aircraft delay
            "hist_p10_late_aircraft_delay": '10th percentile of historical late aircraft delays (minutes)',
            "hist_p50_late_aircraft_delay": '50th percentile (median) of historical late aircraft delays (minutes)',
            "hist_p90_late_aircraft_delay": '90th percentile of historical late aircraft delays (minutes)',
        }
    
    @st.cache_resource
    def get_shap_explainer(_model):
        return shap.Explainer(_model)

    try:
        # Create features once (same for all three percentiles)
        X = create_features_xgb_multi(xgb_date, xgb_dep_hour)

        for percentile in ['10', '50', '90']:
            feature_dfs[percentile] = X
            y_pred_log = models[f'p{percentile}'].predict(X)[0]
            predictions[percentile] = max(0, np.expm1(y_pred_log))

            # Get feature importance for this model
            importances = models[f'p{percentile}'].get_booster().get_score(importance_type='weight')
            feature_importances[percentile] = sorted(importances.items(), key=lambda x: x[1], reverse=True)

            # Get SHAP values for this model
            explainer = get_shap_explainer(models[f'p{percentile}'])
            shap_values[percentile] = explainer(X)
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        st.stop()

    # ---- Display predictions ----
    st.markdown("### 🎯 XGBoost Multi-Percentile Delay Predictions (Minutes)")
    col1, col2, col3 = st.columns(3)

    p10_pred = predictions['10']
    p50_pred = predictions['50']
    p90_pred = predictions['90']

    with col1:
        st.metric(
            "P10 (Best Case)",
            f"{p10_pred:.1f} min",
            help="10th percentile: optimistic scenario"
        )

    with col2:
        status_median = "✅ On-Time Expected" if p50_pred < 10 else (
            "⚠️ Minor Delay Expected" if p50_pred < 30 else "🔴 Significant Delay Expected")
        st.metric(
            "P50 (Median)",
            f"{p50_pred:.1f} min",
            delta=status_median,
            delta_color="off"
        )
        st.markdown(f"**Status:** {status_median}")

    with col3:
        st.metric(
            "P90 (Worst Case)",
            f"{p90_pred:.1f} min",
            help="90th percentile: worst-case scenario"
        )

    st.markdown("---")

    # ---- Feature Importance Analysis for each percentile ----
    st.markdown("### 🔍 Top Feature Contributions (Tree-based Importance)")

    tabs_importance = st.tabs(["P10 Features", "P50 Features", "P90 Features"])

    for idx, percentile in enumerate(['10', '50', '90']):
        with tabs_importance[idx]:
            st.markdown(f"**Prediction:** {predictions[percentile]:.1f} minutes")
            st.markdown("Top features that most influence this percentile's prediction (based on tree splits):")

            # Get the original explanation for instance 0

            # Indices of top 10 features
            importances = feature_importances[percentile][:10]

            if importances:
                imp_df = pd.DataFrame(importances, columns=["Feature", "Importance"])

                fig_imp = px.bar(
                    imp_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    height=350,
                    color="Importance",
                    color_continuous_scale="Blues",
                    title=f"Top 10 Features for P{percentile}"
                )
                fig_imp.update_layout(
                    showlegend=False,
                    xaxis_title="Importance Score",
                    yaxis_title="",
                    margin=dict(l=180, r=50, t=60, b=40),
                    font=dict(size=11),
                )
                st.plotly_chart(fig_imp, use_container_width=True, key=f"barplot-{percentile}")
            else:
                st.info("No feature importance data available for this model.")

            if shap_values[percentile]:
                shap_instance = shap_values[percentile][0]
                top_10_idx = np.argsort(np.abs(shap_instance.values))[-10:]
                top_10_idx_sorted = np.sort(top_10_idx)
                
                top_features_shap = X.columns[top_10_idx_sorted]
                shap_vals = np.round(shap_instance.values[top_10_idx_sorted], 2)
                base_value = shap_instance.base_values

                fig_waterfall = go.Figure(go.Waterfall(
                    orientation="h",
                    measure=["relative"] * len(shap_vals),
                    x=shap_vals,
                    y=top_features_shap,
                    base=base_value,
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": "#3D9970"}},
                    decreasing={"marker": {"color": "#FF4136"}},
                    customdata=np.stack([top_features_shap, shap_vals], axis=-1),
                    hovertemplate="<b>%{customdata[0]}</b><br>SHAP Value: %{customdata[1]:.2f}<extra></extra>",
                ))

                fig_waterfall.update_layout(
                    title=f"Feature Contributions toward Prediction Value",
                    xaxis_title="Feature",
                    yaxis_title="SHAP Value (Impact on Prediction)",
                    height=400,
                    margin=dict(l=180, r=50, t=60, b=40),
                    font=dict(size=11),
                    showlegend=False
                )

                st.plotly_chart(fig_waterfall, use_container_width=True, key=f"shap-waterfall-{percentile}")
                st.markdown("""
                    **Feature Influence Explanation**

                    - Each bar in the waterfall plot shows how much that feature contributed to the predicted delay for this flight scenario.
                    - Bars to the right (positive values) mean the feature increased the predicted delay.
                    - Bars to the left (negative values) mean the feature decreased the predicted delay.
                    - The numbers in the plot are in the model’s log-delay space.
                    - The final predicted delay is calculated by combining all these contributions and converting to minutes.
                """)
            else:
                st.info("No SHAP values available for this model.")

            st.markdown("---")

            st.markdown("### ⭐️ Top Feature Descriptions:")

            importance_set = set([feat[0] for feat in feature_importances[percentile][:10]])
            shap_set = set(top_features_shap)
            combined_features = importance_set.union(shap_set)

            df_top_features = pd.DataFrame({
                "Feature": list(combined_features),
                "Description": [data_dict.get(f, "No description available.") for f in combined_features]
            })

            st.dataframe(df_top_features.rename(columns={"Feature": "Feature Name", "Description": "Description"}),
            use_container_width=True, hide_index=True)

    st.markdown("### Route Information")
    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.metric("Route Distance", f"{stats['avg_distance']:.0f} mi")

    with col_info2:
        st.metric("Average Flight Duration", f"{stats['avg_crs_elapsed_time']:.0f} min")

    st.markdown("---")
    st.caption("✅ Model Performance: XGBoost trained on compressed aggregates with P10, P50, P90 delay drivers")
    st.markdown("**Features Used (15):** temporal, distance, and precomputed delay-driver percentiles")
    st.markdown(
        "**Feature Importance:** Shows which variables have the most influence on delay predictions for each percentile model.")


