import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import os
import warnings
import datetime
from pandas.errors import SettingWithCopyWarning

delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
st.set_page_config(page_title="Flight Route Visualizer (Pandas)", layout="wide")

FLIGHT_DATA_PATH = "data/2024_07_sample.csv.gz"
GEO_DATA_PATH = "flight_geo/L_AIRPORT_ID_with_Coordinates.csv"


@st.cache_data
def load_geo_data():
    if not os.path.exists(GEO_DATA_PATH):
        st.error(f"Geo data file not found at {GEO_DATA_PATH}")
        return pd.DataFrame()
    df_geo = pd.read_csv(GEO_DATA_PATH)
    df_geo = df_geo[["Code", "Airport Name", "Latitude", "Longitude"]]
    return df_geo


@st.cache_data(show_spinner="Loading flight data...")
def load_flight_data():
    if not os.path.exists(FLIGHT_DATA_PATH):
        st.error(f"Flight data file not found at {FLIGHT_DATA_PATH}")
        return pd.DataFrame()
    return pd.read_parquet(FLIGHT_DATA_PATH)


df_geo = load_geo_data()
if df_geo.empty:
    st.stop()

df_flights = load_flight_data()
if df_flights.empty:
    st.stop()


@st.cache_data(show_spinner="Finding date range...")
def get_min_max_dates(_df_flights):
    min_date = datetime.date(
        int(_df_flights["Year"].min()),
        int(_df_flights["Month"].min()),
        int(_df_flights["DayofMonth"].min()),
    )
    max_date = datetime.date(
        int(_df_flights["Year"].max()),
        int(_df_flights["Month"].max()),
        int(_df_flights["DayofMonth"].max()),
    )
    return min_date, max_date


min_date_available, max_date_available = get_min_max_dates(df_flights)


@st.cache_data(show_spinner="Loading airports for date...")
def get_airports_for_date(selected_date, _df_flights, _df_geo):
    """Get only airports that have flights on the selected date"""
    df_date = _df_flights[
        (_df_flights["Year"] == selected_date.year)
        & (_df_flights["Month"] == selected_date.month)
        & (_df_flights["DayofMonth"] == selected_date.day)
        ]

    origin_ids = df_date["OriginAirportID"].unique()
    df_origins = _df_geo[_df_geo["Code"].isin(origin_ids)].copy()
    df_origins["display"] = (
            df_origins["Code"].astype(str) + " - " + df_origins["Airport Name"].fillna("Unknown")
    )
    return dict(zip(df_origins["Code"], df_origins["display"]))


@st.cache_data(show_spinner="Loading destinations...")
def get_destinations_for_origin(origin_id, selected_date, _df_flights, _df_geo):
    """Get destinations for a specific origin on the selected date"""
    df_f = _df_flights[
        (_df_flights["OriginAirportID"] == origin_id)
        & (_df_flights["Year"] == selected_date.year)
        & (_df_flights["Month"] == selected_date.month)
        & (_df_flights["DayofMonth"] == selected_date.day)
        ]

    valid_dest_ids = df_f["DestAirportID"].unique()
    df_dests = _df_geo[_df_geo["Code"].isin(valid_dest_ids)].copy()
    df_dests["display"] = (
            df_dests["Code"].astype(str) + " - " + df_dests["Airport Name"].fillna("Unknown")
    )
    return dict(zip(df_dests["Code"], df_dests["display"]))


@st.cache_data(show_spinner="Loading departure times...")
def get_available_dep_times(origin_id, dest_ids, selected_date, _df_flights):
    df_f = _df_flights[
        (_df_flights["OriginAirportID"] == origin_id)
        & (_df_flights["Year"] == selected_date.year)
        & (_df_flights["Month"] == selected_date.month)
        & (_df_flights["DayofMonth"] == selected_date.day)
        ]

    if dest_ids and dest_ids != ["ALL"]:
        df_f = df_f[df_f["DestAirportID"].isin([int(d) for d in dest_ids])]

    times = df_f["CRSDepTime"].dropna().astype(int).unique()
    return sorted(times.tolist())


def format_time(time_int):
    if pd.isna(time_int) or time_int is None or time_int < 0 or time_int > 2400:
        return "N/A"
    try:
        s = str(int(time_int)).zfill(4)
        return f"{s[:2]}:{s[2:]}"
    except ValueError:
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
def get_routes_for_selection(origin_id, dest_ids, selected_date, delay_filter, time_filter, _df_flights, _geo):
    df_s = _df_flights[
        (_df_flights["OriginAirportID"] == origin_id)
        & (_df_flights["Year"] == selected_date.year)
        & (_df_flights["Month"] == selected_date.month)
        & (_df_flights["DayofMonth"] == selected_date.day)
        ].copy()

    if dest_ids and dest_ids != ["ALL"]:
        df_s = df_s[df_s["DestAirportID"].isin([int(d) for d in dest_ids])]
    if delay_filter == "Delayed":
        df_s = df_s[(df_s["ArrDel15"] == 1) | (df_s["DepDel15"] == 1)]
    elif delay_filter == "On-Time":
        df_s = df_s[(df_s["ArrDel15"] != 1) & (df_s["DepDel15"] != 1)]

    if time_filter != "ALL":
        df_s = df_s[df_s["CRSDepTime"] == int(time_filter)]

    df = df_s.copy()
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()

    df = df.rename(
        columns={
            "Flight_Number_Reporting_Airline": "fl_num",
            "OriginAirportID": "origin",
            "DestAirportID": "dest",
            "CRSDepTime": "crs_dep_time_raw",
        }
    )

    for col_name in delay_cols:
        if col_name not in df.columns:
            df[col_name] = 0

    df["is_delay"] = ((df["ArrDel15"] == 1) | (df["DepDel15"] == 1)).astype(int)
    df["crs_dep_time"] = df["crs_dep_time_raw"].apply(format_time)
    df["crs_dep_time_sort"] = df["crs_dep_time_raw"].fillna(0).astype(int)
    df = df.sort_values("crs_dep_time_sort").reset_index(drop=True)

    df = df.merge(
        _geo.rename(
            columns={
                "Code": "origin",
                "Airport Name": "orig_name",
                "Latitude": "src_lat",
                "Longitude": "src_lon",
            }
        ),
        on="origin",
        how="left",
    )
    df = df.merge(
        _geo.rename(
            columns={
                "Code": "dest",
                "Airport Name": "dest_name",
                "Latitude": "dest_lat",
                "Longitude": "dest_lon",
            }
        ),
        on="dest",
        how="left",
    )

    agg = df.groupby(
        [
            "origin",
            "orig_name",
            "src_lat",
            "src_lon",
            "dest",
            "dest_name",
            "dest_lat",
            "dest_lon",
        ],
        as_index=False,
    ).agg(total_flights=("fl_num", "count"), delayed_count=("is_delay", "sum"))

    agg["delay_ratio"] = agg["delayed_count"] / agg["total_flights"]
    agg["pct_delayed"] = (agg["delay_ratio"] * 100).round(1).astype(str) + "%"
    agg["color"] = agg["delay_ratio"].apply(delay_ratio_to_color)

    arcs = agg[
        [
            "src_lon",
            "src_lat",
            "dest_lon",
            "dest_lat",
            "orig_name",
            "dest_name",
            "total_flights",
            "delayed_count",
            "pct_delayed",
            "color",
        ]
    ].copy()

    origins = (
        agg[["orig_name", "src_lat", "src_lon", "delay_ratio"]]
        .drop_duplicates()
        .rename(columns={"orig_name": "airport", "src_lat": "lat", "src_lon": "lon"})
    )
    origins["type"] = "Origin"
    destinations = (
        agg[["dest_name", "dest_lat", "dest_lon", "delay_ratio"]]
        .drop_duplicates()
        .rename(columns={"dest_name": "airport", "dest_lat": "lat", "dest_lon": "lon"})
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

    return df, arcs, points


st.title("✈️ RouteLens")
st.subheader("Data-driven Flight Delay Prediction with Explainable AI")

tab1 = st.tabs(["📊 History Actuals"])[0]

with tab1:
    st.markdown("### Filters")

    filter_cols = st.columns([1.5, 1.5, 1.5, 1.5, 1.5, 0.5])

    with filter_cols[0]:
        selected_date = st.date_input(
            "Date",
            value=min_date_available,
            min_value=min_date_available,
            max_value=max_date_available,
            format="MM/DD/YYYY",
        )

    airport_id_to_display = get_airports_for_date(selected_date, df_flights, df_geo)

    if not airport_id_to_display:
        st.warning(f"No flights available on {selected_date.strftime('%Y-%m-%d')}")
        st.stop()

    origin_display_names = list(airport_id_to_display.values())

    with filter_cols[1]:
        selected_display_origin = st.selectbox(
            "Origin",
            origin_display_names,
            label_visibility="visible"
        )

    selected_origin_id = [
        k for k, v in airport_id_to_display.items() if v == selected_display_origin
    ][0]

    valid_dest_options = get_destinations_for_origin(selected_origin_id, selected_date, df_flights, df_geo)
    destination_display_names = list(valid_dest_options.values())

    with filter_cols[2]:
        selected_display_dests = st.multiselect(
            "Destination(s)",
            options=destination_display_names,
            default=None,
            label_visibility="visible"
        )

    if selected_display_dests:
        selected_dest_ids = [
            k for k, v in valid_dest_options.items() if v in selected_display_dests
        ]
    else:
        selected_dest_ids = ["ALL"]

    available_times = get_available_dep_times(selected_origin_id, selected_dest_ids, selected_date, df_flights)
    time_options = ["ALL"] + [format_time(t) for t in available_times]
    time_values = ["ALL"] + [str(t) for t in available_times]

    with filter_cols[3]:
        time_display = st.selectbox(
            "Dep Time",
            options=range(len(time_options)),
            format_func=lambda i: time_options[i],
            label_visibility="visible"
        )

    time_filter = time_values[time_display]

    with filter_cols[4]:
        delay_filter = st.selectbox("Delay Status", ["ALL", "On-Time", "Delayed"], label_visibility="visible")

    df_raw, arcs, points = get_routes_for_selection(
        selected_origin_id, selected_dest_ids, selected_date, delay_filter, time_filter, df_flights, df_geo
    )

    if df_raw.empty:
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
        get_width=4,
        pickable=True,
        auto_highlight=True,
    )

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position=["lon", "lat"],
        get_radius=20000,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=df_raw["src_lat"].iloc[0]
        if not df_raw["src_lat"].empty
        else 40.7,
        longitude=df_raw["src_lon"].iloc[0]
        if not df_raw["src_lon"].empty
        else -100.0,
        zoom=5,
        pitch=45,
    )

    r = pdk.Deck(
        layers=[arc_layer, scatter_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v10",
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

    st.markdown("---")

    st.subheader("🗓️ Heatmap View")
    st.markdown(f"**Origin Airport:** {selected_display_origin}")

    heatmap_data = df_raw.groupby(['dest_name', 'crs_dep_time']).agg({
        'is_delay': ['sum', 'count']
    }).reset_index()
    heatmap_data.columns = ['Destination', 'Time', 'Delayed', 'Total']
    heatmap_data['Delay_Ratio'] = heatmap_data['Delayed'] / heatmap_data['Total']

    pivot_data = heatmap_data.pivot(index='Destination', columns='Time', values='Delay_Ratio')

    num_destinations = len(pivot_data.index)
    cell_height = 50 if num_destinations == 1 else 35
    chart_height = max(400, num_destinations * cell_height + 200)

    fig = px.imshow(
        pivot_data,
        labels=dict(x="Scheduled Departure Time", y="Destination Airport", color="Delay Ratio"),
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale=['green', 'yellow', 'red'],
        aspect="auto",
        zmin=0,
        zmax=1
    )

    fig.update_layout(
        height=chart_height,
        xaxis_title="Scheduled Departure Time",
        yaxis_title="Destination Airport",
        coloraxis_colorbar=dict(
            title="Delay<br>Ratio",
            tickvals=[0, 0.5, 1],
            ticktext=['On-Time', 'Mixed', 'Delayed'],
            titlefont=dict(size=14),
            tickfont=dict(size=12)
        ),
        xaxis=dict(
            side="bottom",
            title="Scheduled Departure Time",
            titlefont=dict(size=16),
            tickfont=dict(size=14)
        ),
        xaxis2=dict(
            side="top",
            overlaying="x",
            matches="x",
            showticklabels=True,
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            titlefont=dict(size=16),
            tickfont=dict(size=14)
        ),
        font=dict(size=14)
    )

    PLOTLY_KEY = "heatmap_selection"
    st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key=PLOTLY_KEY
    )

    st.markdown("---")

    st.markdown("### Heatmap Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Destinations", num_destinations)
    col2.metric("Unique Departure Times", len(pivot_data.columns))
    col3.metric("Total Flight Records", len(df_raw))

    df_filtered = df_raw.copy()
    filter_applied = False

    selection_data = st.session_state.get(PLOTLY_KEY)

    if selection_data and 'points' in selection_data:
        points_data = selection_data['points']
        if points_data:
            filter_applied = True
            selected_times = []
            selected_dests = []

            for point in points_data:
                if 'x' in point and 'y' in point:
                    selected_times.append(point['x'])
                    selected_dests.append(point['y'])

            if selected_times and selected_dests:
                df_filtered = df_raw[
                    (df_raw['crs_dep_time'].isin(selected_times)) &
                    (df_raw['dest_name'].isin(selected_dests))
                    ]
                st.info(f"🔍 Filtered to selected cells: {len(df_filtered)} flights")

    st.markdown("---")

    if st.checkbox("Show raw data table (First 500 rows)", value=False):
        display_cols = [
                           "fl_num",
                           "orig_name",
                           "dest_name",
                           "crs_dep_time",
                           "is_delay",
                       ] + delay_cols
        df_display = df_filtered[display_cols].copy()
        df_display["is_delay"] = df_display["is_delay"].map({0: "On-Time", 1: "Delayed"})

        st.dataframe(
            df_display.rename(
                columns={
                    "fl_num": "Flight #",
                    "orig_name": "Origin Airport",
                    "dest_name": "Destination Airport",
                    "crs_dep_time": "Scheduled Dep Time",
                    "is_delay": "Status",
                }
            )
            .head(500),
            column_config={
                "Flight #": st.column_config.TextColumn("Flight #")
            }
        )

    st.markdown("---")

    delay_data = df_filtered[delay_cols].fillna(0)
    has_delays = (delay_data > 0).any().any()

    if has_delays:
        st.subheader("📈 Average Delay Breakdown (Minutes)")

        if filter_applied:
            st.caption("ℹ️ Showing delay breakdown for selected cells in heatmap")
        else:
            st.caption("ℹ️ Showing delay breakdown based on sidebar filters")

        delay_averages = {}
        for col_name in delay_cols:
            non_zero_values = delay_data[col_name][delay_data[col_name] > 0]
            if len(non_zero_values) > 0:
                delay_averages[col_name] = non_zero_values.mean()

        if delay_averages:
            chart_df = pd.DataFrame({
                'Delay Type': [name.replace('Delay', '') for name in delay_averages.keys()],
                'Average Minutes': list(delay_averages.values())
            })

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
                    colorbar=dict(title="Minutes")
                )
            )

            fig_delay.update_layout(
                showlegend=False,
                xaxis_title="Average Delay (Minutes)",
                yaxis_title="",
                margin=dict(l=150, r=100, t=80, b=60),
                plot_bgcolor='rgba(240,240,240,0.5)',
                paper_bgcolor='white',
                font=dict(size=13),
                title_font_size=16,
                xaxis=dict(gridcolor='lightgray', showgrid=True),
                hovermode='closest'
            )

            st.plotly_chart(fig_delay, use_container_width=True)
    else:
        if filter_applied:
            st.info("No delays found in the selected cells.")
