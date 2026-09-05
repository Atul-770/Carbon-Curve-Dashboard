import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from plotly.subplots import make_subplots


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Carbon Curve Dashboard",
    page_icon="🌍",
    layout="wide",
)


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

APP_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = APP_FOLDER / "data"

UKPN_PATH = DATA_FOLDER / "ukpn_cleaned_full.parquet"
NESO_PATH = DATA_FOLDER / "neso_cleaned_full.parquet"


# ---------------------------------------------------------
# LOW-MEMORY DATA ACCESS
# ---------------------------------------------------------

def run_duckdb_query(query, parameters):
    """Run one read-only query and return a small Pandas DataFrame."""
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET threads = 2")
        return connection.execute(query, parameters).fetchdf()
    finally:
        connection.close()


@st.cache_data(show_spinner=False)
def load_neso(file_path):
    """Load the relatively small half-hourly NESO dataset."""
    neso = pd.read_parquet(
        file_path,
        columns=[
            "utc_timestamp",
            "forecast",
            "actual",
            "index",
            "ci_g",
            "ci_source",
        ],
    )

    neso["utc_timestamp"] = pd.to_datetime(
        neso["utc_timestamp"], errors="coerce", utc=True
    )
    neso["ci_g"] = pd.to_numeric(neso["ci_g"], errors="coerce")

    return (
        neso.dropna(subset=["utc_timestamp", "ci_g"])
        .drop_duplicates(subset=["utc_timestamp"])
        .sort_values("utc_timestamp")
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def load_ukpn_metadata(file_path):
    """Read overall metrics and category choices without loading raw rows."""
    summary_query = """
        SELECT
            COUNT(*) AS observations,
            COUNT(DISTINCT anonymised_data_centre_name) AS profiles,
            AVG(hh_utilisation_ratio) AS mean_utilisation,
            MIN(utc_timestamp) AS start_timestamp,
            MAX(utc_timestamp) AS end_timestamp
        FROM read_parquet(?)
        WHERE utc_timestamp IS NOT NULL
          AND hh_utilisation_ratio IS NOT NULL
          AND dc_type IS NOT NULL
          AND cleansed_voltage_level IS NOT NULL
          AND anonymised_data_centre_name IS NOT NULL
    """

    category_query = """
        SELECT
            dc_type,
            cleansed_voltage_level,
            COUNT(DISTINCT anonymised_data_centre_name)
                AS total_profiles_in_category
        FROM read_parquet(?)
        WHERE utc_timestamp IS NOT NULL
          AND hh_utilisation_ratio IS NOT NULL
          AND dc_type IS NOT NULL
          AND cleansed_voltage_level IS NOT NULL
          AND anonymised_data_centre_name IS NOT NULL
        GROUP BY dc_type, cleansed_voltage_level
        ORDER BY dc_type, cleansed_voltage_level
    """

    summary = run_duckdb_query(summary_query, [file_path]).iloc[0]
    categories = run_duckdb_query(category_query, [file_path])

    summary["start_timestamp"] = pd.to_datetime(
        summary["start_timestamp"], utc=True
    )
    summary["end_timestamp"] = pd.to_datetime(
        summary["end_timestamp"], utc=True
    )

    return summary, categories


@st.cache_data(show_spinner=False)
def load_overview(file_path, neso_file_path):
    """Create overview tables inside DuckDB to keep memory usage low."""
    daily_query = """
        WITH ukpn_timestamp AS (
            SELECT
                utc_timestamp,
                AVG(hh_utilisation_ratio) AS mean_utilisation,
                MEDIAN(hh_utilisation_ratio) AS median_utilisation,
                COUNT(DISTINCT anonymised_data_centre_name)
                    AS active_profiles
            FROM read_parquet(?)
            WHERE utc_timestamp IS NOT NULL
              AND hh_utilisation_ratio IS NOT NULL
              AND anonymised_data_centre_name IS NOT NULL
            GROUP BY utc_timestamp
        ),
        neso_timestamp AS (
            SELECT
                utc_timestamp,
                AVG(ci_g) AS ci_g
            FROM read_parquet(?)
            WHERE utc_timestamp IS NOT NULL
              AND ci_g IS NOT NULL
            GROUP BY utc_timestamp
        )
        SELECT
            CAST(date_trunc('day', u.utc_timestamp) AS TIMESTAMP)
                AS date,
            AVG(u.mean_utilisation) AS mean_utilisation,
            AVG(u.median_utilisation) AS median_utilisation,
            AVG(n.ci_g) AS mean_carbon_intensity,
            AVG(u.mean_utilisation * 0.5 * n.ci_g)
                AS mean_normalised_emissions,
            AVG(u.active_profiles) AS mean_active_profiles
        FROM ukpn_timestamp AS u
        LEFT JOIN neso_timestamp AS n
          ON u.utc_timestamp = n.utc_timestamp
        GROUP BY date_trunc('day', u.utc_timestamp)
        ORDER BY date
    """

    type_query = """
        SELECT
            dc_type,
            COUNT(DISTINCT anonymised_data_centre_name) AS profiles,
            COUNT(*) AS observations,
            AVG(hh_utilisation_ratio) AS mean_utilisation,
            MEDIAN(hh_utilisation_ratio) AS median_utilisation
        FROM read_parquet(?)
        WHERE dc_type IS NOT NULL
          AND anonymised_data_centre_name IS NOT NULL
          AND hh_utilisation_ratio IS NOT NULL
        GROUP BY dc_type
        ORDER BY dc_type
    """

    voltage_query = """
        SELECT
            cleansed_voltage_level,
            COUNT(DISTINCT anonymised_data_centre_name) AS profiles,
            COUNT(*) AS observations,
            AVG(hh_utilisation_ratio) AS mean_utilisation,
            MEDIAN(hh_utilisation_ratio) AS median_utilisation
        FROM read_parquet(?)
        WHERE cleansed_voltage_level IS NOT NULL
          AND anonymised_data_centre_name IS NOT NULL
          AND hh_utilisation_ratio IS NOT NULL
        GROUP BY cleansed_voltage_level
        ORDER BY cleansed_voltage_level
    """

    daily = run_duckdb_query(
        daily_query, [file_path, neso_file_path]
    )
    type_summary = run_duckdb_query(type_query, [file_path])
    voltage_summary = run_duckdb_query(voltage_query, [file_path])

    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    type_summary["mean_utilisation_percent"] = (
        type_summary["mean_utilisation"] * 100
    )
    voltage_summary["mean_utilisation_percent"] = (
        voltage_summary["mean_utilisation"] * 100
    )

    return daily, type_summary, voltage_summary


@st.cache_data(show_spinner=False)
def load_category_profile(file_path, dc_type, voltage_level):
    """Aggregate only the category selected by the dashboard user."""
    query = """
        WITH category_rows AS (
            SELECT
                utc_timestamp,
                hh_utilisation_ratio,
                anonymised_data_centre_name
            FROM read_parquet(?)
            WHERE dc_type = ?
              AND cleansed_voltage_level = ?
              AND utc_timestamp IS NOT NULL
              AND hh_utilisation_ratio IS NOT NULL
              AND anonymised_data_centre_name IS NOT NULL
        ),
        category_total AS (
            SELECT
                COUNT(DISTINCT anonymised_data_centre_name)
                    AS total_profiles_in_category
            FROM category_rows
        )
        SELECT
            r.utc_timestamp,
            AVG(r.hh_utilisation_ratio) AS mean_utilisation_ratio,
            MEDIAN(r.hh_utilisation_ratio) AS median_utilisation_ratio,
            STDDEV_SAMP(r.hh_utilisation_ratio) AS utilisation_std,
            COUNT(DISTINCT r.anonymised_data_centre_name)
                AS available_profiles,
            t.total_profiles_in_category
        FROM category_rows AS r
        CROSS JOIN category_total AS t
        GROUP BY r.utc_timestamp, t.total_profiles_in_category
        ORDER BY r.utc_timestamp
    """

    profile = run_duckdb_query(
        query, [file_path, dc_type, voltage_level]
    )

    if profile.empty:
        return profile

    profile["utc_timestamp"] = pd.to_datetime(
        profile["utc_timestamp"], errors="coerce", utc=True
    )
    profile["profile_coverage"] = (
        profile["available_profiles"]
        / profile["total_profiles_in_category"]
    )
    profile["dc_type"] = dc_type
    profile["cleansed_voltage_level"] = voltage_level

    return profile.dropna(subset=["utc_timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------
# CARBON-AWARE SHIFTING MODEL
# ---------------------------------------------------------

def calculate_category_shift(
    category_data,
    carbon_series,
    baseline_start,
    duration_hours,
    flexible_fraction,
    shifting_window_hours,
    capacity_mw,
    minimum_coverage=0.80,
):
    number_of_intervals = int(duration_hours * 2)
    capacity_kw = capacity_mw * 1000

    baseline_start = pd.Timestamp(baseline_start)
    if baseline_start.tzinfo is None:
        baseline_start = baseline_start.tz_localize("UTC")
    else:
        baseline_start = baseline_start.tz_convert("UTC")

    category_series = (
        category_data.set_index("utc_timestamp").sort_index()
    )
    baseline_times = pd.date_range(
        start=baseline_start,
        periods=number_of_intervals,
        freq="30min",
        tz="UTC",
    )
    baseline = category_series.reindex(baseline_times)

    if baseline["mean_utilisation_ratio"].isna().any():
        raise ValueError(
            "The selected baseline period contains missing "
            "UKPN category-profile intervals."
        )

    if baseline["profile_coverage"].isna().any():
        raise ValueError(
            "Category coverage could not be calculated for "
            "the selected period."
        )

    if (baseline["profile_coverage"] < minimum_coverage).any():
        minimum_actual_coverage = baseline["profile_coverage"].min() * 100
        raise ValueError(
            "The selected baseline period does not meet the "
            f"{minimum_coverage * 100:.0f}% category-coverage "
            "requirement. Its minimum coverage is "
            f"{minimum_actual_coverage:.1f}%."
        )

    baseline_ci = carbon_series.reindex(baseline_times)
    if baseline_ci.isna().any():
        raise ValueError(
            "The selected baseline period contains missing "
            "NESO carbon-intensity values."
        )

    utilisation = baseline["mean_utilisation_ratio"].to_numpy()
    baseline_energy = utilisation * capacity_kw * 0.5
    fixed_energy = baseline_energy * (1 - flexible_fraction)
    flexible_energy = baseline_energy * flexible_fraction
    baseline_ci_values = baseline_ci.to_numpy()

    baseline_emissions = (
        baseline_energy * baseline_ci_values / 1000
    ).sum()
    fixed_emissions = (
        fixed_energy * baseline_ci_values / 1000
    ).sum()

    candidate_starts = pd.date_range(
        start=baseline_start,
        end=baseline_start + pd.Timedelta(hours=shifting_window_hours),
        freq="30min",
        tz="UTC",
    )
    candidate_results = []

    for candidate_start in candidate_starts:
        candidate_times = pd.date_range(
            start=candidate_start,
            periods=number_of_intervals,
            freq="30min",
            tz="UTC",
        )
        candidate_ci = carbon_series.reindex(candidate_times)
        if candidate_ci.isna().any():
            continue

        shifted_flexible_emissions = (
            flexible_energy * candidate_ci.to_numpy() / 1000
        ).sum()
        shifted_total_emissions = fixed_emissions + shifted_flexible_emissions
        reduction_kg = baseline_emissions - shifted_total_emissions
        reduction_percentage = (
            reduction_kg / baseline_emissions * 100
            if baseline_emissions > 0
            else 0
        )
        delay_hours = (
            candidate_start - baseline_start
        ).total_seconds() / 3600

        candidate_results.append(
            {
                "Candidate start": candidate_start,
                "Delay (hours)": delay_hours,
                "Baseline emissions (kgCO₂e)": baseline_emissions,
                "Shifted emissions (kgCO₂e)": shifted_total_emissions,
                "Reduction (kgCO₂e)": reduction_kg,
                "Reduction (%)": reduction_percentage,
            }
        )

    results = pd.DataFrame(candidate_results)
    if results.empty:
        raise ValueError(
            "No complete candidate periods were available "
            "within the selected shifting window."
        )

    best_result = results.loc[
        results["Shifted emissions (kgCO₂e)"].idxmin()
    ]
    baseline_output = pd.DataFrame(
        {
            "UTC timestamp": baseline_times,
            "Mean utilisation ratio": utilisation,
            "Median utilisation ratio": baseline[
                "median_utilisation_ratio"
            ].to_numpy(),
            "Profile coverage": baseline["profile_coverage"].to_numpy(),
            "Available profiles": baseline["available_profiles"].to_numpy(),
            "Estimated energy (kWh)": baseline_energy,
            "Carbon intensity (gCO₂e/kWh)": baseline_ci_values,
            "Estimated emissions (kgCO₂e)": (
                baseline_energy * baseline_ci_values / 1000
            ),
        }
    )

    return best_result, results, baseline_output


# ---------------------------------------------------------
# LOAD ESSENTIAL DATA
# ---------------------------------------------------------

st.title("AI & The Planet: Carbon Curve Dashboard")
st.write(
    "Explore representative data-centre categories, time-varying grid "
    "carbon intensity and potential emissions reductions from "
    "carbon-aware workload timing."
)
st.info(
    "Results are scenario-based estimates. UK Power Networks publishes "
    "utilisation ratios but does not disclose each site's maximum import "
    "capacity. The dashboard therefore uses representative category "
    "profiles and a user-defined reference capacity."
)

missing_files = [
    str(path) for path in (UKPN_PATH, NESO_PATH) if not path.exists()
]
if missing_files:
    st.error(
        "A required dataset could not be found. Confirm that both Parquet "
        "files are inside the repository's data folder."
    )
    st.code("\n".join(missing_files))
    st.stop()

try:
    with st.spinner("Reading dataset information..."):
        neso = load_neso(str(NESO_PATH))
        ukpn_summary, category_options = load_ukpn_metadata(str(UKPN_PATH))
except Exception as error:
    st.error("The dashboard could not read the dataset files.")
    st.exception(error)
    st.stop()

carbon_series = neso.set_index("utc_timestamp")["ci_g"].sort_index()


# ---------------------------------------------------------
# DASHBOARD TABS
# ---------------------------------------------------------

overview_tab, scenario_tab, methodology_tab = st.tabs(
    [
        "Overall Overview",
        "Category-Based Scenario",
        "Method and Limitations",
    ]
)


# ---------------------------------------------------------
# OVERALL OVERVIEW TAB
# ---------------------------------------------------------

with overview_tab:
    st.header("Overall data-centre overview")

    with st.spinner("Preparing the overall summary..."):
        daily_overview, type_summary, voltage_summary = load_overview(
            str(UKPN_PATH), str(NESO_PATH)
        )

    common_start = max(
        ukpn_summary["start_timestamp"], neso["utc_timestamp"].min()
    )
    common_end = min(
        ukpn_summary["end_timestamp"], neso["utc_timestamp"].max()
    )
    expected_neso_intervals = len(
        pd.date_range(
            start=common_start, end=common_end, freq="30min", tz="UTC"
        )
    )
    neso_in_common_period = neso[
        neso["utc_timestamp"].between(common_start, common_end)
    ]
    neso_availability = (
        len(neso_in_common_period) / expected_neso_intervals * 100
        if expected_neso_intervals
        else 0
    )

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Data-centre profiles", f"{int(ukpn_summary['profiles']):,}")
    metric2.metric("UKPN observations", f"{int(ukpn_summary['observations']):,}")
    metric3.metric(
        "Mean utilisation",
        f"{float(ukpn_summary['mean_utilisation']) * 100:.2f}%",
    )
    metric4.metric("NESO availability", f"{neso_availability:.2f}%")

    st.caption(
        "UKPN coverage: "
        f"{ukpn_summary['start_timestamp'].strftime('%d %B %Y')} to "
        f"{ukpn_summary['end_timestamp'].strftime('%d %B %Y')}"
    )

    overview_chart = make_subplots(specs=[[{"secondary_y": True}]])
    overview_chart.add_trace(
        go.Scatter(
            x=daily_overview["date"],
            y=daily_overview["mean_utilisation"] * 100,
            name="Mean utilisation",
            mode="lines",
        ),
        secondary_y=False,
    )
    overview_chart.add_trace(
        go.Scatter(
            x=daily_overview["date"],
            y=daily_overview["mean_carbon_intensity"],
            name="Mean carbon intensity",
            mode="lines",
        ),
        secondary_y=True,
    )
    overview_chart.update_layout(
        title="Daily mean utilisation and grid carbon intensity",
        hovermode="x unified",
        legend_title_text="Metric",
    )
    overview_chart.update_yaxes(
        title_text="Mean utilisation (%)", secondary_y=False
    )
    overview_chart.update_yaxes(
        title_text="Carbon intensity (gCO₂e/kWh)", secondary_y=True
    )
    st.plotly_chart(overview_chart, width="stretch")

    left_column, right_column = st.columns(2)

    with left_column:
        type_chart = px.bar(
            type_summary,
            x="dc_type",
            y="mean_utilisation_percent",
            color="dc_type",
            text_auto=".2f",
            title="Mean utilisation by data-centre type",
            labels={
                "dc_type": "Data-centre type",
                "mean_utilisation_percent": "Mean utilisation (%)",
            },
        )
        type_chart.update_layout(showlegend=False)
        st.plotly_chart(type_chart, width="stretch")
        st.dataframe(
            type_summary[
                [
                    "dc_type",
                    "profiles",
                    "observations",
                    "mean_utilisation_percent",
                ]
            ].rename(
                columns={
                    "dc_type": "Data-centre type",
                    "profiles": "Profiles",
                    "observations": "Observations",
                    "mean_utilisation_percent": "Mean utilisation (%)",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with right_column:
        voltage_chart = px.bar(
            voltage_summary,
            x="cleansed_voltage_level",
            y="mean_utilisation_percent",
            color="cleansed_voltage_level",
            text_auto=".2f",
            title="Mean utilisation by voltage level",
            labels={
                "cleansed_voltage_level": "Voltage level",
                "mean_utilisation_percent": "Mean utilisation (%)",
            },
        )
        voltage_chart.update_layout(showlegend=False)
        st.plotly_chart(voltage_chart, width="stretch")
        st.dataframe(
            voltage_summary[
                [
                    "cleansed_voltage_level",
                    "profiles",
                    "observations",
                    "mean_utilisation_percent",
                ]
            ].rename(
                columns={
                    "cleansed_voltage_level": "Voltage level",
                    "profiles": "Profiles",
                    "observations": "Observations",
                    "mean_utilisation_percent": "Mean utilisation (%)",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Normalised emissions trend")
    emissions_chart = px.line(
        daily_overview,
        x="date",
        y="mean_normalised_emissions",
        title="Daily mean normalised emissions per half-hourly interval",
        labels={
            "date": "Date",
            "mean_normalised_emissions": "kgCO₂e per interval per 1 MW",
        },
    )
    st.plotly_chart(emissions_chart, width="stretch")

    with st.expander("How to interpret the overall results"):
        st.write(
            "The overview reports equal-weight mean utilisation across "
            "the available anonymised profiles. The utilisation ratios "
            "cannot be added to obtain total electricity demand because "
            "the maximum capacity of each site is not published."
        )
        st.write(
            "The normalised emissions trend uses a reference capacity of "
            "1 MW. It supports temporal comparison but does not represent "
            "the actual combined emissions of all UKPN data centres."
        )


# ---------------------------------------------------------
# CATEGORY-BASED SCENARIO TAB
# ---------------------------------------------------------

with scenario_tab:
    st.header("Representative category-based scenario")
    st.write(
        "Select a data-centre type and voltage level. The workload is "
        "represented using the equal-weight mean utilisation of all "
        "available anonymised centres within that category."
    )

    st.sidebar.header("Category scenario controls")
    available_types = sorted(category_options["dc_type"].dropna().tolist())
    available_types = sorted(set(available_types))
    selected_type = st.sidebar.selectbox("Data-centre type", available_types)

    available_voltage_levels = sorted(
        category_options.loc[
            category_options["dc_type"] == selected_type,
            "cleansed_voltage_level",
        ]
        .dropna()
        .unique()
        .tolist()
    )
    selected_voltage = st.sidebar.selectbox(
        "Voltage level", available_voltage_levels
    )

    with st.spinner("Loading the selected representative category..."):
        selected_category = load_category_profile(
            str(UKPN_PATH), selected_type, selected_voltage
        )

    if selected_category.empty:
        st.error("No valid observations were found for this category.")
        st.stop()

    total_category_profiles = int(
        selected_category["total_profiles_in_category"].iloc[0]
    )
    available_dates = sorted(selected_category["utc_timestamp"].dt.date.unique())
    selected_date = st.sidebar.selectbox(
        "Baseline date",
        available_dates,
        format_func=lambda value: value.strftime("%d %B %Y"),
    )

    date_rows = selected_category[
        selected_category["utc_timestamp"].dt.date == selected_date
    ].copy()
    available_times = (
        date_rows["utc_timestamp"].dt.strftime("%H:%M").drop_duplicates().tolist()
    )
    selected_time = st.sidebar.selectbox(
        "Baseline start time (UTC)", available_times
    )
    baseline_start = date_rows.loc[
        date_rows["utc_timestamp"].dt.strftime("%H:%M") == selected_time,
        "utc_timestamp",
    ].iloc[0]

    duration_hours = st.sidebar.slider(
        "Workload duration (hours)", 1, 12, 4, 1
    )
    flexible_percentage = st.sidebar.slider(
        "Flexible workload (%)", 10, 30, 20, 5
    )
    shifting_window = st.sidebar.slider(
        "Maximum shifting window (hours)", 12, 48, 24, 12
    )
    capacity_mw = st.sidebar.number_input(
        "Assumed maximum capacity (MW)",
        min_value=0.1,
        max_value=1000.0,
        value=1.0,
        step=0.1,
    )
    minimum_coverage_percentage = st.sidebar.slider(
        "Minimum category coverage (%)", 50, 100, 80, 5
    )

    selected_day_coverage = date_rows["profile_coverage"].mean() * 100
    selected_day_mean_utilisation = (
        date_rows["mean_utilisation_ratio"].mean() * 100
    )
    selected_day_median_utilisation = (
        date_rows["median_utilisation_ratio"].mean() * 100
    )

    category_metric1, category_metric2, category_metric3, category_metric4 = (
        st.columns(4)
    )
    category_metric1.metric("Category", selected_type)
    category_metric2.metric("Voltage level", selected_voltage)
    category_metric3.metric("Contributing profiles", total_category_profiles)
    category_metric4.metric(
        "Mean daily coverage", f"{selected_day_coverage:.1f}%"
    )
    st.caption(
        "Selected-day representative utilisation: "
        f"mean {selected_day_mean_utilisation:.2f}% | "
        f"median {selected_day_median_utilisation:.2f}%"
    )

    if st.button(
        "Calculate category scenario", type="primary", width="stretch"
    ):
        try:
            best, candidates, baseline_output = calculate_category_shift(
                category_data=selected_category,
                carbon_series=carbon_series,
                baseline_start=baseline_start,
                duration_hours=duration_hours,
                flexible_fraction=flexible_percentage / 100,
                shifting_window_hours=shifting_window,
                capacity_mw=capacity_mw,
                minimum_coverage=minimum_coverage_percentage / 100,
            )

            st.subheader("Recommended carbon-aware result")
            result1, result2, result3, result4 = st.columns(4)
            result1.metric(
                "Baseline emissions",
                f"{best['Baseline emissions (kgCO₂e)']:.3f} kgCO₂e",
            )
            result2.metric(
                "Shifted emissions",
                f"{best['Shifted emissions (kgCO₂e)']:.3f} kgCO₂e",
            )
            result3.metric(
                "Estimated reduction", f"{best['Reduction (%)']:.3f}%"
            )
            result4.metric(
                "Required delay", f"{best['Delay (hours)']:.1f} hours"
            )

            st.success(
                "Recommended start: "
                f"{best['Candidate start'].strftime('%d %B %Y, %H:%M UTC')}"
            )
            st.info(
                "This recommendation represents the selected "
                f"{selected_type} / {selected_voltage} category. It is not "
                "attributed to a specific identifiable data centre."
            )

            baseline_chart = make_subplots(specs=[[{"secondary_y": True}]])
            baseline_chart.add_trace(
                go.Scatter(
                    x=baseline_output["UTC timestamp"],
                    y=baseline_output["Estimated energy (kWh)"],
                    name="Estimated energy",
                    mode="lines+markers",
                ),
                secondary_y=False,
            )
            baseline_chart.add_trace(
                go.Scatter(
                    x=baseline_output["UTC timestamp"],
                    y=baseline_output["Carbon intensity (gCO₂e/kWh)"],
                    name="Carbon intensity",
                    mode="lines+markers",
                ),
                secondary_y=True,
            )
            baseline_chart.update_layout(
                title="Representative workload and carbon intensity",
                hovermode="x unified",
            )
            baseline_chart.update_yaxes(
                title_text="Estimated energy (kWh)", secondary_y=False
            )
            baseline_chart.update_yaxes(
                title_text="Carbon intensity (gCO₂e/kWh)", secondary_y=True
            )
            st.plotly_chart(baseline_chart, width="stretch")

            utilisation_chart = go.Figure()
            utilisation_chart.add_trace(
                go.Scatter(
                    x=baseline_output["UTC timestamp"],
                    y=baseline_output["Mean utilisation ratio"] * 100,
                    name="Category mean",
                    mode="lines+markers",
                )
            )
            utilisation_chart.add_trace(
                go.Scatter(
                    x=baseline_output["UTC timestamp"],
                    y=baseline_output["Median utilisation ratio"] * 100,
                    name="Category median",
                    mode="lines+markers",
                    line={"dash": "dash"},
                )
            )
            utilisation_chart.update_layout(
                title="Mean and median representative utilisation",
                xaxis_title="UTC timestamp",
                yaxis_title="Utilisation (%)",
                hovermode="x unified",
            )
            st.plotly_chart(utilisation_chart, width="stretch")

            candidate_chart = px.line(
                candidates,
                x="Candidate start",
                y="Shifted emissions (kgCO₂e)",
                markers=True,
                title="Estimated emissions across eligible start times",
            )
            candidate_chart.add_hline(
                y=best["Baseline emissions (kgCO₂e)"],
                line_dash="dash",
                annotation_text="Baseline",
            )
            st.plotly_chart(candidate_chart, width="stretch")

            st.subheader("Candidate-period results")
            displayed_candidates = candidates.sort_values(
                "Shifted emissions (kgCO₂e)"
            ).copy()
            st.dataframe(
                displayed_candidates, width="stretch", hide_index=True
            )
            csv_data = displayed_candidates.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download category scenario as CSV",
                data=csv_data,
                file_name="category_carbon_aware_results.csv",
                mime="text/csv",
            )

            with st.expander("Scenario assumptions and data quality"):
                st.write(f"Data-centre type: {selected_type}")
                st.write(f"Voltage level: {selected_voltage}")
                st.write(f"Profiles in category: {total_category_profiles}")
                st.write(
                    "Minimum required category coverage: "
                    f"{minimum_coverage_percentage}%"
                )
                st.write(f"Assumed capacity: {capacity_mw} MW")
                st.write(f"Flexible workload: {flexible_percentage}%")
                st.write(f"Workload duration: {duration_hours} hours")
                st.write(f"Maximum shifting window: {shifting_window} hours")
                st.write(
                    "The representative workload uses the equal-weight mean "
                    "utilisation of the available anonymised profiles in "
                    "the selected category."
                )
                st.write(
                    "Actual NESO carbon intensity is preferred. Forecast "
                    "values are used only where actual values are unavailable."
                )
                st.write(
                    "Candidate periods containing missing carbon intensity "
                    "are excluded."
                )

        except ValueError as error:
            st.error(str(error))


# ---------------------------------------------------------
# METHOD AND LIMITATIONS TAB
# ---------------------------------------------------------

with methodology_tab:
    st.header("Calculation method")
    st.subheader("Representative category profile")
    st.markdown(
        """
For each data-centre type, voltage level and half-hourly timestamp,
the representative utilisation is calculated as:
"""
    )
    st.latex(
        r"""
        \bar{U}_{c,t}
        =
        \frac{1}{n_{c,t}}
        \sum_{i=1}^{n_{c,t}} U_{i,t}
        """
    )
    st.markdown(
        r"""
where:

- $\bar{U}_{c,t}$ is the mean utilisation for category $c$ at time $t$;
- $U_{i,t}$ is the utilisation ratio of anonymised data centre $i$;
- $n_{c,t}$ is the number of available profiles in the category at that timestamp.
"""
    )

    st.subheader("Scenario energy")
    st.latex(
        r"""
        \mathrm{Energy}_{t}
        =
        \bar{U}_{c,t}
        \times
        \mathrm{Capacity}
        \times
        0.5
        """
    )
    st.markdown(
        """
The value 0.5 represents the duration of one half-hourly interval.
When capacity is expressed in kW, the resulting energy is expressed in kWh.
"""
    )

    st.subheader("Estimated emissions")
    st.latex(
        r"""
        \mathrm{Emissions}_{t}
        =
        \frac{
        \mathrm{Energy}_{t}
        \times
        \mathrm{CarbonIntensity}_{t}
        }{1000}
        """
    )
    st.markdown(
        """
Carbon intensity is measured in gCO₂e/kWh. Dividing the result by 1,000
converts grams of CO₂ equivalent into kilograms of CO₂ equivalent.
"""
    )

    st.subheader("Carbon-aware comparison")
    st.markdown(
        """
The flexible share of the representative workload is tested at every valid
half-hourly start time within the selected shifting window. Workload energy,
workload duration and assumed capacity remain unchanged. The dashboard
recommends the valid alternative start time that produces the lowest
estimated emissions.
"""
    )

    st.header("Interpretation and limitations")
    st.warning(
        "The results represent modelled scenarios and should not be "
        "interpreted as audited emissions for a specific facility."
    )
    st.markdown(
        """
- UKPN does not publish the identities or actual maximum import capacities
  of the data centres.
- Equal-weight averaging prevents centres with larger but unknown capacities
  from dominating the representative profile.
- The selected data-centre type and voltage level define a comparison
  category rather than a guaranteed forecast for a new site.
- A new data centre may have different hardware, cooling systems, operating
  hours and levels of workload flexibility.
- Utilisation ratios above 100% are retained as a documented limitation of
  the source data.
- Missing UKPN and NESO intervals are not interpolated.
- The model excludes embodied emissions, water consumption, financial cost
  optimisation and geographical workload migration.
- The dashboard does not automatically control or reschedule a live
  data-centre workload.
"""
    )

    st.header("Data sources")
    st.markdown(
        """
- **UK Power Networks:** anonymised half-hourly Data Centre Demand Profiles.
- **National Energy System Operator:** national half-hourly forecast and
  actual carbon-intensity data.
"""
    )

