
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Carbon Curve Dashboard",
    page_icon="🌍",
    layout="wide"
)


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

APP_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = APP_FOLDER / "data"

UKPN_PATH = DATA_FOLDER / "ukpn_cleaned_full.parquet"
NESO_PATH = DATA_FOLDER / "neso_cleaned_full.parquet"


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data():
    ukpn = pd.read_parquet(
        UKPN_PATH,
        columns=[
            "cleansed_voltage_level",
            "anonymised_data_centre_name",
            "dc_type",
            "utc_timestamp",
            "hh_utilisation_ratio"
        ]
    )

    neso = pd.read_parquet(
        NESO_PATH,
        columns=[
            "utc_timestamp",
            "forecast",
            "actual",
            "index",
            "ci_g",
            "ci_source"
        ]
    )

    ukpn["utc_timestamp"] = pd.to_datetime(
        ukpn["utc_timestamp"],
        errors="coerce",
        utc=True
    )

    neso["utc_timestamp"] = pd.to_datetime(
        neso["utc_timestamp"],
        errors="coerce",
        utc=True
    )

    ukpn = ukpn.dropna(
        subset=[
            "utc_timestamp",
            "hh_utilisation_ratio",
            "dc_type",
            "cleansed_voltage_level",
            "anonymised_data_centre_name"
        ]
    ).copy()

    neso = neso.dropna(
        subset=[
            "utc_timestamp",
            "ci_g"
        ]
    ).copy()

    ukpn = ukpn.drop_duplicates().copy()

    neso = (
        neso.drop_duplicates(
            subset=["utc_timestamp"]
        )
        .sort_values("utc_timestamp")
        .reset_index(drop=True)
    )

    return ukpn, neso


# ---------------------------------------------------------
# CATEGORY PROFILE CREATION
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def create_category_profiles(ukpn):
    category_totals = (
        ukpn.groupby(
            [
                "dc_type",
                "cleansed_voltage_level"
            ]
        )["anonymised_data_centre_name"]
        .nunique()
        .rename("total_profiles_in_category")
        .reset_index()
    )

    category_profiles = (
        ukpn.groupby(
            [
                "dc_type",
                "cleansed_voltage_level",
                "utc_timestamp"
            ]
        )
        .agg(
            mean_utilisation_ratio=(
                "hh_utilisation_ratio",
                "mean"
            ),
            median_utilisation_ratio=(
                "hh_utilisation_ratio",
                "median"
            ),
            utilisation_std=(
                "hh_utilisation_ratio",
                "std"
            ),
            available_profiles=(
                "anonymised_data_centre_name",
                "nunique"
            )
        )
        .reset_index()
    )

    category_profiles = category_profiles.merge(
        category_totals,
        on=[
            "dc_type",
            "cleansed_voltage_level"
        ],
        how="left",
        validate="many_to_one"
    )

    category_profiles["profile_coverage"] = (
        category_profiles["available_profiles"] /
        category_profiles["total_profiles_in_category"]
    )

    category_profiles = category_profiles.sort_values(
        [
            "dc_type",
            "cleansed_voltage_level",
            "utc_timestamp"
        ]
    ).reset_index(drop=True)

    return category_profiles, category_totals


# ---------------------------------------------------------
# OVERALL SUMMARY CREATION
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def create_overview(ukpn, neso):
    timestamp_summary = (
        ukpn.groupby("utc_timestamp")
        .agg(
            mean_utilisation=(
                "hh_utilisation_ratio",
                "mean"
            ),
            median_utilisation=(
                "hh_utilisation_ratio",
                "median"
            ),
            active_profiles=(
                "anonymised_data_centre_name",
                "nunique"
            ),
            observations=(
                "hh_utilisation_ratio",
                "size"
            )
        )
        .reset_index()
    )

    timestamp_summary = timestamp_summary.merge(
        neso[
            [
                "utc_timestamp",
                "ci_g"
            ]
        ],
        on="utc_timestamp",
        how="left",
        validate="one_to_one"
    )

    timestamp_summary[
        "normalised_emissions_kg_per_1mw"
    ] = (
        timestamp_summary["mean_utilisation"] *
        1000 *
        0.5 *
        timestamp_summary["ci_g"] /
        1000
    )

    timestamp_summary["date"] = (
        timestamp_summary["utc_timestamp"]
        .dt.floor("D")
    )

    daily_summary = (
        timestamp_summary.groupby("date")
        .agg(
            mean_utilisation=(
                "mean_utilisation",
                "mean"
            ),
            median_utilisation=(
                "median_utilisation",
                "mean"
            ),
            mean_carbon_intensity=(
                "ci_g",
                "mean"
            ),
            mean_normalised_emissions=(
                "normalised_emissions_kg_per_1mw",
                "mean"
            ),
            mean_active_profiles=(
                "active_profiles",
                "mean"
            )
        )
        .reset_index()
    )

    type_summary = (
        ukpn.groupby("dc_type")
        .agg(
            profiles=(
                "anonymised_data_centre_name",
                "nunique"
            ),
            observations=(
                "hh_utilisation_ratio",
                "size"
            ),
            mean_utilisation=(
                "hh_utilisation_ratio",
                "mean"
            ),
            median_utilisation=(
                "hh_utilisation_ratio",
                "median"
            )
        )
        .reset_index()
    )

    type_summary["mean_utilisation_percent"] = (
        type_summary["mean_utilisation"] * 100
    )

    voltage_summary = (
        ukpn.groupby("cleansed_voltage_level")
        .agg(
            profiles=(
                "anonymised_data_centre_name",
                "nunique"
            ),
            observations=(
                "hh_utilisation_ratio",
                "size"
            ),
            mean_utilisation=(
                "hh_utilisation_ratio",
                "mean"
            ),
            median_utilisation=(
                "hh_utilisation_ratio",
                "median"
            )
        )
        .reset_index()
    )

    voltage_summary["mean_utilisation_percent"] = (
        voltage_summary["mean_utilisation"] * 100
    )

    return (
        daily_summary,
        type_summary,
        voltage_summary
    )


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
    minimum_coverage=0.80
):
    number_of_intervals = int(duration_hours * 2)
    capacity_kw = capacity_mw * 1000

    baseline_start = pd.Timestamp(baseline_start)

    if baseline_start.tzinfo is None:
        baseline_start = baseline_start.tz_localize("UTC")
    else:
        baseline_start = baseline_start.tz_convert("UTC")

    category_series = (
        category_data
        .set_index("utc_timestamp")
        .sort_index()
    )

    baseline_times = pd.date_range(
        start=baseline_start,
        periods=number_of_intervals,
        freq="30min",
        tz="UTC"
    )

    baseline = category_series.reindex(
        baseline_times
    )

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

    if (
        baseline["profile_coverage"] <
        minimum_coverage
    ).any():
        minimum_actual_coverage = (
            baseline["profile_coverage"].min() * 100
        )

        raise ValueError(
            "The selected baseline period does not meet the "
            f"{minimum_coverage * 100:.0f}% category-coverage "
            "requirement. Its minimum coverage is "
            f"{minimum_actual_coverage:.1f}%."
        )

    baseline_ci = carbon_series.reindex(
        baseline_times
    )

    if baseline_ci.isna().any():
        raise ValueError(
            "The selected baseline period contains missing "
            "NESO carbon-intensity values."
        )

    utilisation = (
        baseline["mean_utilisation_ratio"]
        .to_numpy()
    )

    baseline_energy = (
        utilisation *
        capacity_kw *
        0.5
    )

    fixed_energy = (
        baseline_energy *
        (1 - flexible_fraction)
    )

    flexible_energy = (
        baseline_energy *
        flexible_fraction
    )

    baseline_ci_values = baseline_ci.to_numpy()

    baseline_emissions = (
        baseline_energy *
        baseline_ci_values /
        1000
    ).sum()

    fixed_emissions = (
        fixed_energy *
        baseline_ci_values /
        1000
    ).sum()

    candidate_starts = pd.date_range(
        start=baseline_start,
        end=baseline_start + pd.Timedelta(
            hours=shifting_window_hours
        ),
        freq="30min",
        tz="UTC"
    )

    candidate_results = []

    for candidate_start in candidate_starts:
        candidate_times = pd.date_range(
            start=candidate_start,
            periods=number_of_intervals,
            freq="30min",
            tz="UTC"
        )

        candidate_ci = carbon_series.reindex(
            candidate_times
        )

        if candidate_ci.isna().any():
            continue

        shifted_flexible_emissions = (
            flexible_energy *
            candidate_ci.to_numpy() /
            1000
        ).sum()

        shifted_total_emissions = (
            fixed_emissions +
            shifted_flexible_emissions
        )

        reduction_kg = (
            baseline_emissions -
            shifted_total_emissions
        )

        if baseline_emissions > 0:
            reduction_percentage = (
                reduction_kg /
                baseline_emissions *
                100
            )
        else:
            reduction_percentage = 0

        delay_hours = (
            candidate_start -
            baseline_start
        ).total_seconds() / 3600

        candidate_results.append({
            "Candidate start":
                candidate_start,
            "Delay (hours)":
                delay_hours,
            "Baseline emissions (kgCO₂e)":
                baseline_emissions,
            "Shifted emissions (kgCO₂e)":
                shifted_total_emissions,
            "Reduction (kgCO₂e)":
                reduction_kg,
            "Reduction (%)":
                reduction_percentage
        })

    results = pd.DataFrame(
        candidate_results
    )

    if results.empty:
        raise ValueError(
            "No complete candidate periods were available "
            "within the selected shifting window."
        )

    best_result = results.loc[
        results[
            "Shifted emissions (kgCO₂e)"
        ].idxmin()
    ]

    baseline_output = pd.DataFrame({
        "UTC timestamp":
            baseline_times,
        "Mean utilisation ratio":
            utilisation,
        "Median utilisation ratio":
            baseline[
                "median_utilisation_ratio"
            ].to_numpy(),
        "Profile coverage":
            baseline[
                "profile_coverage"
            ].to_numpy(),
        "Available profiles":
            baseline[
                "available_profiles"
            ].to_numpy(),
        "Estimated energy (kWh)":
            baseline_energy,
        "Carbon intensity (gCO₂e/kWh)":
            baseline_ci_values,
        "Estimated emissions (kgCO₂e)":
            (
                baseline_energy *
                baseline_ci_values /
                1000
            )
    })

    return (
        best_result,
        results,
        baseline_output
    )


# ---------------------------------------------------------
# LOAD AND PREPARE DATA
# ---------------------------------------------------------

st.title(
    "AI & The Planet: Carbon Curve Dashboard"
)

st.write(
    "Explore representative data-centre categories, "
    "time-varying grid carbon intensity and potential "
    "emissions reductions from carbon-aware workload timing."
)

st.info(
    "Results are scenario-based estimates. UK Power Networks "
    "publishes utilisation ratios but does not disclose each "
    "site's maximum import capacity. The dashboard therefore "
    "uses representative category profiles and a user-defined "
    "reference capacity."
)

try:
    with st.spinner(
        "Loading UKPN and NESO datasets..."
    ):
        ukpn, neso = load_data()

    with st.spinner(
        "Creating representative category profiles..."
    ):
        (
            category_profiles,
            category_totals
        ) = create_category_profiles(ukpn)

except FileNotFoundError as error:
    st.error(
        "A required dataset could not be found. Confirm that "
        "the UKPN and NESO Parquet files are inside the "
        "dashboard/data folder."
    )

    st.code(str(error))
    st.stop()


carbon_series = (
    neso.set_index("utc_timestamp")["ci_g"]
    .sort_index()
)


# ---------------------------------------------------------
# DASHBOARD TABS
# ---------------------------------------------------------

overview_tab, scenario_tab, methodology_tab = st.tabs(
    [
        "Overall Overview",
        "Category-Based Scenario",
        "Method and Limitations"
    ]
)


# ---------------------------------------------------------
# OVERALL OVERVIEW TAB
# ---------------------------------------------------------

with overview_tab:
    st.header(
        "Overall data-centre overview"
    )

    with st.spinner(
        "Preparing overall summary..."
    ):
        (
            daily_overview,
            type_summary,
            voltage_summary
        ) = create_overview(
            ukpn,
            neso
        )

    common_start = max(
        ukpn["utc_timestamp"].min(),
        neso["utc_timestamp"].min()
    )

    common_end = min(
        ukpn["utc_timestamp"].max(),
        neso["utc_timestamp"].max()
    )

    expected_neso_intervals = len(
        pd.date_range(
            start=common_start,
            end=common_end,
            freq="30min",
            tz="UTC"
        )
    )

    neso_availability = (
        len(neso) /
        expected_neso_intervals *
        100
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Data-centre profiles",
        f"{ukpn['anonymised_data_centre_name'].nunique():,}"
    )

    metric2.metric(
        "UKPN observations",
        f"{len(ukpn):,}"
    )

    metric3.metric(
        "Mean utilisation",
        f"{ukpn['hh_utilisation_ratio'].mean() * 100:.2f}%"
    )

    metric4.metric(
        "NESO availability",
        f"{neso_availability:.2f}%"
    )

    st.caption(
        "UKPN coverage: "
        f"{ukpn['utc_timestamp'].min().strftime('%d %B %Y')} "
        "to "
        f"{ukpn['utc_timestamp'].max().strftime('%d %B %Y')}"
    )

    overview_chart = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True
                }
            ]
        ]
    )

    overview_chart.add_trace(
        go.Scatter(
            x=daily_overview["date"],
            y=(
                daily_overview[
                    "mean_utilisation"
                ] * 100
            ),
            name="Mean utilisation",
            mode="lines"
        ),
        secondary_y=False
    )

    overview_chart.add_trace(
        go.Scatter(
            x=daily_overview["date"],
            y=daily_overview[
                "mean_carbon_intensity"
            ],
            name="Mean carbon intensity",
            mode="lines"
        ),
        secondary_y=True
    )

    overview_chart.update_layout(
        title=(
            "Daily mean utilisation and "
            "grid carbon intensity"
        ),
        hovermode="x unified",
        legend_title_text="Metric"
    )

    overview_chart.update_yaxes(
        title_text="Mean utilisation (%)",
        secondary_y=False
    )

    overview_chart.update_yaxes(
        title_text=(
            "Carbon intensity "
            "(gCO₂e/kWh)"
        ),
        secondary_y=True
    )

    st.plotly_chart(
        overview_chart,
        use_container_width=True
    )

    left_column, right_column = st.columns(2)

    with left_column:
        type_chart = px.bar(
            type_summary,
            x="dc_type",
            y="mean_utilisation_percent",
            color="dc_type",
            text_auto=".2f",
            title=(
                "Mean utilisation by "
                "data-centre type"
            ),
            labels={
                "dc_type":
                    "Data-centre type",
                "mean_utilisation_percent":
                    "Mean utilisation (%)"
            }
        )

        type_chart.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            type_chart,
            use_container_width=True
        )

        st.dataframe(
            type_summary[
                [
                    "dc_type",
                    "profiles",
                    "observations",
                    "mean_utilisation_percent"
                ]
            ].rename(
                columns={
                    "dc_type":
                        "Data-centre type",
                    "profiles":
                        "Profiles",
                    "observations":
                        "Observations",
                    "mean_utilisation_percent":
                        "Mean utilisation (%)"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    with right_column:
        voltage_chart = px.bar(
            voltage_summary,
            x="cleansed_voltage_level",
            y="mean_utilisation_percent",
            color="cleansed_voltage_level",
            text_auto=".2f",
            title=(
                "Mean utilisation by "
                "voltage level"
            ),
            labels={
                "cleansed_voltage_level":
                    "Voltage level",
                "mean_utilisation_percent":
                    "Mean utilisation (%)"
            }
        )

        voltage_chart.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            voltage_chart,
            use_container_width=True
        )

        st.dataframe(
            voltage_summary[
                [
                    "cleansed_voltage_level",
                    "profiles",
                    "observations",
                    "mean_utilisation_percent"
                ]
            ].rename(
                columns={
                    "cleansed_voltage_level":
                        "Voltage level",
                    "profiles":
                        "Profiles",
                    "observations":
                        "Observations",
                    "mean_utilisation_percent":
                        "Mean utilisation (%)"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    st.subheader(
        "Normalised emissions trend"
    )

    emissions_chart = px.line(
        daily_overview,
        x="date",
        y="mean_normalised_emissions",
        title=(
            "Daily mean normalised emissions "
            "per half-hourly interval"
        ),
        labels={
            "date":
                "Date",
            "mean_normalised_emissions":
                "kgCO₂e per interval per 1 MW"
        }
    )

    st.plotly_chart(
        emissions_chart,
        use_container_width=True
    )

    with st.expander(
        "How to interpret the overall results"
    ):
        st.write(
            "The overview reports equal-weight mean utilisation "
            "across the available anonymised profiles. The "
            "utilisation ratios cannot be added to obtain total "
            "electricity demand because the maximum capacity of "
            "each site is not published."
        )

        st.write(
            "The normalised emissions trend uses a reference "
            "capacity of 1 MW. It supports temporal comparison "
            "but does not represent the actual combined "
            "emissions of all UKPN data centres."
        )


# ---------------------------------------------------------
# CATEGORY-BASED SCENARIO TAB
# ---------------------------------------------------------

with scenario_tab:
    st.header(
        "Representative category-based scenario"
    )

    st.write(
        "Select a data-centre type and voltage level. "
        "The workload is represented using the equal-weight "
        "mean utilisation of all available anonymised centres "
        "within that category."
    )

    st.sidebar.header(
        "Category scenario controls"
    )

    available_types = sorted(
        category_profiles[
            "dc_type"
        ].unique()
    )

    selected_type = st.sidebar.selectbox(
        "Data-centre type",
        available_types
    )

    available_voltage_levels = sorted(
        category_profiles.loc[
            category_profiles["dc_type"] ==
            selected_type,
            "cleansed_voltage_level"
        ].unique()
    )

    selected_voltage = st.sidebar.selectbox(
        "Voltage level",
        available_voltage_levels
    )

    selected_category = (
        category_profiles[
            (
                category_profiles["dc_type"] ==
                selected_type
            )
            &
            (
                category_profiles[
                    "cleansed_voltage_level"
                ] ==
                selected_voltage
            )
        ]
        .sort_values("utc_timestamp")
        .copy()
    )

    total_category_profiles = int(
        selected_category[
            "total_profiles_in_category"
        ].iloc[0]
    )

    available_dates = sorted(
        selected_category[
            "utc_timestamp"
        ].dt.date.unique()
    )

    selected_date = st.sidebar.selectbox(
        "Baseline date",
        available_dates,
        format_func=lambda value:
            value.strftime("%d %B %Y")
    )

    date_rows = selected_category[
        selected_category[
            "utc_timestamp"
        ].dt.date ==
        selected_date
    ].copy()

    available_times = (
        date_rows["utc_timestamp"]
        .dt.strftime("%H:%M")
        .tolist()
    )

    selected_time = st.sidebar.selectbox(
        "Baseline start time (UTC)",
        available_times
    )

    baseline_start = date_rows.loc[
        (
            date_rows[
                "utc_timestamp"
            ].dt.strftime("%H:%M")
            ==
            selected_time
        ),
        "utc_timestamp"
    ].iloc[0]

    duration_hours = st.sidebar.slider(
        "Workload duration (hours)",
        min_value=1,
        max_value=12,
        value=4,
        step=1
    )

    flexible_percentage = st.sidebar.slider(
        "Flexible workload (%)",
        min_value=10,
        max_value=30,
        value=20,
        step=5
    )

    shifting_window = st.sidebar.slider(
        "Maximum shifting window (hours)",
        min_value=12,
        max_value=48,
        value=24,
        step=12
    )

    capacity_mw = st.sidebar.number_input(
        "Assumed maximum capacity (MW)",
        min_value=0.1,
        max_value=1000.0,
        value=1.0,
        step=0.1
    )

    minimum_coverage_percentage = (
        st.sidebar.slider(
            "Minimum category coverage (%)",
            min_value=50,
            max_value=100,
            value=80,
            step=5
        )
    )

    selected_day_coverage = (
        date_rows["profile_coverage"].mean() *
        100
    )

    selected_day_mean_utilisation = (
        date_rows[
            "mean_utilisation_ratio"
        ].mean() *
        100
    )

    selected_day_median_utilisation = (
        date_rows[
            "median_utilisation_ratio"
        ].mean() *
        100
    )

    category_metric1, category_metric2, category_metric3, category_metric4 = (
        st.columns(4)
    )

    category_metric1.metric(
        "Category",
        selected_type
    )

    category_metric2.metric(
        "Voltage level",
        selected_voltage
    )

    category_metric3.metric(
        "Contributing profiles",
        total_category_profiles
    )

    category_metric4.metric(
        "Mean daily coverage",
        f"{selected_day_coverage:.1f}%"
    )

    st.caption(
        "Selected-day representative utilisation: "
        f"mean {selected_day_mean_utilisation:.2f}% | "
        f"median {selected_day_median_utilisation:.2f}%"
    )

    if st.button(
        "Calculate category scenario",
        type="primary",
        use_container_width=True
    ):
        try:
            (
                best,
                candidates,
                baseline_output
            ) = calculate_category_shift(
                category_data=selected_category,
                carbon_series=carbon_series,
                baseline_start=baseline_start,
                duration_hours=duration_hours,
                flexible_fraction=(
                    flexible_percentage / 100
                ),
                shifting_window_hours=(
                    shifting_window
                ),
                capacity_mw=capacity_mw,
                minimum_coverage=(
                    minimum_coverage_percentage /
                    100
                )
            )

            st.subheader(
                "Recommended carbon-aware result"
            )

            result1, result2, result3, result4 = (
                st.columns(4)
            )

            result1.metric(
                "Baseline emissions",
                (
                    f"{best['Baseline emissions (kgCO₂e)']:.3f} "
                    "kgCO₂e"
                )
            )

            result2.metric(
                "Shifted emissions",
                (
                    f"{best['Shifted emissions (kgCO₂e)']:.3f} "
                    "kgCO₂e"
                )
            )

            result3.metric(
                "Estimated reduction",
                f"{best['Reduction (%)']:.3f}%"
            )

            result4.metric(
                "Required delay",
                f"{best['Delay (hours)']:.1f} hours"
            )

            st.success(
                "Recommended start: "
                f"{best['Candidate start'].strftime('%d %B %Y, %H:%M UTC')}"
            )

            st.info(
                "This recommendation represents the selected "
                f"{selected_type} / {selected_voltage} category. "
                "It is not attributed to a specific identifiable "
                "data centre."
            )

            baseline_chart = make_subplots(
                specs=[
                    [
                        {
                            "secondary_y": True
                        }
                    ]
                ]
            )

            baseline_chart.add_trace(
                go.Scatter(
                    x=baseline_output[
                        "UTC timestamp"
                    ],
                    y=baseline_output[
                        "Estimated energy (kWh)"
                    ],
                    name="Estimated energy",
                    mode="lines+markers"
                ),
                secondary_y=False
            )

            baseline_chart.add_trace(
                go.Scatter(
                    x=baseline_output[
                        "UTC timestamp"
                    ],
                    y=baseline_output[
                        "Carbon intensity (gCO₂e/kWh)"
                    ],
                    name="Carbon intensity",
                    mode="lines+markers"
                ),
                secondary_y=True
            )

            baseline_chart.update_layout(
                title=(
                    "Representative workload and "
                    "carbon intensity"
                ),
                hovermode="x unified"
            )

            baseline_chart.update_yaxes(
                title_text=(
                    "Estimated energy (kWh)"
                ),
                secondary_y=False
            )

            baseline_chart.update_yaxes(
                title_text=(
                    "Carbon intensity "
                    "(gCO₂e/kWh)"
                ),
                secondary_y=True
            )

            st.plotly_chart(
                baseline_chart,
                use_container_width=True
            )

            utilisation_chart = go.Figure()

            utilisation_chart.add_trace(
                go.Scatter(
                    x=baseline_output[
                        "UTC timestamp"
                    ],
                    y=(
                        baseline_output[
                            "Mean utilisation ratio"
                        ] * 100
                    ),
                    name="Category mean",
                    mode="lines+markers"
                )
            )

            utilisation_chart.add_trace(
                go.Scatter(
                    x=baseline_output[
                        "UTC timestamp"
                    ],
                    y=(
                        baseline_output[
                            "Median utilisation ratio"
                        ] * 100
                    ),
                    name="Category median",
                    mode="lines+markers",
                    line={
                        "dash": "dash"
                    }
                )
            )

            utilisation_chart.update_layout(
                title=(
                    "Mean and median representative "
                    "utilisation"
                ),
                xaxis_title="UTC timestamp",
                yaxis_title="Utilisation (%)",
                hovermode="x unified"
            )

            st.plotly_chart(
                utilisation_chart,
                use_container_width=True
            )

            candidate_chart = px.line(
                candidates,
                x="Candidate start",
                y="Shifted emissions (kgCO₂e)",
                markers=True,
                title=(
                    "Estimated emissions across "
                    "eligible start times"
                )
            )

            candidate_chart.add_hline(
                y=best[
                    "Baseline emissions (kgCO₂e)"
                ],
                line_dash="dash",
                annotation_text="Baseline"
            )

            st.plotly_chart(
                candidate_chart,
                use_container_width=True
            )

            st.subheader(
                "Candidate-period results"
            )

            displayed_candidates = (
                candidates.sort_values(
                    "Shifted emissions (kgCO₂e)"
                )
                .copy()
            )

            st.dataframe(
                displayed_candidates,
                use_container_width=True,
                hide_index=True
            )

            csv_data = (
                displayed_candidates
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "Download category scenario as CSV",
                data=csv_data,
                file_name=(
                    "category_carbon_aware_results.csv"
                ),
                mime="text/csv"
            )

            with st.expander(
                "Scenario assumptions and data quality"
            ):
                st.write(
                    f"Data-centre type: {selected_type}"
                )

                st.write(
                    f"Voltage level: {selected_voltage}"
                )

                st.write(
                    "Profiles in category: "
                    f"{total_category_profiles}"
                )

                st.write(
                    "Minimum required category coverage: "
                    f"{minimum_coverage_percentage}%"
                )

                st.write(
                    f"Assumed capacity: {capacity_mw} MW"
                )

                st.write(
                    "Flexible workload: "
                    f"{flexible_percentage}%"
                )

                st.write(
                    "Workload duration: "
                    f"{duration_hours} hours"
                )

                st.write(
                    "Maximum shifting window: "
                    f"{shifting_window} hours"
                )

                st.write(
                    "The representative workload uses the "
                    "equal-weight mean utilisation of the "
                    "available anonymised profiles in the "
                    "selected category."
                )

                st.write(
                    "Actual NESO carbon intensity is preferred. "
                    "Forecast values are used only where actual "
                    "values are unavailable."
                )

                st.write(
                    "Candidate periods containing missing carbon "
                    "intensity are excluded."
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
For each data-centre type, voltage level and half-hourly
timestamp, the representative utilisation is calculated as:
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

- $\bar{U}_{c,t}$ is the mean utilisation for category $c$
  at time $t$;
- $U_{i,t}$ is the utilisation ratio of anonymised
  data centre $i$;
- $n_{c,t}$ is the number of available profiles in the
  category at that timestamp.
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
The value 0.5 represents the duration of one half-hourly
interval. When capacity is expressed in kW, the resulting
energy is expressed in kWh.
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
Carbon intensity is measured in gCO₂e/kWh. Dividing the
result by 1,000 converts grams of CO₂ equivalent into
kilograms of CO₂ equivalent.
"""
    )

    st.subheader("Carbon-aware comparison")

    st.markdown(
        """
The flexible share of the representative workload is tested
at every valid half-hourly start time within the selected
shifting window. Workload energy, workload duration and
assumed capacity remain unchanged. The dashboard recommends
the valid alternative start time that produces the lowest
estimated emissions.
"""
    )

    st.header("Interpretation and limitations")

    st.warning(
        "The results represent modelled scenarios and should "
        "not be interpreted as audited emissions for a specific facility."
    )

    st.markdown(
        """
- UKPN does not publish the identities or actual maximum
  import capacities of the data centres.
- Equal-weight averaging prevents centres with larger but
  unknown capacities from dominating the representative profile.
- The selected data-centre type and voltage level define a
  comparison category rather than a guaranteed forecast for a new site.
- A new data centre may have different hardware, cooling systems,
  operating hours and levels of workload flexibility.
- Utilisation ratios above 100% are retained as a documented
  limitation of the source data.
- Missing UKPN and NESO intervals are not interpolated.
- The model excludes embodied emissions, water consumption,
  financial cost optimisation and geographical workload migration.
- The dashboard does not automatically control or reschedule
  a live data-centre workload.
"""
    )

    st.header("Data sources")

    st.markdown(
        """
- **UK Power Networks:** anonymised half-hourly Data Centre
  Demand Profiles.
- **National Energy System Operator:** national half-hourly
  forecast and actual carbon-intensity data.
"""
    )
