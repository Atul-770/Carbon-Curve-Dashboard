[🌐 Open the hosted Carbon Curve Dashboard](https://carbon-curve-dashboard-ypzlwu44pz6dzrrx3kcngk.streamlit.app/)

# AI & The Planet: Carbon Curve Dashboard

The Carbon Curve Dashboard is a web-based decision-support tool developed as part of an MSc Data Science dissertation. It combines half-hourly UK data-centre demand profiles from UK Power Networks (UKPN) with electricity-grid carbon-intensity data from the National Energy System Operator (NESO).

The dashboard allows users to explore demand and carbon-intensity patterns, estimate scenario-based carbon emissions, and compare an original workload period with lower-carbon alternative periods.

## Problem Statement

Artificial intelligence and other data-centre workloads require substantial electricity, but the carbon emissions associated with that electricity are not constant throughout the day. They change as the mix of renewable, nuclear and fossil-fuel generation supplying the electricity grid changes.

Although UKPN publishes data-centre demand profiles and NESO provides historical and forecast carbon-intensity data, these datasets are published separately and serve different purposes. This makes it difficult for students, sustainability teams and non-specialist decision makers to understand how the timing of a flexible computing workload may affect its estimated carbon emissions.

This project addresses the problem by providing a transparent dashboard that integrates the two data sources and supports baseline-versus-carbon-aware timing comparisons. Because the UKPN dataset provides utilisation ratios rather than the actual maximum electrical capacity of each data centre, the resulting electricity-use and emissions values are scenario-based estimates rather than measurements of identifiable facilities.

## Project Aim

To design, develop and evaluate a web-based dashboard that integrates UK Power Networks’ half-hourly data-centre demand profiles with NESO’s Carbon Intensity API to estimate and visualise time-varying carbon emissions for AI workload scenarios, and to support carbon-aware timing comparisons.

## Project Objectives

1. To examine the relationship between data-centre electricity demand, time-varying grid carbon intensity and the carbon emissions associated with AI workloads.

2. To collect, clean and standardise UKPN half-hourly data-centre demand profiles and NESO carbon-intensity data.

3. To integrate the two datasets using a consistent UTC half-hourly timestamp structure.

4. To implement a transparent emissions-estimation model based on electricity consumption and grid carbon intensity.

5. To develop an interactive Streamlit dashboard for exploring utilisation, carbon intensity and estimated emissions.

6. To provide scenario controls for assumed capacity, workload duration, flexible workload percentage and shifting window.

7. To compare a selected baseline workload period with valid lower-carbon alternative periods and calculate the potential emissions savings.

8. To evaluate the dashboard through data-validation checks, functional testing and scenario-based analysis.



## Data Sources

- **UK Power Networks (UKPN):** Anonymised half-hourly data-centre demand and utilisation profiles.
- **National Energy System Operator (NESO):** Historical and forecast electricity-grid carbon-intensity data.

## Important Interpretation Note

The dashboard is intended for educational, exploratory and decision-support purposes. Its results depend on user-selected assumptions, including data-centre capacity and workload flexibility. The estimates should therefore not be interpreted as verified emissions for a specific real-world data centre.
