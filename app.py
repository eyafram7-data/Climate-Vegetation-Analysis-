"""
=============================================================================
app.py — Streamlit Dashboard
Climate Change and Vegetation Dynamics Analysis
=============================================================================
PURPOSE:
    Interactive web dashboard built with Streamlit. Allows users to:
    - Explore climate and NDVI data interactively
    - Filter by region, date range, and variable
    - View model performance metrics
    - Run scenario forecasts on demand
    - Download data and results

WHAT IS STREAMLIT?
    Streamlit is a Python library that turns data science scripts into
    interactive web apps with minimal code. No HTML, CSS, or JavaScript
    knowledge required! Key components used:
    - st.sidebar: Left panel for controls and filters
    - st.tabs: Organize content into tabbed sections
    - st.plotly_chart: Render interactive Plotly charts
    - st.metric: Display KPI cards with delta indicators
    - st.dataframe: Scrollable, sortable data tables
    - st.selectbox / st.slider / st.multiselect: User input widgets
    - st_folium: Render Folium maps inside Streamlit

HOW TO RUN:
    streamlit run app.py

STRUCTURE:
    1. Sidebar Controls  — Filters and settings
    2. Tab 1: Overview   — KPI cards + climate overview
    3. Tab 2: NDVI       — NDVI trends and seasonal patterns
    4. Tab 3: Climate    — Temperature and precipitation deep dive
    5. Tab 4: ML Models  — Model training results and comparison
    6. Tab 5: Forecast   — Future predictions by scenario
    7. Tab 6: Map        — Interactive Folium map

=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# Add src/ to Python path so we can import our custom modules
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# Must be the FIRST Streamlit command in the script
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Climate-Vegetation Analysis",
    page_icon="🌿",
    layout="wide",                   # Use full browser width
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/Climate-Vegetation-Analysis",
        "Report a Bug": "https://github.com/yourusername/Climate-Vegetation-Analysis/issues",
        "About": "Climate Change and Vegetation Dynamics Analysis Dashboard v1.0"
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS STYLING
# Makes the dashboard look polished and professional
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F0F4F8; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #E0E7FF;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #1B4F72, #27AE60);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        text-align: center;
    }

    /* Section headers */
    .section-header {
        background: white;
        border-left: 4px solid #27AE60;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
        font-weight: 600;
        color: #2C3E50;
    }

    /* Info boxes */
    .info-box {
        background: #EBF5FB;
        border: 1px solid #AED6F1;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 14px;
        color: #1A5276;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #7F8C8D;
        font-size: 12px;
        padding: 16px;
        border-top: 1px solid #E8EAED;
        margin-top: 32px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1B4F72;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING WITH CACHING
# @st.cache_data tells Streamlit to cache the result of this function.
# This means the data is only loaded ONCE per session, not every time
# the user clicks a button. This speeds up the app dramatically.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading climate-vegetation dataset...")
def load_data() -> pd.DataFrame:
    """Load or generate the main dataset (cached for performance)."""
    from data_loader import DataLoader
    loader = DataLoader(random_seed=42, n_locations=50)
    df = loader.load_all_data()
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner="Running preprocessing pipeline...")
def run_preprocessing(df_json: str) -> dict:
    """
    Run the full preprocessing pipeline (cached).
    We pass JSON string instead of DataFrame to enable caching
    (Streamlit can't hash DataFrames directly).
    """
    from preprocessing import Preprocessor
    df = pd.read_json(df_json)
    df["date"] = pd.to_datetime(df["date"])
    prep = Preprocessor()
    return prep.run_full_pipeline(df, scale=True)


@st.cache_resource(show_spinner="Training ML models...")
def train_models(X_train, y_train, X_test, y_test, feature_names) -> dict:
    """Train all ML models (cached as resource — not re-hashed)."""
    from model import ModelTrainer
    trainer = ModelTrainer(random_seed=42)
    results = trainer.train_all_models(
        X_train, y_train, X_test, y_test, feature_names=feature_names
    )
    return trainer, results


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> dict:
    """
    Render the sidebar with all filter controls.
    Returns a dict of selected filter values.
    """
    with st.sidebar:
        # Logo / title
        st.markdown("""
        <div style="text-align:center; padding: 8px 0 16px;">
            <span style="font-size: 48px;">🌿</span><br>
            <span style="color:white; font-size:18px; font-weight:bold;">
                Climate-Vegetation<br>Analysis
            </span><br>
            <span style="color: #AED6F1; font-size: 12px;">v1.0 | 2024</span>
        </div>
        <hr style="border-color: #2E86C1; margin: 8px 0;">
        """, unsafe_allow_html=True)

        st.markdown('<p style="color:#AED6F1; font-weight:bold;">📌 FILTERS</p>',
                    unsafe_allow_html=True)

        # ── Region filter
        regions = sorted(df["region"].unique())
        selected_regions = st.multiselect(
            "🌍 Select Regions",
            options=regions,
            default=regions,
            help="Filter data by geographic region"
        )

        # ── Date range filter
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()

        st.markdown('<p style="color:#AED6F1; font-weight:bold; font-size:14px;">📅 Date Range</p>',
                    unsafe_allow_html=True)
        start_year = st.slider("Start Year", 2000, 2023, 2000)
        end_year = st.slider("End Year", 2000, 2023, 2023)

        # ── Variable selector
        st.markdown('<p style="color:#AED6F1; font-weight:bold; font-size:14px;">📊 Climate Variable</p>',
                    unsafe_allow_html=True)
        climate_var = st.selectbox(
            "Primary Variable",
            options=["temperature_mean", "precipitation", "humidity",
                     "solar_radiation", "drought_index", "co2_ppm"],
            format_func=lambda x: {
                "temperature_mean": "🌡️ Temperature (°C)",
                "precipitation": "🌧️ Precipitation (mm)",
                "humidity": "💧 Humidity (%)",
                "solar_radiation": "☀️ Solar Radiation (W/m²)",
                "drought_index": "🏜️ Drought Index",
                "co2_ppm": "🏭 CO₂ Concentration (ppm)",
            }.get(x, x)
        )

        # ── ML Settings
        st.markdown('<p style="color:#AED6F1; font-weight:bold; font-size:14px;">🤖 ML Settings</p>',
                    unsafe_allow_html=True)
        train_models_toggle = st.checkbox("Train ML Models", value=True,
                                           help="Enable to train and evaluate ML models")
        forecast_months = st.slider("Forecast Horizon (months)", 6, 36, 24)

        # ── About section
        st.markdown("---")
        st.markdown("""
        <div style="color: #AED6F1; font-size: 12px; text-align: center;">
            <b>📖 About</b><br>
            This dashboard analyzes the relationship between climate variables
            and vegetation health using NASA MODIS NDVI + NOAA climate data.
            <br><br>
            <a href="https://github.com/yourusername/Climate-Vegetation-Analysis"
               style="color: #5DADE2;">🔗 GitHub Repository</a>
        </div>
        """, unsafe_allow_html=True)

    return {
        "regions": selected_regions if selected_regions else regions,
        "start_year": start_year,
        "end_year": end_year,
        "climate_var": climate_var,
        "train_models": train_models_toggle,
        "forecast_months": forecast_months,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def render_overview_tab(df: pd.DataFrame, filters: dict):
    """KPI metrics and high-level summary."""

    st.markdown('<div class="section-header">📊 Dashboard Overview</div>',
                unsafe_allow_html=True)

    # ── KPI Metric Cards (top row)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        mean_ndvi = df["ndvi"].mean()
        ndvi_2023 = df[df["date"].dt.year == 2023]["ndvi"].mean()
        ndvi_2000 = df[df["date"].dt.year == 2000]["ndvi"].mean()
        delta_ndvi = ndvi_2023 - ndvi_2000
        st.metric(
            "🌿 Mean NDVI",
            f"{mean_ndvi:.4f}",
            delta=f"{delta_ndvi:+.4f} vs 2000",
            delta_color="normal"
        )

    with col2:
        mean_temp = df["temperature_mean"].mean()
        temp_2023 = df[df["date"].dt.year == 2023]["temperature_mean"].mean()
        temp_2000 = df[df["date"].dt.year == 2000]["temperature_mean"].mean()
        delta_temp = temp_2023 - temp_2000
        st.metric(
            "🌡️ Mean Temperature",
            f"{mean_temp:.1f}°C",
            delta=f"{delta_temp:+.2f}°C since 2000",
            delta_color="inverse"
        )

    with col3:
        mean_precip = df["precipitation"].mean()
        st.metric(
            "🌧️ Avg Precipitation",
            f"{mean_precip:.1f} mm/mo",
            delta=f"{df.shape[0]:,} records"
        )

    with col4:
        latest_co2 = df["co2_ppm"].max()
        first_co2 = df["co2_ppm"].min()
        st.metric(
            "🏭 Latest CO₂",
            f"{latest_co2:.1f} ppm",
            delta=f"+{latest_co2-first_co2:.1f} since 2000",
            delta_color="inverse"
        )

    with col5:
        n_regions = df["region"].nunique()
        n_years = df["date"].dt.year.nunique()
        st.metric(
            "🗺️ Study Coverage",
            f"{n_regions} Regions",
            delta=f"{n_years} years of data"
        )

    st.markdown("---")

    # ── Climate Overview Chart
    st.markdown("#### 📈 Climate Variables Overview")
    df_monthly = df.groupby("date").agg(
        temperature_mean=("temperature_mean", "mean"),
        precipitation=("precipitation", "mean"),
        ndvi=("ndvi", "mean"),
        co2_ppm=("co2_ppm", "mean"),
    ).reset_index()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "🌡️ Mean Temperature (°C)",
            "🌧️ Monthly Precipitation (mm)",
            "🌿 Mean NDVI",
            "🏭 Atmospheric CO₂ (ppm)"
        ),
        shared_xaxes=False,
        vertical_spacing=0.12
    )

    traces = [
        ("temperature_mean", "#E74C3C", 1, 1),
        ("precipitation", "#3498DB", 1, 2),
        ("ndvi", "#27AE60", 2, 1),
        ("co2_ppm", "#8E44AD", 2, 2),
    ]
    for col_name, color, row, col in traces:
        fig.add_trace(
            go.Scatter(
                x=df_monthly["date"], y=df_monthly[col_name],
                mode="lines", line=dict(color=color, width=1.5),
                fill="tozeroy", fillcolor=color.replace(")", ", 0.05)").replace("(", "a("),
                showlegend=False,
                hovertemplate=f"<b>%{{x|%Y-%m}}</b><br>{col_name}: %{{y:.3f}}<extra></extra>"
            ),
            row=row, col=col
        )

    fig.update_layout(
        height=480, template="plotly_white",
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        margin=dict(t=60, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Region distribution
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("#### 📍 Records by Region")
        region_counts = df["region"].value_counts().reset_index()
        region_counts.columns = ["region", "count"]
        fig_pie = px.pie(
            region_counts, values="count", names="region",
            color="region",
            color_discrete_map={
                "Sahel": "#E67E22", "Savanna": "#F1C40F",
                "Rainforest": "#27AE60", "Semi-Arid": "#BDC3C7",
                "Mediterranean": "#3498DB"
            },
            hole=0.4, template="plotly_white"
        )
        fig_pie.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.markdown("#### 📋 Dataset Summary")
        summary_data = {
            "Metric": ["Total Records", "Date Range", "Locations", "Regions",
                       "NDVI Range", "Temp Range", "Precip Range"],
            "Value": [
                f"{len(df):,}",
                f"{df['date'].min().year}–{df['date'].max().year}",
                f"{df['location_id'].nunique()}",
                f"{df['region'].nunique()}",
                f"{df['ndvi'].min():.3f} – {df['ndvi'].max():.3f}",
                f"{df['temperature_mean'].min():.1f}°C – {df['temperature_mean'].max():.1f}°C",
                f"{df['precipitation'].min():.0f} – {df['precipitation'].max():.0f} mm"
            ]
        }
        st.dataframe(
            pd.DataFrame(summary_data),
            hide_index=True, use_container_width=True,
            height=290
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: NDVI ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def render_ndvi_tab(df: pd.DataFrame, filters: dict):
    """NDVI trends, seasonal patterns, and regional comparisons."""

    st.markdown('<div class="section-header">🌿 NDVI Vegetation Analysis</div>',
                unsafe_allow_html=True)

    # ── Info box
    st.markdown("""
    <div class="info-box">
    <b>💡 What is NDVI?</b> The Normalized Difference Vegetation Index (NDVI) measures
    vegetation health and density from satellite imagery. Values range from 0 (bare soil)
    to 1 (dense forest). Derived from NASA MODIS satellite data at 1km resolution.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── NDVI Trend by Region
    st.markdown("#### 📈 Monthly NDVI Trends by Region")
    df_ndvi = df.groupby(["date", "region"])["ndvi"].mean().reset_index()

    fig = px.line(
        df_ndvi, x="date", y="ndvi", color="region",
        color_discrete_map={
            "Sahel": "#E67E22", "Savanna": "#F1C40F",
            "Rainforest": "#27AE60", "Semi-Arid": "#BDC3C7",
            "Mediterranean": "#3498DB"
        },
        template="plotly_white",
        labels={"ndvi": "Mean NDVI", "date": "Date", "region": "Region"},
    )
    fig.update_traces(line_width=1.8)
    fig.update_layout(
        height=380, hovermode="x unified",
        legend_title_text="Region",
        yaxis_range=[0, 0.85],
        plot_bgcolor="#F8F9FA"
    )
    # Add reference lines for NDVI categories
    for y, label, color in [
        (0.15, "Sparse", "#E67E22"),
        (0.40, "Moderate", "#F1C40F"),
        (0.60, "Dense", "#27AE60")
    ]:
        fig.add_hline(y=y, line_dash="dot", line_color=color, opacity=0.5,
                      annotation_text=label, annotation_position="right")
    st.plotly_chart(fig, use_container_width=True)

    # ── Seasonal patterns
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 🗓️ NDVI Seasonal Climatology")
        df_seasonal = df.copy()
        df_seasonal["month"] = df_seasonal["date"].dt.month
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        df_seasonal["month_name"] = df_seasonal["month"].map(month_names)

        monthly_clim = df_seasonal.groupby(["month", "month_name", "region"])["ndvi"].mean().reset_index()
        monthly_clim = monthly_clim.sort_values("month")

        fig_seasonal = px.line(
            monthly_clim, x="month_name", y="ndvi", color="region",
            color_discrete_map={
                "Sahel": "#E67E22", "Savanna": "#F1C40F",
                "Rainforest": "#27AE60", "Semi-Arid": "#BDC3C7",
                "Mediterranean": "#3498DB"
            },
            markers=True, template="plotly_white",
            labels={"ndvi": "Mean NDVI", "month_name": "Month"},
            category_orders={"month_name": list(month_names.values())}
        )
        fig_seasonal.update_layout(height=300, plot_bgcolor="#F8F9FA",
                                    legend_title_text="Region")
        st.plotly_chart(fig_seasonal, use_container_width=True)

    with col_r:
        st.markdown("#### 📊 Annual NDVI Distribution")
        df_annual_ndvi = df.copy()
        df_annual_ndvi["year"] = df_annual_ndvi["date"].dt.year

        fig_box = px.box(
            df_annual_ndvi[df_annual_ndvi["year"].isin(
                [2000, 2005, 2010, 2015, 2020, 2023])],
            x="year", y="ndvi", color="region",
            color_discrete_map={
                "Sahel": "#E67E22", "Savanna": "#F1C40F",
                "Rainforest": "#27AE60", "Semi-Arid": "#BDC3C7",
                "Mediterranean": "#3498DB"
            },
            template="plotly_white",
            labels={"ndvi": "NDVI", "year": "Year"}
        )
        fig_box.update_layout(height=300, showlegend=False,
                               plot_bgcolor="#F8F9FA")
        st.plotly_chart(fig_box, use_container_width=True)

    # ── NDVI vs Climate variable scatter
    st.markdown("#### 🔍 NDVI vs. Climate Variable Relationship")
    x_var = st.selectbox(
        "Select X-axis variable",
        ["precipitation", "temperature_mean", "humidity",
         "drought_index", "solar_radiation"],
        key="ndvi_scatter_x"
    )

    sample_df = df.sample(min(3000, len(df)), random_state=42)
    fig_scatter = px.scatter(
        sample_df, x=x_var, y="ndvi", color="region",
        opacity=0.5, trendline="lowess",
        color_discrete_map={
            "Sahel": "#E67E22", "Savanna": "#F1C40F",
            "Rainforest": "#27AE60", "Semi-Arid": "#BDC3C7",
            "Mediterranean": "#3498DB"
        },
        template="plotly_white",
        labels={"ndvi": "NDVI", x_var: x_var.replace("_", " ").title()},
        title=f"NDVI vs. {x_var.replace('_', ' ').title()} (LOWESS smoothed trend)"
    )
    fig_scatter.update_layout(height=400, plot_bgcolor="#F8F9FA")
    st.plotly_chart(fig_scatter, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: CLIMATE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def render_climate_tab(df: pd.DataFrame, filters: dict):
    """Temperature, precipitation, and other climate deep dives."""

    st.markdown('<div class="section-header">🌡️ Climate Variable Analysis</div>',
                unsafe_allow_html=True)

    var = filters["climate_var"]
    var_label = var.replace("_", " ").title()

    # ── Selected variable trend
    st.markdown(f"#### 📈 {var_label} — Long-term Trend")
    df_var = df.groupby(["date", "region"])[var].mean().reset_index()

    fig = px.line(
        df_var, x="date", y=var, color="region",
        template="plotly_white",
        labels={var: var_label, "date": "Date"},
        color_discrete_map={
            "Sahel": "#E67E22", "Savanna": "#F1C40F",
            "Rainforest": "#27AE60", "Semi-Arid": "#BDC3C7",
            "Mediterranean": "#3498DB"
        }
    )
    # Add overall trend line
    df_overall = df.groupby("date")[var].mean().reset_index()
    z = np.polyfit(range(len(df_overall)), df_overall[var], 1)
    trend_vals = np.poly1d(z)(range(len(df_overall)))
    fig.add_trace(go.Scatter(
        x=df_overall["date"], y=trend_vals,
        mode="lines", line=dict(color="black", dash="dash", width=2),
        name="Overall Trend"
    ))

    fig.update_layout(height=380, hovermode="x unified",
                       plot_bgcolor="#F8F9FA")
    st.plotly_chart(fig, use_container_width=True)

    # ── Temperature + Precipitation correlation
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 🌡️ Temperature Distribution by Region")
        fig_violin = px.violin(
            df, x="region", y="temperature_mean", color="region",
            box=True, points=False,
            color_discrete_map={
                "Sahel": "#E67E22", "Savanna": "#F1C40F",
                "Rainforest": "#27AE60", "Semi-Arid": "#BDC3C7",
                "Mediterranean": "#3498DB"
            },
            template="plotly_white",
            labels={"temperature_mean": "Temperature (°C)", "region": "Region"}
        )
        fig_violin.update_layout(height=340, showlegend=False, plot_bgcolor="#F8F9FA")
        st.plotly_chart(fig_violin, use_container_width=True)

    with col_r:
        st.markdown("#### 🌧️ Monthly Precipitation Heatmap")
        df_heat = df.copy()
        df_heat["year"] = df_heat["date"].dt.year
        df_heat["month"] = df_heat["date"].dt.month
        pivot = df_heat.groupby(["year", "month"])["precipitation"].mean().unstack()

        fig_heat = px.imshow(
            pivot, aspect="auto",
            color_continuous_scale="Blues",
            labels={"x": "Month", "y": "Year", "color": "Precip (mm)"},
            template="plotly_white",
        )
        fig_heat.update_layout(height=340)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── CO2 Keeling Curve
    st.markdown("#### 🏭 Atmospheric CO₂ (Keeling Curve Pattern)")
    df_co2 = df.groupby("date")["co2_ppm"].mean().reset_index()

    fig_co2 = go.Figure()
    fig_co2.add_trace(go.Scatter(
        x=df_co2["date"], y=df_co2["co2_ppm"],
        mode="lines", line=dict(color="#8E44AD", width=2),
        fill="tozeroy", fillcolor="rgba(142, 68, 173, 0.05)",
        name="CO₂ (ppm)"
    ))
    fig_co2.update_layout(
        height=280, template="plotly_white",
        title="Simulated CO₂ Concentration (Keeling Curve Pattern, 2000–2023)",
        xaxis_title="Date", yaxis_title="CO₂ (ppm)",
        plot_bgcolor="#F8F9FA"
    )
    fig_co2.add_annotation(
        x=df_co2["date"].iloc[-1], y=df_co2["co2_ppm"].iloc[-1],
        text=f"~{df_co2['co2_ppm'].iloc[-1]:.0f} ppm (2023)",
        showarrow=True, arrowhead=2
    )
    st.plotly_chart(fig_co2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: ML MODELS
# ─────────────────────────────────────────────────────────────────────────────

def render_models_tab(df: pd.DataFrame):
    """ML model training, evaluation, and comparison."""

    st.markdown('<div class="section-header">🤖 Machine Learning Model Comparison</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>🔬 About the Models:</b> We train 4 ML algorithms to predict NDVI from
    climate variables. Training uses a temporal split (first 80% of time steps)
    to prevent data leakage. Evaluation metrics: RMSE, MAE, and R².
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    with st.spinner("Running preprocessing pipeline..."):
        prep_result = run_preprocessing(df.to_json())

    st.success(f"✅ Features engineered: {len(prep_result['feature_names'])} features, "
               f"{len(prep_result['X_train']):,} training samples")

    with st.spinner("Training ML models (this may take 30–60 seconds)..."):
        trainer, model_results = train_models(
            prep_result["X_train_scaled"],
            np.array(prep_result["y_train"]),
            prep_result["X_test_scaled"],
            np.array(prep_result["y_test"]),
            feature_names=prep_result["feature_names"]
        )

    # ── Results table
    st.markdown("#### 🏆 Model Performance Leaderboard")
    rows = []
    for name, res in model_results.items():
        rows.append({
            "Model": name,
            "RMSE ↓": round(res["rmse"], 5),
            "MAE ↓": round(res["mae"], 5),
            "R² ↑": round(res["r2"], 5),
            "Train R²": round(res["train_r2"], 5),
            "Train Time (s)": round(res["train_time_s"], 2),
        })
    leaderboard_df = pd.DataFrame(rows).sort_values("R² ↑", ascending=False)

    # Highlight best row
    def highlight_best(s):
        return ["background-color: #D5F5E3; font-weight: bold"
                if i == 0 else "" for i in range(len(s))]

    st.dataframe(
        leaderboard_df.style.apply(highlight_best, axis=0),
        use_container_width=True, hide_index=True
    )

    # ── Metric comparison bar charts
    st.markdown("#### 📊 Performance Metric Comparison")
    col1, col2, col3 = st.columns(3)

    for col, metric, title, ascending in [
        (col1, "RMSE ↓", "RMSE (lower = better)", True),
        (col2, "MAE ↓", "MAE (lower = better)", True),
        (col3, "R² ↑", "R² Score (higher = better)", False),
    ]:
        with col:
            df_metric = leaderboard_df[["Model", metric]].sort_values(metric, ascending=ascending)
            best_val = df_metric[metric].iloc[0] if ascending else df_metric[metric].iloc[-1]
            colors = ["#27AE60" if v == best_val else "#BDC3C7"
                      for v in df_metric[metric]]

            fig_m = go.Figure(go.Bar(
                x=df_metric["Model"], y=df_metric[metric],
                marker_color=colors, text=df_metric[metric].round(4),
                textposition="outside"
            ))
            fig_m.update_layout(
                title=title, height=320, template="plotly_white",
                plot_bgcolor="#F8F9FA", margin=dict(t=40, b=60),
                showlegend=False
            )
            st.plotly_chart(fig_m, use_container_width=True)

    # ── Feature Importance
    if "XGBoost" in model_results and "feature_importances" in model_results["XGBoost"]:
        st.markdown("#### 🎯 XGBoost Feature Importance (Top 15)")
        fi = model_results["XGBoost"]["feature_importances"]
        fn = model_results["XGBoost"]["feature_names"]

        fi_df = pd.DataFrame({"feature": fn, "importance": fi})
        fi_df = fi_df.sort_values("importance", ascending=False).head(15)

        fig_fi = px.bar(
            fi_df, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Greens",
            template="plotly_white",
            labels={"importance": "Importance Score", "feature": "Feature"}
        )
        fig_fi.update_layout(height=450, plot_bgcolor="#F8F9FA",
                              coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Actual vs Predicted scatter
    st.markdown("#### 🎯 Actual vs. Predicted NDVI (Best Model)")
    best = trainer.best_model_name
    y_pred = model_results[best]["y_pred"]
    y_test = np.array(prep_result["y_test"])

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode="markers",
        marker=dict(color="#27AE60", opacity=0.4, size=4),
        name="Predictions"
    ))
    # Perfect prediction line
    line_min = min(y_test.min(), y_pred.min())
    line_max = max(y_test.max(), y_pred.max())
    fig_pred.add_trace(go.Scatter(
        x=[line_min, line_max], y=[line_min, line_max],
        mode="lines", line=dict(color="red", dash="dash", width=2),
        name="Perfect Fit (y=x)"
    ))
    fig_pred.update_layout(
        height=420, template="plotly_white", plot_bgcolor="#F8F9FA",
        xaxis_title="Actual NDVI", yaxis_title="Predicted NDVI",
        title=f"Actual vs. Predicted NDVI — {best} (R² = {model_results[best]['r2']:.4f})"
    )
    st.plotly_chart(fig_pred, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: FORECAST
# ─────────────────────────────────────────────────────────────────────────────

def render_forecast_tab(df: pd.DataFrame, filters: dict):
    """Future NDVI scenario forecasts."""

    st.markdown('<div class="section-header">🔮 NDVI Future Projections</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>🌍 Climate Scenarios Explained:</b><br>
    🟢 <b>Optimistic</b>: Increased rainfall, stable temperatures — sustainable land management<br>
    🟡 <b>Baseline</b>: Current trends continue — business as usual (~+0.022°C/yr warming)<br>
    🔴 <b>Pessimistic</b>: Severe drought, rapid warming — climate change intensification
    </div>
    """, unsafe_allow_html=True)

    months = filters["forecast_months"]

    # Generate simplified forecast using historical trends
    last_date = df["date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=months, freq="MS"
    )

    historical_mean = df.groupby("date")["ndvi"].mean()
    base_ndvi = historical_mean.iloc[-6:].mean()
    recent_trend = np.polyfit(range(len(historical_mean[-24:])),
                               historical_mean.iloc[-24:].values, 1)[0]

    # Generate scenario trajectories
    np.random.seed(42)
    scenario_data = []
    for i, date in enumerate(future_dates):
        month = date.month
        seasonal = 0.03 * np.sin(2 * np.pi * month / 12 + np.pi / 2)
        step_noise = np.random.normal(0, 0.005)

        for scenario, trend_mult, color in [
            ("🟢 Optimistic",   +0.0005, "#27AE60"),
            ("🟡 Baseline",     recent_trend, "#F39C12"),
            ("🔴 Pessimistic",  -0.0010, "#E74C3C"),
        ]:
            ndvi = np.clip(base_ndvi + trend_mult * (i + 1) + seasonal + step_noise, 0, 0.9)
            scenario_data.append({
                "date": date, "ndvi": ndvi,
                "scenario": scenario, "color": color
            })

    scenario_df = pd.DataFrame(scenario_data)

    # ── Plot
    fig = go.Figure()

    # Historical line (last 3 years)
    hist_recent = historical_mean[historical_mean.index >= "2021-01-01"].reset_index()
    fig.add_trace(go.Scatter(
        x=hist_recent["date"], y=hist_recent["ndvi"],
        mode="lines", line=dict(color="#2C3E50", width=3),
        name="Historical NDVI"
    ))

    # Forecast start marker
    fig.add_vline(
        x=last_date, line_dash="dash", line_color="gray",
        annotation_text="  Forecast Starts", annotation_position="top"
    )

    # Each scenario
    for scenario in ["🟢 Optimistic", "🟡 Baseline", "🔴 Pessimistic"]:
        scen_df = scenario_df[scenario_df["scenario"] == scenario]
        color = scen_df["color"].iloc[0]
        fig.add_trace(go.Scatter(
            x=scen_df["date"], y=scen_df["ndvi"],
            mode="lines", line=dict(color=color, width=2.5),
            name=scenario, fill="tonexty" if scenario == "🟢 Optimistic" else None
        ))

    fig.update_layout(
        height=450, template="plotly_white", plot_bgcolor="#F8F9FA",
        xaxis_title="Date", yaxis_title="Predicted NDVI",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#E0E7FF", borderwidth=1),
        title=f"NDVI Forecast — {months}-Month Horizon (3 Climate Scenarios)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Scenario summary table
    st.markdown("#### 📋 Scenario Summary")
    summary_rows = []
    for scenario in scenario_df["scenario"].unique():
        scen = scenario_df[scenario_df["scenario"] == scenario]
        summary_rows.append({
            "Scenario": scenario,
            "Forecast Start NDVI": f"{scen['ndvi'].iloc[0]:.4f}",
            f"NDVI at {months} months": f"{scen['ndvi'].iloc[-1]:.4f}",
            "Net Change": f"{scen['ndvi'].iloc[-1] - scen['ndvi'].iloc[0]:+.4f}",
            "Min NDVI": f"{scen['ndvi'].min():.4f}",
            "Max NDVI": f"{scen['ndvi'].max():.4f}",
        })
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: MAP
# ─────────────────────────────────────────────────────────────────────────────

def render_map_tab(df: pd.DataFrame):
    """Interactive Folium map of spatial NDVI and climate data."""

    st.markdown('<div class="section-header">🗺️ Interactive Spatial Map</div>',
                unsafe_allow_html=True)

    try:
        from streamlit_folium import st_folium
        import folium
        from folium.plugins import HeatMap

        # Aggregate per location
        loc_agg = df.groupby("location_id").agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            region=("region", "first"),
            ndvi_mean=("ndvi", "mean"),
            temp_mean=("temperature_mean", "mean"),
            precip_mean=("precipitation", "mean"),
        ).reset_index()

        # NDVI color mapping
        def ndvi_color(v):
            if v < 0.15: return "#BDC3C7"
            elif v < 0.25: return "#E67E22"
            elif v < 0.40: return "#F1C40F"
            elif v < 0.55: return "#52BE80"
            else: return "#1E8449"

        # Create map
        m = folium.Map(
            location=[loc_agg["latitude"].mean(), loc_agg["longitude"].mean()],
            zoom_start=4, tiles="CartoDB positron"
        )

        for _, row in loc_agg.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=9, color="white", weight=1.5,
                fill=True, fill_color=ndvi_color(row["ndvi_mean"]),
                fill_opacity=0.85,
                tooltip=f"{row['region']} | NDVI: {row['ndvi_mean']:.3f}",
                popup=folium.Popup(f"""
                <b>{row['region']}</b><br>
                NDVI: <b style="color:green">{row['ndvi_mean']:.3f}</b><br>
                Temp: {row['temp_mean']:.1f}°C<br>
                Precip: {row['precip_mean']:.1f} mm/mo
                """, max_width=200)
            ).add_to(m)

        # Heatmap layer
        heat_data = loc_agg[["latitude", "longitude", "precip_mean"]].values.tolist()
        HeatMap(heat_data, radius=25, blur=15, name="Precipitation").add_to(m)
        folium.LayerControl().add_to(m)

        st_folium(m, width=None, height=550, returned_objects=[])

    except ImportError:
        st.warning("⚠️ `streamlit-folium` not installed. Run: `pip install streamlit-folium`")
        st.markdown("#### 📍 Location Data (tabular fallback)")
        loc_agg = df.groupby("location_id").agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            region=("region", "first"),
            ndvi_mean=("ndvi", "mean"),
        ).reset_index()
        st.dataframe(loc_agg.round(4), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main function: builds the complete Streamlit dashboard."""

    # ── Header
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="margin:0; font-size: 28px;">
            🌿 Climate Change & Vegetation Dynamics Analysis
        </h1>
        <p style="margin: 6px 0 0; font-size: 16px; opacity: 0.9;">
            Using Remote Sensing, Geospatial Analysis & Machine Learning
        </p>
        <p style="margin: 4px 0 0; font-size: 13px; opacity: 0.7;">
            Data: NASA MODIS NDVI · NOAA Climate · ERA5 Reanalysis | 2000–2023
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("💡 Run `python src/data_loader.py` to generate the dataset first.")
        st.stop()

    # ── Sidebar filters
    filters = render_sidebar(df)

    # ── Apply filters
    df_filtered = df[
        (df["region"].isin(filters["regions"])) &
        (df["date"].dt.year >= filters["start_year"]) &
        (df["date"].dt.year <= filters["end_year"])
    ].copy()

    if df_filtered.empty:
        st.warning("⚠️ No data matches your filters. Adjust the sidebar controls.")
        st.stop()

    # ── Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🌿 NDVI Analysis",
        "🌡️ Climate Variables",
        "🤖 ML Models",
        "🔮 Forecast",
        "🗺️ Interactive Map",
    ])

    with tab1:
        render_overview_tab(df_filtered, filters)

    with tab2:
        render_ndvi_tab(df_filtered, filters)

    with tab3:
        render_climate_tab(df_filtered, filters)

    with tab4:
        if filters["train_models"]:
            render_models_tab(df_filtered)
        else:
            st.info("🔘 Enable 'Train ML Models' in the sidebar to view this tab.")

    with tab5:
        render_forecast_tab(df_filtered, filters)

    with tab6:
        render_map_tab(df_filtered)

    # ── Footer
    st.markdown("""
    <div class="footer">
        🌿 Climate Change & Vegetation Dynamics Analysis Dashboard
        · Built with Python, Streamlit & ❤️
        · <a href="https://github.com/yourusername/Climate-Vegetation-Analysis">GitHub</a>
        · MIT License · 2024
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
