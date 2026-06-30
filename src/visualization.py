"""
=============================================================================
visualization.py — Climate Vegetation Analysis Project
=============================================================================
PURPOSE:
    All visualization functions for the project. This module creates:
    1. Temperature trend plots
    2. Precipitation distribution plots
    3. NDVI time series with seasonal decomposition
    4. Correlation heatmaps
    5. Feature importance plots
    6. Model performance comparison charts
    7. Future prediction plots
    8. Interactive Folium maps

DESIGN PHILOSOPHY:
    - Each function is self-contained (accepts data, returns/saves figure)
    - All plots use a consistent color scheme (professional palette)
    - Figures are saved to the images/ directory automatically
    - Seaborn style applied globally for a clean academic look

BEGINNER TIP:
    matplotlib is like MS Paint — low-level drawing functions.
    seaborn is like Canva — beautiful prebuilt chart templates.
    plotly makes interactive charts you can zoom and hover over.
    folium creates interactive maps powered by OpenStreetMap/Leaflet.js.

USAGE:
    from src.visualization import Visualizer
    viz = Visualizer()
    viz.plot_temperature_trends(df)
    viz.plot_ndvi_timeseries(df)
    viz.plot_correlation_heatmap(df)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

# Set a clean, professional style for all matplotlib/seaborn plots
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F9FA",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
})

# Color palette (colorblind-friendly)
PALETTE = {
    "temperature": "#E74C3C",   # Red
    "precipitation": "#3498DB", # Blue
    "ndvi": "#27AE60",          # Green
    "trend": "#2C3E50",         # Dark navy
    "highlight": "#F39C12",     # Orange
    "neutral": "#95A5A6",       # Gray
}

REGION_COLORS = {
    "Sahel": "#E67E22",
    "Savanna": "#F1C40F",
    "Rainforest": "#27AE60",
    "Semi-Arid": "#BDC3C7",
    "Mediterranean": "#3498DB",
}

PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)


class Visualizer:
    """
    Creates all project visualizations.

    Methods:
        plot_temperature_trends()       — Long-term temperature change
        plot_precipitation_patterns()   — Rainfall distribution
        plot_ndvi_timeseries()          — Vegetation index over time
        plot_correlation_heatmap()      — Feature correlations
        plot_model_comparison()         — ML model performance
        plot_feature_importance()       — XGBoost feature ranking
        plot_future_predictions()       — Forecast with confidence
        create_interactive_map()        — Folium HTML map
        create_dashboard_figures()      — Plotly for Streamlit
    """

    def __init__(self):
        print("✅ Visualizer initialized")

    # ─────────────────────────────────────────────────────────────────────
    # 1. TEMPERATURE TRENDS
    # ─────────────────────────────────────────────────────────────────────

    def plot_temperature_trends(self,
                                  df: pd.DataFrame,
                                  save: bool = True) -> plt.Figure:
        """
        Plot long-term temperature trends with seasonal decomposition.

        Shows:
        - Annual mean temperature time series
        - Linear warming trend overlay
        - Shaded uncertainty range (±1 std dev)
        - Regional comparison as subplots

        Args:
            df: DataFrame with 'date', 'temperature_mean', 'region' columns
            save: Whether to save to images/

        Returns:
            matplotlib Figure object
        """
        print("  📈 Plotting temperature trends...")

        # Aggregate: annual mean temperature per region
        df_annual = (
            df.groupby([df["date"].dt.year, "region"])["temperature_mean"]
            .agg(["mean", "std"])
            .reset_index()
        )
        df_annual.columns = ["year", "region", "temp_mean", "temp_std"]

        regions = df["region"].unique()
        n_regions = len(regions)

        fig, axes = plt.subplots(
            nrows=(n_regions + 1) // 2, ncols=2,
            figsize=(14, 4 * ((n_regions + 1) // 2)),
            constrained_layout=True
        )
        axes = axes.flatten()

        fig.suptitle(
            "🌡️  Long-Term Temperature Trends by Region (2000–2023)",
            fontsize=16, fontweight="bold", y=1.02
        )

        for idx, region in enumerate(regions):
            ax = axes[idx]
            data = df_annual[df_annual["region"] == region]

            # ── Temperature line
            ax.plot(
                data["year"], data["temp_mean"],
                color=REGION_COLORS.get(region, PALETTE["temperature"]),
                linewidth=2.5, label="Annual Mean Temp", zorder=3
            )

            # ── Shaded uncertainty (mean ± 1 std dev)
            ax.fill_between(
                data["year"],
                data["temp_mean"] - data["temp_std"],
                data["temp_mean"] + data["temp_std"],
                alpha=0.2,
                color=REGION_COLORS.get(region, PALETTE["temperature"]),
                label="±1 Std Dev"
            )

            # ── Linear trend line (using numpy polyfit = least squares regression)
            z = np.polyfit(data["year"], data["temp_mean"], 1)  # fit degree-1 polynomial
            p = np.poly1d(z)
            trend_label = f"Trend: {z[0]:+.3f}°C/yr"
            ax.plot(data["year"], p(data["year"]),
                    "--", color=PALETTE["trend"],
                    linewidth=1.5, label=trend_label, zorder=4)

            # ── Formatting
            ax.set_title(f"Region: {region}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Temperature (°C)")
            ax.legend(fontsize=8)

            # Annotate warming rate
            warming_total = z[0] * (data["year"].max() - data["year"].min())
            ax.annotate(
                f"Total: {warming_total:+.1f}°C",
                xy=(0.98, 0.05), xycoords="axes fraction",
                ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8)
            )

        # Hide unused axes if odd number of regions
        for i in range(n_regions, len(axes)):
            axes[i].set_visible(False)

        if save:
            path = IMAGES_DIR / "temperature_trends.png"
            fig.savefig(path, bbox_inches="tight", dpi=150)
            print(f"     Saved: {path}")

        return fig

    # ─────────────────────────────────────────────────────────────────────
    # 2. PRECIPITATION PATTERNS
    # ─────────────────────────────────────────────────────────────────────

    def plot_precipitation_patterns(self,
                                      df: pd.DataFrame,
                                      save: bool = True) -> plt.Figure:
        """
        Visualize rainfall distributions and trends.

        Shows:
        - Monthly precipitation distribution (violin plots)
        - Annual total precipitation trend
        - Wet/dry season separation

        Args:
            df: Climate DataFrame
            save: Save figure to images/

        Returns:
            matplotlib Figure
        """
        print("  🌧️  Plotting precipitation patterns...")

        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        fig.suptitle("🌧️  Precipitation Patterns Analysis (2000–2023)",
                     fontsize=16, fontweight="bold")

        gs = gridspec.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])   # Top row: full width
        ax2 = fig.add_subplot(gs[1, 0])   # Bottom left
        ax3 = fig.add_subplot(gs[1, 1])   # Bottom right

        # ── Plot 1: Annual precipitation trends by region
        df_annual = (
            df.groupby([df["date"].dt.year, "region"])["precipitation"]
            .sum()
            .reset_index()
        )
        df_annual.columns = ["year", "region", "annual_precip"]

        for region in df["region"].unique():
            data = df_annual[df_annual["region"] == region]
            ax1.plot(data["year"], data["annual_precip"],
                     color=REGION_COLORS.get(region, "gray"),
                     linewidth=2, label=region, marker="o", markersize=3)

        ax1.set_title("Annual Total Precipitation by Region")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Precipitation (mm/year)")
        ax1.legend(ncol=3)

        # ── Plot 2: Seasonal distribution (box plots by month)
        df["month_name"] = df["date"].dt.strftime("%b")
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        sns.boxplot(
            data=df, x="month_name", y="precipitation",
            order=month_order, ax=ax2,
            palette="Blues", flierprops={"markersize": 2}
        )
        ax2.set_title("Monthly Precipitation Distribution")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Precipitation (mm)")
        ax2.tick_params(axis="x", rotation=45)

        # ── Plot 3: Rainfall variability (coefficient of variation)
        df["year"] = df["date"].dt.year
        cv_by_region = df.groupby("region")["precipitation"].apply(
            lambda x: (x.std() / x.mean() * 100)
        ).reset_index()
        cv_by_region.columns = ["region", "cv_percent"]

        bars = ax3.barh(
            cv_by_region["region"],
            cv_by_region["cv_percent"],
            color=[REGION_COLORS.get(r, "gray") for r in cv_by_region["region"]],
            edgecolor="white", linewidth=0.5
        )
        ax3.set_title("Rainfall Variability (Coefficient of Variation)")
        ax3.set_xlabel("CV (%)")
        ax3.set_ylabel("Region")

        # Add value labels on bars
        for bar, val in zip(bars, cv_by_region["cv_percent"]):
            ax3.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=9)

        if save:
            path = IMAGES_DIR / "precipitation_patterns.png"
            fig.savefig(path, bbox_inches="tight", dpi=150)
            print(f"     Saved: {path}")

        return fig

    # ─────────────────────────────────────────────────────────────────────
    # 3. NDVI TIME SERIES
    # ─────────────────────────────────────────────────────────────────────

    def plot_ndvi_timeseries(self,
                               df: pd.DataFrame,
                               save: bool = True) -> plt.Figure:
        """
        Plot NDVI time series with seasonal decomposition.

        WHAT IS SEASONAL DECOMPOSITION?
        Any time series can be broken into:
        - Trend: The long-term direction (going up or down)
        - Seasonality: The repeating annual pattern
        - Residual: What's left after removing trend + seasonality

        We do a simple decomposition manually:
        1. Trend = 12-month rolling average
        2. Seasonal = original - trend
        3. Residual = original - trend - mean(seasonal by month)

        Args:
            df: DataFrame with NDVI and date
            save: Save to images/

        Returns:
            matplotlib Figure
        """
        print("  🌿 Plotting NDVI time series...")

        fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
        fig.suptitle("🌿  NDVI Dynamics and Seasonal Decomposition",
                     fontsize=16, fontweight="bold")

        regions = df["region"].unique()

        for col_idx, region in enumerate(list(regions)[:2]):  # Show 2 regions
            df_reg = (
                df[df["region"] == region]
                .groupby("date")["ndvi"].mean()
                .reset_index()
                .sort_values("date")
            )
            dates = df_reg["date"]
            ndvi = df_reg["ndvi"]

            # ── Panel 1: Raw NDVI + 12-month rolling average (the TREND)
            ax = axes[0, col_idx]
            rolling_trend = ndvi.rolling(window=12, center=True, min_periods=6).mean()
            ax.plot(dates, ndvi, color=PALETTE["ndvi"], alpha=0.4,
                    linewidth=1, label="Monthly NDVI")
            ax.plot(dates, rolling_trend, color=PALETTE["trend"],
                    linewidth=2.5, label="12-month Trend")
            ax.set_title(f"{region} — NDVI Time Series")
            ax.set_ylabel("NDVI")
            ax.legend()

            # Shade above-average months green, below-average red
            ndvi_mean = ndvi.mean()
            ax.axhline(y=ndvi_mean, color="gray", linestyle=":", linewidth=1, alpha=0.7)
            ax.fill_between(dates, ndvi, ndvi_mean,
                            where=(ndvi >= ndvi_mean), alpha=0.15,
                            color=PALETTE["ndvi"], label="Above avg")
            ax.fill_between(dates, ndvi, ndvi_mean,
                            where=(ndvi < ndvi_mean), alpha=0.15,
                            color=PALETTE["temperature"], label="Below avg")

            # ── Panel 2: Seasonal climatology (average NDVI by month)
            ax = axes[1, col_idx]
            df_reg["month"] = df_reg["date"].dt.month
            monthly_clim = df_reg.groupby("month")["ndvi"].agg(["mean", "std"])
            month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
            ax.bar(monthly_clim.index, monthly_clim["mean"],
                   color=PALETTE["ndvi"], alpha=0.7, edgecolor="white")
            ax.errorbar(monthly_clim.index, monthly_clim["mean"],
                        yerr=monthly_clim["std"], fmt="none",
                        color=PALETTE["trend"], capsize=4, linewidth=1.5)
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_names, rotation=45)
            ax.set_title(f"{region} — NDVI Seasonal Climatology")
            ax.set_ylabel("Mean NDVI")
            ax.set_xlabel("Month")

            # ── Panel 3: Annual average NDVI trend
            ax = axes[2, col_idx]
            df_reg["year"] = df_reg["date"].dt.year
            annual = df_reg.groupby("year")["ndvi"].mean()
            ax.plot(annual.index, annual.values, "o-",
                    color=PALETTE["ndvi"], linewidth=2, markersize=4)

            # Trend line
            z = np.polyfit(annual.index, annual.values, 1)
            p = np.poly1d(z)
            ax.plot(annual.index, p(annual.index), "--",
                    color=PALETTE["trend"], linewidth=1.5,
                    label=f"Trend: {z[0]:+.4f}/yr")
            ax.set_title(f"{region} — Annual NDVI Trend")
            ax.set_ylabel("Mean Annual NDVI")
            ax.set_xlabel("Year")
            ax.legend()

        if save:
            path = IMAGES_DIR / "ndvi_trends.png"
            fig.savefig(path, bbox_inches="tight", dpi=150)
            print(f"     Saved: {path}")

        return fig

    # ─────────────────────────────────────────────────────────────────────
    # 4. CORRELATION HEATMAP
    # ─────────────────────────────────────────────────────────────────────

    def plot_correlation_heatmap(self,
                                   df: pd.DataFrame,
                                   save: bool = True) -> plt.Figure:
        """
        Plot Pearson correlation matrix between all numeric variables.

        WHAT IS A CORRELATION HEATMAP?
        A correlation heatmap shows how strongly pairs of variables are
        related. Pearson correlation (r) ranges from -1 to +1:
        - r = +1: Perfect positive relationship (as X rises, Y rises)
        - r = 0: No linear relationship
        - r = -1: Perfect negative relationship (as X rises, Y falls)

        The diagonal is always 1.0 (a variable is perfectly correlated
        with itself). We mask the upper triangle to avoid duplication.

        Args:
            df: DataFrame with numeric columns
            save: Save to images/

        Returns:
            matplotlib Figure
        """
        print("  🔥 Plotting correlation heatmap...")

        # Select relevant numeric variables for correlation
        cols_of_interest = [
            "ndvi", "temperature_mean", "temperature_max",
            "precipitation", "humidity", "solar_radiation",
            "drought_index", "wind_speed", "co2_ppm",
            "latitude", "elevation_m"
        ]
        # Keep only columns that exist in the DataFrame
        cols = [c for c in cols_of_interest if c in df.columns]
        corr_data = df[cols].copy()

        # Rename columns for cleaner labels
        rename_map = {
            "temperature_mean": "Temp Mean",
            "temperature_max": "Temp Max",
            "precipitation": "Precipitation",
            "solar_radiation": "Solar Rad.",
            "drought_index": "Drought Idx",
            "wind_speed": "Wind Speed",
            "co2_ppm": "CO₂ (ppm)",
            "elevation_m": "Elevation",
        }
        corr_data = corr_data.rename(columns=rename_map)
        cols_renamed = [rename_map.get(c, c.title()) for c in cols]

        # Compute Pearson correlation matrix
        corr_matrix = corr_data.corr(method="pearson")

        # Create mask for upper triangle (to avoid duplicate information)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
        fig.suptitle("🔥  Feature Correlation Analysis",
                     fontsize=16, fontweight="bold")

        # ── Left: Masked heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",      # Red=positive, Blue=negative, White=no correlation
            center=0,
            vmin=-1, vmax=1,
            linewidths=0.5,
            linecolor="white",
            square=True,
            ax=axes[0],
            cbar_kws={"label": "Pearson r", "shrink": 0.8}
        )
        axes[0].set_title("Pearson Correlation Matrix\n(lower triangle)")
        axes[0].tick_params(axis="x", rotation=45)

        # ── Right: NDVI correlations sorted by strength
        ndvi_col = "ndvi" if "ndvi" in corr_matrix.columns else corr_matrix.columns[0]
        ndvi_corr = corr_matrix[ndvi_col].drop(ndvi_col).sort_values()

        colors = [PALETTE["temperature"] if v < 0 else PALETTE["ndvi"]
                  for v in ndvi_corr.values]
        bars = axes[1].barh(ndvi_corr.index, ndvi_corr.values,
                             color=colors, edgecolor="white", linewidth=0.5)
        axes[1].axvline(x=0, color="black", linewidth=1)
        axes[1].set_title("Correlation with NDVI\n(sorted by strength)")
        axes[1].set_xlabel("Pearson Correlation Coefficient (r)")

        # Add value labels
        for bar, val in zip(bars, ndvi_corr.values):
            ha = "right" if val < 0 else "left"
            x = val - 0.01 if val < 0 else val + 0.01
            axes[1].text(x, bar.get_y() + bar.get_height() / 2,
                         f"{val:.3f}", va="center", ha=ha, fontsize=9)

        if save:
            path = IMAGES_DIR / "correlation_heatmap.png"
            fig.savefig(path, bbox_inches="tight", dpi=150)
            print(f"     Saved: {path}")

        return fig

    # ─────────────────────────────────────────────────────────────────────
    # 5. MODEL COMPARISON
    # ─────────────────────────────────────────────────────────────────────

    def plot_model_comparison(self,
                               results: dict,
                               save: bool = True) -> plt.Figure:
        """
        Compare ML model performance using RMSE, MAE, and R².

        Args:
            results: Dict of {model_name: {"rmse": x, "mae": y, "r2": z}}
            save: Save to images/

        Returns:
            matplotlib Figure
        """
        print("  📊 Plotting model comparison...")

        models = list(results.keys())
        metrics = ["rmse", "mae", "r2"]
        metric_labels = ["RMSE ↓ (lower is better)",
                         "MAE ↓ (lower is better)",
                         "R² ↑ (higher is better)"]
        metric_colors = [PALETTE["temperature"], PALETTE["highlight"], PALETTE["ndvi"]]

        fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
        fig.suptitle("🤖  Machine Learning Model Performance Comparison",
                     fontsize=16, fontweight="bold")

        for ax, metric, label, color in zip(axes, metrics, metric_labels, metric_colors):
            values = [results[m][metric] for m in models]

            # Highlight best model
            if metric == "r2":
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)

            bar_colors = [color if i == best_idx else PALETTE["neutral"]
                          for i in range(len(models))]

            bars = ax.bar(models, values, color=bar_colors,
                          edgecolor="white", linewidth=0.5)
            ax.set_title(label)
            ax.set_ylabel(metric.upper())
            ax.tick_params(axis="x", rotation=15)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(values) * 0.01,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                        fontweight="bold" if bar.get_facecolor() != PALETTE["neutral"] else "normal")

            # Mark best model
            ax.text(best_idx, values[best_idx] + max(values) * 0.03,
                    "🏆", ha="center", fontsize=14)

        if save:
            path = IMAGES_DIR / "model_comparison.png"
            fig.savefig(path, bbox_inches="tight", dpi=150)
            print(f"     Saved: {path}")

        return fig

    # ─────────────────────────────────────────────────────────────────────
    # 6. FEATURE IMPORTANCE
    # ─────────────────────────────────────────────────────────────────────

    def plot_feature_importance(self,
                                  feature_names: list,
                                  importances: np.ndarray,
                                  model_name: str = "XGBoost",
                                  top_n: int = 20,
                                  save: bool = True) -> plt.Figure:
        """
        Plot feature importance scores from tree-based models.

        WHAT IS FEATURE IMPORTANCE?
        In tree-based models (Random Forest, XGBoost), each feature gets an
        importance score based on how much it reduces prediction error when
        used as a split point. Higher importance = more predictive power.

        Args:
            feature_names: List of feature names
            importances: Array of importance scores
            model_name: Name for plot title
            top_n: How many top features to display
            save: Save to images/

        Returns:
            matplotlib Figure
        """
        print(f"  🎯 Plotting feature importance for {model_name}...")

        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        sorted_names = [feature_names[i] for i in indices]
        sorted_vals = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

        # Color gradient: top features are darker green
        n = len(sorted_vals)
        colors = plt.cm.Greens(np.linspace(0.4, 0.9, n))[::-1]

        bars = ax.barh(range(n), sorted_vals[::-1],
                       color=colors[::-1], edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(n))
        ax.set_yticklabels(sorted_names[::-1], fontsize=9)
        ax.set_title(f"🎯  {model_name} Feature Importance (Top {top_n})",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Feature Importance Score")

        # Add value labels
        for bar, val in zip(bars, sorted_vals[::-1]):
            ax.text(val + max(sorted_vals) * 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

        if save:
            path = IMAGES_DIR / "feature_importance.png"
            fig.savefig(path, bbox_inches="tight", dpi=150)
            print(f"     Saved: {path}")

        return fig

    # ─────────────────────────────────────────────────────────────────────
    # 7. PREDICTION PLOT
    # ─────────────────────────────────────────────────────────────────────

    def plot_predictions(self,
                          y_test: np.ndarray,
                          y_pred: np.ndarray,
                          y_future: np.ndarray = None,
                          model_name: str = "XGBoost",
                          save: bool = True) -> plt.Figure:
        """
        Plot actual vs predicted values and future forecast.

        Args:
            y_test: True NDVI values
            y_pred: Predicted NDVI values
            y_future: Future predictions (optional)
            model_name: For plot title
            save: Save to images/

        Returns:
            matplotlib Figure
        """
        print("  🔮 Plotting predictions vs actuals...")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
        fig.suptitle(f"🔮  {model_name} — Prediction Analysis",
                     fontsize=16, fontweight="bold")

        # ── Left: Scatter plot (actual vs predicted)
        ax = axes[0]
        ax.scatter(y_test, y_pred, alpha=0.4, s=15,
                   color=PALETTE["ndvi"], edgecolors="none")

        # Perfect prediction line (45° diagonal)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                "r--", linewidth=2, label="Perfect Prediction (y=x)")

        ax.set_xlabel("Actual NDVI")
        ax.set_ylabel("Predicted NDVI")
        ax.set_title("Actual vs Predicted NDVI")
        ax.legend()

        # Add R² annotation
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_pred)
        ax.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
                fontsize=11, fontweight="bold", verticalalignment="top")

        # ── Right: Time series of predictions
        ax = axes[1]
        x = np.arange(len(y_test))
        ax.plot(x, y_test, color=PALETTE["ndvi"], linewidth=1.5,
                alpha=0.8, label="Actual NDVI")
        ax.plot(x, y_pred, color=PALETTE["temperature"], linewidth=1.5,
                alpha=0.8, label="Predicted NDVI", linestyle="--")

        # Residuals shading
        ax.fill_between(x, y_test, y_pred, alpha=0.15, color=PALETTE["highlight"],
                        label="Prediction Error")

        ax.set_xlabel("Sample Index")
        ax.set_ylabel("NDVI")
        ax.set_title("Prediction Timeline")
        ax.legend()

        if save:
            path = IMAGES_DIR / "predictions.png"
            fig.savefig(path, bbox_inches="tight", dpi=150)
            print(f"     Saved: {path}")

        return fig

    # ─────────────────────────────────────────────────────────────────────
    # 8. INTERACTIVE FOLIUM MAP
    # ─────────────────────────────────────────────────────────────────────

    def create_interactive_map(self,
                                 df: pd.DataFrame,
                                 save: bool = True) -> folium.Map:
        """
        Create an interactive Folium map showing spatial NDVI and climate data.

        WHAT IS FOLIUM?
        Folium is a Python library that creates interactive Leaflet.js maps.
        It lets you zoom, pan, and click on map elements in your browser.
        The output is an HTML file that works without any server.

        Map features:
        - Color-coded circles for each location (colored by mean NDVI)
        - Popup windows showing location details on click
        - Heatmap layer showing precipitation intensity
        - Layer control to toggle features

        Args:
            df: DataFrame with latitude, longitude, ndvi, region columns
            save: Save HTML to images/

        Returns:
            folium.Map object
        """
        print("  🗺️  Creating interactive Folium map...")

        # Aggregate spatial summary per location
        spatial_summary = df.groupby("location_id").agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            region=("region", "first"),
            ndvi_mean=("ndvi", "mean"),
            temp_mean=("temperature_mean", "mean"),
            precip_mean=("precipitation", "mean"),
            land_cover=("land_cover", "first") if "land_cover" in df.columns else ("region", "first"),
        ).reset_index()

        # Center the map on the mean coordinates
        map_center = [spatial_summary["latitude"].mean(),
                      spatial_summary["longitude"].mean()]

        # Create map with OpenStreetMap tiles
        m = folium.Map(
            location=map_center,
            zoom_start=4,
            tiles="CartoDB positron",  # Clean, minimal basemap
            control_scale=True
        )

        # ── Add title
        title_html = """
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                    z-index: 1000; background: white; padding: 8px 16px;
                    border-radius: 8px; box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
                    font-family: Arial; font-size: 14px; font-weight: bold; color: #2C3E50;">
            🌿 Climate-Vegetation Analysis — Interactive Map
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))

        # ── NDVI color scale
        def ndvi_to_color(ndvi_val: float) -> str:
            """Map NDVI value to color (red=low, yellow=medium, green=high)."""
            if ndvi_val < 0.15:
                return "#BDC3C7"   # Gray (bare)
            elif ndvi_val < 0.25:
                return "#E67E22"   # Orange (sparse)
            elif ndvi_val < 0.40:
                return "#F1C40F"   # Yellow (shrubland)
            elif ndvi_val < 0.55:
                return "#52BE80"   # Light green (moderate vegetation)
            else:
                return "#1E8449"   # Dark green (dense forest)

        # ── Add location markers
        marker_cluster = MarkerCluster(name="Climate Stations").add_to(m)

        for _, row in spatial_summary.iterrows():
            color = ndvi_to_color(row["ndvi_mean"])

            # Popup HTML with rich formatting
            popup_html = f"""
            <div style="font-family: Arial; min-width: 200px;">
                <h4 style="color: #2C3E50; margin: 0 0 8px 0;">
                    📍 Location {int(row['location_id'])}
                </h4>
                <table style="border-collapse: collapse; width: 100%;">
                    <tr><td style="padding: 2px 8px 2px 0; color: #7F8C8D;">Region:</td>
                        <td><b>{row['region']}</b></td></tr>
                    <tr><td style="padding: 2px 8px 2px 0; color: #7F8C8D;">Coordinates:</td>
                        <td>{row['latitude']:.3f}°N, {row['longitude']:.3f}°E</td></tr>
                    <tr><td style="padding: 2px 8px 2px 0; color: #27AE60;">Mean NDVI:</td>
                        <td><b style="color: #27AE60;">{row['ndvi_mean']:.3f}</b></td></tr>
                    <tr><td style="padding: 2px 8px 2px 0; color: #E74C3C;">Mean Temp:</td>
                        <td><b>{row['temp_mean']:.1f}°C</b></td></tr>
                    <tr><td style="padding: 2px 8px 2px 0; color: #3498DB;">Mean Precip:</td>
                        <td><b>{row['precip_mean']:.1f} mm/mo</b></td></tr>
                </table>
            </div>
            """

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=8,
                color="white",
                weight=1.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                tooltip=f"{row['region']} | NDVI: {row['ndvi_mean']:.3f}",
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(marker_cluster)

        # ── Heatmap layer (based on precipitation)
        heat_data = spatial_summary[["latitude", "longitude", "precip_mean"]].values.tolist()
        HeatMap(
            heat_data,
            name="Precipitation Heatmap",
            radius=25,
            blur=15,
            gradient={"0.4": "blue", "0.65": "cyan", "1": "deepskyblue"},
            min_opacity=0.3
        ).add_to(m)

        # ── Layer control
        folium.LayerControl(collapsed=False).add_to(m)

        # ── Legend
        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                    background: white; padding: 12px; border-radius: 8px;
                    box-shadow: 2px 2px 8px rgba(0,0,0,0.3); font-family: Arial; font-size: 12px;">
            <b>🌿 NDVI Legend</b><br>
            <span style="color: #BDC3C7;">●</span> < 0.15: Bare land<br>
            <span style="color: #E67E22;">●</span> 0.15–0.25: Sparse veg.<br>
            <span style="color: #F1C40F;">●</span> 0.25–0.40: Shrubland<br>
            <span style="color: #52BE80;">●</span> 0.40–0.55: Moderate<br>
            <span style="color: #1E8449;">●</span> > 0.55: Dense forest
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        if save:
            path = IMAGES_DIR / "interactive_map.html"
            m.save(str(path))
            print(f"     Saved: {path}")

        return m

    # ─────────────────────────────────────────────────────────────────────
    # 9. PLOTLY DASHBOARD FIGURES (for Streamlit)
    # ─────────────────────────────────────────────────────────────────────

    def plotly_ndvi_trend(self, df: pd.DataFrame) -> go.Figure:
        """Create interactive Plotly line chart of NDVI trends."""
        df_monthly = df.groupby(["date", "region"])["ndvi"].mean().reset_index()

        fig = px.line(
            df_monthly, x="date", y="ndvi", color="region",
            title="🌿 NDVI Trends by Region (2000–2023)",
            labels={"ndvi": "Mean NDVI", "date": "Date", "region": "Region"},
            color_discrete_map=REGION_COLORS,
            template="plotly_white"
        )
        fig.update_traces(line_width=2)
        fig.update_layout(
            hovermode="x unified",
            legend_title_text="Region",
            yaxis_range=[0, 0.9],
            plot_bgcolor="#F8F9FA"
        )
        return fig

    def plotly_climate_overview(self, df: pd.DataFrame) -> go.Figure:
        """Create 2x2 subplot figure: temp, precip, humidity, solar."""
        df_monthly = df.groupby("date").agg(
            temperature_mean=("temperature_mean", "mean"),
            precipitation=("precipitation", "mean"),
            humidity=("humidity", "mean"),
            solar_radiation=("solar_radiation", "mean")
        ).reset_index()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Temperature (°C)", "Precipitation (mm)",
                            "Humidity (%)", "Solar Radiation (W/m²)"),
            shared_xaxes=False
        )

        traces = [
            (df_monthly["temperature_mean"], "#E74C3C", "Temp Mean", 1, 1),
            (df_monthly["precipitation"], "#3498DB", "Precipitation", 1, 2),
            (df_monthly["humidity"], "#9B59B6", "Humidity", 2, 1),
            (df_monthly["solar_radiation"], "#F39C12", "Solar Rad.", 2, 2),
        ]

        for values, color, name, row, col in traces:
            fig.add_trace(
                go.Scatter(x=df_monthly["date"], y=values,
                           mode="lines", name=name,
                           line=dict(color=color, width=1.5)),
                row=row, col=col
            )

        fig.update_layout(
            title_text="🌡️ Climate Variable Overview",
            showlegend=False,
            template="plotly_white",
            height=500
        )
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import DataLoader

    print("=" * 60)
    print("  VISUALIZATION MODULE TEST")
    print("=" * 60)

    loader = DataLoader()
    df = loader.load_all_data()

    viz = Visualizer()
    viz.plot_temperature_trends(df)
    viz.plot_precipitation_patterns(df)
    viz.plot_ndvi_timeseries(df)
    viz.plot_correlation_heatmap(df)
    viz.create_interactive_map(df)

    print("\n✅ All visualizations saved to images/")
