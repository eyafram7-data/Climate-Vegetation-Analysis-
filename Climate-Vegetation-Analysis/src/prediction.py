"""
=============================================================================
prediction.py — Climate Vegetation Analysis Project
=============================================================================
PURPOSE:
    This module handles future NDVI prediction and scenario analysis.
    Once the best ML model is trained, we use it to:
    1. Generate rolling predictions on historical data
    2. Forecast future NDVI under different climate scenarios
    3. Quantify prediction uncertainty
    4. Analyze NDVI sensitivity to individual climate variables

WHAT IS SCENARIO ANALYSIS?
    Instead of predicting one future, we project three plausible futures:
    - 🟢 Optimistic: Precipitation increases, temperature stable
      (rainfall recovery, sustainable land use)
    - 🟡 Baseline: Current trends continue unchanged
      (business as usual, mild warming)
    - 🔴 Pessimistic: Severe drought, continued warming
      (climate change intensification, land degradation)

    This is analogous to IPCC climate scenarios (SSP1, SSP2, SSP5).

WHAT IS UNCERTAINTY QUANTIFICATION?
    No prediction is perfectly certain. We estimate uncertainty using:
    - Bootstrap sampling: train many models on slightly different data
    - Prediction interval: range that captures 95% of true values
    - This tells decision-makers how confident we should be in forecasts

USAGE:
    from src.prediction import Predictor
    predictor = Predictor(model=best_model, feature_names=features)
    forecast = predictor.forecast_future(df_features, months_ahead=24)
    scenarios = predictor.scenario_analysis(df_features)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# Each scenario defines monthly delta values applied to base climate variables.
# These represent plausible future trajectories over a 24-month forecast period.
# Values are *multipliers* or *additive shifts* applied to the last observed value.

SCENARIOS = {
    "Optimistic": {
        "description": "Increased rainfall, stable temperatures, reduced drought",
        "color": "#27AE60",   # Green
        "temp_trend_monthly": +0.001,     # +0.012°C/year (slow warming)
        "precip_multiplier": 1.005,       # +0.5%/month rainfall increase
        "drought_trend": -0.005,          # Drought index improving
        "co2_monthly_increase": 0.18,     # CO2 still rising, but slowly
        "icon": "🟢",
    },
    "Baseline": {
        "description": "Current trends continue (business as usual)",
        "color": "#F39C12",   # Orange
        "temp_trend_monthly": +0.0018,    # +0.022°C/year (IPCC current rate)
        "precip_multiplier": 1.000,       # No change in precipitation
        "drought_trend": 0.000,           # Drought index unchanged
        "co2_monthly_increase": 0.183,    # ~2.2 ppm/year (current rate)
        "icon": "🟡",
    },
    "Pessimistic": {
        "description": "Severe drought, rapid warming, precipitation decline",
        "color": "#E74C3C",   # Red
        "temp_trend_monthly": +0.004,     # +0.048°C/year (accelerated warming)
        "precip_multiplier": 0.997,       # -0.3%/month rainfall decrease
        "drought_trend": +0.015,          # Drought worsening
        "co2_monthly_increase": 0.22,     # Higher CO2 emissions
        "icon": "🔴",
    },
}


class Predictor:
    """
    Generates NDVI forecasts and scenario analyses using a trained ML model.

    Attributes:
        model: Trained sklearn/xgboost model
        feature_names: List of feature column names the model expects
        forecast_months: Default forecast horizon (months)
    """

    def __init__(self, model, feature_names: list, forecast_months: int = 24):
        """
        Initialize the predictor.

        Args:
            model: A trained sklearn-compatible model (has .predict() method)
            feature_names: Ordered list of feature names (must match training)
            forecast_months: How many months into the future to predict
        """
        self.model = model
        self.feature_names = feature_names
        self.forecast_months = forecast_months
        print(f"✅ Predictor initialized")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Forecast horizon: {forecast_months} months")

    # ─────────────────────────────────────────────────────────────────────
    # ROLLING PREDICTION ON HISTORICAL DATA
    # ─────────────────────────────────────────────────────────────────────

    def predict_historical(self,
                             X: pd.DataFrame,
                             y: pd.Series) -> pd.DataFrame:
        """
        Generate model predictions on the historical test period.

        This is different from training evaluation — we return a DataFrame
        with timestamps attached for time-series plotting.

        Args:
            X: Feature matrix (test set)
            y: True NDVI values

        Returns:
            pd.DataFrame: Columns [actual, predicted, residual, abs_error]
        """
        print("\n🔍 Generating historical predictions...")

        X_arr = X.values if hasattr(X, "values") else X
        y_arr = np.array(y)

        # Predict
        y_pred = self.model.predict(X_arr)

        # Compute error metrics
        residuals = y_arr - y_pred
        abs_errors = np.abs(residuals)

        df_pred = pd.DataFrame({
            "actual": y_arr,
            "predicted": y_pred,
            "residual": residuals,
            "abs_error": abs_errors,
        })

        rmse = np.sqrt(mean_squared_error(y_arr, y_pred))
        r2 = r2_score(y_arr, y_pred)
        print(f"   RMSE: {rmse:.5f} | R²: {r2:.5f}")
        print(f"   Mean absolute error: {abs_errors.mean():.5f}")
        print(f"   Max error: {abs_errors.max():.5f}")

        return df_pred

    # ─────────────────────────────────────────────────────────────────────
    # FUTURE FORECAST
    # ─────────────────────────────────────────────────────────────────────

    def forecast_future(self,
                         df_features: pd.DataFrame,
                         months_ahead: int = None,
                         scenario: str = "Baseline") -> pd.DataFrame:
        """
        Forecast future NDVI values using a specified climate scenario.

        HOW ROLLING FORECASTING WORKS:
        At each step, we:
        1. Take the last known feature values
        2. Apply scenario-based changes (temp shift, precip change, etc.)
        3. Update lag features with the newly predicted NDVI
        4. Feed updated features into the model for next prediction
        5. Repeat for each future month

        This is called 'auto-regressive' forecasting — the model's own
        predictions become inputs for future predictions.

        Args:
            df_features: Feature-engineered historical DataFrame
            months_ahead: Forecast horizon (default: self.forecast_months)
            scenario: One of "Optimistic", "Baseline", "Pessimistic"

        Returns:
            pd.DataFrame: Future predictions with dates and scenario info
        """
        if months_ahead is None:
            months_ahead = self.forecast_months

        if scenario not in SCENARIOS:
            raise ValueError(f"Scenario must be one of {list(SCENARIOS.keys())}")

        scen = SCENARIOS[scenario]
        print(f"\n🔮 Forecasting {months_ahead} months ahead [{scenario}]...")
        print(f"   {scen['icon']} {scen['description']}")

        # ── Get the last row of historical data as our starting point
        # We'll evolve it forward month by month
        last_row = df_features[self.feature_names].iloc[-1].copy()
        last_date = df_features["date"].iloc[-1] if "date" in df_features.columns else pd.Timestamp("2023-12-01")

        # Track recent NDVI predictions for updating lag features
        recent_ndvi = list(df_features["ndvi"].iloc[-6:].values)

        forecast_records = []

        for step in range(months_ahead):
            # ── Advance the date by one month
            forecast_date = last_date + pd.DateOffset(months=step + 1)
            month = forecast_date.month

            # ── Build the feature row for this future step
            row = last_row.copy()

            # Apply scenario-based climate changes
            if "temperature_mean" in self.feature_names:
                row["temperature_mean"] = (
                    row.get("temperature_mean", 25) +
                    scen["temp_trend_monthly"] * (step + 1)
                )
            if "temperature_max" in self.feature_names:
                row["temperature_max"] = row.get("temperature_max", 30) + scen["temp_trend_monthly"] * (step + 1)
            if "temperature_min" in self.feature_names:
                row["temperature_min"] = row.get("temperature_min", 18) + scen["temp_trend_monthly"] * (step + 1)

            if "precipitation" in self.feature_names:
                row["precipitation"] = row.get("precipitation", 60) * (scen["precip_multiplier"] ** (step + 1))
                row["precipitation"] = max(0, row["precipitation"])

            if "drought_index" in self.feature_names:
                row["drought_index"] = np.clip(
                    row.get("drought_index", 0) + scen["drought_trend"] * (step + 1),
                    -4, 4
                )

            if "co2_ppm" in self.feature_names:
                row["co2_ppm"] = row.get("co2_ppm", 415) + scen["co2_monthly_increase"] * (step + 1)

            # Update cyclical month encoding
            if "month_sin" in self.feature_names:
                row["month_sin"] = np.sin(2 * np.pi * month / 12)
            if "month_cos" in self.feature_names:
                row["month_cos"] = np.cos(2 * np.pi * month / 12)

            # Update lag features using recent predictions
            for lag in [1, 2, 3]:
                lag_key = f"ndvi_lag_{lag}"
                if lag_key in self.feature_names and len(recent_ndvi) >= lag:
                    row[lag_key] = recent_ndvi[-lag]

            # Update rolling means using recent predictions
            for window in [3, 6]:
                roll_key = f"ndvi_roll_mean_{window}"
                if roll_key in self.feature_names and len(recent_ndvi) >= window:
                    row[roll_key] = np.mean(recent_ndvi[-window:])

            # ── Prepare feature vector (same order as training)
            feature_vector = np.array([row.get(f, 0) for f in self.feature_names]).reshape(1, -1)

            # ── Predict NDVI for this month
            ndvi_pred = float(self.model.predict(feature_vector)[0])
            ndvi_pred = np.clip(ndvi_pred, 0.0, 0.9)  # Physical constraint

            # ── Store the prediction
            forecast_records.append({
                "date": forecast_date,
                "predicted_ndvi": ndvi_pred,
                "scenario": scenario,
                "step": step + 1,
                "temperature_forecast": row.get("temperature_mean", np.nan),
                "precipitation_forecast": row.get("precipitation", np.nan),
            })

            # ── Update recent NDVI with this prediction (for next lag)
            recent_ndvi.append(ndvi_pred)
            last_row = row  # Roll forward

        forecast_df = pd.DataFrame(forecast_records)
        print(f"   Forecast complete: {len(forecast_df)} months predicted")
        print(f"   NDVI range: {forecast_df['predicted_ndvi'].min():.3f} – {forecast_df['predicted_ndvi'].max():.3f}")
        return forecast_df

    # ─────────────────────────────────────────────────────────────────────
    # SCENARIO ANALYSIS (all 3 scenarios)
    # ─────────────────────────────────────────────────────────────────────

    def scenario_analysis(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Run all three climate scenarios and return combined DataFrame.

        Args:
            df_features: Feature-engineered historical DataFrame

        Returns:
            pd.DataFrame: All scenario forecasts stacked together
        """
        print("\n🌍 Running full scenario analysis (3 scenarios)...")

        all_forecasts = []
        for scenario_name in SCENARIOS:
            forecast = self.forecast_future(df_features, scenario=scenario_name)
            all_forecasts.append(forecast)

        combined = pd.concat(all_forecasts, ignore_index=True)
        print(f"\n   Total forecast records: {len(combined)}")
        return combined

    # ─────────────────────────────────────────────────────────────────────
    # UNCERTAINTY ESTIMATION (Bootstrap)
    # ─────────────────────────────────────────────────────────────────────

    def estimate_uncertainty(self,
                               X_train: np.ndarray,
                               y_train: np.ndarray,
                               X_forecast: pd.DataFrame,
                               n_bootstrap: int = 30) -> dict:
        """
        Estimate prediction uncertainty using bootstrap sampling.

        HOW BOOTSTRAP WORKS:
        1. Sample the training data WITH replacement (n_bootstrap times)
        2. Each bootstrap sample is slightly different from the original
        3. Train a separate model on each bootstrap sample
        4. Generate predictions from all bootstrap models
        5. The spread of predictions = our uncertainty estimate

        A 95% prediction interval = 2.5th to 97.5th percentile of predictions.

        Args:
            X_train: Training features
            y_train: Training targets
            X_forecast: Feature matrix for future time steps
            n_bootstrap: Number of bootstrap iterations (30 is fast but sufficient)

        Returns:
            dict: {"mean": array, "lower_95": array, "upper_95": array}
        """
        print(f"\n📊 Estimating uncertainty with {n_bootstrap} bootstrap samples...")

        from sklearn.ensemble import GradientBoostingRegressor

        if hasattr(X_train, "values"):
            X_train = X_train.values
        if hasattr(X_forecast, "values"):
            X_forecast = X_forecast.values

        y_train = np.array(y_train)
        n_train = len(X_train)

        all_predictions = []

        for i in range(n_bootstrap):
            # Sample with replacement
            boot_indices = np.random.choice(n_train, size=n_train, replace=True)
            X_boot = X_train[boot_indices]
            y_boot = y_train[boot_indices]

            # Train a fast GBM model on bootstrap sample
            boot_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=i
            )
            boot_model.fit(X_boot, y_boot)
            preds = boot_model.predict(X_forecast)
            preds = np.clip(preds, 0.0, 0.9)
            all_predictions.append(preds)

            if (i + 1) % 10 == 0:
                print(f"   Bootstrap {i + 1}/{n_bootstrap} complete")

        all_predictions = np.array(all_predictions)

        result = {
            "mean": all_predictions.mean(axis=0),
            "std": all_predictions.std(axis=0),
            "lower_95": np.percentile(all_predictions, 2.5, axis=0),
            "upper_95": np.percentile(all_predictions, 97.5, axis=0),
            "lower_50": np.percentile(all_predictions, 25, axis=0),
            "upper_50": np.percentile(all_predictions, 75, axis=0),
        }

        mean_width = (result["upper_95"] - result["lower_95"]).mean()
        print(f"   Mean 95% interval width: ±{mean_width / 2:.4f} NDVI units")

        return result

    # ─────────────────────────────────────────────────────────────────────
    # SENSITIVITY ANALYSIS
    # ─────────────────────────────────────────────────────────────────────

    def sensitivity_analysis(self,
                               df_features: pd.DataFrame,
                               variables: list = None) -> pd.DataFrame:
        """
        Analyze NDVI sensitivity to individual climate variables.

        WHAT IS SENSITIVITY ANALYSIS?
        We vary one variable at a time (holding all others constant) and
        observe how much the NDVI prediction changes. This tells us:
        "If precipitation increases by 10%, NDVI changes by X units."
        "If temperature rises by 1°C, NDVI changes by Y units."

        This is called 'partial dependence' analysis.

        Args:
            df_features: Feature-engineered DataFrame
            variables: Which variables to test (default: key climate vars)

        Returns:
            pd.DataFrame: Sensitivity scores per variable
        """
        if variables is None:
            variables = [
                "temperature_mean", "precipitation", "humidity",
                "drought_index", "solar_radiation", "co2_ppm"
            ]

        # Only keep variables that exist in feature set
        variables = [v for v in variables if v in self.feature_names]

        print(f"\n🔬 Running sensitivity analysis on {len(variables)} variables...")

        # Use median row as baseline
        X_base = df_features[self.feature_names].median().values.reshape(1, -1)
        baseline_ndvi = float(self.model.predict(X_base)[0])

        results = []
        perturbation_pct = 0.10  # 10% perturbation

        for var in variables:
            if var not in self.feature_names:
                continue

            var_idx = self.feature_names.index(var)
            base_val = X_base[0, var_idx]
            perturb = base_val * perturbation_pct if base_val != 0 else perturbation_pct

            # Increase variable by 10%
            X_up = X_base.copy()
            X_up[0, var_idx] += perturb
            ndvi_up = float(self.model.predict(X_up)[0])

            # Decrease variable by 10%
            X_down = X_base.copy()
            X_down[0, var_idx] -= perturb
            ndvi_down = float(self.model.predict(X_down)[0])

            sensitivity = (ndvi_up - ndvi_down) / (2 * perturb)
            ndvi_change_up = ndvi_up - baseline_ndvi
            ndvi_change_down = ndvi_down - baseline_ndvi

            results.append({
                "variable": var,
                "base_value": round(base_val, 4),
                "ndvi_when_up_10pct": round(ndvi_up, 5),
                "ndvi_when_down_10pct": round(ndvi_down, 5),
                "ndvi_change_up": round(ndvi_change_up, 5),
                "ndvi_change_down": round(ndvi_change_down, 5),
                "sensitivity_slope": round(sensitivity, 6),
                "abs_sensitivity": round(abs(sensitivity), 6),
            })

        df_sensitivity = pd.DataFrame(results).sort_values(
            "abs_sensitivity", ascending=False
        )

        print("\n   Sensitivity Results (sorted by impact):")
        for _, row in df_sensitivity.iterrows():
            direction = "↑" if row["sensitivity_slope"] > 0 else "↓"
            print(f"   {row['variable']:25s}: {direction} {abs(row['sensitivity_slope']):.5f} NDVI per unit")

        return df_sensitivity

    # ─────────────────────────────────────────────────────────────────────
    # PLOT SCENARIO FORECAST
    # ─────────────────────────────────────────────────────────────────────

    def plot_scenario_forecast(self,
                                 df_historical: pd.DataFrame,
                                 scenario_df: pd.DataFrame,
                                 uncertainty: dict = None,
                                 save: bool = True) -> plt.Figure:
        """
        Create the final forecast visualization showing all 3 scenarios.

        This is the 'money plot' of the project — it shows:
        - Historical NDVI trend (2000-2023)
        - Three future trajectories (optimistic/baseline/pessimistic)
        - Optional uncertainty bands
        - Clear annotation of key features

        Args:
            df_historical: Historical data with 'date' and 'ndvi' columns
            scenario_df: Output from scenario_analysis()
            uncertainty: Output from estimate_uncertainty() (optional)
            save: Save to images/

        Returns:
            matplotlib Figure
        """
        print("\n📉 Plotting scenario forecast...")

        fig, axes = plt.subplots(2, 1, figsize=(16, 12), constrained_layout=True,
                                  gridspec_kw={"height_ratios": [3, 1]})

        fig.suptitle(
            "🔮  NDVI Future Projections Under Climate Scenarios (2024–2025)",
            fontsize=16, fontweight="bold"
        )

        ax_main = axes[0]
        ax_diff = axes[1]

        # ── Historical data (aggregated to monthly mean)
        hist = (
            df_historical.groupby("date")["ndvi"].mean()
            .reset_index().sort_values("date")
        )

        # Plot last 5 years of history for context
        hist_recent = hist[hist["date"] >= "2019-01-01"]
        ax_main.plot(hist_recent["date"], hist_recent["ndvi"],
                     color="#2C3E50", linewidth=2.5,
                     label="Historical NDVI (2019–2023)", zorder=5)

        # Vertical line at forecast start
        forecast_start = scenario_df["date"].min()
        ax_main.axvline(x=forecast_start, color="gray", linestyle="--",
                        linewidth=1.5, alpha=0.7, label="Forecast Start")
        ax_main.text(forecast_start, ax_main.get_ylim()[1] if ax_main.get_ylim()[1] != 1 else 0.75,
                     " Forecast →", color="gray", fontsize=9, va="top")

        # ── Plot each scenario
        baseline_preds = None
        for scenario_name, scen_info in SCENARIOS.items():
            scen_data = scenario_df[scenario_df["scenario"] == scenario_name].sort_values("date")

            ax_main.plot(
                scen_data["date"], scen_data["predicted_ndvi"],
                color=scen_info["color"],
                linewidth=2.5, linestyle="-",
                label=f"{scen_info['icon']} {scenario_name}: {scen_info['description'][:40]}...",
                zorder=4
            )

            # Fill area under each scenario line
            ax_main.fill_between(
                scen_data["date"], scen_data["predicted_ndvi"],
                alpha=0.08, color=scen_info["color"]
            )

            if scenario_name == "Baseline":
                baseline_preds = scen_data["predicted_ndvi"].values

        # ── Uncertainty band (if provided, plot around baseline)
        if uncertainty is not None and baseline_preds is not None:
            baseline_dates = scenario_df[
                scenario_df["scenario"] == "Baseline"
            ].sort_values("date")["date"]

            ax_main.fill_between(
                baseline_dates,
                uncertainty["lower_95"],
                uncertainty["upper_95"],
                alpha=0.12, color="#F39C12",
                label="95% Prediction Interval"
            )
            ax_main.fill_between(
                baseline_dates,
                uncertainty["lower_50"],
                uncertainty["upper_50"],
                alpha=0.20, color="#F39C12",
                label="50% Prediction Interval"
            )

        # ── Main plot formatting
        ax_main.set_ylabel("NDVI (Normalized Difference Vegetation Index)", fontsize=12)
        ax_main.set_ylim(0, 0.85)
        ax_main.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax_main.set_title("NDVI Forecast: Three Climate Scenarios")

        # Add NDVI interpretation zones as horizontal bands
        ndvi_zones = [
            (0.0, 0.15, "#F5CBA7", "Bare/Sparse"),
            (0.15, 0.30, "#FAD7A0", "Low Vegetation"),
            (0.30, 0.50, "#A9DFBF", "Moderate"),
            (0.50, 0.85, "#82E0AA", "Dense Vegetation"),
        ]
        for ymin, ymax, color, label in ndvi_zones:
            ax_main.axhspan(ymin, ymax, alpha=0.05, color=color)
            ax_main.text(
                hist_recent["date"].min(), (ymin + ymax) / 2,
                label, fontsize=7, color="gray", alpha=0.7, va="center"
            )

        # ── Difference plot (Optimistic minus Pessimistic)
        opt = scenario_df[scenario_df["scenario"] == "Optimistic"].sort_values("date")
        pes = scenario_df[scenario_df["scenario"] == "Pessimistic"].sort_values("date")

        if len(opt) == len(pes):
            diff = opt["predicted_ndvi"].values - pes["predicted_ndvi"].values
            dates = opt["date"].values

            ax_diff.bar(dates, diff, color="#27AE60", alpha=0.7,
                        label="Scenario Gap (Optimistic − Pessimistic)",
                        width=20)
            ax_diff.axhline(y=0, color="black", linewidth=1)
            ax_diff.set_ylabel("NDVI Difference")
            ax_diff.set_xlabel("Date")
            ax_diff.set_title("Optimistic vs. Pessimistic NDVI Gap")
            ax_diff.legend(fontsize=9)

            # Annotate max gap
            max_gap_idx = np.argmax(diff)
            ax_diff.annotate(
                f"Max gap: {diff[max_gap_idx]:.3f}",
                xy=(dates[max_gap_idx], diff[max_gap_idx]),
                xytext=(dates[max_gap_idx], diff[max_gap_idx] + 0.005),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=9
            )

        if save:
            path = IMAGES_DIR / "scenario_forecast.png"
            fig.savefig(path, bbox_inches="tight", dpi=150)
            print(f"   Saved: {path}")

        return fig

    # ─────────────────────────────────────────────────────────────────────
    # PLOT SENSITIVITY ANALYSIS
    # ─────────────────────────────────────────────────────────────────────

    def plot_sensitivity(self,
                          df_sensitivity: pd.DataFrame,
                          save: bool = True) -> plt.Figure:
        """
        Plot sensitivity analysis results as a diverging bar chart.

        Args:
            df_sensitivity: Output from sensitivity_analysis()
            save: Save to images/

        Returns:
            matplotlib Figure
        """
        print("\n🔬 Plotting sensitivity analysis...")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
        fig.suptitle("🔬  NDVI Sensitivity to Climate Variable Changes",
                     fontsize=15, fontweight="bold")

        # ── Left: Absolute sensitivity (which variable matters most?)
        ax = axes[0]
        df_plot = df_sensitivity.sort_values("abs_sensitivity", ascending=True)
        colors = ["#E74C3C" if s < 0 else "#27AE60"
                  for s in df_plot["sensitivity_slope"]]

        ax.barh(df_plot["variable"], df_plot["abs_sensitivity"],
                color=colors, edgecolor="white", linewidth=0.3)
        ax.set_title("Absolute Sensitivity\n(NDVI change per unit change in variable)")
        ax.set_xlabel("|dNDVI / d(variable)|")

        # Add direction indicators
        for i, (_, row) in enumerate(df_plot.iterrows()):
            direction = "▲ positive" if row["sensitivity_slope"] > 0 else "▼ negative"
            ax.text(row["abs_sensitivity"] + row["abs_sensitivity"] * 0.02,
                    i, direction, va="center", fontsize=7, color="gray")

        # ── Right: +10% vs -10% impact on NDVI
        ax = axes[1]
        df_plot2 = df_sensitivity.sort_values("abs_sensitivity", ascending=False).head(8)

        x = np.arange(len(df_plot2))
        width = 0.35

        bars_up = ax.bar(x - width / 2, df_plot2["ndvi_change_up"],
                         width, label="+10% change", color="#27AE60", alpha=0.8,
                         edgecolor="white")
        bars_down = ax.bar(x + width / 2, df_plot2["ndvi_change_down"],
                            width, label="-10% change", color="#E74C3C", alpha=0.8,
                            edgecolor="white")

        ax.axhline(y=0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot2["variable"], rotation=30, ha="right", fontsize=9)
        ax.set_title("NDVI Response to ±10% Variable Change")
        ax.set_ylabel("NDVI Change (absolute)")
        ax.legend()

        if save:
            path = IMAGES_DIR / "sensitivity_analysis.png"
            fig.savefig(path, bbox_inches="tight", dpi=150)
            print(f"   Saved: {path}")

        return fig


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import DataLoader
    from preprocessing import Preprocessor
    from model import ModelTrainer

    print("=" * 60)
    print("  PREDICTION MODULE TEST")
    print("=" * 60)

    # Load and preprocess
    loader = DataLoader()
    df = loader.load_all_data()
    prep = Preprocessor()
    result = prep.run_full_pipeline(df, scale=True)

    # Train models
    trainer = ModelTrainer()
    model_results = trainer.train_all_models(
        result["X_train_scaled"], result["y_train"],
        result["X_test_scaled"], result["y_test"],
        feature_names=result["feature_names"]
    )
    trainer.print_leaderboard()

    # Initialize predictor with best model
    predictor = Predictor(
        model=trainer.best_model,
        feature_names=result["feature_names"],
        forecast_months=24
    )

    # Run scenario analysis
    df_features = result["df_features"]
    df_features["date"] = pd.to_datetime(df_features["date"]) if "date" in df_features.columns else None

    scenarios = predictor.scenario_analysis(df_features)
    sensitivity = predictor.sensitivity_analysis(df_features)

    # Plot results
    if "date" in df_features.columns:
        predictor.plot_scenario_forecast(df_features, scenarios)
    predictor.plot_sensitivity(sensitivity)

    print("\n✅ Prediction module complete!")
