"""
=============================================================================
preprocessing.py — Climate Vegetation Analysis Project
=============================================================================
PURPOSE:
    Data rarely arrives clean and analysis-ready. This module handles:
    1. Data quality checks (missing values, outliers, duplicates)
    2. Data cleaning (imputation, filtering)
    3. Feature engineering (creating new predictive variables)
    4. Data normalization/standardization (for ML models)
    5. Train/test splitting

WHAT IS FEATURE ENGINEERING?
    Feature engineering means creating new variables from existing ones that
    better capture the underlying patterns. For example:
    - Lag features: "What was the precipitation 3 months ago?" (plants
      respond to past rainfall, not just current)
    - Rolling averages: "What's the 6-month average temperature?" (captures
      medium-term climate stress)
    - Interaction terms: "How does rainfall × temperature affect vegetation?"

DESIGN PHILOSOPHY:
    Each function is small, focused, and documented. This makes it easy to
    test each step independently and understand what's happening.

USAGE:
    from src.preprocessing import Preprocessor
    prep = Preprocessor()
    df_clean = prep.clean_data(df)
    df_features = prep.engineer_features(df_clean)
    X_train, X_test, y_train, y_test = prep.split_data(df_features)
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


class Preprocessor:
    """
    Handles all data preprocessing and feature engineering.

    Attributes:
        scaler: sklearn scaler fitted on training data
        feature_columns: List of feature names used in ML
        target_column: The variable we're trying to predict (NDVI)
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = "ndvi"
        self.is_fitted = False
        print("✅ Preprocessor initialized")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: DATA QUALITY CHECK
    # ─────────────────────────────────────────────────────────────────────

    def quality_check(self, df: pd.DataFrame) -> dict:
        """
        Run a comprehensive data quality assessment.

        Think of this as the 'health checkup' for your dataset before any
        analysis begins. We check for common data problems.

        Args:
            df: Input DataFrame

        Returns:
            dict: Quality metrics report
        """
        print("\n📋 Running data quality check...")

        report = {}

        # 1. Dataset shape
        report["shape"] = df.shape
        report["n_rows"] = df.shape[0]
        report["n_cols"] = df.shape[1]

        # 2. Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        report["missing_values"] = missing[missing > 0].to_dict()
        report["missing_pct"] = missing_pct[missing_pct > 0].to_dict()

        # 3. Duplicate rows
        report["duplicates"] = df.duplicated().sum()

        # 4. NDVI range check (should be 0.0 to 0.9 for vegetation)
        if "ndvi" in df.columns:
            ndvi_out_of_range = ((df["ndvi"] < 0) | (df["ndvi"] > 1)).sum()
            report["ndvi_outliers"] = int(ndvi_out_of_range)
            report["ndvi_range"] = (df["ndvi"].min(), df["ndvi"].max())
            report["ndvi_mean"] = df["ndvi"].mean().round(4)

        # 5. Temperature range check (shouldn't be below -50 or above 60°C)
        if "temperature_mean" in df.columns:
            temp_invalid = ((df["temperature_mean"] < -50) |
                            (df["temperature_mean"] > 60)).sum()
            report["temp_invalid"] = int(temp_invalid)

        # 6. Date coverage
        if "date" in df.columns:
            report["date_range"] = (str(df["date"].min()), str(df["date"].max()))
            report["n_months"] = df["date"].nunique()

        # Print summary
        print(f"   Shape: {report['shape']}")
        print(f"   Missing values: {sum(report['missing_values'].values()) if report['missing_values'] else 0} cells")
        print(f"   Duplicates: {report['duplicates']}")
        if "ndvi_range" in report:
            print(f"   NDVI range: {report['ndvi_range'][0]:.3f} – {report['ndvi_range'][1]:.3f}")

        return report

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: DATA CLEANING
    # ─────────────────────────────────────────────────────────────────────

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw dataset.

        Cleaning steps:
        1. Remove exact duplicate rows
        2. Filter poor-quality NDVI observations (cloud contamination)
        3. Clip NDVI to valid vegetation range
        4. Impute missing values using forward-fill (temporal interpolation)
        5. Remove any remaining rows with NaN values

        Args:
            df: Raw input DataFrame

        Returns:
            pd.DataFrame: Cleaned dataset
        """
        print("\n🧹 Cleaning data...")
        original_len = len(df)

        df = df.copy()

        # ── Step 1: Remove duplicate rows
        df = df.drop_duplicates()
        print(f"   Removed {original_len - len(df)} duplicates")

        # ── Step 2: Filter poor-quality NDVI data
        # In real MODIS data, poor-quality pixels are cloud-contaminated.
        # We exclude them rather than use unreliable values.
        if "ndvi_quality" in df.columns:
            before = len(df)
            df = df[df["ndvi_quality"] != "poor"]
            print(f"   Removed {before - len(df)} poor-quality NDVI observations")

        # ── Step 3: Clip NDVI to valid range
        # While NDVI can theoretically go negative (water), we focus on land
        # vegetation, so we clip to 0.0–0.9
        df["ndvi"] = df["ndvi"].clip(0.0, 0.90)

        # ── Step 4: Handle missing values using forward fill within each location
        # Forward fill means: fill a missing value with the previous valid value.
        # This is appropriate for monthly time series (last month's NDVI ≈ this month's)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df.sort_values(["location_id", "date"])

        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df.groupby("location_id")[col].transform(
                    lambda x: x.ffill().bfill()
                )

        # ── Step 5: Drop any remaining NaN rows (shouldn't be many)
        before = len(df)
        df = df.dropna(subset=["ndvi", "temperature_mean", "precipitation"])
        if before > len(df):
            print(f"   Dropped {before - len(df)} rows with remaining NaN values")

        print(f"   ✅ Clean dataset: {len(df):,} rows ({len(df)/original_len*100:.1f}% retained)")
        return df.reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────────────────

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing data.

        Feature engineering is often the most impactful step in ML.
        Good features = better model performance.

        Features created:
        ── Temporal
            - month_sin, month_cos: Cyclical encoding of month
        ── Lag features (vegetation memory)
            - ndvi_lag_1, ndvi_lag_2, ndvi_lag_3: NDVI from 1-3 months ago
            - precip_lag_1, precip_lag_2: Rainfall 1-2 months ago
        ── Rolling statistics (recent trends)
            - ndvi_roll_mean_3, ndvi_roll_mean_6: Rolling average NDVI
            - precip_roll_sum_3: Cumulative 3-month rainfall
            - temp_roll_mean_3: Rolling 3-month temperature
        ── Interaction features
            - temp_x_precip: Temperature × precipitation
            - vpd: Vapor Pressure Deficit (temperature + humidity)
        ── Climate anomaly features
            - temp_anomaly: Deviation from monthly climatological mean
            - precip_anomaly: Deviation from monthly average

        Args:
            df: Cleaned DataFrame

        Returns:
            pd.DataFrame: DataFrame with new engineered features
        """
        print("\n⚙️  Engineering features...")
        df = df.copy().sort_values(["location_id", "date"])

        # ── CYCLICAL MONTH ENCODING
        # Important: Month 12 and Month 1 are adjacent, but if we use raw
        # month numbers (1-12), the model thinks they are 11 apart.
        # Sine/cosine encoding captures the circular nature of months.
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        print("   ✓ Cyclical month encoding")

        # ── LAG FEATURES
        # Plants respond to past climate conditions, not just current ones.
        # This is called the "vegetation memory" effect.
        for lag in [1, 2, 3]:
            df[f"ndvi_lag_{lag}"] = df.groupby("location_id")["ndvi"].shift(lag)
            df[f"precip_lag_{lag}"] = df.groupby("location_id")["precipitation"].shift(lag)

        df["temp_lag_1"] = df.groupby("location_id")["temperature_mean"].shift(1)
        print("   ✓ Lag features (1-3 months)")

        # ── ROLLING STATISTICS
        # Smooth out monthly variability to capture medium-term trends.
        # min_periods=1 allows calculation even at the start of the series.
        for window in [3, 6]:
            df[f"ndvi_roll_mean_{window}"] = (
                df.groupby("location_id")["ndvi"]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            df[f"precip_roll_sum_{window}"] = (
                df.groupby("location_id")["precipitation"]
                .transform(lambda x: x.rolling(window, min_periods=1).sum())
            )

        df["temp_roll_mean_3"] = (
            df.groupby("location_id")["temperature_mean"]
            .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )
        print("   ✓ Rolling statistics (3-month and 6-month windows)")

        # ── VAPOR PRESSURE DEFICIT (VPD)
        # VPD measures atmospheric dryness — a key driver of plant water stress.
        # Higher VPD = more atmospheric demand for water = more plant stress.
        # Simplified approximation: VPD ≈ saturation_VP × (1 - RH/100)
        # Where saturation_VP = 0.6108 × exp(17.27T / (T + 237.3)) kPa
        T = df["temperature_mean"]
        RH = df["humidity"]
        sat_vp = 0.6108 * np.exp(17.27 * T / (T + 237.3))
        df["vpd_kpa"] = sat_vp * (1 - RH / 100)
        print("   ✓ Vapor pressure deficit (VPD)")

        # ── TEMPERATURE × PRECIPITATION INTERACTION
        # The combined effect of heat and drought is often worse than either alone
        df["temp_x_precip"] = df["temperature_mean"] * df["precipitation"]

        # ── ARIDITY INDEX
        # Aridity Index = Precipitation / Potential Evapotranspiration
        # We approximate PET using temperature (Thornthwaite method)
        # AI < 0.2: Desert, 0.2-0.5: Semi-arid, 0.5-0.75: Sub-humid
        pet_approx = 0.0023 * (T + 17.8) * (df["solar_radiation"] / 30)
        pet_approx = pet_approx.replace(0, 0.001)
        df["aridity_index"] = df["precipitation"] / pet_approx.clip(lower=0.001)
        df["aridity_index"] = df["aridity_index"].clip(0, 10)
        print("   ✓ Aridity index")

        # ── CLIMATE ANOMALIES
        # Anomaly = actual value - climatological mean for that month
        # This removes the seasonal cycle and highlights unusual conditions
        df["temp_anomaly"] = (
            df["temperature_mean"] -
            df.groupby(["location_id", "month"])["temperature_mean"].transform("mean")
        )
        df["precip_anomaly"] = (
            df["precipitation"] -
            df.groupby(["location_id", "month"])["precipitation"].transform("mean")
        )
        print("   ✓ Climate anomaly features")

        # ── NDVI CHANGE RATE
        # How fast is vegetation changing? (1st derivative of NDVI)
        df["ndvi_change"] = df.groupby("location_id")["ndvi"].diff()
        print("   ✓ NDVI change rate")

        # ── REGION ENCODING (one-hot)
        # Machine learning models need numbers, not text categories.
        # One-hot encoding creates a binary column for each region.
        df = pd.get_dummies(df, columns=["region"], prefix="region", drop_first=True)
        print("   ✓ Region one-hot encoding")

        # ── DROP ROWS WITH NaN FROM LAGGING (first few months of each location)
        before = len(df)
        df = df.dropna()
        print(f"   Dropped {before - len(df)} rows due to lag/rolling NaN values")

        # Save engineered features
        df.to_csv(DATA_PROCESSED / "features_engineered.csv", index=False)
        print(f"\n   ✅ Feature engineering complete: {df.shape[1]} features, {len(df):,} samples")

        return df.reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: FEATURE SELECTION
    # ─────────────────────────────────────────────────────────────────────

    def select_features(self, df: pd.DataFrame) -> tuple:
        """
        Define which columns are features (X) and which is the target (y).

        We exclude:
        - Date columns (non-numeric time info)
        - ID columns
        - Quality flags
        - Redundant columns

        Args:
            df: Feature-engineered DataFrame

        Returns:
            tuple: (X DataFrame of features, y Series of target)
        """
        print("\n🎯 Selecting features...")

        # Columns to exclude from features
        exclude_cols = [
            "date", "location_id", "land_cover", "ndvi_quality",
            "ndvi",          # This is our TARGET, not a feature
            "ndvi_change",   # Derived from future knowledge — exclude to prevent leakage
        ]

        # Feature columns = everything not excluded
        self.feature_columns = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in [np.float64, np.int64, bool]
        ]

        X = df[self.feature_columns]
        y = df[self.target_column]

        print(f"   Features selected: {len(self.feature_columns)}")
        print(f"   Feature names: {self.feature_columns[:8]}...")
        print(f"   Target: {self.target_column}")
        print(f"   X shape: {X.shape}, y shape: {y.shape}")

        return X, y

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5: TRAIN/TEST SPLIT
    # ─────────────────────────────────────────────────────────────────────

    def split_data(self,
                    X: pd.DataFrame,
                    y: pd.Series,
                    test_size: float = 0.2,
                    method: str = "temporal") -> tuple:
        """
        Split data into training and testing sets.

        WHY NOT RANDOM SPLIT FOR TIME SERIES?
        For time series data, we must split by time, not randomly.
        If we randomly shuffle, the model can "see the future" through
        lag features derived from test data points. This would make
        performance metrics unrealistically high (data leakage).

        We use the last 20% of time steps as the test set.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data for testing (0.2 = 20%)
            method: 'temporal' (recommended) or 'random'

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print(f"\n✂️  Splitting data ({method} split, test_size={test_size})...")

        if method == "temporal":
            # Temporal split: use last 20% of rows as test set
            split_idx = int(len(X) * (1 - test_size))
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
        else:
            # Random split (use only for non-temporal baseline comparison)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        print(f"   Training samples: {len(X_train):,}")
        print(f"   Test samples:     {len(X_test):,}")

        return X_train, X_test, y_train, y_test

    # ─────────────────────────────────────────────────────────────────────
    # STEP 6: SCALING
    # ─────────────────────────────────────────────────────────────────────

    def scale_features(self,
                        X_train: pd.DataFrame,
                        X_test: pd.DataFrame) -> tuple:
        """
        Standardize features to zero mean and unit variance.

        WHY SCALE?
        Some ML algorithms (SVR, Linear Regression) are sensitive to the
        magnitude of features. Temperature (0-40°C) and CO2 (370-420 ppm)
        are on very different scales. Standardization makes them comparable.

        IMPORTANT: Fit the scaler ONLY on training data. Then transform both
        train and test. This prevents 'data leakage' — where information from
        the test set influences training.

        Formula: x_scaled = (x - mean_train) / std_train

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            tuple: (X_train_scaled, X_test_scaled) as numpy arrays
        """
        print("\n📏 Scaling features with StandardScaler...")

        # Fit on training data ONLY
        X_train_scaled = self.scaler.fit_transform(X_train)
        # Transform test data using training statistics
        X_test_scaled = self.scaler.transform(X_test)

        self.is_fitted = True
        print(f"   Train shape: {X_train_scaled.shape}")
        print(f"   Test shape:  {X_test_scaled.shape}")

        return X_train_scaled, X_test_scaled

    # ─────────────────────────────────────────────────────────────────────
    # CONVENIENCE: FULL PIPELINE
    # ─────────────────────────────────────────────────────────────────────

    def run_full_pipeline(self, df: pd.DataFrame, scale: bool = True) -> dict:
        """
        Run the complete preprocessing pipeline in one call.

        This is the 'one-stop shop' method that runs all steps in order:
        clean → engineer → select → split → (optionally) scale

        Args:
            df: Raw input DataFrame
            scale: Whether to apply StandardScaler

        Returns:
            dict: Contains X_train, X_test, y_train, y_test (scaled if requested)
        """
        print("\n" + "=" * 50)
        print("  FULL PREPROCESSING PIPELINE")
        print("=" * 50)

        # Run pipeline steps
        report = self.quality_check(df)
        df_clean = self.clean_data(df)
        df_features = self.engineer_features(df_clean)
        X, y = self.select_features(df_features)
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        result = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": self.feature_columns,
            "quality_report": report,
            "df_features": df_features,
        }

        if scale:
            X_train_s, X_test_s = self.scale_features(X_train, X_test)
            result["X_train_scaled"] = X_train_s
            result["X_test_scaled"] = X_test_s

        print("\n✅ Preprocessing pipeline complete!")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import DataLoader

    print("=" * 60)
    print("  PREPROCESSING MODULE TEST")
    print("=" * 60)

    # Load data
    loader = DataLoader()
    df = loader.load_all_data()

    # Run preprocessing
    prep = Preprocessor()
    result = prep.run_full_pipeline(df)

    print(f"\n📦 Output keys: {list(result.keys())}")
    print(f"   X_train shape: {result['X_train'].shape}")
    print(f"   y_train range: {result['y_train'].min():.3f} – {result['y_train'].max():.3f}")
