"""
=============================================================================
data_loader.py — Climate Vegetation Analysis Project
=============================================================================
PURPOSE:
    This module handles all data acquisition for the project. It serves two
    roles:
    1. Download real datasets from NASA, NOAA, and ERA5 (when APIs available)
    2. Generate realistic synthetic data when real data is unavailable

WHY SYNTHETIC DATA?
    Real satellite and climate datasets require API keys, large storage space,
    and complex preprocessing. For learning and demonstration purposes, we
    generate synthetic data that preserves the statistical properties and
    seasonal patterns of real-world climate-vegetation data.

WHAT IS NDVI?
    NDVI (Normalized Difference Vegetation Index) is a satellite-derived
    measure of vegetation health. It ranges from -1 to +1:
    - Values < 0: Water, ice, or bare rock
    - Values 0.0-0.2: Sparse vegetation or bare soil
    - Values 0.2-0.4: Shrubland or grassland
    - Values 0.4-0.7: Moderate to dense vegetation (forests)
    - Values > 0.7: Dense tropical forest

USAGE:
    from src.data_loader import DataLoader
    loader = DataLoader()
    df = loader.load_all_data()
=============================================================================
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# Define study parameters — these match real-world dataset specifications
# ─────────────────────────────────────────────────────────────────────────────

# Time range: 2000-2023 (matches MODIS NDVI Terra product availability)
START_YEAR = 2000
END_YEAR = 2023

# Study region: Sub-Saharan Africa (Sahel region) — a well-documented
# region showing clear climate-vegetation coupling
STUDY_REGIONS = {
    "Sahel":       {"lat": (10, 20),  "lon": (-10, 30), "base_ndvi": 0.25},
    "Savanna":     {"lat": (5,  15),  "lon": (-5,  35), "base_ndvi": 0.45},
    "Rainforest":  {"lat": (-5, 5),   "lon": (10,  30), "base_ndvi": 0.70},
    "Semi-Arid":   {"lat": (20, 30),  "lon": (-5,  25), "base_ndvi": 0.15},
    "Mediterranean": {"lat": (30, 40),"lon": (-5,  40), "base_ndvi": 0.35},
}

# Paths — using pathlib for cross-platform compatibility
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Ensure directories exist
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def generate_date_range(start_year: int, end_year: int, freq: str = "MS") -> pd.DatetimeIndex:
    """
    Generate a monthly date range for the study period.

    Args:
        start_year (int): First year of data
        end_year (int): Last year of data
        freq (str): Frequency — 'MS' means Month Start

    Returns:
        pd.DatetimeIndex: Array of monthly timestamps

    Example:
        dates = generate_date_range(2000, 2023)
        # Returns: DatetimeIndex from 2000-01-01 to 2023-12-01
    """
    return pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-01",
        freq=freq
    )


def add_seasonal_signal(dates: pd.DatetimeIndex,
                         amplitude: float,
                         phase_shift: float = 0.0) -> np.ndarray:
    """
    Generate a sinusoidal seasonal signal.

    Most climate and vegetation variables show annual cycles. We model
    this with a sine wave: value = amplitude * sin(2π * month/12 + phase)

    Args:
        dates: Array of timestamps
        amplitude: Strength of the seasonal cycle
        phase_shift: Offset in radians (shifts when peak occurs)

    Returns:
        np.ndarray: Seasonal component values
    """
    # Convert month number (1-12) to radians on a circle
    # Dividing by 12 maps one year to 2π radians
    months = dates.month
    seasonal = amplitude * np.sin(2 * np.pi * months / 12 + phase_shift)
    return seasonal


def add_long_term_trend(n_points: int,
                         trend_per_year: float,
                         freq: str = "monthly") -> np.ndarray:
    """
    Add a linear long-term trend to a time series.

    Climate change has caused measurable shifts in temperature (+0.02°C/yr)
    and NDVI (-0.001/yr in semi-arid regions, +0.002/yr in boreal zones).

    Args:
        n_points (int): Number of time steps
        trend_per_year (float): Annual trend magnitude
        freq (str): 'monthly' or 'annual'

    Returns:
        np.ndarray: Trend component
    """
    # Convert annual trend to per-timestep trend
    points_per_year = 12 if freq == "monthly" else 1
    trend_per_step = trend_per_year / points_per_year
    return np.linspace(0, trend_per_step * n_points, n_points)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DATA LOADER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Handles all data loading and generation for the Climate-Vegetation project.

    This class follows the design pattern of a 'data pipeline entry point':
    - First tries to load existing processed data
    - Falls back to loading raw data
    - Falls back to generating synthetic data

    Attributes:
        random_seed (int): For reproducibility
        n_locations (int): Number of spatial grid points to generate
    """

    def __init__(self, random_seed: int = 42, n_locations: int = 50):
        """
        Initialize the DataLoader.

        Args:
            random_seed: Set this for reproducible results
            n_locations: How many spatial grid cells to simulate
        """
        self.random_seed = random_seed
        self.n_locations = n_locations
        np.random.seed(random_seed)
        print("✅ DataLoader initialized")
        print(f"   Study period: {START_YEAR}–{END_YEAR}")
        print(f"   Spatial locations: {n_locations}")

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC METHOD: load_all_data
    # ─────────────────────────────────────────────────────────────────────

    def load_all_data(self, force_regenerate: bool = False) -> pd.DataFrame:
        """
        Main entry point: load or generate all data.

        Checks for existing processed data first. If not found, generates
        fresh synthetic data and saves it to disk.

        Args:
            force_regenerate (bool): If True, regenerate even if data exists

        Returns:
            pd.DataFrame: Complete merged dataset with all features
        """
        processed_path = DATA_PROCESSED / "climate_vegetation_merged.csv"

        # Check if processed data already exists
        if processed_path.exists() and not force_regenerate:
            print(f"📂 Loading existing data from: {processed_path}")
            df = pd.read_csv(processed_path, parse_dates=["date"])
            print(f"   Loaded {len(df):,} records")
            return df

        print("🔄 Generating synthetic climate-vegetation dataset...")

        # Generate each component
        ndvi_df = self.generate_ndvi_data()
        climate_df = self.generate_climate_data()
        spatial_df = self.generate_spatial_data()

        # Merge all components into one dataset
        df = self._merge_datasets(ndvi_df, climate_df, spatial_df)

        # Save to disk
        df.to_csv(processed_path, index=False)
        print(f"💾 Data saved to: {processed_path}")
        print(f"   Total records: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")

        return df

    # ─────────────────────────────────────────────────────────────────────
    # NDVI DATA GENERATION
    # ─────────────────────────────────────────────────────────────────────

    def generate_ndvi_data(self) -> pd.DataFrame:
        """
        Generate synthetic NDVI time series mimicking NASA MODIS data.

        MODIS (Moderate Resolution Imaging Spectroradiometer) provides
        monthly global NDVI at 1km resolution since February 2000.

        The synthetic data includes:
        1. Base NDVI level (varies by region/ecosystem type)
        2. Annual seasonal cycle (growing season peaks)
        3. Long-term declining trend (climate stress)
        4. Inter-annual variability (El Niño, drought years)
        5. Random noise (sensor noise, cloud contamination)

        Returns:
            pd.DataFrame: Monthly NDVI data with columns:
                          [date, location_id, region, ndvi, ndvi_quality]
        """
        print("  🌿 Generating NDVI data...")

        dates = generate_date_range(START_YEAR, END_YEAR)
        n_dates = len(dates)
        records = []

        for loc_id in range(self.n_locations):
            # Assign each location to a region
            region_name = list(STUDY_REGIONS.keys())[loc_id % len(STUDY_REGIONS)]
            region = STUDY_REGIONS[region_name]

            # ── Base NDVI level (ecosystem baseline)
            base_ndvi = region["base_ndvi"]
            base_ndvi += np.random.uniform(-0.05, 0.05)  # location variability

            # ── Seasonal cycle
            # NDVI peaks during rainy/growing season
            # Phase shift = π/2 → peak in month 4 (April) for Northern Hemisphere
            seasonal = add_seasonal_signal(
                dates,
                amplitude=0.12 * base_ndvi / 0.45,  # amplitude scales with density
                phase_shift=np.pi / 2
            )

            # ── Long-term trend
            # Semi-arid regions show greening trends in some areas (Sahel greening)
            # and browning trends in others (deforestation zones)
            trend_sign = 1 if region_name in ["Sahel", "Savanna"] else -1
            trend = add_long_term_trend(n_dates, trend_per_year=trend_sign * 0.002)

            # ── Inter-annual variability (e.g., ENSO, drought)
            # Uses a slow-varying random walk to simulate multi-year patterns
            interannual = np.cumsum(np.random.normal(0, 0.003, n_dates))
            interannual -= interannual.mean()  # center around zero

            # ── Random noise (sensor noise, cloud artifacts)
            noise = np.random.normal(0, 0.015, n_dates)

            # ── Combine all components
            ndvi = base_ndvi + seasonal + trend + interannual + noise

            # ── Clip to valid NDVI range (0.0 to 0.9 for vegetated land)
            ndvi = np.clip(ndvi, 0.0, 0.90)

            # ── Data quality flag (simulating cloud contamination)
            # MODIS uses a quality bit flag; we simulate with a binary variable
            quality = np.random.choice(
                ["good", "marginal", "poor"],
                size=n_dates,
                p=[0.75, 0.18, 0.07]
            )

            # Create records for this location
            for i, date in enumerate(dates):
                records.append({
                    "date": date,
                    "location_id": loc_id,
                    "region": region_name,
                    "ndvi": round(ndvi[i], 4),
                    "ndvi_quality": quality[i]
                })

        df = pd.DataFrame(records)
        df.to_csv(DATA_RAW / "ndvi_raw.csv", index=False)
        print(f"     NDVI records generated: {len(df):,}")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # CLIMATE DATA GENERATION
    # ─────────────────────────────────────────────────────────────────────

    def generate_climate_data(self) -> pd.DataFrame:
        """
        Generate synthetic climate data mimicking NOAA/ERA5 records.

        Variables generated:
        - temperature_mean, temperature_max, temperature_min (°C)
        - precipitation (mm/month)
        - humidity (%)
        - solar_radiation (W/m²)
        - drought_index (Palmer Drought Severity Index, -4 to +4)
        - wind_speed (m/s)
        - co2_ppm (atmospheric CO₂ concentration)

        All variables follow known climatological relationships:
        - Temperature rises over time (global warming trend)
        - Precipitation shows high variability
        - Solar radiation follows strict annual cycle
        - CO₂ increases monotonically (Keeling Curve pattern)

        Returns:
            pd.DataFrame: Monthly climate records
        """
        print("  🌡️ Generating climate data...")

        dates = generate_date_range(START_YEAR, END_YEAR)
        n_dates = len(dates)
        records = []

        for loc_id in range(self.n_locations):
            region_name = list(STUDY_REGIONS.keys())[loc_id % len(STUDY_REGIONS)]
            region = STUDY_REGIONS[region_name]

            # ── Base temperature by latitude band
            lat_center = np.mean(region["lat"])
            # Tropics: ~28°C, mid-latitudes: ~15°C
            base_temp = 30 - 0.4 * abs(lat_center - 5)

            # ── Temperature seasonal cycle
            # In tropics, seasonal range is small (~3°C); larger at higher latitudes
            temp_amplitude = 3 + 0.3 * abs(lat_center - 10)
            temp_seasonal = add_seasonal_signal(dates, temp_amplitude, phase_shift=0)

            # ── Long-term warming trend (IPCC AR6: ~0.02°C/year)
            temp_trend = add_long_term_trend(n_dates, trend_per_year=0.022)

            # ── Random variability (year-to-year temperature anomalies)
            temp_noise = np.random.normal(0, 0.8, n_dates)

            temperature_mean = base_temp + temp_seasonal + temp_trend + temp_noise

            # Max and min temperatures (with diurnal range based on region)
            diurnal_range = 10 + np.random.normal(0, 1, n_dates)
            temperature_max = temperature_mean + diurnal_range / 2
            temperature_min = temperature_mean - diurnal_range / 2

            # ── Precipitation
            # Semi-arid gets 200-400mm/yr, Rainforest 1500-3000mm/yr
            base_precip_annual = {
                "Sahel": 350, "Savanna": 800, "Rainforest": 1800,
                "Semi-Arid": 200, "Mediterranean": 500
            }
            base_precip_monthly = base_precip_annual[region_name] / 12

            # Precipitation has a strong seasonal cycle and high variability
            precip_seasonal = add_seasonal_signal(dates, base_precip_monthly * 0.6, np.pi / 2)
            precip_noise = np.random.exponential(base_precip_monthly * 0.3, n_dates)
            precipitation = np.maximum(0, base_precip_monthly + precip_seasonal + precip_noise)

            # ── Humidity (correlated with precipitation)
            base_humidity = {"Sahel": 45, "Savanna": 60, "Rainforest": 82,
                             "Semi-Arid": 35, "Mediterranean": 55}[region_name]
            humidity_noise = np.random.normal(0, 5, n_dates)
            humidity = np.clip(base_humidity + 0.01 * precipitation + humidity_noise, 10, 100)

            # ── Solar Radiation (W/m²)
            # Peaks at June solstice in Northern Hemisphere (~250-350 W/m²)
            solar_base = 220 + 10 * (1 - abs(lat_center) / 90)
            solar_seasonal = add_seasonal_signal(dates, 60, phase_shift=np.pi)
            solar_noise = np.random.normal(0, 10, n_dates)
            solar_radiation = solar_base + solar_seasonal + solar_noise

            # ── Palmer Drought Severity Index
            # Calculated as: (precip anomaly - temp-driven demand) / variability
            precip_anomaly = precipitation - base_precip_monthly
            drought_index = (precip_anomaly / 50) - (temp_trend * 0.5)
            drought_index += np.random.normal(0, 0.5, n_dates)
            drought_index = np.clip(drought_index, -4, 4)

            # ── Wind Speed (m/s)
            wind_speed = np.abs(np.random.normal(3.5, 1.5, n_dates))

            # ── CO₂ (atmospheric concentration in ppm)
            # The Keeling Curve: started ~370ppm in 2000, ~420ppm in 2023
            # Also has a seasonal cycle of ~7ppm amplitude
            years_elapsed = (dates.year - START_YEAR) + (dates.month - 1) / 12
            co2_trend = 370 + 2.2 * years_elapsed  # ~2.2 ppm/year increase
            co2_seasonal = -3.5 * np.sin(2 * np.pi * dates.month / 12)  # land biosphere uptake
            co2_ppm = co2_trend + co2_seasonal

            for i, date in enumerate(dates):
                records.append({
                    "date": date,
                    "location_id": loc_id,
                    "region": region_name,
                    "temperature_mean": round(float(temperature_mean[i]), 2),
                    "temperature_max": round(float(temperature_max[i]), 2),
                    "temperature_min": round(float(temperature_min[i]), 2),
                    "precipitation": round(float(precipitation[i]), 2),
                    "humidity": round(float(humidity[i]), 1),
                    "solar_radiation": round(float(solar_radiation[i]), 1),
                    "drought_index": round(float(drought_index[i]), 3),
                    "wind_speed": round(float(wind_speed[i]), 2),
                    "co2_ppm": round(float(co2_ppm[i]), 2),
                })

        df = pd.DataFrame(records)
        df.to_csv(DATA_RAW / "climate_raw.csv", index=False)
        print(f"     Climate records generated: {len(df):,}")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # SPATIAL DATA GENERATION
    # ─────────────────────────────────────────────────────────────────────

    def generate_spatial_data(self) -> pd.DataFrame:
        """
        Generate spatial metadata (lat/lon) for each location.

        In real data, each MODIS pixel has geographic coordinates.
        We simulate a grid of 50 locations across our study regions.

        Returns:
            pd.DataFrame: Location metadata [location_id, latitude, longitude,
                          region, elevation, land_cover]
        """
        print("  🗺️  Generating spatial metadata...")

        records = []
        for loc_id in range(self.n_locations):
            region_name = list(STUDY_REGIONS.keys())[loc_id % len(STUDY_REGIONS)]
            region = STUDY_REGIONS[region_name]

            # Random lat/lon within region bounds
            lat = np.random.uniform(region["lat"][0], region["lat"][1])
            lon = np.random.uniform(region["lon"][0], region["lon"][1])

            # Elevation: higher elevations in Mediterranean, lower in Sahel
            base_elevation = {"Sahel": 300, "Savanna": 500, "Rainforest": 200,
                              "Semi-Arid": 400, "Mediterranean": 600}[region_name]
            elevation = max(0, base_elevation + np.random.normal(0, 150))

            # Land cover classification (FAO categories)
            land_covers = {
                "Sahel": ["Sparse Grassland", "Shrubland", "Bare Land"],
                "Savanna": ["Savanna", "Woodland Savanna", "Grassland"],
                "Rainforest": ["Tropical Forest", "Dense Forest", "Riverine Forest"],
                "Semi-Arid": ["Desert Shrub", "Bare Land", "Rocky Terrain"],
                "Mediterranean": ["Shrubland", "Cropland", "Mixed Forest"],
            }
            land_cover = np.random.choice(land_covers[region_name])

            records.append({
                "location_id": loc_id,
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "region": region_name,
                "elevation_m": round(elevation, 1),
                "land_cover": land_cover,
            })

        df = pd.DataFrame(records)
        df.to_csv(DATA_RAW / "spatial_raw.csv", index=False)
        print(f"     Spatial locations created: {len(df)}")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # MERGE ALL DATASETS
    # ─────────────────────────────────────────────────────────────────────

    def _merge_datasets(self,
                         ndvi_df: pd.DataFrame,
                         climate_df: pd.DataFrame,
                         spatial_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge NDVI, climate, and spatial data into a single DataFrame.

        This creates the 'master dataset' used for all analysis.
        Merging is done on [date, location_id] as the composite key.

        Args:
            ndvi_df: NDVI time series
            climate_df: Climate time series
            spatial_df: Location metadata (no date column)

        Returns:
            pd.DataFrame: Fully merged dataset
        """
        print("  🔗 Merging all datasets...")

        # Merge NDVI + Climate on date and location
        merged = pd.merge(
            ndvi_df,
            climate_df.drop(columns=["region"]),  # avoid duplicate column
            on=["date", "location_id"],
            how="inner"
        )

        # Merge spatial metadata (one row per location, broadcast across dates)
        merged = pd.merge(
            merged,
            spatial_df.drop(columns=["region"]),  # avoid duplicate
            on="location_id",
            how="left"
        )

        # Add derived time features
        merged["year"] = merged["date"].dt.year
        merged["month"] = merged["date"].dt.month
        merged["quarter"] = merged["date"].dt.quarter
        merged["day_of_year"] = merged["date"].dt.dayofyear

        # Sort chronologically
        merged = merged.sort_values(["location_id", "date"]).reset_index(drop=True)

        print(f"     Final dataset shape: {merged.shape}")
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT ENTRY POINT
# When you run: python src/data_loader.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CLIMATE-VEGETATION DATA LOADER")
    print("=" * 60)

    loader = DataLoader(random_seed=42, n_locations=50)
    df = loader.load_all_data(force_regenerate=True)

    print("\n📊 Dataset Summary:")
    print(f"  Shape:   {df.shape}")
    print(f"  Dates:   {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Regions: {df['region'].unique().tolist()}")
    print(f"\n  First 3 rows:")
    print(df.head(3).to_string())

    print("\n✅ Data generation complete!")
