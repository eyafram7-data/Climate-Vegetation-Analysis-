"""
=============================================================================
run_pipeline.py — Master Pipeline Script
Climate Change and Vegetation Dynamics Analysis
=============================================================================
PURPOSE:
    Run the complete project pipeline from data generation to model training
    and visualization in a single command.

    This is the easiest entry point for first-time users. It orchestrates:
    1. Data generation (or loading if already exists)
    2. Data quality check and cleaning
    3. Feature engineering
    4. Model training and evaluation
    5. Visualization generation (saves to images/)
    6. Future scenario forecasting
    7. Summary report printing

USAGE:
    python run_pipeline.py                    # Full pipeline
    python run_pipeline.py --skip-training    # Skip ML (faster)
    python run_pipeline.py --regenerate       # Force regenerate data

OUTPUT:
    ├── data/raw/            ← Generated raw datasets
    ├── data/processed/      ← Cleaned + feature-engineered data
    ├── images/              ← All plots (PNG files)
    └── Console leaderboard  ← Model performance metrics

ESTIMATED RUNTIME:
    Data generation  : ~5s
    Preprocessing    : ~15s
    Model training   : ~90s  (XGBoost 500 trees is slowest)
    Visualization    : ~30s
    Total            : ~2-3 minutes
=============================================================================
"""

import sys
import argparse
import time
from pathlib import Path

# ── Ensure src/ is importable ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Climate-Vegetation Analysis — Full Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                  Run everything
  python run_pipeline.py --skip-training  Skip ML model training
  python run_pipeline.py --regenerate     Force regenerate synthetic data
  python run_pipeline.py --no-plots       Skip plot generation
        """
    )
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip ML model training (faster iteration)")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regenerate synthetic data even if it exists")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating visualization images")
    parser.add_argument("--n-locations", type=int, default=50,
                        help="Number of spatial grid locations (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


def print_banner():
    print("\n" + "=" * 65)
    print("  🌿  CLIMATE-VEGETATION ANALYSIS PIPELINE")
    print("  Climate Change & Vegetation Dynamics Using ML")
    print("=" * 65)
    print("  Data:    NASA MODIS NDVI + NOAA/ERA5 (synthetic)")
    print("  Regions: Sahel, Savanna, Rainforest, Semi-Arid, Mediterranean")
    print("  Period:  2000–2023  |  Resolution: Monthly")
    print("=" * 65 + "\n")


def step(n, title, emoji="🔷"):
    print(f"\n{'─'*55}")
    print(f"  {emoji}  STEP {n}: {title}")
    print(f"{'─'*55}")


def main():
    args = parse_args()
    print_banner()
    total_start = time.time()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: DATA LOADING / GENERATION
    # ─────────────────────────────────────────────────────────────────────
    step(1, "DATA LOADING & GENERATION", "📦")
    from data_loader import DataLoader
    import pandas as pd

    t0 = time.time()
    loader = DataLoader(random_seed=args.seed, n_locations=args.n_locations)
    df = loader.load_all_data(force_regenerate=args.regenerate)
    df["date"] = pd.to_datetime(df["date"])
    print(f"\n  ✅ Data ready in {time.time()-t0:.1f}s")
    print(f"     Shape   : {df.shape}")
    print(f"     Regions : {sorted(df['region'].unique())}")
    print(f"     Dates   : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"     NDVI    : {df['ndvi'].min():.3f} – {df['ndvi'].max():.3f}  "
          f"(mean={df['ndvi'].mean():.3f})")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: PREPROCESSING & FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────────────────
    step(2, "PREPROCESSING & FEATURE ENGINEERING", "⚙️")
    from preprocessing import Preprocessor

    t0 = time.time()
    prep = Preprocessor()
    pipeline_result = prep.run_full_pipeline(df, scale=True)
    print(f"\n  ✅ Preprocessing complete in {time.time()-t0:.1f}s")
    print(f"     Features      : {len(pipeline_result['feature_names'])}")
    print(f"     Training rows : {len(pipeline_result['X_train']):,}")
    print(f"     Test rows     : {len(pipeline_result['X_test']):,}")
    print(f"\n  📋 Quality Report:")
    qr = pipeline_result["quality_report"]
    print(f"     Missing values    : {sum(qr.get('missing_values',{}).values()) or 0} cells")
    print(f"     Duplicates removed: {qr.get('duplicates', 0)}")
    print(f"     NDVI range        : {qr.get('ndvi_range', ('?','?'))[0]:.3f} – "
          f"{qr.get('ndvi_range', ('?','?'))[1]:.3f}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: VISUALIZATIONS
    # ─────────────────────────────────────────────────────────────────────
    step(3, "GENERATING VISUALIZATIONS", "📊")
    if args.no_plots:
        print("  ⏭️  Skipped (--no-plots flag set)")
    else:
        from visualization import Visualizer
        t0 = time.time()
        viz = Visualizer()
        print("  Creating plots...")
        viz.plot_temperature_trends(df)
        viz.plot_precipitation_patterns(df)
        viz.plot_ndvi_timeseries(df)
        viz.plot_correlation_heatmap(df)
        viz.create_interactive_map(df)
        print(f"\n  ✅ Visualizations saved in {time.time()-t0:.1f}s → images/")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: MODEL TRAINING
    # ─────────────────────────────────────────────────────────────────────
    step(4, "MACHINE LEARNING MODEL TRAINING", "🤖")
    if args.skip_training:
        print("  ⏭️  Skipped (--skip-training flag set)")
        trainer = None
        model_results = {}
    else:
        import numpy as np
        from model import ModelTrainer

        t0 = time.time()
        trainer = ModelTrainer(random_seed=args.seed)
        model_results = trainer.train_all_models(
            pipeline_result["X_train_scaled"],
            pipeline_result["y_train"].values
                if hasattr(pipeline_result["y_train"], "values")
                else pipeline_result["y_train"],
            pipeline_result["X_test_scaled"],
            pipeline_result["y_test"].values
                if hasattr(pipeline_result["y_test"], "values")
                else pipeline_result["y_test"],
            feature_names=pipeline_result["feature_names"]
        )
        print(f"\n  ✅ All models trained in {time.time()-t0:.1f}s")
        trainer.print_leaderboard()

        # Save best model
        trainer.save_best_model("best_model.pkl")

        # Plot model comparison if plots enabled
        if not args.no_plots:
            from visualization import Visualizer
            viz = Visualizer()
            viz.plot_model_comparison(model_results)
            if "XGBoost" in model_results and "feature_importances" in model_results["XGBoost"]:
                viz.plot_feature_importance(
                    model_results["XGBoost"]["feature_names"],
                    model_results["XGBoost"]["feature_importances"],
                    model_name="XGBoost"
                )
            viz.plot_predictions(
                pipeline_result["y_test"].values
                    if hasattr(pipeline_result["y_test"], "values")
                    else pipeline_result["y_test"],
                model_results[trainer.best_model_name]["y_pred"],
                model_name=trainer.best_model_name
            )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5: FUTURE FORECASTING
    # ─────────────────────────────────────────────────────────────────────
    step(5, "FUTURE SCENARIO FORECASTING", "🔮")
    if args.skip_training or trainer is None:
        print("  ⏭️  Skipped (no trained model available)")
    else:
        import numpy as np
        from prediction import Predictor

        t0 = time.time()
        df_features = pipeline_result["df_features"]
        if "date" not in df_features.columns and "date" in df.columns:
            df_features = df_features.copy()

        predictor = Predictor(
            model=trainer.best_model,
            feature_names=pipeline_result["feature_names"],
            forecast_months=24
        )

        # Run full scenario analysis
        scenarios_df = predictor.scenario_analysis(df_features)

        # Sensitivity analysis
        sensitivity_df = predictor.sensitivity_analysis(df_features)

        # Plot forecast
        if not args.no_plots:
            df_with_date = df_features.copy()
            if "date" not in df_with_date.columns:
                df_with_date["date"] = df["date"].values[:len(df_with_date)]
            predictor.plot_scenario_forecast(df_with_date, scenarios_df)
            predictor.plot_sensitivity(sensitivity_df)

        print(f"\n  ✅ Forecast complete in {time.time()-t0:.1f}s")
        print(f"\n  📋 Scenario Summary (24-month horizon):")
        for scenario in ["Optimistic", "Baseline", "Pessimistic"]:
            scen = scenarios_df[scenarios_df["scenario"] == scenario]
            if len(scen) > 0:
                start_v = scen["predicted_ndvi"].iloc[0]
                end_v   = scen["predicted_ndvi"].iloc[-1]
                print(f"     {scenario:12s}: NDVI {start_v:.3f} → {end_v:.3f}  "
                      f"(net change: {end_v-start_v:+.3f})")

    # ─────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print("\n" + "=" * 65)
    print("  ✅  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Total runtime : {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"  Dataset       : {df.shape[0]:,} records  ×  {df.shape[1]} columns")
    print(f"  Features      : {len(pipeline_result['feature_names'])}")

    if model_results:
        best = trainer.best_model_name
        r2   = model_results[best]["r2"]
        rmse = model_results[best]["rmse"]
        print(f"  Best model    : {best}  (R²={r2:.4f}, RMSE={rmse:.5f})")

    print("\n  📂 Output files:")
    print("     data/raw/                ← Raw generated datasets")
    print("     data/processed/          ← Cleaned + feature data")
    print("     images/                  ← All visualization plots")
    print("     data/processed/best_model.pkl  ← Serialised best model")
    print("\n  🚀 Next steps:")
    print("     streamlit run app.py     ← Launch interactive dashboard")
    print("     jupyter notebook notebooks/  ← Explore analysis notebooks")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
