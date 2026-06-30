"""
=============================================================================
model.py — Climate Vegetation Analysis Project
=============================================================================
PURPOSE:
    Train, evaluate, and compare machine learning models to predict NDVI
    from climate variables. This module implements a complete ML pipeline.

MODELS IMPLEMENTED:
    1. Linear Regression (baseline) — Simple, interpretable
    2. Random Forest Regressor      — Ensemble of decision trees
    3. XGBoost Regressor            — Gradient-boosted trees (best performance)
    4. Support Vector Regression    — Kernel-based method

WHY MULTIPLE MODELS?
    No single model is best for every problem. By comparing multiple
    algorithms, we can:
    1. Identify which model type suits the data structure best
    2. Understand performance trade-offs (accuracy vs. interpretability)
    3. Build an ensemble that combines strengths of multiple models

EVALUATION METRICS:
    - RMSE (Root Mean Squared Error): √(mean of squared errors)
      Penalizes large errors heavily. Good when big errors are costly.
    - MAE (Mean Absolute Error): mean of |actual - predicted|
      Robust to outliers. Easy to interpret ("off by X NDVI units on average")
    - R² (Coefficient of Determination): fraction of variance explained
      R²=1 is perfect, R²=0 means model is no better than predicting the mean

USAGE:
    from src.model import ModelTrainer
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    trainer.print_leaderboard(results)
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import joblib
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "processed"


class ModelTrainer:
    """
    Handles training, evaluation, and comparison of all ML models.

    This class follows the 'strategy pattern': each model follows the
    same interface (fit/predict/score), making comparison straightforward.

    Attributes:
        models: Dictionary of initialized model objects
        results: Dictionary storing evaluation metrics for each model
        best_model_name: Name of the top-performing model
        best_model: The best-performing model object
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize model trainer with all algorithms.

        Args:
            random_seed: For reproducibility across runs
        """
        self.random_seed = random_seed
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        self.feature_names = None

        # ── Define all models with tuned hyperparameters
        # Hyperparameters are set based on typical best practices for
        # regression on environmental/climate data
        self.models = {

            # ── BASELINE: Linear Regression
            # Assumes a linear relationship between features and NDVI.
            # Simple and interpretable but limited by linearity assumption.
            "Linear Regression": LinearRegression(),

            # ── Ridge Regression (L2 regularization)
            # Like Linear Regression but adds a penalty for large coefficients.
            # Helps when many correlated features exist (multicollinearity).
            # alpha=1.0 controls regularization strength
            "Ridge Regression": Ridge(alpha=1.0),

            # ── Random Forest
            # An ensemble of decision trees, each trained on a random subset
            # of data and features. The final prediction is the average.
            # Strengths: handles nonlinearity, robust to outliers, feature importance
            # n_estimators=200: 200 trees (more = better but slower)
            # max_depth=12: maximum tree depth (controls overfitting)
            # min_samples_leaf=5: prevents overfitting on tiny subsets
            "Random Forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=5,
                max_features="sqrt",    # Use sqrt(n_features) per split
                random_state=random_seed,
                n_jobs=-1              # Use all CPU cores
            ),

            # ── XGBoost (eXtreme Gradient Boosting)
            # Builds trees sequentially, each correcting errors of the previous.
            # Generally the best-performing model for tabular data.
            # learning_rate: how much each tree contributes (lower = more robust)
            # n_estimators: total number of boosting rounds
            # max_depth: depth of each tree (3-6 is typical for boosting)
            # subsample: fraction of samples per tree (prevents overfitting)
            # colsample_bytree: fraction of features per tree
            "XGBoost": xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,         # L1 regularization
                reg_lambda=1.0,        # L2 regularization
                random_state=random_seed,
                verbosity=0,
                n_jobs=-1
            ),

            # ── SVR (Support Vector Regression)
            # Uses a 'kernel trick' to fit complex nonlinear relationships.
            # Good for small-to-medium datasets with complex patterns.
            # kernel='rbf': Radial Basis Function kernel (Gaussian)
            # C: regularization (larger = less regularization)
            # epsilon: tube width within which no penalty is applied
            "SVR": SVR(
                kernel="rbf",
                C=10.0,
                epsilon=0.01,
                gamma="scale"
            ),
        }

        print(f"✅ ModelTrainer initialized with {len(self.models)} models:")
        for name in self.models:
            print(f"   - {name}")

    # ─────────────────────────────────────────────────────────────────────
    # EVALUATION METRICS
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def evaluate_model(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        model_name: str = "") -> dict:
        """
        Calculate all evaluation metrics for a model.

        Args:
            y_true: True NDVI values
            y_pred: Model predictions
            model_name: For printing

        Returns:
            dict: {"rmse": float, "mae": float, "r2": float}
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {"rmse": rmse, "mae": mae, "r2": r2}

    # ─────────────────────────────────────────────────────────────────────
    # TRAIN SINGLE MODEL
    # ─────────────────────────────────────────────────────────────────────

    def train_model(self,
                     model_name: str,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     feature_names: list = None) -> dict:
        """
        Train a single model and return its evaluation results.

        Args:
            model_name: Key in self.models dict
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            feature_names: For XGBoost feature importance

        Returns:
            dict: Training results including metrics and predictions
        """
        model = self.models[model_name]
        print(f"\n  🔧 Training {model_name}...")

        # ── Train (fit the model on training data)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # ── Predict on test set
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # ── Calculate metrics
        test_metrics = self.evaluate_model(y_test, y_pred, model_name)
        train_metrics = self.evaluate_model(y_train, y_pred_train, model_name)

        # ── Check for overfitting
        # If train R² >> test R², the model has memorized training data
        overfit_gap = train_metrics["r2"] - test_metrics["r2"]

        result = {
            "model": model,
            "model_name": model_name,
            "y_pred": y_pred,
            "y_pred_train": y_pred_train,
            "rmse": test_metrics["rmse"],
            "mae": test_metrics["mae"],
            "r2": test_metrics["r2"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "overfit_gap": overfit_gap,
            "train_time_s": train_time,
        }

        # ── Feature importance (for tree-based models)
        if hasattr(model, "feature_importances_") and feature_names:
            result["feature_importances"] = model.feature_importances_
            result["feature_names"] = feature_names

        # ── Print results
        print(f"     RMSE:  {test_metrics['rmse']:.5f}")
        print(f"     MAE:   {test_metrics['mae']:.5f}")
        print(f"     R²:    {test_metrics['r2']:.5f}")
        print(f"     Train R²: {train_metrics['r2']:.5f}")
        print(f"     Overfit gap: {overfit_gap:.4f} {'⚠️' if overfit_gap > 0.1 else '✅'}")
        print(f"     Time: {train_time:.2f}s")

        return result

    # ─────────────────────────────────────────────────────────────────────
    # TRAIN ALL MODELS
    # ─────────────────────────────────────────────────────────────────────

    def train_all_models(self,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          feature_names: list = None) -> dict:
        """
        Train and evaluate all models, then identify the best performer.

        Args:
            X_train: Training feature matrix
            y_train: Training targets (NDVI)
            X_test: Test feature matrix
            y_test: Test targets (NDVI)
            feature_names: Column names (for feature importance)

        Returns:
            dict: All results keyed by model name
        """
        print("\n" + "=" * 55)
        print("  TRAINING ALL MODELS")
        print("=" * 55)

        self.feature_names = feature_names

        # Convert to numpy arrays if DataFrames
        if hasattr(X_train, "values"):
            X_train_arr = X_train.values
            X_test_arr = X_test.values
        else:
            X_train_arr = X_train
            X_test_arr = X_test

        y_train_arr = np.array(y_train)
        y_test_arr = np.array(y_test)

        # Train each model
        for name in self.models:
            try:
                self.results[name] = self.train_model(
                    name, X_train_arr, y_train_arr,
                    X_test_arr, y_test_arr, feature_names
                )
            except Exception as e:
                print(f"     ❌ Error training {name}: {e}")
                continue

        # Find the best model (highest R²)
        self.best_model_name = max(
            self.results,
            key=lambda k: self.results[k]["r2"]
        )
        self.best_model = self.results[self.best_model_name]["model"]

        print(f"\n🏆 Best model: {self.best_model_name}")
        print(f"   R² = {self.results[self.best_model_name]['r2']:.5f}")

        return self.results

    # ─────────────────────────────────────────────────────────────────────
    # CROSS-VALIDATION
    # ─────────────────────────────────────────────────────────────────────

    def cross_validate_best_model(self,
                                    X: np.ndarray,
                                    y: np.ndarray,
                                    n_splits: int = 5) -> dict:
        """
        Perform k-fold cross-validation on the best model.

        WHAT IS CROSS-VALIDATION?
        Instead of one train/test split, we split the data k times.
        Each fold is used as the test set once, and the model is trained
        on the remaining k-1 folds. This gives a more robust estimate
        of real-world performance.

        Args:
            X: Full feature matrix
            y: Full target vector
            n_splits: Number of folds (5 is standard)

        Returns:
            dict: CV scores and statistics
        """
        if not self.best_model_name:
            raise ValueError("Train models first with train_all_models()")

        print(f"\n🔄 Cross-validating {self.best_model_name} ({n_splits}-fold)...")

        model = self.models[self.best_model_name]
        kf = KFold(n_splits=n_splits, shuffle=False)  # shuffle=False for time series

        if hasattr(X, "values"):
            X = X.values
        y = np.array(y)

        cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)

        result = {
            "scores": cv_scores,
            "mean_r2": cv_scores.mean(),
            "std_r2": cv_scores.std(),
        }

        print(f"   CV R² scores: {cv_scores.round(4)}")
        print(f"   Mean R²: {result['mean_r2']:.4f} ± {result['std_r2']:.4f}")

        return result

    # ─────────────────────────────────────────────────────────────────────
    # LEADERBOARD PRINTING
    # ─────────────────────────────────────────────────────────────────────

    def print_leaderboard(self) -> pd.DataFrame:
        """
        Print a formatted model performance leaderboard.

        Returns:
            pd.DataFrame: Performance table sorted by R²
        """
        if not self.results:
            print("No results yet. Run train_all_models() first.")
            return

        print("\n" + "=" * 70)
        print("  📊 MODEL PERFORMANCE LEADERBOARD")
        print("=" * 70)

        rows = []
        for name, res in self.results.items():
            rows.append({
                "Model": name,
                "RMSE ↓": f"{res['rmse']:.5f}",
                "MAE ↓": f"{res['mae']:.5f}",
                "R² ↑": f"{res['r2']:.5f}",
                "Train R²": f"{res['train_r2']:.5f}",
                "Time (s)": f"{res['train_time_s']:.2f}",
                "🏆": "★ BEST" if name == self.best_model_name else ""
            })

        df = pd.DataFrame(rows).sort_values("R² ↑", ascending=False)
        print(df.to_string(index=False))
        print("=" * 70)

        return df

    # ─────────────────────────────────────────────────────────────────────
    # SAVE BEST MODEL
    # ─────────────────────────────────────────────────────────────────────

    def save_best_model(self, filename: str = "best_model.pkl") -> str:
        """
        Save the best model to disk using joblib.

        Joblib is better than pickle for large numpy arrays (like Random Forest).
        The saved model can be loaded and used for predictions without retraining.

        Args:
            filename: Output filename (saved to data/processed/)

        Returns:
            str: Path to saved model file
        """
        if not self.best_model:
            raise ValueError("No best model to save. Run train_all_models() first.")

        save_data = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "feature_names": self.feature_names,
            "metrics": {
                "rmse": self.results[self.best_model_name]["rmse"],
                "mae": self.results[self.best_model_name]["mae"],
                "r2": self.results[self.best_model_name]["r2"],
            }
        }

        path = MODELS_DIR / filename
        joblib.dump(save_data, path)
        print(f"\n💾 Best model saved to: {path}")
        return str(path)

    @staticmethod
    def load_model(filepath: str) -> dict:
        """
        Load a previously saved model from disk.

        Args:
            filepath: Path to .pkl file

        Returns:
            dict: {"model": ..., "model_name": ..., "feature_names": ...}
        """
        data = joblib.load(filepath)
        print(f"✅ Model loaded: {data['model_name']}")
        return data


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import DataLoader
    from preprocessing import Preprocessor

    print("=" * 60)
    print("  MODEL TRAINING MODULE TEST")
    print("=" * 60)

    # Load and preprocess data
    loader = DataLoader()
    df = loader.load_all_data()

    prep = Preprocessor()
    result = prep.run_full_pipeline(df, scale=True)

    # Train all models
    trainer = ModelTrainer()
    results = trainer.train_all_models(
        result["X_train_scaled"],
        result["y_train"],
        result["X_test_scaled"],
        result["y_test"],
        feature_names=result["feature_names"]
    )

    trainer.print_leaderboard()
    trainer.save_best_model()

    print("\n✅ Model training complete!")
