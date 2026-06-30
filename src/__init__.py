"""
=============================================================================
Climate Change and Vegetation Dynamics Analysis
src/__init__.py
=============================================================================
This file makes the src/ directory a Python package, allowing imports like:
    from src.data_loader import DataLoader
    from src.model import ModelTrainer

Module index:
    data_loader    — Data acquisition and synthetic generation
    preprocessing  — Cleaning, feature engineering, train/test split
    visualization  — All matplotlib, seaborn, plotly, and folium plots
    model          — ML model training, evaluation, and comparison
    prediction     — Future forecasting and scenario analysis
=============================================================================
"""

__version__ = "1.0.0"
__author__  = "Afram Yaw Emmanuel"
__email__   = "eyafram7@gmail.com"
__license__ = "MIT"

# Expose key classes at package level for clean imports
from .data_loader   import DataLoader
from .preprocessing import Preprocessor
from .visualization import Visualizer
from .model         import ModelTrainer
from .prediction    import Predictor

__all__ = [
    "DataLoader",
    "Preprocessor",
    "Visualizer",
    "ModelTrainer",
    "Predictor",
]
