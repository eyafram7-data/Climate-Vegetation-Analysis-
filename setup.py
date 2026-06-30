"""
setup.py — Climate Vegetation Analysis
Allows the project to be installed as a package with: pip install -e .
This makes imports like `from src.data_loader import DataLoader` work
from anywhere, not just the project root directory.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README for the long description on PyPI
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="climate-vegetation-analysis",
    version="1.0.0",
    author="Afram Yaw Emmanuel",
    author_email="eyafram7@gmail.com",
    description="Climate Change and Vegetation Dynamics Analysis Using Remote Sensing and ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eyafram7-data/Climate-Vegetation-Analysis",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "streamlit>=1.28.0",
        "folium>=0.14.0",
        "geopandas>=0.14.0",
        "rasterio>=1.3.0",
        "xarray>=2023.6.0",
        "scipy>=1.11.0",
        "statsmodels>=0.14.0",
        "joblib>=1.3.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "climate-veg-pipeline=run_pipeline:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "climate-change", "ndvi", "remote-sensing", "machine-learning",
        "vegetation", "xgboost", "streamlit", "geospatial", "modis", "era5"
    ],
    project_urls={
        "Bug Reports": "https://github.com/eyafarm7-data/Climate-Vegetation-Analysis/issues",
        "Source":      "https://github.com/eyafram7-data/Climate-Vegetation-Analysis",
        "Dashboard":   "https://eyafram7-data-climate-veg.streamlit.app",
    },
)
