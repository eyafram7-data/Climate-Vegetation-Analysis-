# 🤝 Contributing to Climate-Vegetation Analysis

Thank you for considering contributing to this project! Every contribution —
whether it's a bug fix, new feature, documentation improvement, or data source
addition — is valued and appreciated.

---

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Project Structure Guide](#project-structure-guide)
- [Coding Standards](#coding-standards)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Good First Issues](#good-first-issues)

---

## 📜 Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/).
Please be respectful and constructive in all interactions.

---

## 🛠️ How to Contribute

### 🐛 Reporting Bugs
1. Check [existing issues](https://github.com/yourusername/Climate-Vegetation-Analysis/issues)
2. Open a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behaviour
   - Python version and OS

### 💡 Suggesting Features
- Open a [feature request issue](https://github.com/yourusername/Climate-Vegetation-Analysis/issues/new)
- Describe the use case and why it adds value

### 📝 Improving Documentation
- Fix typos, clarify explanations, add examples
- Improve docstrings in any `src/*.py` file

### 🔬 Adding Data Sources
- See `src/data_loader.py` — add a new `load_<source>()` method
- Include fallback to synthetic data if API is unavailable
- Document the data format in the README

---

## 💻 Development Setup

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Climate-Vegetation-Analysis.git
cd Climate-Vegetation-Analysis

# 2. Create a feature branch
git checkout -b feature/my-awesome-feature

# 3. Set up virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# OR: venv\Scripts\activate     # Windows

# 4. Install dependencies (including dev tools)
pip install -r requirements.txt
pip install pytest black flake8 isort

# 5. Generate the dataset
python src/data_loader.py

# 6. Run the test suite
pytest tests/ -v

# 7. Run the dashboard
streamlit run app.py
```

---

## 📁 Project Structure Guide

| File / Folder | What to edit here |
|---|---|
| `src/data_loader.py` | New data sources, synthetic data improvements |
| `src/preprocessing.py` | New feature engineering, cleaning steps |
| `src/visualization.py` | New plot types, colour schemes |
| `src/model.py` | New ML models, evaluation metrics |
| `src/prediction.py` | New scenarios, uncertainty methods |
| `app.py` | New dashboard tabs or widgets |
| `notebooks/` | New EDA sections, tutorial notebooks |
| `README.md` | Documentation, results, references |

---

## ✅ Coding Standards

### Style
- Follow **PEP 8** (use `black` for auto-formatting)
- Max line length: **100 characters**
- Use `isort` for import ordering

```bash
# Auto-format before committing
black src/ app.py --line-length 100
isort src/ app.py
flake8 src/ app.py --max-line-length 100
```

### Docstrings
Every function must have a docstring explaining:
- **PURPOSE**: What does this function do?
- **Args**: Every parameter (name, type, description)
- **Returns**: What is returned (type and description)
- **Example** (optional but encouraged)

```python
def my_function(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Filter DataFrame rows where NDVI exceeds a threshold.

    Args:
        df (pd.DataFrame): Input DataFrame with 'ndvi' column
        threshold (float): Minimum NDVI value to keep (default: 0.5)

    Returns:
        pd.DataFrame: Filtered DataFrame with only rows above threshold

    Example:
        df_dense = my_function(df, threshold=0.6)
    """
    return df[df["ndvi"] > threshold]
```

### Comments
- Explain **why**, not just **what**
- Target audience: beginner data scientists
- Every non-obvious step should be commented

### Tests
- Add tests in `tests/` for any new function
- Use `pytest` style (function names start with `test_`)

```python
# tests/test_preprocessing.py
def test_clean_data_removes_duplicates():
    """Verify that clean_data() removes duplicate rows."""
    import pandas as pd
    from src.preprocessing import Preprocessor

    df_with_dupes = pd.DataFrame({"ndvi": [0.5, 0.5], "date": ["2020-01-01", "2020-01-01"]})
    prep = Preprocessor()
    df_clean = prep.clean_data(df_with_dupes)
    assert len(df_clean) == 1
```

---

## 🚀 Submitting a Pull Request

1. **Commit** your changes with a clear message:
   ```bash
   git add .
   git commit -m "feat: add ERA5 data loader with automatic retry"
   ```
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation
   - `refactor:` code restructure (no behaviour change)
   - `test:` adding tests
   - `chore:` maintenance tasks

2. **Push** to your fork:
   ```bash
   git push origin feature/my-awesome-feature
   ```

3. **Open a Pull Request** on GitHub:
   - Write a clear description of changes
   - Reference any related issue (e.g. "Closes #42")
   - Include screenshots or output if applicable

4. **Respond** to reviewer feedback promptly

---

## 🌱 Good First Issues

Perfect for first-time contributors:

| Issue | Difficulty | Skills Needed |
|---|---|---|
| Add unit tests for `preprocessing.py` | ⭐ Easy | Python, pytest |
| Improve README with more visualisation examples | ⭐ Easy | Markdown |
| Add a new seaborn theme option to `visualization.py` | ⭐⭐ Medium | matplotlib |
| Implement NDVI anomaly detection | ⭐⭐ Medium | pandas, statistics |
| Add WorldClim data loader | ⭐⭐⭐ Hard | API, rasterio |
| Implement LSTM model for time series | ⭐⭐⭐ Hard | TensorFlow/PyTorch |

Look for issues tagged [`good first issue`](https://github.com/yourusername/Climate-Vegetation-Analysis/labels/good%20first%20issue).

---

## 🙏 Recognition

All contributors will be acknowledged in:
- The `README.md` Contributors section
- The project's release notes

Thank you for helping advance open-source climate science! 🌍
