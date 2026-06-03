# ML Pipeline

A machine learning pipeline combining scikit-learn and PyTorch for tabular data classification and regression tasks.

## Features

- **Data Processing** - Automated data loading, cleaning, and preprocessing
- **Feature Engineering** - Scaling, encoding, and feature selection
- **ML Models** - scikit-learn pipelines (Random Forest, XGBoost)
- **Deep Learning** - PyTorch MLP for tabular data
- **Training** - Unified training loop with experiment tracking
- **Evaluation** - Comprehensive metrics and visualization

## Project Structure

```
ml-pipeline/
├── configs/          # Hyperparameter configurations
├── data/             # Datasets (git-ignored)
├── notebooks/        # Jupyter notebooks for EDA
├── outputs/          # Model outputs and results
├── src/
│   ├── models/       # ML and DL model definitions
│   ├── data.py       # Data loading and preprocessing
│   ├── features.py   # Feature engineering
│   ├── train.py      # Training scripts
│   └── evaluate.py   # Evaluation utilities
└── tests/            # Unit tests
```

## Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-pipeline.git
cd ml-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train ML Model (scikit-learn)

```python
from src.train import train_ml_pipeline
train_ml_pipeline("configs/default.yaml")
```

### Train DL Model (PyTorch)

```python
from src.train import train_dl_model
train_dl_model("configs/default.yaml")
```

### Run Notebooks

```bash
jupyter notebook notebooks/
```

## License

MIT
