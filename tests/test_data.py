import pandas as pd
import numpy as np
import tempfile
import os
from src.data import load_csv, load_data, scale_features


def test_load_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(f, index=False)
        path = f.name

    df = load_csv(path)
    assert df.shape == (3, 2)
    os.unlink(path)


def test_load_data():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        pd.DataFrame({
            "feat1": range(100),
            "feat2": range(100),
            "target": [0] * 50 + [1] * 50,
        }).to_csv(f, index=False)
        path = f.name

    X_train, X_test, y_train, y_test, encoders = load_data(path, "target")
    assert len(X_train) + len(X_test) == 100
    assert len(y_train) + len(y_test) == 100
    os.unlink(path)


def test_scale_features():
    X_train = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    X_test = pd.DataFrame({"a": [7, 8], "b": [9, 10]})

    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    assert X_train_s["a"].mean() < 0.01
    assert X_test_s.shape == (2, 2)
