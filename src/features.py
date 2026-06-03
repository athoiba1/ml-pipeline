import numpy as np
import pandas as pd


def add_polynomial_features(X: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
    X_poly = X.copy()
    for col in X.columns:
        for d in range(2, degree + 1):
            X_poly[f"{col}^{d}"] = X[col] ** d
    return X_poly


def add_interaction_features(X: pd.DataFrame) -> pd.DataFrame:
    X_interact = X.copy()
    cols = X.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            X_interact[f"{cols[i]}_x_{cols[j]}"] = X[cols[i]] * X[cols[j]]
    return X_interact


def add_log_features(X: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    X_log = X.copy()
    target_cols = columns or X.select_dtypes(include=[np.number]).columns
    for col in target_cols:
        X_log[f"{col}_log"] = np.log1p(X_log[col].clip(lower=0))
    return X_log
