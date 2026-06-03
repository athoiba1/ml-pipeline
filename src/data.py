import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_data(path: str, target: str, test_size: float = 0.2, random_state: int = 42):
    df = load_csv(path)

    X = df.drop(columns=[target])
    y = df[target]

    label_encoders = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target)
        label_encoders[target] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() < 50 else None
    )

    return X_train, X_test, y_train, y_test, label_encoders


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler
