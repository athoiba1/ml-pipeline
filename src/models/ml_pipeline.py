from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression


def build_classification_pipeline(model_name: str = "random_forest", **kwargs) -> Pipeline:
    if model_name == "random_forest":
        clf = RandomForestClassifier(**kwargs)
    elif model_name == "logistic_regression":
        clf = LogisticRegression(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])


def build_regression_pipeline(model_name: str = "random_forest", **kwargs) -> Pipeline:
    if model_name == "random_forest":
        reg = RandomForestRegressor(**kwargs)
    elif model_name == "linear_regression":
        reg = LinearRegression(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", reg),
    ])
