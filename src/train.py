import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from src.data import load_data, scale_features
from src.models.ml_pipeline import build_classification_pipeline, build_regression_pipeline
from src.models.dl_model import TabularMLP


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_ml_pipeline(config_path: str, data_path: str = None):
    config = load_config(config_path)
    ml_config = config.get("ml", {})
    data_cfg = config.get("data", {})

    path = data_path or data_cfg.get("path", "data/dataset.csv")
    target = data_cfg.get("target", "target")

    X_train, X_test, y_train, y_test, _ = load_data(
        path, target, test_size=data_cfg.get("test_size", 0.2)
    )
    X_train, X_test, scaler = scale_features(X_train, X_test)

    task = ml_config.get("task", "classification")
    model_name = ml_config.get("model", "random_forest")

    if task == "classification":
        pipeline = build_classification_pipeline(model_name)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"[ML] {model_name} Accuracy: {acc:.4f}")
    else:
        pipeline = build_regression_pipeline(model_name)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"[ML] {model_name} MSE: {mse:.4f} | R2: {r2:.4f}")

    return pipeline


def train_dl_model(config_path: str, data_path: str = None):
    config = load_config(config_path)
    dl_config = config.get("dl", {})
    data_cfg = config.get("data", {})

    path = data_path or data_cfg.get("path", "data/dataset.csv")
    target = data_cfg.get("target", "target")

    X_train, X_test, y_train, y_test, _ = load_data(
        path, target, test_size=data_cfg.get("test_size", 0.2)
    )
    X_train, X_test, scaler = scale_features(X_train, X_test)

    task = dl_config.get("task", "classification")
    hidden_dims = dl_config.get("hidden_dims", [128, 64, 32])
    epochs = dl_config.get("epochs", 50)
    lr = dl_config.get("lr", 1e-3)
    batch_size = dl_config.get("batch_size", 32)

    X_train_t = torch.FloatTensor(X_train.values)
    X_test_t = torch.FloatTensor(X_test.values)
    y_train_t = torch.FloatTensor(y_train.values)
    y_test_t = torch.FloatTensor(y_test.values)

    if task == "classification":
        y_train_t = y_train_t.long()
        y_test_t = y_test_t.long()
        output_dim = len(np.unique(y_train))
    else:
        output_dim = 1

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = TabularMLP(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        task=task,
    )

    if task == "classification":
        criterion = nn.CrossEntropyLoss() if output_dim > 1 else nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            if task == "classification" and output_dim == 1:
                loss = criterion(output.squeeze(), batch_y.float())
            else:
                loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t)
        if task == "classification":
            if output_dim == 1:
                preds = (torch.sigmoid(preds.squeeze()) > 0.5).float()
            else:
                preds = preds.argmax(dim=1).float()
            acc = accuracy_score(y_test.values, preds.numpy())
            print(f"[DL] MLP Accuracy: {acc:.4f}")
        else:
            mse = mean_squared_error(y_test.values, preds.numpy())
            r2 = r2_score(y_test.values, preds.numpy())
            print(f"[DL] MLP MSE: {mse:.4f} | R2: {r2:.4f}")

    return model
