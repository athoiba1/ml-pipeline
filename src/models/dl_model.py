import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = None, output_dim: int = 1, task: str = "classification"):
        super().__init__()
        self.task = task

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)

        if task == "classification":
            self.head = nn.Linear(prev_dim, output_dim)
        else:
            self.head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.head(features)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if self.task == "classification" and logits.shape[1] == 1:
                return torch.sigmoid(logits)
            return torch.softmax(logits, dim=-1)
