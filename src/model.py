from typing import Optional
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Binary classification (logit)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)