from __future__ import annotations
import torch
import torch.nn as nn


class TrajectoryLifter(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.xyz_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 3),
        )
        self.eot_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.pre(x)
        y, _ = self.lstm(x)
        xyz = self.xyz_head(y)
        eot = self.eot_head(y).squeeze(-1)
        return {"xyz": xyz, "eot_prob": eot}
