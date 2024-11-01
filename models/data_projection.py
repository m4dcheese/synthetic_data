from __future__ import annotations

from torch import Tensor, nn


class DataProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.data_projection_x = nn.Linear(self.input_dim, self.hidden_dim)
        self.data_projection_y = nn.Linear(1, self.hidden_dim)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        print(f"DataProjection: x.shape={x.shape}, y.shape={y.shape}")
        x = self.data_projection_x(x)
        y = self.data_projection_y(y)
        return x + y
