from torch import nn


class WeightProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.weight_projection = nn.Linear(1, self.hidden_dim)

    def forward(self, x):
        print(f"WeightProjection: x.shape={x.shape}")
        return self.weight_projection(x)
