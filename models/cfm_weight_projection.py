from models.positional_encoding import PositionalEncoding
from torch import nn


class WeightProjection(nn.Module):
    """Simple linear weight projection of compact form."""
    def __init__(self, input_dim, hidden_dim, positional_encoding: PositionalEncoding):
        super().__init__()
        self.positional_encoding = positional_encoding
        self.weight_projection = nn.Linear(input_dim, hidden_dim)

    def forward(self, weights):
        projected_weights = self.weight_projection(weights)
        return self.positional_encoding(projected_weights)
