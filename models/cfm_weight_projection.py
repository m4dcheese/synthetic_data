from torch import nn

from config import config
from models.positional_encoding import PositionalEncoding
from models.target_mlp import TargetMLP


class WeightProjection(nn.Module):
    def __init__(self, hidden_dim, positional_encoding: PositionalEncoding):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.positional_encoding = positional_encoding

        input_dim = TargetMLP.compact_shape()[1]
        self.input_dim = input_dim

        self.weight_projection = nn.Linear(input_dim, self.hidden_dim)

    def forward(self, weights):
        projected_weights = self.weight_projection(weights)
        return self.positional_encoding(projected_weights)
