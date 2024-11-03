from torch import nn

from config import config

class WeightProjection(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        input_dim = (
            1
            + config.data.features.max
            + (config.target_mlp.hidden_dim * (config.target_mlp.num_layers - 2))
            + 1
        )
        self.input_dim = input_dim

        self.weight_projection = nn.Linear(input_dim, self.hidden_dim)

    def forward(self, weights):
        return self.weight_projection(weights)
