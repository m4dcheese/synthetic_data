from torch import nn


class WeightProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim, data_config, mlp_config):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        input_dim = (
            1
            + data_config.features.max
            + (mlp_config.hidden_dim * (mlp_config.num_layers - 2))
            + 1
        )

        self.weight_projection = nn.Linear(input_dim, self.hidden_dim)

    def forward(self, weights):
        return self.weight_projection(weights)
