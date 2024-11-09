import torch
from config import config
from models.mlp import MLP


class TargetMLP(MLP):
    """Model to predict by CFM."""
    def __init__(self, in_features: int = config.data.features.max):
        # in_features causes unused input dimensions to be ignored by setting weights to zero (see below)
        super().__init__(in_features=config.data.features.max,
                         out_features=config.target_mlp.output_dim,
                         hidden_dim=config.target_mlp.hidden_dim,
                         num_layers=config.target_mlp.num_layers,
                         activation=config.target_mlp.activation_str,
                         initialization=config.target_mlp.initialization,
                         bias=config.target_mlp.bias)

        if config.data.features.max > in_features:
            layer_1 = self.model[0] if config.target_mlp.num_layers > 1 else self.model
            layer_1[0].weight.data[:, in_features :].zero_()

    def compact_shape() -> torch.Tensor:
        """Compute shape of compact representation."""
        return (
            (config.target_mlp.hidden_dim if config.target_mlp.num_layers > 1 else 0) + 1,
            (1 if config.target_mlp.num_layers > 1 else 0) # Extra column for bias of layer 1
            + config.data.features.max
            + (config.target_mlp.hidden_dim * max(0, config.target_mlp.num_layers - 2))
            + 1,
        )

    def load_compact_form(self, compact_form: torch.Tensor) -> None:
        """Load weights and biases from compact form."""
        self.model[0][0].bias.data = compact_form[:-1, 0]
        self.model[0][0].weight.data = compact_form[:-1, 1 : config.data.features.max + 1]
        for i in range(config.target_mlp.num_layers - 2):
            column_start = (
                1 + config.data.features.max + i * config.target_mlp.hidden_dim
            )
            column_end = (
                1 + config.data.features.max + (i + 1) * config.target_mlp.hidden_dim
            )
            self.model[i + 1][0].weight.data = compact_form[:-1, column_start:column_end]
            self.model[i + 1][0].bias.data = compact_form[-1, column_start:column_end]

        # Finish the puzzle by transposing last layer weights
        self.model[-1].weight.data = compact_form[:-1, -1:].T
        self.model[-1].bias.data = compact_form[-1, -1]

    def get_compact_form(self) -> torch.Tensor:
        """Return compact form representing model weights and biases."""
        size = TargetMLP.compact_shape()

        compact_form = torch.zeros(size=size)

        # first layer
        # # double 0 index if there are more than 1 layers, because nested sequentials!
        layer_1 = self.model[0] if config.target_mlp.num_layers > 1 else self.model
        # Define bias row
        b = -1 if config.target_mlp.num_layers > 1 else 1
        compact_form[:b, 1 : config.data.features.max + 1] = layer_1[0].weight.data
        compact_form[:b, 0] = layer_1[0].bias.data

        # hidden layers
        for i in range(config.target_mlp.num_layers - 2):
            column_start = (
                1 + config.data.features.max + i * config.target_mlp.hidden_dim
            )
            column_end = (
                1 + config.data.features.max + (i + 1) * config.target_mlp.hidden_dim
            )
            compact_form[:-1, column_start:column_end] = self.model[i + 1][0].weight.data
            compact_form[-1, column_start:column_end] = self.model[i + 1][0].bias.data

        # Finish the puzzle by transposing last layer weights
        if config.target_mlp.num_layers > 1:
            compact_form[:-1, -1:] = self.model[-1].weight.data.T
            compact_form[-1, -1] = self.model[-1].bias.data

        return compact_form
