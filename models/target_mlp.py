import torch
from config import DotDict
from models.mlp import MLP


class TargetMLP(MLP):
    """Model to predict by CFM."""

    def __init__(
        self, mlp_config: DotDict, data_config: DotDict, in_features: int = -1,
    ):
        """in_features causes unused input dimensions to be ignored
        by setting weights to zero (see below).
        """
        super().__init__(
            in_features=data_config.features.max,
            out_features=mlp_config.output_dim,
            hidden_dim=mlp_config.hidden_dim,
            num_layers=mlp_config.num_layers,
            activation_str=mlp_config.activation_str,
            initialization=mlp_config.initialization,
            bias=mlp_config.bias,
        )

        self.hidden_dim = mlp_config.hidden_dim
        self.num_layers = mlp_config.num_layers
        self.max_features = data_config.features.max

        if data_config.features.max > in_features and in_features > 0:
            layer_1 = self.model[0] if self.num_layers > 1 else self.model
            layer_1[0].weight.data[:, mlp_config.in_features :].zero_()

    def compact_shape(self) -> torch.Tensor:
        """Compute shape of compact representation."""
        return (
            (self.hidden_dim if self.num_layers > 1 else 0) + 1,
            (1 if self.num_layers > 1 else 0)  # Extra column for bias of layer 1
            + self.max_features
            + (self.hidden_dim * max(0, self.num_layers - 2))
            + 1,
        )

    def load_compact_form(self, compact_form: torch.Tensor) -> None:
        """Load weights and biases from compact form."""
        self.model[0][0].bias.data = compact_form[:-1, 0]
        self.model[0][0].weight.data = compact_form[:-1, 1 : self.max_features + 1]

        for i in range(self.num_layers - 2):
            col_start = 1 + self.max_features + i * self.hidden_dim
            col_end = 1 + self.max_features + (i + 1) * self.hidden_dim
            self.model[i + 1][0].weight.data = compact_form[:-1, col_start:col_end]
            self.model[i + 1][0].bias.data = compact_form[-1, col_start:col_end]

        # Finish the puzzle by transposing last layer weights
        self.model[-1].weight.data = compact_form[:-1, -1:].T
        self.model[-1].bias.data = compact_form[-1, -1]

    def get_compact_form(self) -> torch.Tensor:
        """Return compact form representing model weights and biases."""
        size = TargetMLP.compact_shape()

        compact_form = torch.zeros(size=size)

        # first layer
        # # double 0 index if there are more than 1 layers, because nested sequentials!
        layer_1 = self.model[0] if self.num_layers > 1 else self.model
        # Define bias row
        b = -1 if self.num_layers > 1 else 1
        compact_form[:b, 1 : self.max_features + 1] = layer_1[0].weight.data
        compact_form[:b, 0] = layer_1[0].bias.data

        # hidden layers
        for i in range(self.num_layers - 2):
            col_start = 1 + self.max_features + i * self.hidden_dim
            col_end = 1 + self.max_features + (i + 1) * self.hidden_dim
            compact_form[:-1, col_start:col_end] = self.model[i + 1][0].weight.data
            compact_form[-1, col_start:col_end] = self.model[i + 1][0].bias.data

        # Finish the puzzle by transposing last layer weights
        if self.num_layers > 1:
            compact_form[:-1, -1:] = self.model[-1].weight.data.T
            compact_form[-1, -1] = self.model[-1].bias.data

        return compact_form
