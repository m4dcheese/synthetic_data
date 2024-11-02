from __future__ import annotations

import torch
from torch import nn
from utils import get_activation


class PredictionMLP(nn.Module):
    def __init__(
        self,
        prediction_mlp_config,
        transformer_decoder_config,
        data_config,
        mlp_config,
    ):
        super().__init__()

        self.prediction_mlp_config = prediction_mlp_config
        self.transformer_decoder_config = transformer_decoder_config
        self.data_config = data_config
        self.mlp_config = mlp_config

        self.mlp = self._generate_mlp()

    def _generate_mlp(self):
        activation = get_activation(self.prediction_mlp_config.activation_str)
        layers = []

        weight_output_dim = (
            1
            + self.data_config.features.max
            + (self.mlp_config.hidden_dim * (self.mlp_config.num_layers - 2))
            + 1
        )
        # Define layers with specified initialization
        for i in range(self.prediction_mlp_config.num_layers):
            in_features = (
                self.transformer_decoder_config.decoder_layer.d_model
                if i == 0
                else self.prediction_mlp_config.hidden_dim
            )
            out_features = (
                weight_output_dim
                if i == self.prediction_mlp_config.num_layers - 1
                else self.prediction_mlp_config.hidden_dim
            )

            # Initialize layer and apply uniform weight initialization
            layer = nn.Linear(
                in_features,
                out_features,
                bias=self.prediction_mlp_config.bias,
            )

            # Append layer with optional activation
            layers.append(
                layer
                if i == self.prediction_mlp_config.num_layers - 1
                else nn.Sequential(layer, activation()),
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
