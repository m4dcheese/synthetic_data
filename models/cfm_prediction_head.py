from __future__ import annotations

from config import config
from models.mlp import MLP


class PredictionHead(MLP):

    def __init__(self):
        weight_output_dim = (
            1
            + config.data.features.max
            + (config.target_mlp.hidden_dim * (config.target_mlp.num_layers - 2))
            + 1
        )

        super().__init__(in_features=config.cfm.transformer.decoder.d_model,
                         out_features=weight_output_dim,
                         hidden_dim=config.cfm.prediction_head.hidden_dim,
                         num_layers=config.cfm.prediction_head.num_layers,
                         activation=config.cfm.prediction_head.activation_str,
                         bias=config.cfm.prediction_head.bias)
