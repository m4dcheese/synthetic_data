from __future__ import annotations

from models.mlp import MLP


class PredictionHead(MLP):
    """Basic MLP prediction head mapping tokens to rows in compact target mlp form."""
    def __init__(
        self,
        weight_output_dim: int,
        transformer_config,
        prediction_head_config,
    ):
        """Specify configs and weight_output_dim (width of compact form)."""
        super().__init__(
            in_features=transformer_config.decoder.d_model,
            out_features=weight_output_dim,
            hidden_dim=prediction_head_config.hidden_dim,
            num_layers=prediction_head_config.num_layers,
            activation_str=prediction_head_config.activation_str,
            bias=prediction_head_config.bias,
        )
