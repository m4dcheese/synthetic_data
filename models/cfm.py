from __future__ import annotations

from models.cfm_data_projection import DataProjection
from models.cfm_prediction_head import PredictionHead
from models.cfm_weight_projection import WeightProjection
from models.positional_encoding import PositionalEncoding
from models.target_mlp import TargetMLP
from torch.nn import (
    Module,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class CFM(Module):
    """Conditional Flow Matching model class."""
    def __init__(
        self,
        data_projection: Module,
        weight_projection: Module,
        encoder: Module,
        decoder: Module,
        prediction_head: Module,
    ):
        """Composition of projections, transformer and prediction head."""
        super().__init__()
        self.data_projection = data_projection
        self.weight_projection = weight_projection
        self.encoder = encoder
        self.decoder = decoder
        self.prediction_head = prediction_head

    def forward(self, xs, ys, ts, weights):
        """Encode data and t, decode to projected weights."""
        data = self.data_projection(xs=xs, ys=ys, ts=ts)
        encoded = self.encoder(data)

        weights = self.weight_projection(weights=weights)
        decoded = self.decoder(tgt=weights, memory=encoded)
        return self.prediction_head(decoded)


def build_cfm_from_config(config) -> CFM:
    """Build a CFM model given a config DotDict."""
    target_mlp = TargetMLP(mlp_config=config.target_mlp, data_config=config.data)

    return CFM(
        data_projection=DataProjection(
            input_dim=config.data.features.max,
            hidden_dim=config.cfm.data_projection.hidden_dim,
        ),
        weight_projection=WeightProjection(
            input_dim=target_mlp.compact_shape()[1],
            hidden_dim=config.cfm.weight_projection.hidden_dim,
            positional_encoding=PositionalEncoding(
                d_model=config.cfm.transformer.decoder.d_model,
                dropout=config.cfm.positional_encoding.dropout,
                max_len=config.data.samples.max,
            ),
        ),
        encoder=TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=config.cfm.transformer.encoder.d_model,
                nhead=config.cfm.transformer.encoder.nhead,
                dim_feedforward=config.cfm.transformer.encoder.dim_feedforward,
                dropout=config.cfm.transformer.encoder.dropout,
                activation=config.cfm.transformer.encoder.activation_str,
                layer_norm_eps=config.cfm.transformer.encoder.layer_norm_eps,
                batch_first=config.cfm.transformer.encoder.batch_first,
                norm_first=config.cfm.transformer.encoder.norm_first,
                bias=config.cfm.transformer.encoder.bias,
            ),
            num_layers=config.cfm.transformer.encoder.num_layers,
        ),
        decoder=TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(
                d_model=config.cfm.transformer.decoder.d_model,
                nhead=config.cfm.transformer.decoder.nhead,
                dim_feedforward=config.cfm.transformer.decoder.dim_feedforward,
                dropout=config.cfm.transformer.decoder.dropout,
                activation=config.cfm.transformer.decoder.activation_str,
                layer_norm_eps=config.cfm.transformer.decoder.layer_norm_eps,
                batch_first=config.cfm.transformer.decoder.batch_first,
                norm_first=config.cfm.transformer.decoder.norm_first,
                bias=config.cfm.transformer.decoder.bias,
            ),
            num_layers=config.cfm.transformer.decoder.num_layers,
        ),
        prediction_head=PredictionHead(
            weight_output_dim=target_mlp.compact_shape()[1],
            transformer_config=config.cfm.transformer,
            prediction_head_config=config.cfm.prediction_head,
        ),
    )
