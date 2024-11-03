from __future__ import annotations

import torch.multiprocessing as mp
from config import config
from gym.train import train
from models.cfm import CFM
from models.cfm_data_projection import DataProjection
from models.cfm_prediction_head import PredictionHead
from models.cfm_weight_projection import WeightProjection
from torch.nn import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from utils import get_criterion, get_experiment_path, set_global_seed


def main():
    experiment_path = get_experiment_path(base_path=config.results.base_path)

    cfm_model = CFM(
        data_projection=DataProjection(
            input_dim=config.data.features.max,
            hidden_dim=config.cfm.data_projection.hidden_dim,
        ),
        weight_projection=WeightProjection(
            hidden_dim=config.cfm.weight_projection.hidden_dim,
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
        prediction_head=PredictionHead(),
    )

    loss_fn = get_criterion(criterion_str=config.criterion.criterion_str)()

    if config.training.world_size > 1:
        mp.spawn(
            train,
            args=(
                config.training.world_size,
                cfm_model,
                loss_fn,
                # Configurations
                config.training,
                config.optimizer,
                config.data,
                config.target_mlp,
                # Rank & World Size
                config.training.world_size,
                # Path to save the model
                experiment_path,
            ),
            nprocs=config.training.world_size,
            join=True,
        )
    elif config.training.world_size == 1:
        cfm_model = train(
            cfm_model=cfm_model,
            loss_fn=loss_fn,
            # Configurations
            training_config=config.training,
            optimizer_config=config.optimizer,
            data_config=config.data,
            mlp_config=config.target_mlp,
            # Rank & World Size
            world_size=config.training.world_size,
            rank="cuda:0",
            # Path to save the model
            save_path=experiment_path,
        )
    else:
        cfm_model = train(
            cfm_model=cfm_model,
            loss_fn=loss_fn,
            # Configurations
            training_config=config.training,
            optimizer_config=config.optimizer,
            data_config=config.data,
            mlp_config=config.target_mlp,
            # Rank & World Size
            world_size=config.training.world_size,
            rank="cpu",
            # Path to save the model
            save_path=experiment_path,
        )


if __name__ == "__main__":
    set_global_seed(42)
    main()
