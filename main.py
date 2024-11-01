from __future__ import annotations

import torch
from config import config
from data.data_hyperparameter_sampler import DataHyperparameterSampler
from data.training_dataset import TrainingDataset
from gym.train import train
from models.cfm import CFM
from models.data_projection import DataProjection
from models.weight_projection import WeightProjection
from torch.nn import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch.utils.data import DataLoader
from utils import get_criterion, get_number_parameters, get_optimizer


def main():
    cfm_model = CFM(
        data_projection=DataProjection(
            input_dim=config.data.features.max,
            hidden_dim=config.cfm.data_projection.hidden_dim,
        ),
        weight_projection=WeightProjection(
            input_dim=1,
            hidden_dim=config.cfm.weight_projection.hidden_dim,
        ),
        encoder=TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=config.cfm.transformer.encoder.encoder_layer.d_model,
                nhead=config.cfm.transformer.encoder.encoder_layer.nhead,
                dim_feedforward=config.cfm.transformer.encoder.encoder_layer.dim_feedforward,
                dropout=config.cfm.transformer.encoder.encoder_layer.dropout,
                activation=config.cfm.transformer.encoder.encoder_layer.activation_str,
                layer_norm_eps=config.cfm.transformer.encoder.encoder_layer.layer_norm_eps,
                batch_first=config.cfm.transformer.encoder.encoder_layer.batch_first,
                norm_first=config.cfm.transformer.encoder.encoder_layer.norm_first,
                bias=config.cfm.transformer.encoder.encoder_layer.bias,
            ),
            num_layers=config.cfm.transformer.encoder.num_layers,
        ),
        decoder=TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(
                d_model=config.cfm.transformer.decoder.decoder_layer.d_model,
                nhead=config.cfm.transformer.decoder.decoder_layer.nhead,
                dim_feedforward=config.cfm.transformer.decoder.decoder_layer.dim_feedforward,
                dropout=config.cfm.transformer.decoder.decoder_layer.dropout,
                activation=config.cfm.transformer.decoder.decoder_layer.activation_str,
                layer_norm_eps=config.cfm.transformer.decoder.decoder_layer.layer_norm_eps,
                batch_first=config.cfm.transformer.decoder.decoder_layer.batch_first,
                norm_first=config.cfm.transformer.decoder.decoder_layer.norm_first,
                bias=config.cfm.transformer.decoder.decoder_layer.bias,
            ),
            num_layers=config.cfm.transformer.decoder.num_layers,
        ),
    )

    rank = "cpu"  # TODO MultiGPU Training
    world_size = 1

    dataset = TrainingDataset(
        data_hp_sampler=DataHyperparameterSampler(config.data),
        training_config=config.training,
        data_config=config.data,
        mlp_config=config.mlp,
        rank=rank,
        worldsize=world_size,
        num_data_workers=config.training.num_data_workers,
    )

    # "When both batch_size and batch_sampler are None (default value for batch_sampler
    # is already None), automatic batching is disabled." (https://pytorch.org/docs/stable/data.html)
    data_loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        worker_init_fn=dataset.worker_init_fn,
        num_workers=config.training.num_data_workers,
        prefetch_factor=10 if config.training.num_data_workers > 0 else None,
        persistent_workers=config.training.num_data_workers > 0,
    )

    optimizer = get_optimizer(optimizer_str=config.optimizer.optimizer_str)(
        cfm_model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )

    loss_fn = get_criterion(criterion_str=config.criterion.criterion_str)()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Total Parameters CFM Model: {get_number_parameters(cfm_model)}")

    cfm_model = train(
        cfm=cfm_model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        dataloader=data_loader,
        training_config=config.training,
        device=device,
    )
    print("")


if __name__ == "__main__":
    main()
