from __future__ import annotations
from argparse import ArgumentParser

import torch.multiprocessing as mp
from config import config
from gym.train import train
from models.cfm import build_cfm_from_config
from utils import (
    get_criterion,
    get_experiment_path,
    set_global_seed,
)

config_changes = [
    [
        "config.data.features.min=1",
        "config.data.samples.min=100",
    ],
    [
        "config.target_mlp.num_layers=2",
        "config.target_mlp.hidden_dim=4",
    ]
]

def main(run: int):
    """Main function for training process."""
    # Adapt config for different runs
    for config_change in config_changes[run]:
        eval(config_change)
    experiment_path = get_experiment_path(base_path=config.results.base_path)
    cfm_model = build_cfm_from_config(config=config)

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
    parser = ArgumentParser()
    parser.add_argument("--run", type=int, default=0, help="Index of config adaptation used in main function")
    args = parser.parse_args()
    main(run=args.run)
