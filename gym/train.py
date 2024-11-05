from __future__ import annotations

from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader

import numpy as np
import torch
from data.data_hyperparameter_sampler import DataHyperparameterSampler
from data.training_dataset import TrainingDataset
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from utils import ddp_cleanup, ddp_setup, get_optimizer, save_trained_model


def sample_z1(z0_shape, mean=0, std=1):
    return torch.randn(z0_shape) * std + mean


def generate_trajectory(z0, z1, t, sigma):
    return (1 - t) * z0 + (sigma + (1 - sigma) * t) * z1


def match_closest_samples(z0, z1):
    batch_size = z0.shape[0]
    # Flatten the tensors to shape (batch_size, -1)
    A_flat = z0.reshape(batch_size, -1)
    B_flat = z1.reshape(batch_size, -1)

    # Compute the (batch_size, batch_size) distance matrix
    D = np.linalg.norm(A_flat[:, None] - B_flat[None, :], axis=2)

    # Use the linear sum assignment algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(D)
    # permute the rows of B to match the columns of A
    B_matched = B_flat[col_ind]
    B_matched = B_matched.reshape(z1.shape)
    return z0, B_matched


def train(
    cfm_model: nn.Module,
    loss_fn,
    # Configurations
    training_config,
    optimizer_config,
    data_config,
    mlp_config,
    # Rank & Worldsize
    world_size,
    rank,
    # Path to save the model
    save_path,
):
    # First Setup the Distributed Data Parallel
    ddp_setup(rank=rank, world_size=world_size)
    cfm_model.to(rank)

    if world_size > 1:
        # find unused parameters=True Bug: https://stackoverflow.com/questions/68000761/pytorch-ddp-finding-the-cause-of-expected-to-mark-a-variable-ready-only-once
        cfm_model = DDP(
            cfm_model,
            device_ids=[rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    # initialize optimizer with model parameters
    optimizer = get_optimizer(optimizer_config.optimizer_str)(
        cfm_model.parameters(),
        lr=optimizer_config.lr,
        weight_decay=optimizer_config.weight_decay,
    )

    # Dataset and DataLoader
    dataset = TrainingDataset(
        data_hp_sampler=DataHyperparameterSampler(data_config),
        training_config=training_config,
        data_config=data_config,
        mlp_config=mlp_config,
        rank=rank,
        worldsize=training_config.world_size,
        num_data_workers=training_config.num_data_workers,
    )

    # "When both batch_size and batch_sampler are None (default value for batch_sampler
    # is already None), automatic batching is disabled." (https://pytorch.org/docs/stable/data.html)
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        worker_init_fn=dataset.worker_init_fn,
        num_workers=training_config.num_data_workers,
        prefetch_factor=training_config.data_prefetch_factor
        if training_config.num_data_workers > 0
        else None,
        persistent_workers=training_config.num_data_workers > 0,
    )

    for _iteration_i in tqdm(range(training_config.total_iterations)):
        for _batch_i, batch in enumerate(dataloader):
            xs = batch.xs.to(rank)
            ys = batch.ys.to(rank)
            ts = batch.t.to(rank).reshape(-1, 1, 1)

            weights = batch.weights.to(rank)
            noise = sample_z1(z0_shape=weights.shape, mean=0, std=1).to(rank)

            z0, z1 = match_closest_samples(z0=weights, z1=noise)

            z_t = generate_trajectory(
                z0=z0,
                z1=z1,
                t=ts,
                sigma=training_config.sigma,
            ).to(rank)

            # target is not the new position but the difference between the new position and the old position
            target = ((1 - training_config.sigma) * z1 - z0).to(rank)

            v = cfm_model(xs=xs, ys=ys, ts=ts, weights=z_t)

            loss = loss_fn(v, target)

            optimizer.zero_grad()
            loss.backward()
            # Update the model parameters across all GPUs (triggers the all-reduce)
            optimizer.step()

    ddp_cleanup(world_size=world_size)

    if rank in ["cpu", "cuda:0"]:
        save_trained_model(model=cfm_model, optimizer=optimizer, save_path=save_path)

    return cfm_model
