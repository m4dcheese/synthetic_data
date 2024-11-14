from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from data.data_hyperparameter_sampler import DataHyperparameterSampler
from data.training_dataset import TrainingDataset
from model_io import load_trained_model
from scipy.optimize import linear_sum_assignment
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import ddp_cleanup, ddp_setup


def reparam_normal(shape, mean=0, std=1):
    """Sample from normal distribution with reparameterization trick."""
    return torch.randn(shape) * std + mean


def generate_trajectory(z0, z1, t, sigma):
    """Linear interpolation between z0 and z1."""
    return (1 - t) * z0 + (sigma + (1 - sigma) * t) * z1


def match_closest_samples(z0: torch.Tensor, z1: torch.Tensor):
    """Find best sample pairs to avoid intersecting trajecrories."""
    # Flatten the tensors to shape (batch_size, -1)
    z0_flat = z0.flatten(start_dim=1)
    z1_flat = z1.flatten(start_dim=1)

    # Compute the (batch_size, batch_size) distance matrix
    D = np.linalg.norm(z0_flat[:, None] - z1_flat[None, :], axis=2)

    # Use the linear sum assignment algorithm to find the optimal assignment
    _, col_ind = linear_sum_assignment(D)
    # permute the rows of B to match the columns of A
    b_matched = z1_flat[col_ind]
    b_matched = b_matched.reshape(z1.shape)
    return z0, b_matched


def evaluate(path: str, eval_config):
    """Run the main training loop for conditional flow matching."""
    rank = "cpu" if eval_config.world_size == 0 else "cuda:0"
    # First Setup the Distributed Data Parallel
    ddp_setup(rank=rank, world_size=eval_config.world_size)
    model_config, cfm_model, optimizer = load_trained_model(path=path, rank=rank)

    if eval_config.world_size > 1:
        # find unused parameters=True Bug: https://stackoverflow.com/questions/68000761/pytorch-ddp-finding-the-cause-of-expected-to-mark-a-variable-ready-only-once
        cfm_model = DistributedDataParallel(
            cfm_model,
            device_ids=[rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    # Set to eval mode
    cfm_model.eval()

    # Dataset and DataLoader
    dataset = TrainingDataset(
        data_hp_sampler=DataHyperparameterSampler(model_config.data),
        training_config=eval_config,
        data_config=model_config.data,
        mlp_config=model_config.target_mlp,
        rank=rank,
        worldsize=eval_config.world_size,
        num_data_workers=eval_config.num_data_workers,
    )

    # "When both batch_size and batch_sampler are None (default value for batch_sampler
    # is already None), automatic batching is disabled." (https://pytorch.org/docs/stable/data.html)
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        worker_init_fn=dataset.worker_init_fn,
        num_workers=eval_config.num_data_workers,
        prefetch_factor=eval_config.data_prefetch_factor
        if eval_config.num_data_workers > 0
        else None,
        persistent_workers=eval_config.num_data_workers > 0,
    )

    for _, batch in enumerate(dataloader):
        xs = batch.xs.to(rank)
        ys = batch.ys.to(rank)
        # Save target weights only for comparison, batch.ts is not used
        weights_0 = batch.weights.to(rank)

        # Start with random MLP weights
        weights_t = reparam_normal(shape=weights_0.shape, mean=0, std=1)

        # Naive forward euler
        diff_list_vn = []
        ode_steps = 50
        with torch.no_grad():
            for i in tqdm(range(ode_steps)):
                # TODO Is t correct, or other direction?
                t_value = 1 - (i / ode_steps)
                v = cfm_model(
                    xs=xs,
                    ys=ys,
                    ts=torch.full(
                        size=(eval_config.batch_size, 1, 1), fill_value=t_value,
                    ),
                    weights=weights_t,
                )
                weights_t -= v / ode_steps
                diff_list_vn.append(
                    torch.linalg.vector_norm(
                        weights_0 - weights_t, dim=(1, 2), keepdim=True,
                    ).reshape(1, -1),
                )
        diff_list_vn = torch.cat(diff_list_vn)
        plt.figure()
        x = np.arange(0, 0.9999999, 1 / ode_steps)
        plt.plot(x, diff_list_vn[:, 0], label="Vector norm sample 0")
        plt.plot(x, diff_list_vn[:, 1], label="Vector norm sample 1")
        plt.legend()
        plt.show()

    ddp_cleanup(world_size=eval_config.world_size)

    return cfm_model
