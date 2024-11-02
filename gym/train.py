from __future__ import annotations

from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


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
    cfm: nn.Module,
    dataloader: DataLoader,
    loss_fn,
    optimizer,
    training_config,
    device,
):
    cfm.to(device)

    for _iteration_i in tqdm(range(training_config.total_iterations)):
        for _batch_i, batch in enumerate(dataloader):
            xs = batch.xs.to(device)
            ys = batch.ys.to(device)
            ts = batch.t.to(device).reshape(-1, 1, 1)

            weights = batch.weights.to(device)
            noise = sample_z1(z0_shape=weights.shape, mean=0, std=1).to(device)

            z0, z1 = match_closest_samples(z0=weights, z1=noise)

            z_t = generate_trajectory(
                z0=z0,
                z1=z1,
                t=ts,
                sigma=training_config.sigma,
            ).to(device)

            target = ((1 - training_config.sigma) * z1 - z0).to(device)

            optimizer.zero_grad()

            v = cfm(xs=xs, ys=ys, ts=ts, weights=z_t)

            loss = loss_fn(v, target)
            loss.backward()
            optimizer.step()

    return cfm
