from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from data.data_hyperparameter_sampler import DataHyperparameterSampler
from data.training_dataset import TrainingDataset
from model_io import load_trained_model
from models.target_mlp import TargetMLP
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import ddp_cleanup, ddp_setup


def reparam_normal(shape, mean=0, std=1):
    """Sample from normal distribution with reparameterization trick."""
    return torch.randn(shape) * std + mean


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
        ode_steps = 20
        with torch.no_grad():
            for i in tqdm(range(ode_steps)):
                # TODO Is t correct, or other direction?
                t_value = 1 - (i / ode_steps)
                v = cfm_model(
                    xs=xs,
                    ys=ys,
                    ts=torch.full(
                        size=(eval_config.batch_size, 1, 1),
                        fill_value=t_value,
                    ),
                    weights=weights_t,
                )
                weights_t -= v / ode_steps
                diff_list_vn.append(
                    torch.linalg.vector_norm(
                        weights_0 - weights_t,
                        dim=(1, 2),
                        keepdim=True,
                    ).reshape(1, -1),
                )
        # Let's try the generated MLPs
        loss_fn = torch.nn.MSELoss()
        for mlp_i, compact_form in enumerate(weights_t):
            target_mlp = TargetMLP(
                mlp_config=model_config.target_mlp, data_config=model_config.data,
            ).to(rank)
            target_mlp.eval()
            pred = target_mlp(xs[mlp_i])
            loss = loss_fn(pred, ys[mlp_i])
            print(f"model {mlp_i} before fit: {loss.cpu().detach().numpy()}")
            target_mlp.load_compact_form(compact_form=compact_form)
            target_mlp.eval()
            pred = target_mlp(xs[mlp_i])
            loss = loss_fn(pred, ys[mlp_i])
            print(f"model {mlp_i} after fit: {loss.cpu().detach().numpy()}")

        # See how close we got to the ground truth:
        diff_list_vn = torch.cat(diff_list_vn)
        plt.figure()
        x = np.arange(0, 0.9999999, 1 / ode_steps)
        plt.plot(x, diff_list_vn[:, 0], label="Vector norm sample 0")
        plt.plot(x, diff_list_vn[:, 1], label="Vector norm sample 1")
        plt.legend()
        plt.show()

    ddp_cleanup(world_size=eval_config.world_size)

    return cfm_model



# def sample_from_model(model, x_0):
#     t = torch.tensor([1.0, 0.0], dtype=x_0.dtype, device="cuda")
#     fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5, adjoint_params=model.func.parameters())
#     return fake_image