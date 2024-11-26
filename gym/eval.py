from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import torch
from data.data_hyperparameter_sampler import DataHyperparameterSampler
from data.training_dataset import TrainingDataset
from model_io import load_trained_model
from models.target_mlp import TargetMLP
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from tqdm import tqdm
from utils import ddp_cleanup, ddp_setup


def reparam_normal(shape, mean=0, std=1):
    """Sample from normal distribution with reparameterization trick."""
    return torch.randn(shape) * std + mean


def binary_decision(prediction, threshold):
    """Perform binary decision using ground truth threshold."""
    pred_mask = prediction < threshold
    prediction[pred_mask] = 0
    prediction[~pred_mask] = 1
    return prediction


class ODEFunc(torch.nn.Module):
    def __init__(self, cfm_model, xs, ys):
        super().__init__()
        self.cfm_model = cfm_model
        self.xs = xs
        self.ys = ys

    def forward(self, t, weights):
        t_tensor = torch.full(
            size=(weights.shape[0], 1, 1),
            fill_value=t,
            dtype=weights.dtype,
            device=weights.device,
        )
        v = self.cfm_model(xs=self.xs, ys=self.ys, ts=t_tensor, weights=weights)
        return v


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
    with torch.no_grad():

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
            ys_raw = batch.ys_raw
            threshold = batch.threshold

            # Save target weights only for comparison, batch.ts is not used
            weights_0 = batch.weights.to(rank)

            # Start with random MLP weights
            weights_t = reparam_normal(shape=weights_0.shape, mean=0, std=1)

            # Naive forward euler
            diff_list_vn = []
            ode_steps = 20

            if eval_config.solver == "naive":
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
            elif eval_config.solver == "odeint":
                ode_cfm = ODEFunc(cfm_model, xs, ys)

                # Solve ODE
                with torch.no_grad():
                    t_span = torch.tensor([1.0, 0.0], device=rank)  # Integration interval
                    weights_t_trajectory = odeint(
                        ode_cfm,
                        weights_t,
                        t_span,
                        atol=1e-5,
                        rtol=1e-5,
                    )
                # Use the final weights from the trajectory
                weights_t = weights_t_trajectory[-1]

            # Let's try the generated MLPs
            loss_fn = torch.nn.MSELoss()
            for mlp_i, compact_form in enumerate(weights_t):
                target_mlp = TargetMLP(
                    mlp_config=model_config.target_mlp,
                    data_config=model_config.data,
                ).to(rank)
                target_mlp.eval()
                pred = binary_decision(target_mlp(xs[mlp_i]), threshold[mlp_i])
                loss = loss_fn(pred, ys[mlp_i])
                print(f"model {mlp_i} before fit: {loss.cpu().detach().numpy()}")

                target_mlp.load_compact_form(compact_form=compact_form)
                target_mlp.eval()
                pred = target_mlp(xs[mlp_i])
                # Threshold can be replaced with 0 when target model weights include decision boundary
                pred_bin = binary_decision(pred.clone(), threshold[mlp_i]).squeeze().detach()
                gt = ys[mlp_i].detach()
                loss = loss_fn(pred_bin, gt)
                print(f"model {mlp_i} after fit: {loss.cpu().detach().numpy()}")

                confusion = []
                for x in (0, 1):
                    confusion_part = []
                    for y in (0, 1):
                        confusion_part.append(((pred_bin == x) & (gt == y)).sum())
                    confusion.append(confusion_part)

                sn.heatmap(confusion, annot=True)
                plt.show()
                plt.figure()
                plt.scatter(pred.detach(), ys_raw[mlp_i].detach(), label="Predictions over ground truth")
                plt.xlabel("True Y")
                plt.ylabel("Prediction")
                plt.show()

            # See how close we got to the ground truth:
            if eval_config.solver == "naive":
                diff_list_vn = torch.cat(diff_list_vn)
                plt.figure()
                x = np.arange(0, 0.9999999, 1 / ode_steps)
                plt.plot(x, diff_list_vn[:, 0], label="Vector norm sample 0")
                plt.plot(x, diff_list_vn[:, 1], label="Vector norm sample 1")
                plt.legend()
                plt.show()

    ddp_cleanup(world_size=eval_config.world_size)

    return cfm_model
