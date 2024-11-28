from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from data.data_hyperparameter_sampler import DataHyperparameterSampler
from data.training_dataset import TrainingDataset
from model_io import load_trained_model
from models.target_mlp import TargetMLP
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchdiffeq import odeint
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
    """Encapsulate X-Y data with model for ODE solver."""

    def __init__(self, cfm_model, xs, ys):
        """Initialize new model with partial arguments x and y."""
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
        return self.cfm_model(xs=self.xs, ys=self.ys, ts=t_tensor, weights=weights)


def evaluate_target_network(
    mlp_config,
    data_config,
    x: torch.Tensor,
    threshold: int = 0,
    compact_form: torch.Tensor = None,
):
    """Load target network and evaluate."""
    model = TargetMLP(mlp_config=mlp_config, data_config=data_config)
    if compact_form is not None:
        model.load_compact_form(compact_form=compact_form)
    model.eval()
    with torch.no_grad():
        pred = model(x).detach().cpu()
        pred_bin = binary_decision(pred.clone(), threshold=threshold).squeeze()

    return pred, pred_bin


def solve_ode_naive(
    cfm_model: torch.nn.Module,
    initial_weights: torch.Tensor,
    steps: int,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Solve ODE with naive explicit euler."""
    weights_t = initial_weights
    trajectory = [initial_weights.clone().unsqueeze(0)]
    for i in range(steps):
        t_value = 1 - (i / steps)
        v = cfm_model(
            xs=x,
            ys=y,
            ts=torch.full(
                size=(x.shape[0], 1, 1),
                fill_value=t_value,
            ),
            weights=weights_t,
        )
        weights_t -= v / steps
        trajectory.append(weights_t.clone().unsqueeze(0))
    return torch.cat(trajectory)


def solve_ode_odeint(
    cfm_model: torch.nn.Module,
    initial_weights: torch.Tensor,
    steps: int,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Solve ODE using torchdiffeq."""
    ode_cfm = ODEFunc(cfm_model, x, y).to(x.device)
    with torch.no_grad():
        t_span = torch.tensor(np.linspace(1.0, 0.0, steps + 1), device=x.device)
        return odeint(
            ode_cfm,
            initial_weights,
            t_span,
            atol=1e-5,
            rtol=1e-5,
        )


def plot_confusion_matrix(pred_bin, gt_bin):
    """Display confusion matrix."""
    plt.figure()
    confusion = []
    for x in (0, 1):
        confusion_part = []
        for y in (0, 1):
            confusion_part.append(((pred_bin == x) & (gt_bin == y)).sum())
        confusion.append(confusion_part)
    sns.heatmap(confusion, annot=True)
    plt.show()


def plot_prediction_scatter(pred, pred_bin, gt, threshold):
    """Scatter prediction correlation and draw/color decision boundaries."""
    plt.figure()
    plt.scatter(
        x=gt,
        y=pred,
        c=pred_bin,
        label="Predictions over ground truth",
    )
    plt.vlines([threshold], -10, 10, label="Threshold")
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.show()


def plot_flow_trajectory(gt: torch.Tensor, trajectory: torch.Tensor):
    """See how close we got to the ground truth through flow matching."""
    diff_list_vn = np.array(
        [torch.linalg.vector_norm(t - gt).numpy() for t in trajectory],
    )
    plt.figure()
    x = np.linspace(start=0, stop=1, num=len(trajectory))
    plt.plot(x, diff_list_vn, label="Distance of pred weights from gt weights")
    plt.legend()
    plt.show()


def plot_roc_auc_scores(roc_auc_scores):
    """Plot ROC AUC scores."""
    # use seaborn to plot the ROC AUC scores (roc_auc_scores is list of scalars)
    # figure size 16x12
    # make nice seaborn style barplot
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    # histplot
    sns.histplot(roc_auc_scores, bins=10, kde=False)  # Adjust bins as needed
    plt.xlabel("Model")
    plt.ylabel("ROC AUC Score")
    # rotate x-axis labels for better readability
    plt.xticks(rotation=25)
    plt.title(f"Histogram of ROC AUC Scores across {len(roc_auc_scores)} models")
    plt.show()


def plot_decision_boundary(
    mlp_config, data_config, compact_form, gt_weights, x, threshold
):
    # basic approach:
    # 1. zero out all but the first two features because we can only plot in 2D
    # 2. create a mesh grid in the 2d space
    # 3. predict over the grid (create 8 zero features for the grid)
    # 4. plot the decision boundary

    x[:, :, 2:] = 0

    y, y_bin = evaluate_target_network(
        mlp_config=mlp_config,
        data_config=data_config,
        x=x,
        threshold=threshold,
        compact_form=gt_weights,
    )

    # Step 2: Create a mesh grid in the 2d space
    x_min, x_max = x[0, :, 0].min() - 1, x[0, :, 0].max() + 1
    y_min, y_max = x[0, :, 1].min() - 1, x[0, :, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Flatten the mesh grid and inverse transform back to the original feature space
    grid = np.c_[xx.ravel(), yy.ravel()]

    # add 8 zero padding on the right on the last dimension for grid
    grid = np.concatenate([grid, np.zeros((grid.shape[0], 8))], axis=1)

    # Step 3: Predict over the grid
    pred, pred_bin = evaluate_target_network(
        mlp_config=mlp_config,
        data_config=data_config,
        x=torch.tensor(grid, dtype=torch.float32).unsqueeze(0),
        threshold=threshold,
        compact_form=compact_form,
    )

    # Reshape predictions to the shape of the mesh grid
    pred_bin = pred_bin.reshape(xx.shape)

    # Step 4: Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, pred_bin, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.9)
    plt.colorbar(label="Target Network Decision Boundary")

    # get the true labels for the data points

    x1 = x[0, :, 0].numpy()
    x2 = x[0, :, 1].numpy()

    scatter = plt.scatter(
        x1,
        x2,
        c=y_bin,
        cmap="coolwarm",
        edgecolor="k",
        alpha=0.8,
    )
    plt.legend(*scatter.legend_elements(), title="Ground Truth")
    plt.title("Decision Boundary in 2D Space")
    plt.xlabel("Features 1")
    plt.ylabel("Features 2")
    plt.grid(True)
    # I think the plot looks so noisy due to biases not being zeroes in the model
    plt.show()


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

            if eval_config.solver == "naive":
                solve_fn = solve_ode_naive
            else:
                solve_fn = solve_ode_odeint

            trajectory = solve_fn(
                cfm_model=cfm_model,
                initial_weights=weights_t,
                steps=eval_config.solve_steps,
                x=xs,
                y=ys,
            )

            weights_t = trajectory[-1]

            # Let's try the generated MLPs
            loss_fn = torch.nn.MSELoss()

            roc_auc_scores = []
            for mlp_i, compact_form in enumerate(weights_t):
                threshold = (
                    0
                    if model_config.data.shift_for_threshold
                    else batch.threshold[mlp_i]
                )
                xs_inference = dataset.generate_features(
                    features=batch.data_hps[mlp_i].features,
                    samples=batch.data_hps[mlp_i].samples,
                    max_sampled_sequence_length=max(
                        [data_hps.samples for data_hps in batch.data_hps],
                    ),
                )

                # Use ground truth model to generate new y labels
                gt_weights = weights_0[mlp_i]
                gt, gt_bin = evaluate_target_network(
                    mlp_config=model_config.target_mlp,
                    data_config=model_config.data,
                    x=xs_inference,
                    threshold=0,
                    compact_form=gt_weights,
                )
                # Use predicted model to generate predicted y labels
                pred, pred_bin = evaluate_target_network(
                    mlp_config=model_config.target_mlp,
                    data_config=model_config.data,
                    x=xs_inference,
                    threshold=0,
                    compact_form=compact_form,
                )

                # measure ROC AUC score for each model and store it
                roc_auc_scores.append(
                    roc_auc_score(
                        gt_bin.squeeze().detach().numpy(),
                        pred_bin.squeeze().detach().numpy(),
                    )
                )

                loss = loss_fn(pred_bin, gt_bin.squeeze())
                print(f"model {mlp_i} after fit: {loss.cpu().detach().numpy()}")

                plot_confusion_matrix(pred_bin=pred_bin, gt_bin=gt_bin)
                plot_prediction_scatter(
                    pred=pred,
                    pred_bin=pred_bin,
                    gt=gt,
                    threshold=threshold,
                )
                plot_flow_trajectory(
                    batch.weights[mlp_i],
                    trajectory=trajectory[:, mlp_i],
                )

                plot_decision_boundary(
                    mlp_config=model_config.target_mlp,
                    data_config=model_config.data,
                    compact_form=compact_form,
                    gt_weights=gt_weights,
                    x=xs_inference,
                    threshold=threshold,
                )

            plot_roc_auc_scores(roc_auc_scores)
    ddp_cleanup(world_size=eval_config.world_size)

    return cfm_model
