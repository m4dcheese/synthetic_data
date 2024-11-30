from __future__ import annotations

import numpy as np
import seaborn as sns
import torch
from gym.eval_utils import evaluate_target_network
from matplotlib import pyplot as plt


def plot_confusion_matrix(
    pred_bin, gt_bin, model: str = "", roc_auc: int = -1, instant_show: bool = True,
):
    """Display confusion matrix."""
    plt.figure()
    confusion = []
    for x in (0, 1):
        confusion_part = []
        for y in (0, 1):
            confusion_part.append(((pred_bin == x) & (gt_bin == y)).sum())
        confusion.append(confusion_part)

    sns.heatmap(confusion, annot=True, cmap="Blues", alpha=0.8)
    plt.xlabel("True Class")
    plt.ylabel("Predicted Class")
    plt.title(f"Confusion Matrix of {model}")
    if roc_auc > 0:
        plt.suptitle(f"ROC AUC Score: {roc_auc:.3f}")
    plt.grid(visible=False)
    if instant_show:
        plt.show()


def plot_prediction_scatter(pred, pred_bin, gt, threshold):
    """Scatter prediction correlation and draw/color decision boundaries."""
    x_min, x_max = gt.min() - 1, gt.max() + 1
    y_min, y_max = pred.min() - 1, pred.max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    plt.figure()
    plt.contourf(xx, yy, xx > threshold, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.9)
    plt.colorbar(label="True Decision Boundary")
    scatter = plt.scatter(
        x=gt,
        y=pred,
        c=pred_bin,
        label="Examples",
        cmap="coolwarm",
        alpha=0.8,
        edgecolor="k",
    )
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.legend(*scatter.legend_elements(), title="Predicted Class")
    plt.grid(visible=True)
    plt.title("True vs. Predicted Class")
    plt.show()


def plot_flow_trajectory(gt: torch.Tensor, trajectory: torch.Tensor):
    """See how close we got to the ground truth through flow matching."""
    diff_list_vn = np.array(
        [torch.linalg.vector_norm(t - gt).numpy() for t in trajectory],
    )
    plt.figure()
    x = np.linspace(start=0, stop=1, num=len(trajectory))
    plt.plot(x, diff_list_vn)
    plt.xlabel("Timestep t of ODE Solver")
    plt.ylabel("Distance to True MLP Weights")
    plt.title("ODE Solver Trajectory Distance")
    plt.show()


def plot_roc_auc_scores(roc_auc_scores):
    """Plot ROC AUC scores."""
    # use seaborn to plot the ROC AUC scores (roc_auc_scores is list of scalars)
    # figure size 16x12
    # make nice seaborn style barplot
    sns.set_theme(style="whitegrid")
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
    mlp_config,
    data_config,
    compact_form,
    gt_weights,
    x,
    threshold,
):
    """Compare model decision boundary to true classes."""
    # basic approach:
    # 1. zero out all but the first two features because we can only plot in 2D
    # 2. create a mesh grid in the 2d space
    # 3. predict over the grid (create 8 zero features for the grid)
    # 4. plot the decision boundary

    x[:, 2:] = 0

    y, y_bin = evaluate_target_network(
        mlp_config=mlp_config,
        data_config=data_config,
        x=x,
        threshold=threshold,
        compact_form=gt_weights,
    )

    # Step 2: Create a mesh grid in the 2d space
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Flatten the mesh grid and inverse transform back to the original feature space
    grid = np.c_[xx.ravel(), yy.ravel()]

    # add zero padding on the right on the last dimension for grid
    to_pad = x.shape[-1] - 2
    if to_pad > 0:
        grid = np.concatenate([grid, np.zeros((grid.shape[0], to_pad))], axis=1)

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

    x1 = x[:, 0].numpy()
    x2 = x[:, 1].numpy()

    scatter = plt.scatter(
        x1,
        x2,
        c=y_bin,
        cmap="coolwarm",
        edgecolor="k",
        alpha=0.8,
    )
    plt.legend(*scatter.legend_elements(), title="Ground Truth")
    plt.title("Decision Boundary for 2 Features")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(visible=True)
    # I think the plot looks so noisy due to biases not being zeroes in the model
    plt.show()
