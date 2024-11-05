from __future__ import annotations

import wandb


def init_wandb(project_name, config, experiment_name=None):
    """Initializes a Weights & Biases run with the given project name, config, and optional experiment name.

    Args:
        project_name (str): The name of the W&B project.
        config (dict): Configuration dictionary for the W&B run.
        experiment_name (str, optional): Name of the specific experiment or run.
    """
    wandb.init(project=project_name, config=config, name=experiment_name)


def log_metrics(metrics):
    """Logs metrics to Weights & Biases.

    Args:
        metrics (dict): A dictionary of metrics to log (e.g., {"loss": loss_value}).
    """
    wandb.log(metrics)


def finish_wandb():
    """Finishes the W&B run, ensuring all data is saved."""
    wandb.finish()
