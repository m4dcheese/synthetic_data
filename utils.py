from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn, optim


def get_optimizer(optimizer_str: str):
    optimizer_str = optimizer_str.lower()
    if optimizer_str == "adam":
        return optim.Adam
    if optimizer_str == "sgd":
        return optim.SGD
    if optimizer_str == "adamw":
        return optim.AdamW
    raise ValueError(f"Unknown optimizer: {optimizer_str}")


def get_criterion(criterion_str: str):
    criterion_str = criterion_str.lower()
    if criterion_str == "mse":
        return nn.MSELoss
    if criterion_str == "crossentropy":
        return nn.CrossEntropyLoss
    if criterion_str == "nll":
        return nn.NLLLoss
    if criterion_str == "bce":
        return nn.BCELoss
    if criterion_str == "bcewithlogits":
        return nn.BCEWithLogitsLoss
    raise ValueError(f"Unknown criterion: {criterion_str}")


def get_activation(activation_str: str):
    activation_str = activation_str.lower()
    if activation_str == "relu":
        return nn.ReLU
    if activation_str == "leakyrelu":
        return nn.LeakyReLU
    if activation_str == "sigmoid":
        return nn.Sigmoid
    if activation_str == "tanh":
        return nn.Tanh
    if activation_str == "softmax":
        return nn.Softmax

    raise ValueError(f"Unknown activation: {activation_str}")


def get_number_parameters(module: nn.Module):
    total_params = sum(p.numel() for p in module.parameters())

    # Format the number of parameters with appropriate units
    if total_params >= 1e9:
        params_str = f"{total_params / 1e9:,.2f} Billion"
    elif total_params >= 1e6:
        params_str = f"{total_params / 1e6:,.2f} Million"
    elif total_params >= 1e3:
        params_str = f"{total_params / 1e3:,.2f} Thousand"
    else:
        params_str = f"{total_params:,}"

    # Return the formatted output string
    return params_str


def set_global_seed(seed):
    """Set the seed for all necessary libraries to ensure reproducibility."""
    # Set the seed for numpy
    np.random.seed(seed)
    random.seed(seed)
    # Pandas uses numpy for its randomness, so setting the numpy seed is usually enough.
    # If using random functions from pandas that do not utilize numpy, set the seed directly if possible.
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
