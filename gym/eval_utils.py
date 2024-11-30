from __future__ import annotations

import numpy as np
import torch
from models.target_mlp import TargetMLP
from torchdiffeq import odeint


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
        """Torch forward."""
        t_tensor = torch.full(
            size=(weights.shape[0], 1, 1),
            fill_value=t,
            dtype=weights.dtype,
            device=weights.device,
        )
        return -self.cfm_model(xs=self.xs, ys=self.ys, ts=t_tensor, weights=weights)


def evaluate_target_network(
    mlp_config,
    data_config,
    x: torch.Tensor,
    threshold: int = 0,
    compact_form: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load target network and evaluate."""
    model = TargetMLP(mlp_config=mlp_config, data_config=data_config)
    if compact_form is not None:
        model.load_compact_form(compact_form=compact_form)
    model.eval()
    with torch.no_grad():
        pred = model(x).detach().cpu().squeeze()
        pred_bin = binary_decision(pred.clone(), threshold=threshold)

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
        t_span = torch.tensor(np.linspace(0.0, 1.0, steps + 1), device=x.device)
        return odeint(
            ode_cfm,
            initial_weights,
            t_span,
            atol=1e-5,
            rtol=1e-5,
        )


def run_cfm(
    cfm_model: torch.nn.Module,
    compact_form_shape: tuple,
    x: torch.Tensor,
    y: torch.Tensor,
    solver: str,
    solve_steps: int,
) -> torch.Tensor:
    """Run CFM to predict MLP weights."""
    weights_t = reparam_normal(shape=[x.shape[0], *compact_form_shape], mean=0, std=1)

    solve_fn = solve_ode_naive if solver == "naive" else solve_ode_odeint

    return solve_fn(
        cfm_model=cfm_model,
        initial_weights=weights_t,
        steps=solve_steps,
        x=x,
        y=y,
    )