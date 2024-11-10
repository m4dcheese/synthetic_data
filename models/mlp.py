import torch
from utils import get_activation


class MLP(torch.nn.Module):
    """MLP base class."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int,
        num_layers: int,
        activation_str: str,
        initialization: str = "default",
        bias: bool = True,
    ):
        super().__init__()
        activation = get_activation(activation_str=activation_str)
        layers = []
        for i in range(num_layers):
            layer_in = in_features if i == 0 else hidden_dim
            layer_out = out_features if i == num_layers - 1 else hidden_dim
            layer = torch.nn.Linear(
                in_features=layer_in,
                out_features=layer_out,
                bias=bias,
            )
            if initialization == "uniform":
                torch.nn.init.uniform_(layer.weight, -1, 1)
                if bias:
                    torch.nn.init.uniform_(layer.bias, -1, 1)
            layers.append(
                layer
                if i == num_layers - 1
                else torch.nn.Sequential(layer, activation()),
            )
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
