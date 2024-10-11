from config import config
from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential, Softmax, Sigmoid


class TargetNetwork(Module):
    """Target-network architeecture. Weights are trained on synthetic data and
    predicted for real-world tasks by the hyper-network.
    """

    def __init__(self, num_x: int, num_y: int):
        """Initialize network with given input and output dimensions."""
        super().__init__()
        layers = [Linear(in_features=num_x, out_features=config.target.hidden_dim)]

        for _i in range(config.target.num_hidden_layers):
            layers += [ReLU(), Linear(in_features=config.target.hidden_dim,
                                      out_features=config.target.hidden_dim)]
        layers += [ReLU(),
                   Linear(in_features=config.target.hidden_dim, out_features=num_y),
                   Sigmoid()]

        self.model = Sequential(*layers)

    def forward(self, x: Tensor):
        """Pytorch forward implementation."""
        return self.model(x)
