"""File for SCM class."""

import numpy as np
import torch
from config import config
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential


class SCM:
    """Structured Causal Model, able to generate random but meaningful data."""

    def __init__(self, num_x: int, num_y: int, activation: torch.nn.Module = ReLU):
        """Initialize SCM as an MLP with dropped weights, randomly selecting nodes to
        observe.

        Args:
            num_x: Number of 'independent' variables to observe (features)
            num_y: Number of 'dependent' variables to observe (labels)
            allow_xy_overlap: Whether the same node can appear in x and y, default True
            activation: The activation function to use, default torch.nn.ReLU
        """
        first_linear = Linear(in_features=config.scm.num_causes,
                              out_features=config.scm.layer_dim)

        layers = [Sequential(first_linear, activation())]

        for _i in range(config.scm.num_layers):
            linear = Linear(in_features=config.scm.layer_dim,
                            out_features=config.scm.layer_dim)

            # Drop some connections to obtain a sparse but random SCM
            dropout_mask = torch.bernoulli(torch.full_like(linear.weight,
                                                           1 - config.scm.p_dropout))
            linear.weight.data = linear.weight.data * dropout_mask
            layers.append(Sequential(linear, activation()))


        # Select observed x variables that stay fixed for all samples
        generator = np.random.Generator(np.random.PCG64())
        x_var_locations = []
        while len(x_var_locations) < num_x:
            layer = generator.integers(0, config.scm.num_layers + 1)
            max_d = config.scm.num_causes if layer == 0 else config.scm.layer_dim
            d = generator.integers(0, max_d)

            location = (layer, d)
            if location not in x_var_locations:
                x_var_locations.append(location)

        # Select observed y variables that stay fixed for all samples
        y_var_locations = []
        while len(y_var_locations) < num_y:
            layer = generator.integers(0, config.scm.num_layers + 1)
            max_d = config.scm.num_causes if layer == 0 else config.scm.layer_dim
            d = generator.integers(0, max_d)

            location = (layer, d)
            # Check for overlap with x variables
            xy_cond = config.scm.allow_xy_overlap or location not in x_var_locations
            if location not in y_var_locations and xy_cond:
                y_var_locations.append(location)

        self.x_var_locations = np.array(x_var_locations).T
        self.y_var_locations = np.array(y_var_locations).T
        self.layers = layers
        self.model = Sequential(*layers)


    def generate_data(self, size: int) -> tuple[Tensor, Tensor]:
        """Generate data in a batch of given size, using the selected observables."""
        x = torch.rand((size, config.scm.num_causes))
        data = torch.empty((size, config.scm.num_layers + 1, config.scm.layer_dim))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            data[:, i, :x.shape[-1]] = x

        x_data = data[:, self.x_var_locations[0], self.x_var_locations[1]]
        y_data = data[:, self.y_var_locations[0], self.y_var_locations[1]]

        return x_data, y_data
