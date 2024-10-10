from torch.nn import Module, Sequential, Linear, ReLU, Softmax

from config import config

class TargetNetwork(Module):

    def __init__(self, num_x: int, num_y: int):
        layers = [Linear(in_features=num_x, out_features=config.target.hidden_dim)]

        for _i in range(config.target.hidden_layers):
            layers += [ReLU(), Linear(in_features=config.target.hidden_dim,
                                      out_features=config.target.hidden_dim)]
        layers += [ReLU(),
                   Linear(in_features=config.target.hidden_dim, out_features=num_y),
                   Softmax(dim=num_y)]

        self.model = Sequential(layers)