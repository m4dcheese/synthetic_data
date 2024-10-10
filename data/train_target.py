# trainign loop for MLP models

from torch import Tensor

from models.target_network import TargetNetwork

class TargetNetworkTrainer:

    def __init__(self, model: Target, data: Tensor)