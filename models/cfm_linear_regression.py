from config import config
from models.cfm import CFM
from models.cfm_data_projection import DataProjection
from models.target_mlp import TargetMLP
import torch


class FullyConnectedWeightProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        shape = TargetMLP.compact_shape()
        self.model = torch.nn.Linear(
            in_features=shape[0] * shape[1],
            out_features=config.cfm.weight_projection.hidden_dim,
        )


class CFMLinReg(CFM):
    def __init__(self):
        super().__init__(
            data_projection=DataProjection(),
            weight_projection=FullyConnectedWeightProjection(),
            encoder=TODO,
            decoder=TODO,
            prediction_head=TODO,
        )
    
    def forward(self, xs, ys, ts, weights):
        weights = weights.flatten(start_dim=-2)
        return super().forward(xs, ys, ts, weights)
