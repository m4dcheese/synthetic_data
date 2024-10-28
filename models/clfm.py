from __future__ import annotations

from typing import *

import torch
from torch import nn
from zuko.utils import odeint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# https://colab.research.google.com/github/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb


# @title ⏳ Summary: please run this cell which contains the ```CondVF``` class.
class CondVF(nn.Module):
    def __init__(self, net: nn.Module, n_steps: int = 100) -> None:
        super().__init__()
        self.net = net

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(t, x)

    def wrapper(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = t * torch.ones(len(x), device=x.device)
        return self(t, x)

    def decode_t0_t1(self, x_0, t0, t1):
        # from x_0 to x_1
        return odeint(self.wrapper, x_0, t0, t1, self.parameters())

    def encode(self, x_1: torch.Tensor) -> torch.Tensor:
        # from x_0 to x_1
        return odeint(self.wrapper, x_1, 1.0, 0.0, self.parameters())

    def decode(self, x_0: torch.Tensor) -> torch.Tensor:
        # from x_0 to x_1
        return odeint(self.wrapper, x_0, 0.0, 1.0, self.parameters())


class Net(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        h_dims: List[int],
        n_frequencies: int,
    ) -> None:
        super().__init__()

        ins = [in_dim + 2 * n_frequencies, *h_dims]
        outs = [*h_dims, out_dim]
        self.n_frequencies = n_frequencies

        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(in_d, out_d), nn.LeakyReLU())
                for in_d, out_d in zip(ins, outs, strict=False)
            ],
        )
        self.top = nn.Sequential(nn.Linear(out_dim, out_dim))
