from __future__ import annotations

from torch import nn


class CFM(nn.Module):
    def __init__(
        self,
        data_projection: nn.Module,
        weight_projection: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.data_projection = data_projection
        self.weight_projection = weight_projection
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, xs, ys, ts, weights):
        data = self.data_projection(xs=xs, ys=ys, ts=ts)
        encoded = self.encoder(data)

        weights = self.weight_projection(weights=weights)
        decoded = self.decoder(tgt=weights, memory=encoded)
        return decoded
