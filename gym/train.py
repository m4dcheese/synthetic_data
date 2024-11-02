from __future__ import annotations

from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader


def train(
    cfm: nn.Module,
    dataloader: DataLoader,
    loss_fn,
    optimizer,
    training_config,
    device,
):
    for iteration_i in tqdm(range(training_config.total_iterations)):
        for _batch_i, batch in enumerate(dataloader):
            xs = batch.xs.to(device)
            ys = batch.ys.to(device)
            ts = batch.t.to(device)
            weights = batch.weights.to(device)
            data_hps = batch.data_hps

            optimizer.zero_grad()

            output = cfm(xs=xs, ys=ys, ts=ts, weights=weights)

            loss = loss_fn(output, batch["y"])
            loss.backward()
            optimizer.step()

            if iteration_i % training_config.log_interval == 0:
                pass
