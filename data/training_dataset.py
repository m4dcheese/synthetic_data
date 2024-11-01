from __future__ import annotations

import math

import torch
from config import make_dotdict_recursive
from torch import nn
from torch.utils.data import IterableDataset
from utils import get_activation

import time


class TrainingDataset(IterableDataset):
    def __init__(
        self,
        data_hp_sampler,
        training_config,
        data_config,
        mlp_config,
        # multi gpu training
        rank,
        worldsize,
        num_data_workers,
    ):
        self.data_hp_sampler = data_hp_sampler
        self.training_config = training_config
        self.data_config = data_config
        self.mlp_config = mlp_config

        self.rank = rank
        self.worldsize = worldsize
        self.num_data_workers = num_data_workers

        self.random_generator = None

    def worker_init_fn(self, worker_id):
        if self.random_generator is None:  # none is default value
            self.random_generator = torch.Generator()
            initial_seed = self._create_seed(
                gpu_rank=self.rank if self.worldsize > 1 else 0,
                pid=worker_id,
            )
            self.random_generator.manual_seed(initial_seed)

    def __iter__(self):
        num_batches = int(
            math.ceil(
                self.training_config.batches_per_iteration
                / max(1, self.num_data_workers),
            ),
        )

        for _ in range(num_batches):
            yield self.generate_batch_of_data()

    def collate_fn(self, batch):
        # This function is called after the __iter__ function has been called
        # and the data has been generated. This function is called on the main process
        # only use last element from list, as this already contains the batch
        return batch[-1]

    def _create_seed(self, gpu_rank, pid):
        # Combine the gpu_rank and pid using a bitwise OR operation
        # seed: 32 bit integer such that the most significant 16 bits are the gpu_rank and the least significant 16 bits are the pid
        return gpu_rank | (pid << 16)

    def generate_batch_of_data(self):
        # we are sampling data on the cpu, so that if we are pre-generating data
        # the cuda-memory is not unnecessarily filled. We only want to have the data
        # on the gpu, if we process it (after loading from dataloader).

        start_time = time.monotonic_ns()
        # ----- ----- D A T A - H Y P E R P A R A M E T E R S

        data_hps = self.data_hp_sampler.sample()

        # ----- ----- F E A T U R E S
        xs = torch.randn(
            size=(
                self.training_config.batch_size,
                data_hps.samples,
                data_hps.features,
            ),
            device=self.rank,
            generator=self.random_generator,
        )

        # ----- ----- M O D E L
        model = self._generate_mlp(data_hps=data_hps)

        # ----- ----- L A B E L S
        ys_regression = model(xs).squeeze(-1)  # batch_size, sequence_length

        # to make sure, that each class is represented enough times
        quantile_25 = torch.quantile(ys_regression, 0.25, dim=1)
        quantile_75 = torch.quantile(ys_regression, 0.75, dim=1)

        # threshold is uniformly sampled between 25% and 75% quantile
        threshold = quantile_25 + torch.rand(
            size=(self.training_config.batch_size,),
            device=self.rank,
        ) * (quantile_75 - quantile_25)

        # use threshold to create binary labels
        ys_labels = (ys_regression > threshold.unsqueeze(-1)).long()

        final_batch = make_dotdict_recursive(
            {
                "xs": xs,
                "ys_labels": ys_labels,
                "data_hps": data_hps,
                "model": model,
            },
        )

        end_time = time.monotonic_ns()
        time_millis = (end_time - start_time) / 1e6
        print(f"Data Generation Time: {time_millis:.2f} ms")
        return final_batch

    def _generate_mlp(self, data_hps):
        activation = get_activation(self.mlp_config.activation_str)
        layers = []

        # Define layers with specified initialization
        for i in range(self.mlp_config.num_layers):
            in_features = data_hps.features if i == 0 else self.mlp_config.hidden_dim
            out_features = (
                self.mlp_config.output_dim
                if i == self.mlp_config.num_layers - 1
                else self.mlp_config.hidden_dim
            )

            # Initialize layer and apply uniform weight initialization
            layer = nn.Linear(
                in_features,
                out_features,
                bias=self.mlp_config.bias,
            )
            torch.nn.init.uniform_(layer.weight, -1, 1)
            if self.mlp_config.bias:
                torch.nn.init.uniform_(layer.bias, -1, 1)

            # Append layer with optional activation
            layers.append(
                layer
                if i == self.mlp_config.num_layers - 1
                else nn.Sequential(layer, activation()),
            )
        return nn.Sequential(*layers)
