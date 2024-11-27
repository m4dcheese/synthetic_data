from __future__ import annotations

import math
import time

import torch
from config import make_dotdict_recursive
from models.target_mlp import TargetMLP
from torch import nn
from torch.utils.data import IterableDataset


class TrainingDataset(IterableDataset):
    """Generate synthetic data of model-data combinations."""

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
        """Initialize Dataset generator class."""
        self.data_hp_sampler = data_hp_sampler
        self.training_config = training_config
        self.data_config = data_config
        self.mlp_config = mlp_config

        self.rank = rank
        self.worldsize = worldsize
        self.num_data_workers = num_data_workers

        self.random_generator = None

    def worker_init_fn(self, worker_id):
        """Initialize worker per GPU."""
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
        """This function is called after the __iter__ function has been called
        and the data has been generated. This function is called on the main process
        only use last element from list, as this already contains the batch.
        """
        return make_dotdict_recursive(batch[-1])

    def _create_seed(self, gpu_rank, pid):
        """Combine the gpu_rank and pid using a bitwise OR operation.
        seed: 32 bit integer such that the most significant 16 bits are the gpu_rank
        and the least significant 16 bits are the pid.
        """
        return gpu_rank | (pid << 16)

    def generate_features(
        self,
        features: int,
        samples: int,
        max_sampled_sequence_length: int,
    ) -> torch.Tensor:
        """Generate input features (x) for target network."""
        xs = torch.randn(
            size=(
                1,  # Was batch size
                samples,
                features,
            ),
            device=self.rank,
            generator=self.random_generator,
        )
        # pad xs to max features and max sampled sequence_length
        if features < self.data_config.features.max:
            xs = nn.functional.pad(
                xs,
                pad=(0, self.data_config.features.max - features),
                mode="constant",
                value=0,
            )
        if samples < max_sampled_sequence_length:
            xs = nn.functional.pad(
                xs,
                pad=(0, 0, 0, max_sampled_sequence_length - samples),
                mode="constant",
                value=0,
            )
        return xs

    def generate_batch_of_data(self):
        """We are sampling data on the cpu, so that if we are pre-generating data
        the cuda-memory is not unnecessarily filled. We only want to have the data
        on the gpu, if we process it (after loading from dataloader).
        """
        start_time = time.monotonic_ns()
        # ----- ----- D A T A - H Y P E R P A R A M E T E R S

        data_hps_list = [
            self.data_hp_sampler.sample()
            for _ in range(self.training_config.batch_size)
        ]
        max_sampled_sequence_length = max(
            [data_hps.samples for data_hps in data_hps_list],
        )

        xs_list = []
        ys_list = []
        ys_raw_list = []
        threshold_list = []
        weights_list = []
        t_list = []

        for _, data_hps in enumerate(data_hps_list):
            # ----- ----- F E A T U R E S
            xs = self.generate_features(
                data_hps.features, data_hps.samples, max_sampled_sequence_length,
            )

            # ----- ----- M O D E L
            model = TargetMLP(
                mlp_config=self.mlp_config,
                data_config=self.data_config,
                in_features=data_hps.features,
            ).to(self.rank)

            # ----- ----- L A B E L S
            ys_regression = model(xs).squeeze(-1)  # batch_size, sequence_length

            # to make sure, that each class is represented enough times
            quantile_25 = torch.quantile(ys_regression, 0.25, dim=1)
            quantile_75 = torch.quantile(ys_regression, 0.75, dim=1)

            # threshold is uniformly sampled between 25% and 75% quantile
            threshold = quantile_25 + torch.rand(
                size=(xs.shape[0],),
                device=self.rank,
            ) * (quantile_75 - quantile_25)

            # if shift option was unknown at model training point, this is None
            if self.data_config.shift_for_threshold:
                # use threshold to shift model output for <|> 0 classification
                model.shift_for_threshold(threshold=threshold)

            # use threshold to create binary labels
            ys_labels = (ys_regression > threshold.unsqueeze(-1)).long()

            total_weights = model.get_compact_form()

            # ----- ----- A P P E N D
            xs_list.append(xs)
            ys_list.append(ys_labels)
            ys_raw_list.append(ys_regression)
            threshold_list.append(threshold)
            weights_list.append(total_weights)
            t_list.append(data_hps.t)

        # ----- ----- C O N V E R T - T O - T E N S O R
        xs_tensor = torch.cat(xs_list, dim=0)
        ys_tensor = torch.cat(ys_list, dim=0)
        ys_raw_tensor = torch.cat(ys_raw_list, dim=0)
        threshold_tensor = torch.cat(threshold_list, dim=0)
        weights_tensor = torch.stack(weights_list, dim=0)
        t_tensor = torch.tensor(t_list, device=self.rank)

        end_time = time.monotonic_ns()
        (end_time - start_time) / 1e6

        return {
            "xs": xs_tensor.detach(),
            "ys": ys_tensor.detach(),
            "ys_raw": ys_raw_tensor.detach(),
            "threshold": threshold_tensor.detach(),
            "weights": weights_tensor.detach(),
            "data_hps": data_hps_list,
            "t": t_tensor.detach(),
        }
