from __future__ import annotations

import torch
from data.data_hyperparameter_sampler import DataHyperparameterSampler
from data.training_dataset import TrainingDataset
from gym.eval_utils import evaluate_target_network, run_cfm
from gym.plotting import (
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_flow_trajectory,
    plot_prediction_scatter,
    plot_roc_auc_scores,
)
from gym.real_world_data import load_dataset
from model_io import load_trained_model
from models.target_mlp import TargetMLP
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from utils import ddp_cleanup, ddp_setup


def evaluate_on_synthetic(path: str, eval_config):
    """Run the main training loop for conditional flow matching."""
    rank = "cpu" if eval_config.world_size == 0 else "cuda:0"
    # First Setup the Distributed Data Parallel
    ddp_setup(rank=rank, world_size=eval_config.world_size)
    model_config, cfm_model, optimizer = load_trained_model(path=path, rank=rank)

    if eval_config.world_size > 1:
        # find unused parameters=True Bug: https://stackoverflow.com/questions/68000761/pytorch-ddp-finding-the-cause-of-expected-to-mark-a-variable-ready-only-once
        cfm_model = DistributedDataParallel(
            cfm_model,
            device_ids=[rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    cfm_model.eval()
    with torch.no_grad():
        # Dataset and DataLoader
        dataset = TrainingDataset(
            data_hp_sampler=DataHyperparameterSampler(model_config.data),
            training_config=eval_config,
            data_config=model_config.data,
            mlp_config=model_config.target_mlp,
            rank=rank,
            worldsize=eval_config.world_size,
            num_data_workers=eval_config.num_data_workers,
        )

        dataloader = DataLoader(
            dataset=dataset,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            worker_init_fn=dataset.worker_init_fn,
            num_workers=eval_config.num_data_workers,
            prefetch_factor=eval_config.data_prefetch_factor
            if eval_config.num_data_workers > 0
            else None,
            persistent_workers=eval_config.num_data_workers > 0,
        )

        for _, batch in enumerate(dataloader):
            xs = batch.xs.to(rank)
            ys = batch.ys.to(rank)

            # Save target weights only for comparison, batch.ts is not used
            weights_0 = batch.weights.to(rank)

            trajectory = run_cfm(
                cfm_model=cfm_model,
                compact_form_shape=weights_0.shape[-2:],
                x=xs,
                y=ys,
                solver=eval_config.solver,
                solve_steps=eval_config.solve_steps,
            )

            weights_t = trajectory[-1]

            # Let's try the generated MLPs
            loss_fn = torch.nn.MSELoss()

            roc_auc_scores = []
            for mlp_i, compact_form in enumerate(weights_t):
                threshold = (
                    0
                    if model_config.data.shift_for_threshold
                    else batch.threshold[mlp_i]
                )
                xs_inference = dataset.generate_features(
                    features=batch.data_hps[mlp_i].features,
                    samples=batch.data_hps[mlp_i].samples,
                    max_sampled_sequence_length=max(
                        [data_hps.samples for data_hps in batch.data_hps],
                    ),
                )

                # Use ground truth model to generate new y labels
                gt_weights = weights_0[mlp_i]
                gt, gt_bin = evaluate_target_network(
                    mlp_config=model_config.target_mlp,
                    data_config=model_config.data,
                    x=xs_inference,
                    threshold=threshold,
                    compact_form=gt_weights,
                )
                # Use predicted model to generate predicted y labels
                pred, pred_bin = evaluate_target_network(
                    mlp_config=model_config.target_mlp,
                    data_config=model_config.data,
                    x=xs_inference,
                    threshold=threshold,
                    compact_form=compact_form,
                )

                # measure ROC AUC score for each model and store it
                roc_auc = roc_auc_score(
                    gt_bin.numpy(),
                    pred_bin.numpy(),
                )
                roc_auc_scores.append(roc_auc)

                loss = loss_fn(pred_bin, gt_bin)
                print(f"model {mlp_i} after fit: {loss.cpu().detach().numpy()}")

                # Linear regression - compare closed form
                logreg = LogisticRegression(penalty=None).fit(
                    X=xs[mlp_i], y=ys[mlp_i],
                )
                pred_logreg = logreg.predict(X=xs_inference)
                roc_auc_logreg = roc_auc_score(
                    gt_bin.numpy(),
                    pred_logreg,
                )
                plot_confusion_matrix(
                    pred_bin=pred_logreg,
                    gt_bin=gt_bin.numpy(),
                    model="Logistic Regression",
                    roc_auc=roc_auc_logreg,
                    instant_show=False,
                )
                plot_confusion_matrix(
                    pred_bin=pred_bin,
                    gt_bin=gt_bin,
                    model="CFM-Predicted MLP",
                    roc_auc=roc_auc,
                )
                plot_prediction_scatter(
                    pred=pred,
                    pred_bin=pred_bin,
                    gt=gt,
                    threshold=threshold,
                )
                plot_flow_trajectory(
                    batch.weights[mlp_i],
                    trajectory=trajectory[:, mlp_i],
                )

                plot_decision_boundary(
                    mlp_config=model_config.target_mlp,
                    data_config=model_config.data,
                    compact_form=compact_form,
                    gt_weights=gt_weights,
                    x=xs_inference,
                    threshold=threshold,
                )

            plot_roc_auc_scores(roc_auc_scores)
    ddp_cleanup(world_size=eval_config.world_size)

    return cfm_model


def evaluate_on_real(path: str, eval_config: dict, datasets: list[str]):
    """Run the main eval loop for conditional flow matching on real-world data."""
    rank = "cpu" if eval_config.world_size == 0 else "cuda:0"
    # First Setup the Distributed Data Parallel
    ddp_setup(rank=rank, world_size=eval_config.world_size)
    model_config, cfm_model, optimizer = load_trained_model(path=path, rank=rank)

    if eval_config.world_size > 1:
        # find unused parameters=True Bug: https://stackoverflow.com/questions/68000761/pytorch-ddp-finding-the-cause-of-expected-to-mark-a-variable-ready-only-once
        cfm_model = DistributedDataParallel(
            cfm_model,
            device_ids=[rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    cfm_model.eval()
    with torch.no_grad():
        # Dataset and DataLoader
        example_mlp = TargetMLP(
            mlp_config=model_config.target_mlp, data_config=model_config.data,
        )
        compact_form_shape = example_mlp.compact_shape()

        for dataset_str in datasets:
            train_x, train_y, test_x, test_y = load_dataset(
                dataset_str=dataset_str,
                train_size=model_config.data.samples.max,
                test_size=model_config.data.samples.max,
                features_max=model_config.data.features.max,
            )

            xs = train_x.to(rank)
            ys = train_y.to(rank)

            trajectory = run_cfm(
                cfm_model=cfm_model,
                compact_form_shape=compact_form_shape,
                x=xs.unsqueeze(0),
                y=ys.unsqueeze(0),
                solver=eval_config.solver,
                solve_steps=eval_config.solve_steps,
            )

            weights_t = trajectory[-1][0]

            # Let's try the generated MLPs
            loss_fn = torch.nn.MSELoss()

            xs_inference = test_x.to(rank)
            gt_bin = test_y.to(rank)

            # Use predicted model to generate predicted y labels
            pred, pred_bin = evaluate_target_network(
                mlp_config=model_config.target_mlp,
                data_config=model_config.data,
                x=xs_inference,
                threshold=0,
                compact_form=weights_t,
            )

            roc_auc = roc_auc_score(
                gt_bin.numpy(),
                pred_bin.numpy(),
            )

            loss = loss_fn(pred_bin, gt_bin)
            print(f"model after fit: {loss.cpu().detach().numpy()}")

            # Linear regression - compare closed form
            logreg = LogisticRegression(penalty=None).fit(X=xs, y=ys)
            pred_logreg = logreg.predict(X=xs_inference)
            roc_auc_logreg = roc_auc_score(
                gt_bin.numpy(),
                pred_logreg,
            )
            plot_confusion_matrix(
                pred_bin=pred_logreg,
                gt_bin=gt_bin.numpy(),
                model="Logistic Regression",
                roc_auc=roc_auc_logreg,
                instant_show=False,
            )
            plot_confusion_matrix(
                pred_bin=pred_bin,
                gt_bin=gt_bin,
                model="CFM-Predicted MLP",
                roc_auc=roc_auc,
            )
            plot_prediction_scatter(
                pred=pred,
                pred_bin=pred_bin,
                gt=gt_bin,
                threshold=0.5,
            )
    ddp_cleanup(world_size=eval_config.world_size)

    return cfm_model
