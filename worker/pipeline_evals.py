import torch
from torch import nn
from typing import Any

from config.config import Config
from utils.losses import ReconLoss, anomaly_score_l1

from worker.pipelines import run_epoch_ce, run_epoch_ae
from utils.data_utils import get_data_loader, get_test_loader_jsonl


def eval_all_ce(
    model: nn.Module,
    config: Config,
    device: torch.device,
):
    all_tests: dict[str, Any] = {}
    order = [ds for ds, _ratio in config.evaluate_datasets]

    if config.use_dataset_sum:
        loader_sum = get_test_loader_jsonl(config, dataset_name=None)
        all_tests["SUM"] = run_epoch_ce(
            model=model,
            loader=loader_sum,
            device=device,
            split="test_SUM",
            opt=None,
            probs_threshold=config.probs_threshold,
        )

    for ds in order:
        loader = get_data_loader(config, "test", dataset_name=ds)
        all_tests[ds.value] = run_epoch_ce(
            model=model,
            loader=loader,
            device=device,
            split=f"test_{ds.value}",
            opt=None,
            probs_threshold=config.probs_threshold,
        )

    return {
        "test_order": (
            (["SUM"] + [d.value for d in order])
            if config.use_dataset_sum
            else [d.value for d in order]
        ),
        "tests_all": all_tests,
    }


def eval_all_ae(
    model: nn.Module,
    config: Config,
    device: torch.device,
    recon_loss: ReconLoss,
    threshold: float,
):
    all_tests: dict[str, Any] = {}
    order = [ds for ds, _ratio in config.evaluate_datasets]

    if config.use_dataset_sum:
        loader_sum = get_test_loader_jsonl(config, dataset_name=None)
        all_tests["SUM"] = run_epoch_ae(
            model=model,
            loader=loader_sum,
            device=device,
            split="test_SUM",
            recon_loss=recon_loss,
            opt=None,
            threshold=threshold,
        )

    for ds in order:
        loader = get_data_loader(config, "test", dataset_name=ds)
        all_tests[ds.value] = run_epoch_ae(
            model=model,
            loader=loader,
            device=device,
            split=f"test_{ds.value}",
            recon_loss=recon_loss,
            opt=None,
            threshold=threshold,
        )

    return {
        "test_order": (
            (["SUM"] + [d.value for d in order])
            if config.use_dataset_sum
            else [d.value for d in order]
        ),
        "tests_all": all_tests,
    }
