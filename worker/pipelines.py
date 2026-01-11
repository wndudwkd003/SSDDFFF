# /workspace/SSDDFF/worker/pipelines.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config.config import Config, DatasetName
from utils.data_utils import get_data_loader, get_test_loader_jsonl
from utils.strategy import confident_strategy
from utils.losses import ReconLoss, anomaly_score_l1
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F

from utils.viz_features import render_feature_images


@torch.no_grad()
def save_correct_wrong_images_ce(model, loader, config, device, out_dir: str):
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    model.eval()

    for batch in loader:
        x = batch["pixel_values"].to(device)
        y = batch["labels"].to(device)
        paths = batch["path"]

        out = model(x)
        logits = out["logits"] if isinstance(out, dict) and "logits" in out else out
        if logits.dim() == 2 and logits.size(-1) == 2:
            prob = F.softmax(logits, dim=-1)[:, 1]
        else:
            prob = torch.sigmoid(logits.view(-1))

        pred = (prob >= 0.5).long()

        for i in range(x.size(0)):
            ok = int(pred[i].item() == y[i].item())
            group = "correct" if ok == 1 else "wrong"
            base = Path(paths[i]).stem

            img_pil = Image.open(paths[i]).convert("RGB")
            feats = render_feature_images(img_pil, config)

            for k, im in feats.items():
                d = out_root / k / group
                d.mkdir(parents=True, exist_ok=True)
                im.save(
                    d
                    / f"{base}_y{int(y[i])}_p{int(pred[i])}_prob{float(prob[i]):.5f}.png"
                )


def build_optim_and_scheduler(model: nn.Module, config: Config):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    if config.scheduler == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.num_epochs)
    elif config.scheduler == "linear":
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda e: max(0.0, 1.0 - (e / float(max(1, config.num_epochs)))),
        )
    elif config.scheduler == "step":
        sch = torch.optim.lr_scheduler.StepLR(
            opt, step_size=max(1, config.num_epochs // 3), gamma=0.1
        )
    else:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.num_epochs)

    return opt, sch


def run_epoch_ce(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    split: str,
    opt=None,
    probs_threshold: float = 0.5,
):
    is_train = split == "train"
    model.train() if is_train else model.eval()

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    all_probs: list[float] = []
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in tqdm(loader, desc=f"Epoch {split}"):
        x = batch["pixel_values"].to(device)
        y = batch["labels"].to(device)

        out = model(x)
        logits = out["logits"]

        loss = loss_fn(logits, y)

        if is_train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_loss += float(loss.item()) * x.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= float(probs_threshold)).long()

        all_probs.extend(probs.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())

    avg_loss = total_loss / float(len(loader.dataset))
    return {
        "loss": float(avg_loss),
        "probs": all_probs,
        "preds": all_preds,
        "labels": all_labels,
    }


def run_epoch_ae(
    epoch: int,
    max_epoch: int,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    split: str,
    recon_loss: ReconLoss,
    opt=None,
    threshold: float | None = None,
):
    is_train = split == "train"
    model.train() if is_train else model.eval()

    total_loss = 0.0

    all_scores: list[float] = []
    all_labels: list[int] = []
    all_preds: list[int] = []

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{max_epoch} {split}")
    for batch in pbar:
        x = batch["pixel_values"].to(device)
        y = batch["labels"].to(device)

        out = model(x)
        recon = out["recon"]

        loss = recon_loss(recon, x)["loss"]
        pbar.set_postfix({"loss": float(loss.item())})
        pbar.refresh()

        if is_train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_loss += float(loss.item()) * x.size(0)

        scores = anomaly_score_l1(recon.detach(), x.detach())
        all_scores.extend(scores.detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())

        if threshold is not None:
            preds = (scores >= float(threshold)).long()
            all_preds.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / float(len(loader.dataset))
    out_dict: dict[str, Any] = {
        "loss": float(avg_loss),
        "scores": all_scores,
        "labels": all_labels,
    }
    if threshold is not None:
        out_dict["preds"] = all_preds
        out_dict["threshold"] = float(threshold)

    return out_dict


def train_ce(
    *,
    model: nn.Module,
    train_loader,
    valid_loader,
    config: Config,
    device: torch.device,
    run_dir: Path,
):
    opt, sch = build_optim_and_scheduler(model, config)

    best_valid_loss = float("inf")
    best_path = run_dir / "best_stage1.pth"

    last_train = None
    last_valid = None

    for _epoch in range(config.num_epochs):
        last_train = run_epoch_ce(
            model=model,
            loader=train_loader,
            device=device,
            split="train",
            opt=opt,
            probs_threshold=config.probs_threshold,
        )
        last_valid = run_epoch_ce(
            model=model,
            loader=valid_loader,
            device=device,
            split="valid",
            opt=None,
            probs_threshold=config.probs_threshold,
        )
        sch.step()

        v = float(last_valid["loss"])
        if v < best_valid_loss:
            best_valid_loss = v
            torch.save(model.state_dict(), best_path)

    return {
        "run_dir": str(run_dir),
        "best_valid_loss": float(best_valid_loss),
        "best_ckpt_path": str(best_path),
        "train": last_train,
        "valid": last_valid,
    }


def train_ae(
    *,
    model: nn.Module,
    train_loader,
    valid_loader,
    config: Config,
    device: torch.device,
    run_dir: Path,
    recon_loss: ReconLoss,
):
    opt, sch = build_optim_and_scheduler(model, config)

    best_valid_loss = float("inf")
    best_path = run_dir / "best_ae.pth"
    best_scores = None

    last_train = None
    last_valid = None

    for epoch in range(config.num_epochs):
        last_train = run_epoch_ae(
            epoch=epoch,
            max_epoch=config.num_epochs,
            model=model,
            loader=train_loader,
            device=device,
            split="train",
            recon_loss=recon_loss,
            opt=opt,
            threshold=None,
        )
        last_valid = run_epoch_ae(
            epoch=epoch,
            max_epoch=config.num_epochs,
            model=model,
            loader=valid_loader,
            device=device,
            split="valid",
            recon_loss=recon_loss,
            opt=None,
            threshold=None,
        )
        sch.step()

        v = float(last_valid["loss"])
        if v < best_valid_loss:
            best_valid_loss = v
            best_scores = np.asarray(last_valid["scores"], dtype=np.float32)
            torch.save(model.state_dict(), best_path)

    thr = float(np.quantile(best_scores, 0.99))
    np.save(run_dir / "threshold.npy", np.asarray(thr, dtype=np.float32))

    return {
        "run_dir": str(run_dir),
        "best_valid_loss": float(best_valid_loss),
        "best_ckpt_path": str(best_path),
        "threshold": float(thr),
        "train": last_train,
        "valid": last_valid,
    }, float(thr)


def eval_all_ce(
    *,
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
    *,
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


@torch.no_grad()
def collect_meta_probs_from_model(model: nn.Module, loader, device: torch.device):
    model.eval()

    prob_list = []
    filename_list = []
    media_type_list = []

    for batch in tqdm(loader, desc="Collect meta probs (model)"):
        x = batch["pixel_values"].to(device)
        out = model(x)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=1)[:, 1]
        prob_list.extend(probs.detach().cpu().numpy().tolist())

        filename_list.extend(list(batch["filename"]))
        media_type_list.extend(list(batch["media_type"]))

    return prob_list, filename_list, media_type_list


@torch.no_grad()
def collect_meta_scores_from_ae(model: nn.Module, loader, device: torch.device):
    model.eval()

    score_list = []
    filename_list = []
    media_type_list = []

    for batch in tqdm(loader, desc="Collect meta scores (ae)"):
        x = batch["pixel_values"].to(device)
        out = model(x)
        recon = out["recon"]
        scores = anomaly_score_l1(recon.detach(), x.detach())
        score_list.extend(scores.detach().cpu().numpy().tolist())

        filename_list.extend(list(batch["filename"]))
        media_type_list.extend(list(batch["media_type"]))

    return score_list, filename_list, media_type_list


def aggregate_per_file(
    values, filenames, media_types, *, kind: str, t_video: float = 0.8
):
    per_file: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    key = "probs" if kind == "probs" else "scores"

    for fn, mt, v in zip(filenames, media_types, values):
        if fn not in per_file:
            per_file[fn] = {"media_type": mt, key: []}
            order.append(fn)
        per_file[fn][key].append(float(v))

    out_filenames: list[str] = []
    out_values: list[float] = []

    for fn in order:
        mt = per_file[fn]["media_type"]
        vs = per_file[fn][key]

        if mt == "image":
            final_v = float(vs[0])
        else:
            final_v = float(confident_strategy(vs, t=float(t_video)))

        out_filenames.append(fn)
        out_values.append(final_v)

    return out_filenames, out_values, {k: v[key] for k, v in per_file.items()}
