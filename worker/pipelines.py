# /workspace/SSDDFF/worker/pipelines.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config.config import Config, DatasetName
from utils.losses import ReconLoss, anomaly_score_l1
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F

from utils.viz_features import render_feature_images
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image


def _atomic_savefig(fig, out_path: Path, dpi: int = 160):
    out_path = Path(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    fig.savefig(tmp, dpi=dpi)
    plt.close(fig)
    tmp.replace(out_path)


def update_loss_curve_image(
    train_losses: list[float], valid_losses: list[float], out_path: Path
):
    epochs = list(range(1, len(train_losses) + 1))

    fig = plt.figure()
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, valid_losses, label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()

    _atomic_savefig(fig, Path(out_path))


@torch.no_grad()
def update_recon_preview_image(
    model: nn.Module, loader, device: torch.device, out_path: Path
):
    model.eval()

    batch = next(iter(loader))
    x = batch["pixel_values"].to(device)

    out = model(x)
    recon = out["recon"]

    # 첫 샘플만: (원본 | 복원) 2장 그리드로 저장
    x0 = x[:1].detach().cpu()
    r0 = recon[:1].detach().cpu()

    grid = make_grid(torch.cat([x0, r0], dim=0), nrow=2, normalize=True)
    tmp = Path(out_path).with_suffix(Path(out_path).suffix + ".tmp")
    save_image(grid, tmp)
    tmp.replace(out_path)


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
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    split: str,
    epoch: int = 0,
    max_epoch: int = 0,
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

    epoch_str = f"Epoch {epoch}/{max_epoch} {split}" if split != "test" else "Test"

    pbar = tqdm(loader, desc=epoch_str)
    for batch in pbar:
        x = batch["pixel_values"].to(device)
        y = batch["labels"].to(device)

        out = model(x)
        logits = out["logits"]

        loss = loss_fn(logits, y)
        pbar.set_postfix({"loss": float(loss.item())})
        pbar.refresh()

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
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    split: str,
    recon_loss: ReconLoss,
    epoch: int = 0,
    max_epoch: int = 0,
    opt=None,
    threshold: float | None = None,
):
    is_train = split == "train"
    model.train() if is_train else model.eval()

    total_loss = 0.0

    all_scores: list[float] = []
    all_labels: list[int] = []
    all_preds: list[int] = []

    epoch_str = f"Epoch {epoch}/{max_epoch} {split}" if split != "test" else "Test"

    pbar = tqdm(loader, desc=epoch_str)
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

    for epoch in range(config.num_epochs):
        last_train = run_epoch_ce(
            epoch=epoch + 1,
            max_epoch=config.num_epochs,
            model=model,
            loader=train_loader,
            device=device,
            split="train",
            opt=opt,
            probs_threshold=config.probs_threshold,
        )
        last_valid = run_epoch_ce(
            epoch=epoch + 1,
            max_epoch=config.num_epochs,
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

    # 곡선 저장용 리스트
    train_losses: list[float] = []
    valid_losses: list[float] = []

    best_thr: float | None = None

    last_train = None
    last_valid = None

    loss_curve_path = run_dir / "loss_curve.png"
    recon_preview_path = run_dir / "recon_preview.png"
    thr_path = run_dir / "threshold.npy"

    for epoch in range(config.num_epochs):
        last_train = run_epoch_ae(
            epoch=epoch + 1,
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
            epoch=epoch + 1,
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

        train_losses.append(float(last_train["loss"]))
        valid_losses.append(float(last_valid["loss"]))
        update_loss_curve_image(train_losses, valid_losses, loss_curve_path)
        update_recon_preview_image(model, valid_loader, device, recon_preview_path)

        v = float(last_valid["loss"])
        if v < best_valid_loss:
            best_valid_loss = v
            torch.save(model.state_dict(), best_path)
            best_thr = get_threshold(last_valid, thr_path)

    if best_thr is None:
        best_thr = get_threshold(last_valid, thr_path)

    return {
        "run_dir": str(run_dir),
        "best_valid_loss": float(best_valid_loss),
        "best_ckpt_path": str(best_path),
        "threshold": float(best_thr),
        "loss_curve_path": str(loss_curve_path),
        "recon_preview_path": str(recon_preview_path),
        "train": last_train,
        "valid": last_valid,
    }, float(best_thr)


def get_threshold(last_valid, thr_path):
    scores = np.asarray(last_valid["scores"], dtype=np.float32)
    best_thr = float(np.quantile(scores, 0.99))
    np.save(thr_path, np.asarray(best_thr, dtype=np.float32))
    return best_thr
