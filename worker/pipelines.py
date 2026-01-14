# /workspace/SSDDFF/worker/pipelines.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from config.config import Config
from utils.losses import ReconLoss, anomaly_score_l1
from utils.viz_features import render_feature_images, compose_feature_grid
from utils.train_utils import build_optim_and_scheduler, update_loss_curve_image

from utils.viz_features import tensor_to_pil_rgb
from torchvision.transforms.functional import to_pil_image


@torch.no_grad()
def save_input_patch_preview_folder(
    loader,
    out_dir: Path,
    config: Config,
    *,
    epoch: int,
    max_items: int = 16,
    overwrite_each_epoch: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = len(config.input_features) + 1  # INPUT_RGB + features
    saved = 0

    for batch in loader:
        # --- Case A) patch preview exists ---
        if "debug_patch_rgb01" in batch and "debug_patch_names" in batch:
            patch_rgb01 = batch["debug_patch_rgb01"]  # (B,P,3,H,W)
            patch_names = batch["debug_patch_names"][0]  # list[str]

            B, P = patch_rgb01.shape[0], patch_rgb01.shape[1]

            for i in range(B):
                if saved >= max_items:
                    return {"saved": saved, "out_dir": str(out_dir)}

                blocks: list[Image.Image] = []
                for p in range(P):
                    pil_patch = to_pil_image(patch_rgb01[i, p])
                    feats = render_feature_images(pil_patch, config)
                    feats = {"INPUT_RGB": pil_patch, **feats}

                    grid = compose_feature_grid(
                        feats,
                        tile_size=int(config.image_size),
                        cols=cols,
                        pad=8,
                        bg=(0, 0, 0),
                    )
                    blocks.append(grid)

                # 세로 결합
                gap = 14
                W = max(im.size[0] for im in blocks)
                H = sum(im.size[1] for im in blocks) + gap * (len(blocks) - 1)
                panel = Image.new("RGB", (W, H), (0, 0, 0))

                y0 = 0
                for im in blocks:
                    panel.paste(im, (0, y0))
                    y0 += im.size[1] + gap

                fn = (
                    f"idx{saved:03d}.png"
                    if overwrite_each_epoch
                    else f"epoch{epoch:04d}_idx{saved:03d}_{np.random.randint(0,10**9):09d}.png"
                )
                out_path = out_dir / fn
                tmp = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
                panel.save(tmp)
                os.replace(tmp, out_path)

                saved += 1

        # --- Case B) no patches: fallback to actual model input ---
        elif "debug_input_rgb01" in batch:
            inp_rgb01 = batch["debug_input_rgb01"]  # (B,3,H,W)
            B = inp_rgb01.shape[0]

            for i in range(B):
                if saved >= max_items:
                    return {"saved": saved, "out_dir": str(out_dir)}

                pil_in = to_pil_image(inp_rgb01[i])
                feats = render_feature_images(pil_in, config)
                feats = {"INPUT_RGB": pil_in, **feats}

                grid = compose_feature_grid(
                    feats,
                    tile_size=int(config.image_size),
                    cols=cols,
                    pad=8,
                    bg=(0, 0, 0),
                )

                fn = (
                    f"idx{saved:03d}.png"
                    if overwrite_each_epoch
                    else f"epoch{epoch:04d}_idx{saved:03d}_{np.random.randint(0,10**9):09d}.png"
                )
                out_path = out_dir / fn
                tmp = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
                grid.save(tmp)
                os.replace(tmp, out_path)

                saved += 1

        else:
            # 디버그 텐서가 없다면 그냥 스킵(데이터셋 쪽에서 debug_input_rgb01은 항상 넣는 게 정상)
            continue

    return {"saved": saved, "out_dir": str(out_dir)}


@torch.no_grad()
def save_recon_preview_folder(
    model: nn.Module,
    loader,
    device: torch.device,
    out_dir: Path,
    config: Config,
    *,
    epoch: int,
    max_items: int = 16,
    overwrite_each_epoch: bool = True,
):
    model.eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = len(config.input_features)

    saved = 0
    for batch in loader:
        if saved >= max_items:
            break

        x = batch["pixel_values"].to(device)
        out = model(x)
        recon = out["recon"]

        for i in range(x.size(0)):
            if saved >= max_items:
                break

            x_pil = tensor_to_pil_rgb(x[i : i + 1])
            r_pil = tensor_to_pil_rgb(recon[i : i + 1])

            feats_x = render_feature_images(x_pil, config)
            feats_r = render_feature_images(r_pil, config)

            merged: dict[str, Image.Image] = {}
            for k, im in feats_x.items():
                merged[f"IN_{k}"] = im
            for k, im in feats_r.items():
                merged[f"RECON_{k}"] = im

            grid = compose_feature_grid(
                merged,
                tile_size=int(config.image_size),
                cols=cols,
                pad=8,
                bg=(0, 0, 0),
            )

            if overwrite_each_epoch:
                fn = f"idx{saved:03d}.png"
            else:
                fn = f"epoch{epoch:04d}_idx{saved:03d}_{np.random.randint(0, 10**9):09d}.png"

            out_path = out_dir / fn
            tmp = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
            grid.save(tmp)
            os.replace(tmp, out_path)

            saved += 1

    return {"saved": saved, "out_dir": str(out_dir)}


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

    train_losses: list[float] = []
    valid_losses: list[float] = []

    loss_curve_path = run_dir / "loss_curve.png"
    input_patch_preview_dir = run_dir / "train" / "input_patches"

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

        train_losses.append(float(last_train["loss"]))
        valid_losses.append(float(last_valid["loss"]))
        update_loss_curve_image(train_losses, valid_losses, loss_curve_path)

        save_input_patch_preview_folder(
            loader=train_loader,
            out_dir=input_patch_preview_dir,
            config=config,
            epoch=epoch + 1,
            max_items=config.input_patch_preview_max_items,
            overwrite_each_epoch=True,
        )

        v = float(last_valid["loss"])
        if v < best_valid_loss:
            best_valid_loss = v
            torch.save(model.state_dict(), best_path)

    return {
        "run_dir": str(run_dir),
        "best_valid_loss": float(best_valid_loss),
        "best_ckpt_path": str(best_path),
        "loss_curve_path": str(loss_curve_path),
        "train": last_train,
        "valid": last_valid,
    }


def get_threshold(last_valid, thr_path: Path):
    scores = np.asarray(last_valid["scores"], dtype=np.float32)
    best_thr = float(np.quantile(scores, 0.99))
    np.save(thr_path, np.asarray(best_thr, dtype=np.float32))
    return best_thr


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

    train_losses: list[float] = []
    valid_losses: list[float] = []

    best_thr: float | None = None
    last_train = None
    last_valid = None

    loss_curve_path = run_dir / "loss_curve.png"
    recon_preview_dir = run_dir / "recon_preview"
    thr_path = run_dir / "threshold.npy"
    input_patch_preview_dir = run_dir / "train" / "input_patches"

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

        save_input_patch_preview_folder(
            loader=train_loader,
            out_dir=input_patch_preview_dir,
            config=config,
            epoch=epoch + 1,
            max_items=config.input_patch_preview_max_items,
            overwrite_each_epoch=True,
        )

        save_recon_preview_folder(
            model=model,
            loader=valid_loader,
            device=device,
            out_dir=recon_preview_dir,
            config=config,
            epoch=epoch + 1,
            max_items=config.recon_preview_max_items,
            overwrite_each_epoch=True,
        )

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
        "recon_preview_dir": str(recon_preview_dir),
        "train": last_train,
        "valid": last_valid,
    }, float(best_thr)
