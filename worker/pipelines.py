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

from config.config import Config, InputFeature
from utils.losses import ReconLoss, anomaly_score_l1
from utils.viz_features import render_feature_images, compose_feature_grid
from utils.train_utils import build_optim_and_scheduler, update_loss_curve_image

from utils.viz_features import tensor_to_pil_rgb


@torch.no_grad()
def save_input_patch_preview_folder(
    loader,
    out_dir: Path,
    config: Config,
    max_items: int = 16,
):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tile = int(config.image_size)  # 각 타일 크기 (큰 사이즈)
    bg = (0, 0, 0)

    def _feat_ch(f) -> int:
        if f == InputFeature.RGB:
            return 3
        if f == InputFeature.NPR:
            return 3
        if f == InputFeature.RESIDUAL:
            return 1
        if f == InputFeature.WAVELET:
            return 4
        raise ValueError(f"Unknown InputFeature: {f}")

    def _to_uint8_gray(x_hw: torch.Tensor) -> Image.Image:
        # 시각화 목적: min-max normalize
        x = x_hw.detach().float().cpu()
        mn = float(x.min().item())
        mx = float(x.max().item())
        if mx - mn < 1e-12:
            x01 = torch.zeros_like(x)
        else:
            x01 = (x - mn) / (mx - mn)
        u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
        return Image.fromarray(u8, mode="L")

    def _to_uint8_rgb(x_3hw: torch.Tensor) -> Image.Image:
        # 시각화 목적: 전체 min-max normalize (색 균형 유지)
        x = x_3hw.detach().float().cpu()
        mn = float(x.min().item())
        mx = float(x.max().item())
        if mx - mn < 1e-12:
            x01 = torch.zeros_like(x)
        else:
            x01 = (x - mn) / (mx - mn)
        u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)  # (3,H,W)
        u8 = u8.permute(1, 2, 0).numpy()  # (H,W,3)
        return Image.fromarray(u8, mode="RGB")

    def _fit_tile(im: Image.Image) -> Image.Image:
        # 타일 크기 강제 통일
        if im.size != (tile, tile):
            im = im.resize((tile, tile), resample=Image.BICUBIC)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return im

    saved = 0

    for batch in loader:
        if saved >= max_items:
            break

        if "pixel_values" not in batch:
            continue

        x = batch["pixel_values"]
        if not torch.is_tensor(x):
            continue

        # (B,C,H,W) 보정
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"Expected pixel_values (B,C,H,W), got {tuple(x.shape)}")

        B, C, H, W = map(int, x.shape)

        for i in range(B):
            if saved >= max_items:
                break

            xi = x[i]  # (C,H,W)

            # feature 순서대로 타일 리스트 구성
            tiles: list[Image.Image] = []
            off = 0

            for feat in list(config.input_features):
                ch = _feat_ch(feat)
                xj = xi[off : off + ch]  # (ch,H,W)

                if int(xj.shape[0]) != ch:
                    raise RuntimeError(
                        f"Channel mismatch: feat={feat}, need={ch}, got={int(xj.shape[0])}, off={off}, C={C}"
                    )

                if ch == 3:
                    im = _to_uint8_rgb(xj)
                    tiles.append(_fit_tile(im))
                elif ch == 1:
                    im = _to_uint8_gray(xj[0])
                    tiles.append(_fit_tile(im))
                elif ch == 4:
                    # WAVELET: 채널별로 4장 “그대로” 나열
                    for k in range(4):
                        im = _to_uint8_gray(xj[k])
                        tiles.append(_fit_tile(im))
                else:
                    # 규칙상 없음
                    im = _to_uint8_gray(xj[0])
                    tiles.append(_fit_tile(im))

                off += ch

            # 가로 패널 생성
            if len(tiles) == 0:
                continue

            gap = 16  # 타일 간격(가로)
            panel_w = len(tiles) * tile + (len(tiles) - 1) * gap
            panel_h = tile

            panel = Image.new("RGB", (panel_w, panel_h), bg)

            x0 = 0
            for im in tiles:
                panel.paste(im, (x0, 0))
                x0 += tile + gap

            out_path = out_dir / f"idx{saved:03d}.png"
            tmp = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
            panel.save(tmp)
            os.replace(tmp, out_path)

            saved += 1


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

    save_input_patch_preview_folder(
        loader=train_loader,
        out_dir=input_patch_preview_dir,
        config=config,
        max_items=config.input_patch_preview_max_items,
    )

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
