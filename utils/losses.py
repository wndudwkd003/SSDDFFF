# /workspace/SSDDFF/utils/losses.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconL1(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(recon, target, reduction=self.reduction)


class ReconMSE(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, target, reduction=self.reduction)


def _ssim_per_channel(
    x: torch.Tensor, y: torch.Tensor, c1: float, c2: float
) -> torch.Tensor:
    mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim = num / (den + 1e-12)
    return ssim


class SSIMLoss(nn.Module):
    def __init__(self, data_range: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.data_range = float(data_range)
        self.reduction = reduction

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = recon
        y = target

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        ssim = _ssim_per_channel(x, y, c1=c1, c2=c2)
        ssim = ssim.mean(dim=1, keepdim=False)

        loss = 1.0 - ssim

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ReconLoss(nn.Module):
    def __init__(
        self,
        w_l1: float = 1.0,
        w_mse: float = 0.0,
        w_ssim: float = 0.0,
        ssim_data_range: float = 1.0,
    ):
        super().__init__()
        self.w_l1 = float(w_l1)
        self.w_mse = float(w_mse)
        self.w_ssim = float(w_ssim)

        self.l1 = ReconL1(reduction="mean")
        self.mse = ReconMSE(reduction="mean")
        self.ssim = SSIMLoss(data_range=ssim_data_range, reduction="mean")

    def forward(
        self, recon: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        out = {}

        total = recon.new_tensor(0.0)

        if self.w_l1 != 0.0:
            l1 = self.l1(recon, target)
            out["l1"] = l1
            total = total + self.w_l1 * l1

        if self.w_mse != 0.0:
            mse = self.mse(recon, target)
            out["mse"] = mse
            total = total + self.w_mse * mse

        if self.w_ssim != 0.0:
            ssim = self.ssim(recon, target)
            out["ssim"] = ssim
            total = total + self.w_ssim * ssim

        out["loss"] = total
        return out


def anomaly_score_l1(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (recon - target).abs().flatten(1).mean(dim=1)


def anomaly_score_mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (recon - target).pow(2).flatten(1).mean(dim=1)


def anomaly_score_mix(
    recon: torch.Tensor, target: torch.Tensor, alpha: float = 0.5
) -> torch.Tensor:
    a = float(alpha)
    l1 = anomaly_score_l1(recon, target)
    mse = anomaly_score_mse(recon, target)
    return a * l1 + (1.0 - a) * mse
