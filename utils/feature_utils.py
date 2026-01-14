# utils/feature_utils.py
from __future__ import annotations

from typing import List
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from config.config import Config, InputFeature

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_mean_std(config: Config):
    m = str(config.model_name).lower()
    if "clip" in m:
        return _CLIP_MEAN, _CLIP_STD
    return _IMAGENET_MEAN, _IMAGENET_STD


def _normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    c = int(x.shape[0])
    base_m = torch.tensor(mean, dtype=x.dtype, device=x.device)
    base_s = torch.tensor(std, dtype=x.dtype, device=x.device)

    rep = (c + base_m.numel() - 1) // base_m.numel()
    m = base_m.repeat(rep)[:c][:, None, None]
    s = base_s.repeat(rep)[:c][:, None, None]
    return (x - m) / s


def _rgb01_to_gray01(rgb01: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb01[0], rgb01[1], rgb01[2]
    return (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0)


def _npr_from_rgb01(rgb01: torch.Tensor) -> torch.Tensor:
    x = rgb01.unsqueeze(0)
    h = x.shape[-2]
    w = x.shape[-1]
    if h % 2 == 1:
        x = x[:, :, :-1, :]
    if w % 2 == 1:
        x = x[:, :, :, :-1]
    down = F.interpolate(
        x, scale_factor=0.5, mode="nearest", recompute_scale_factor=True
    )
    up = F.interpolate(
        down, scale_factor=2.0, mode="nearest", recompute_scale_factor=True
    )
    return (x - up).squeeze(0)  # 3xHxW


def _residual_from_gray01(gray01: torch.Tensor) -> torch.Tensor:
    k = torch.tensor(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        dtype=gray01.dtype,
        device=gray01.device,
    )
    k = (k / k.sum()).view(1, 1, 3, 3)
    x = gray01.view(1, 1, gray01.shape[-2], gray01.shape[-1])
    blur = F.conv2d(x, k, padding=1)
    res = x - blur
    return res.squeeze(0)  # 1xHxW


def _residual_from_rgb01(rgb01: torch.Tensor) -> torch.Tensor:
    gray01 = _rgb01_to_gray01(rgb01)
    return _residual_from_gray01(gray01)


def _haar_wavelet_1level_gray(gray01: torch.Tensor) -> torch.Tensor:
    x = gray01
    H, W = x.shape[-2], x.shape[-1]
    if H % 2 == 1:
        x = x[:-1, :]
        H -= 1
    if W % 2 == 1:
        x = x[:, :-1]
        W -= 1

    a = x[0::2, 0::2]
    b = x[0::2, 1::2]
    c = x[1::2, 0::2]
    d = x[1::2, 1::2]

    ll = (a + b + c + d) * 0.5
    lh = (a + b - c - d) * 0.5
    hl = (a - b + c - d) * 0.5
    hh = (a - b - c + d) * 0.5

    def up(z):
        z = z[None, None, :, :]
        z = F.interpolate(
            z, scale_factor=2.0, mode="nearest", recompute_scale_factor=True
        )
        return z.squeeze(0).squeeze(0)

    ll = up(ll).clamp(0.0, 1.0)
    lh = up(lh)
    hl = up(hl)
    hh = up(hh)

    return torch.stack([ll, lh, hl, hh], dim=0)  # 4xHxW


def _wavelet_from_rgb01(rgb01: torch.Tensor) -> torch.Tensor:
    gray01 = _rgb01_to_gray01(rgb01)
    return _haar_wavelet_1level_gray(gray01)


def build_input_tensor(img_pil: Image.Image, config: Config) -> torch.Tensor:
    rgb01 = T.Compose(
        [
            T.Resize(config.image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(config.image_size),
            T.ToTensor(),
        ]
    )(img_pil.convert("RGB"))

    mean, std = _get_mean_std(config)
    feats: List[torch.Tensor] = []

    for f in config.input_features:
        if f == InputFeature.RGB:
            feats.append(_normalize(rgb01, mean, std))

        elif f == InputFeature.NPR:
            npr = _npr_from_rgb01(rgb01)
            feats.append(_normalize(npr, mean, std))

        elif f == InputFeature.RESIDUAL:
            res = _residual_from_rgb01(rgb01)
            feats.append(_normalize(res, mean, std))

        elif f == InputFeature.WAVELET:
            wav = _wavelet_from_rgb01(rgb01)
            feats.append(_normalize(wav, mean, std))

        else:
            raise ValueError(f"Unknown input feature: {f}")

    return torch.cat(feats, dim=0)
