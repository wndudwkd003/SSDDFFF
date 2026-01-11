# utils/viz_features.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from config.config import Config, InputFeature
from utils.data_utils import (
    _get_mean_std,
    _rgb01_to_gray01,
    _npr_from_rgb01,
    _residual_from_rgb01,
    _wavelet_from_rgb01,
)


def _to_uint8_01(x01: torch.Tensor) -> np.ndarray:
    x01 = x01.clamp(0.0, 1.0)
    x = (x01 * 255.0).round().to(torch.uint8)
    return x.permute(1, 2, 0).cpu().numpy()


def _to_uint8_signed(x: torch.Tensor) -> np.ndarray:
    mn = float(x.min().item())
    mx = float(x.max().item())
    if mx - mn < 1e-12:
        x01 = torch.zeros_like(x)
    else:
        x01 = (x - mn) / (mx - mn)
    if x01.dim() == 2:
        x01 = x01.unsqueeze(0)
    if x01.size(0) == 1:
        x01 = x01.repeat(3, 1, 1)
    return _to_uint8_01(x01)


def render_feature_images(
    img_pil: Image.Image, config: Config
) -> dict[str, Image.Image]:
    rgb01 = T.Compose(
        [
            T.Resize(config.image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(config.image_size),
            T.ToTensor(),
        ]
    )(img_pil.convert("RGB"))

    out: dict[str, Image.Image] = {}

    for f in config.input_features:
        if f == InputFeature.RGB:
            out["RGB"] = Image.fromarray(_to_uint8_01(rgb01), mode="RGB")

        elif f == InputFeature.NPR:
            npr = _npr_from_rgb01(rgb01)
            out["NPR"] = Image.fromarray(_to_uint8_signed(npr), mode="RGB")

        elif f == InputFeature.RESIDUAL:
            res = _residual_from_rgb01(rgb01)  # 1xHxW signed
            out["RESIDUAL"] = Image.fromarray(_to_uint8_signed(res), mode="RGB")

        elif f == InputFeature.WAVELET:
            wav = _wavelet_from_rgb01(rgb01)  # 4xHxW [LL, LH, HL, HH]
            ll = wav[0:1]
            lh = wav[1]
            hl = wav[2]
            hh = wav[3]
            ll_img = Image.fromarray(_to_uint8_01(ll.repeat(3, 1, 1)), mode="RGB")
            lh_img = Image.fromarray(_to_uint8_signed(lh), mode="RGB")
            hl_img = Image.fromarray(_to_uint8_signed(hl), mode="RGB")
            hh_img = Image.fromarray(_to_uint8_signed(hh), mode="RGB")

            W, H = ll_img.size
            canvas = Image.new("RGB", (W * 2, H * 2))
            canvas.paste(ll_img, (0, 0))
            canvas.paste(lh_img, (W, 0))
            canvas.paste(hl_img, (0, H))
            canvas.paste(hh_img, (W, H))
            out["WAVELET"] = canvas

    return out
