# utils/viz_features.py
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from config.config import Config, InputFeature
from utils.data_utils import (
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
            res = _residual_from_rgb01(rgb01)
            out["RESIDUAL"] = Image.fromarray(_to_uint8_signed(res), mode="RGB")

        elif f == InputFeature.WAVELET:
            wav = _wavelet_from_rgb01(rgb01)
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


def compose_feature_grid(
    feats: dict[str, Image.Image],
    tile_size: int,
    cols: int = 2,
    pad: int = 8,
    bg: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    items = list(feats.items())
    n = len(items)
    rows = int(math.ceil(n / cols))

    W = cols * tile_size + (cols + 1) * pad
    H = rows * tile_size + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), color=bg)

    for i, (_k, im) in enumerate(items):
        r = i // cols
        c = i % cols
        tile = im.convert("RGB").resize((tile_size, tile_size), resample=Image.BICUBIC)
        x0 = pad + c * (tile_size + pad)
        y0 = pad + r * (tile_size + pad)
        canvas.paste(tile, (x0, y0))

    return canvas


@torch.no_grad()
def save_correct_wrong_images_ce(model, loader, config, device, out_dir: str):
    out_root = Path(out_dir)
    (out_root / "correct").mkdir(parents=True, exist_ok=True)
    (out_root / "wrong").mkdir(parents=True, exist_ok=True)

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
            grid = compose_feature_grid(
                feats, tile_size=int(config.image_size), cols=2, pad=8
            )

            grid.save(
                out_root
                / group
                / f"{base}_y{int(y[i])}_p{int(pred[i])}_prob{float(prob[i]):.5f}.png"
            )
