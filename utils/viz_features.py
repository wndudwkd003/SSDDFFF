# utils/viz_features.py
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import random
import torch
import torch.nn.functional as F
from PIL import Image
from config.config import Config, InputFeature
from utils.data_utils import (
    _npr_from_rgb01,
    _residual_from_rgb01,
    _wavelet_from_rgb01,
)
from pathlib import Path
import random
import torch
import torch.nn.functional as F
from PIL import Image


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


def tensor_to_pil_rgb(t: torch.Tensor) -> Image.Image:
    if t.dim() == 4:
        t = t[0]

    t = t.detach().float().cpu()
    C, H, W = t.shape

    if C == 1:
        t = t.repeat(3, 1, 1)
    elif C == 2:
        t = torch.cat([t, t[:1]], dim=0)
    elif C >= 3:
        t = t[:3]

    mn = float(t.min().item())
    mx = float(t.max().item())
    if mn < 0.0 or mx > 1.0:
        t01 = (t - mn) / (mx - mn) if (mx - mn) > 1e-12 else torch.zeros_like(t)
    else:
        t01 = t.clamp(0.0, 1.0)

    arr = (t01 * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")


@torch.no_grad()
def save_correct_wrong_images_ce(
    model,
    loader,
    config,
    device,
    out_dir: str,
    *,
    max_total: int = 200,
    max_correct: int | None = None,
    max_wrong: int | None = None,
    thr: float = 0.5,
    shuffle: bool = False,
):
    """
    맞춘/틀린 예시 이미지 저장 (개수 제한 지원)

    - max_total: 전체 저장 최대 개수
    - max_correct: correct 폴더 저장 최대 개수 (None이면 제한 없음)
    - max_wrong: wrong 폴더 저장 최대 개수 (None이면 제한 없음)
    - thr: 분류 threshold
    - shuffle: True면 loader 순서가 고정되어 있을 때 다양하게 뽑히도록 저장 순서를 랜덤화(단, loader 자체 shuffle이 아니면 완전 랜덤은 아님)
    """
    out_root = Path(out_dir)
    (out_root / "correct").mkdir(parents=True, exist_ok=True)
    (out_root / "wrong").mkdir(parents=True, exist_ok=True)

    model.eval()

    saved_total = 0
    saved_correct = 0
    saved_wrong = 0

    for batch in loader:
        if saved_total >= max_total:
            break

        x = batch["pixel_values"].to(device)
        y = batch["labels"].to(device)
        paths = batch["path"]

        out = model(x)
        logits = out["logits"] if isinstance(out, dict) and "logits" in out else out

        if logits.dim() == 2 and logits.size(-1) == 2:
            prob = F.softmax(logits, dim=-1)[:, 1]
        else:
            prob = torch.sigmoid(logits.view(-1))

        pred = (prob >= float(thr)).long()

        idxs = list(range(x.size(0)))
        if shuffle:
            random.shuffle(idxs)

        for i in idxs:
            if saved_total >= max_total:
                break

            ok = int(pred[i].item() == y[i].item())
            if ok == 1:
                if max_correct is not None and saved_correct >= max_correct:
                    continue
                group = "correct"
            else:
                if max_wrong is not None and saved_wrong >= max_wrong:
                    continue
                group = "wrong"

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

            saved_total += 1
            if ok == 1:
                saved_correct += 1
            else:
                saved_wrong += 1

            # 두 그룹 모두 상한에 도달했고 total도 의미 없으면 빠르게 종료
            if (max_correct is not None and saved_correct >= max_correct) and (
                max_wrong is not None and saved_wrong >= max_wrong
            ):
                # 두 상한을 모두 채웠으면 더 볼 필요 없음
                return

    return {
        "saved_total": saved_total,
        "saved_correct": saved_correct,
        "saved_wrong": saved_wrong,
        "out_dir": str(out_root),
    }
