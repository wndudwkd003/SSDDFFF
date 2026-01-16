# utils/patch_utils.py

from __future__ import annotations

import hashlib
from typing import List, Tuple

from PIL import Image
import torch
from torchvision import transforms as T

from config.config import Config, InputPatchType


def _face_box_from_meta(face: dict, w: int, h: int) -> Tuple[int, int, int, int]:
    """
    전제:
    - face는 항상 존재
    - face["bbox_xyxy"]는 항상 존재
    """
    x1, y1, x2, y2 = face["bbox_xyxy"]

    x1 = max(0, int(round(float(x1))))
    y1 = max(0, int(round(float(y1))))
    x2 = min(w, int(round(float(x2))))
    y2 = min(h, int(round(float(y2))))
    return x1, y1, x2, y2


def _resize_center_crop(pil: Image.Image, out_size: int) -> Image.Image:
    tfm = T.Compose(
        [
            T.Resize(out_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(out_size),
        ]
    )
    return tfm(pil.convert("RGB"))


def _crop_bbox_B(
    pilA: Image.Image,
    face: dict,
    *,
    out_size: int,
    pad_frac: float = 0.25,
) -> Image.Image:
    """
    B: bbox 기반 crop(+pad) -> out_size로 resize
    """
    w, h = pilA.size
    x1, y1, x2, y2 = _face_box_from_meta(face, w, h)

    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))

    x1p = int(max(0, round(x1 - bw * pad_frac)))
    y1p = int(max(0, round(y1 - bh * pad_frac)))
    x2p = int(min(w, round(x2 + bw * pad_frac)))
    y2p = int(min(h, round(y2 + bh * pad_frac)))

    crop = pilA.crop((x1p, y1p, x2p, y2p))
    return crop.resize((out_size, out_size), resample=Image.BICUBIC)


def _crop_kp_tile(
    pilA: Image.Image,
    face: dict,
    kp_idx: int,
    *,
    tile_size: int,
    kp_pad_frac: float = 0.18,
) -> Image.Image:
    """
    A에서 keypoint 주변 crop 타일 생성.

    전제:
    - face["kps_5"]는 항상 존재
    - len(face["kps_5"]) == 5
    """
    kps = face["kps_5"]

    w, h = pilA.size
    x1, y1, x2, y2 = _face_box_from_meta(face, w, h)
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    base = max(bw, bh)

    px, py = kps[kp_idx]
    px = float(px)
    py = float(py)

    half = 0.5 * base * float(kp_pad_frac)

    cx1 = int(max(0, round(px - half)))
    cy1 = int(max(0, round(py - half)))
    cx2 = int(min(w, round(px + half)))
    cy2 = int(min(h, round(py + half)))

    # 전제상 거의 없겠지만, 최소한의 안전 처리
    if cx2 <= cx1 + 1 or cy2 <= cy1 + 1:
        return _resize_center_crop(pilA, tile_size)

    crop = pilA.crop((cx1, cy1, cx2, cy2))
    return crop.resize((tile_size, tile_size), resample=Image.BICUBIC)


def _stable_sample_seed(config_seed: int, path: str) -> int:
    """
    샘플별로 랜덤이 달라지되 재현 가능하도록 seed 구성.
    """
    h = hashlib.md5(path.encode("utf-8")).hexdigest()
    v = int(h[:8], 16)  # 32-bit
    return int(config_seed) ^ v


def _make_k_tiles_8(
    pilA: Image.Image,
    face: dict,
    *,
    tile_size: int,
    seed: int,
) -> List[Image.Image]:
    """
    K1~K8 생성:
    - kp 5개는 고정(kp_idx 0..4)
    - 추가 3개는 0..4 중 랜덤(중복 허용)
    """
    base = [_crop_kp_tile(pilA, face, k, tile_size=tile_size) for k in range(5)]

    g = torch.Generator()
    g.manual_seed(int(seed))
    extra_idx = torch.randint(low=0, high=5, size=(3,), generator=g).tolist()
    extra = [base[i].copy() for i in extra_idx]

    out = base + extra
    # base 5 + extra 3 = 8
    return out


def _build_3x3_grid(
    *,
    center: Image.Image,
    outer8: List[Image.Image],
    out_size: int,
) -> Image.Image:
    """
    3x3 타일(콜라주) 한 장 생성.

    배치:
      K1 K2 K3
      K4  C K5
      K6 K7 K8
    """
    tile = out_size // 3
    canvas = Image.new("RGB", (tile * 3, tile * 3))

    cimg = center.resize((tile, tile), resample=Image.BICUBIC)

    outer_pos = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]

    for t, (r, c) in zip(outer8, outer_pos):
        tt = t.resize((tile, tile), resample=Image.BICUBIC)
        canvas.paste(tt, (c * tile, r * tile))

    canvas.paste(cimg, (tile, tile))

    # out_size가 3으로 나누어 떨어지지 않으면 최종 보정
    if canvas.size != (out_size, out_size):
        canvas = canvas.resize((out_size, out_size), resample=Image.Resampling.BICUBIC)

    return canvas


def apply_input_patch(
    imgA: Image.Image,
    face: dict,
    config: Config,
    *,
    path_for_seed: str,
) -> Image.Image:
    """
    최종 입력 이미지 생성 규칙(요구사항 반영):

    1) config.input_patches == []:
       - 반환: A

    2) CENTER_ZOOM만 포함:
       - 반환: 3x3 grid
         center = A
         outer  = K(A) 8개

    3) CENTER_ZOOM + OUTTER_FEATURE_ZOOM 포함:
       - 반환: 3x3 grid
         center = B (bbox crop from A)
         outer  = K(A) 8개  (kp는 항상 A에서 crop)
    """
    patches = list(config.input_patches or [])
    if len(patches) == 0:
        return imgA

    has_center = InputPatchType.CENTER_ZOOM in patches
    has_outer = InputPatchType.OUTTER_FEATURE_ZOOM in patches

    # 요구사항 상 CENTER_ZOOM이 없으면 아무것도 안 함
    if not has_center:
        return imgA

    out_size = int(config.image_size)
    tile_size = out_size // 3

    seed = _stable_sample_seed(config.seed, path_for_seed)
    k_tiles = _make_k_tiles_8(imgA, face, tile_size=tile_size, seed=seed)

    if has_outer:
        center = _crop_bbox_B(imgA, face, out_size=tile_size, pad_frac=0.25)  # B
    else:
        center = _resize_center_crop(imgA, tile_size)  # A

    return _build_3x3_grid(center=center, outer8=k_tiles, out_size=out_size)
