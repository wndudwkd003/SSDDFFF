# utils/aug/augmenter.py
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .ops_independent import (
    IndependentAugConfig,
    iso_resize_and_pad,
    rotate,
    jpeg_compress,
    apply_hom_to_points,
    occlude_rect_from_two_kps,
)
from .ops_limited import LimitedAugConfig, apply_limited_color_ops


@dataclass
class AugmentConfig:
    independent: IndependentAugConfig = IndependentAugConfig()
    limited_color: LimitedAugConfig = LimitedAugConfig()


def _get_kps5(face: dict | None) -> np.ndarray | None:
    if face is None:
        return None
    kps = face.get("kps_5", None)
    if kps is None:
        return None
    kps = np.asarray(kps, dtype=np.float32)
    if kps.shape != (5, 2):
        return None
    return kps


class ImageAugmenter:
    def __init__(self, cfg: AugmentConfig, *, seed: int | None = None):
        self.cfg = cfg
        self.base_seed = seed

    def __call__(self, img_pil: Image.Image, face: dict | None = None) -> Image.Image:
        rgb = np.asarray(img_pil.convert("RGB"))
        rgb_aug, _applied = self.augment_rgb_uint8(rgb, face)
        return Image.fromarray(rgb_aug, mode="RGB")

    def augment_rgb_uint8(self, rgb: np.ndarray, face: dict | None):
        ind = self.cfg.independent
        applied: list[str] = []

        # (옵션) per-call seed 고정이 필요하면 여기서 처리 가능
        # 현재는 DataLoader worker마다 random seed가 다르도록 두는 편이 일반적입니다.

        # 누적 좌표 변환 (원본 kps -> 현재 rgb)
        H_acc = np.eye(3, dtype=np.float32)

        # 1) ISO resize/pad
        if ind.iso_enable and random.random() < float(ind.iso_p):
            rgb, H_iso = iso_resize_and_pad(rgb, ind.iso_max_side)
            H_acc = H_iso @ H_acc
            applied.append(f"ISO({ind.iso_max_side})")

        # 2) Rotate
        if ind.rot_enable and random.random() < float(ind.rot_p):
            deg = random.uniform(-float(ind.rot_deg), float(ind.rot_deg))
            rgb, H_rot = rotate(rgb, deg)
            H_acc = H_rot @ H_acc
            applied.append(f"Rot({deg:+.1f})")

        # 3) Keypoint occlusion (두 키포인트)
        if ind.occ_enable and random.random() < float(ind.occ_p):
            kps = _get_kps5(face)
            if kps is not None:
                kps_t = apply_hom_to_points(H_acc, kps)  # (5,2)
                Hh, Ww = rgb.shape[:2]

                for _ in range(int(max(1, ind.occ_num_rects))):
                    idx1, idx2 = random.sample(range(5), 2)
                    kx1, ky1 = float(kps_t[idx1, 0]), float(kps_t[idx1, 1])
                    kx2, ky2 = float(kps_t[idx2, 0]), float(kps_t[idx2, 1])

                    if not (0.0 <= kx1 < float(Ww) and 0.0 <= ky1 < float(Hh)):
                        continue
                    if not (0.0 <= kx2 < float(Ww) and 0.0 <= ky2 < float(Hh)):
                        continue

                    rgb, (x1, y1, x2, y2) = occlude_rect_from_two_kps(
                        rgb,
                        (kx1, ky1),
                        (kx2, ky2),
                        ind.occ_area_frac_min,
                        ind.occ_area_frac_max,
                        ind.occ_ar_min,
                        ind.occ_ar_max,
                        fill_rgb=ind.occ_fill_rgb,
                    )
                    applied.append(f"OCC(kp{idx1+1}&kp{idx2+1},{x2-x1}x{y2-y1})")

        # 4) 제한 그룹(color): 후보들 중 최대 N개만
        rgb, applied = apply_limited_color_ops(rgb, self.cfg.limited_color, applied)

        # 5) JPEG
        if ind.jpeg_enable and random.random() < float(ind.jpeg_p):
            rgb = jpeg_compress(rgb, ind.jpeg_qmin, ind.jpeg_qmax)
            applied.append("JPEG")

        return rgb, applied
