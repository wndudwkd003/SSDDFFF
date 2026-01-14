# utils/aug/ops_independent.py
from __future__ import annotations

import random
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class IndependentAugConfig:
    # ISO resize/pad
    iso_enable: bool = True
    iso_p: float = 1.0
    iso_max_side: int = 380

    # rotate
    rot_enable: bool = True
    rot_p: float = 0.5
    rot_deg: float = 10.0

    # jpeg
    jpeg_enable: bool = True
    jpeg_p: float = 0.5
    jpeg_qmin: int = 80
    jpeg_qmax: int = 95

    # keypoint occlusion
    occ_enable: bool = True
    occ_p: float = 0.5
    occ_area_frac_min: float = 0.01
    occ_area_frac_max: float = 0.04
    occ_ar_min: float = 0.4
    occ_ar_max: float = 2.0
    occ_num_rects: int = 1
    occ_fill_rgb: tuple[int, int, int] = (0, 0, 0)


def mat2x3_to_hom(M2x3: np.ndarray) -> np.ndarray:
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = M2x3.astype(np.float32)
    return H


def apply_hom_to_points(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts = np.concatenate(
        [pts_xy.astype(np.float32), np.ones((pts_xy.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    out = (H @ pts.T).T
    return out[:, :2]


def iso_resize_and_pad(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray]:
    """returns (rgb_out, H) where H maps input coords -> output coords"""
    h, w = rgb.shape[:2]
    m = max(h, w)
    size = int(max_side)

    if m != size:
        s = float(size) / float(m)
        nh = int(round(h * s))
        nw = int(round(w * s))
        interp = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA
        rgb_rs = cv2.resize(rgb, (nw, nh), interpolation=interp)
    else:
        s = 1.0
        nh, nw = h, w
        rgb_rs = rgb

    top = (size - nh) // 2
    left = (size - nw) // 2
    bottom = size - nh - top
    right = size - nw - left

    rgb_out = cv2.copyMakeBorder(
        rgb_rs, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    H = np.array(
        [[s, 0.0, float(left)], [0.0, s, float(top)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return rgb_out, H


def rotate(rgb: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray]:
    """returns (rgb_out, H_rot) where H_rot maps input coords -> output coords"""
    h, w = rgb.shape[:2]
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    M = cv2.getRotationMatrix2D((cx, cy), float(deg), 1.0)
    out = cv2.warpAffine(
        rgb,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return out, mat2x3_to_hom(M)


def jpeg_compress(rgb: np.ndarray, qmin: int, qmax: int) -> np.ndarray:
    q = int(random.randint(int(qmin), int(qmax)))
    ok, enc = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), q],
    )
    if not ok:
        return rgb
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def occlude_rect_from_two_kps(
    rgb: np.ndarray,
    kp1_xy: tuple[float, float],
    kp2_xy: tuple[float, float],
    area_frac_min: float,
    area_frac_max: float,
    ar_min: float,
    ar_max: float,
    fill_rgb=(0, 0, 0),
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    H, W = rgb.shape[:2]
    x1f, y1f = float(kp1_xy[0]), float(kp1_xy[1])
    x2f, y2f = float(kp2_xy[0]), float(kp2_xy[1])

    xmin, xmax = min(x1f, x2f), max(x1f, x2f)
    ymin, ymax = min(y1f, y2f), max(y1f, y2f)

    bw = max(2.0, xmax - xmin)
    bh = max(2.0, ymax - ymin)

    area_frac_min = max(0.0001, min(float(area_frac_min), 1.0))
    area_frac_max = max(area_frac_min, min(float(area_frac_max), 1.0))
    ar_min = max(0.05, float(ar_min))
    ar_max = max(ar_min, float(ar_max))

    target_area = random.uniform(area_frac_min, area_frac_max) * float(H * W)
    ar = random.uniform(ar_min, ar_max)

    target_w = np.sqrt(target_area * ar)
    target_h = np.sqrt(target_area / ar)

    target_w = max(target_w, bw)
    target_h = max(target_h, bh)

    target_w = max(2.0, min(float(W), target_w))
    target_h = max(2.0, min(float(H), target_h))

    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5

    x1 = int(round(cx - target_w * 0.5))
    x2 = int(round(cx + target_w * 0.5))
    y1 = int(round(cy - target_h * 0.5))
    y2 = int(round(cy + target_h * 0.5))

    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))

    if x2 <= x1 + 1:
        x2 = min(W, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(H, y1 + 2)

    out = rgb.copy()
    out[y1:y2, x1:x2, :] = np.array(fill_rgb, dtype=np.uint8)[None, None, :]
    return out, (x1, y1, x2, y2)
