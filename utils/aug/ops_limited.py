# utils/aug/ops_limited.py
from __future__ import annotations

import random
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class LimitedAugConfig:
    # 제한 그룹: color 계열에서 최대 N개만
    enable: bool = True
    max_ops: int = 1
    random_order: bool = True

    # brightness/contrast
    bc_enable: bool = True
    bc_p: float = 0.5
    b_delta: float = 0.10
    c_delta: float = 0.10

    # fancy PCA
    fancypca_enable: bool = True
    fancypca_p: float = 0.5
    fancypca_std: float = 0.10

    # hsv shift
    hsv_enable: bool = True
    hsv_p: float = 1.0
    h_shift: int = 5
    s_shift: int = 7
    v_shift: int = 5


def brightness_contrast(rgb: np.ndarray, b_delta: float, c_delta: float) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    b = random.uniform(-float(b_delta), float(b_delta))
    c = random.uniform(-float(c_delta), float(c_delta))
    x = x * (1.0 + c) + b
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)


def hsv_shift(rgb: np.ndarray, h_lim: int, s_lim: int, v_lim: int) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
    dh = int(random.randint(-int(h_lim), int(h_lim)))
    ds = int(random.randint(-int(s_lim), int(s_lim)))
    dv = int(random.randint(-int(v_lim), int(v_lim)))
    hsv[..., 0] = (hsv[..., 0] + dh) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + ds, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + dv, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def fancy_pca(rgb: np.ndarray, alpha_std: float) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    flat = x.reshape(-1, 3)
    mean = flat.mean(axis=0, keepdims=True)
    flat0 = flat - mean

    cov = (flat0.T @ flat0) / float(max(1, flat0.shape[0] - 1))
    w, v = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]

    alpha = np.random.normal(0.0, float(alpha_std), size=(3,)).astype(np.float32)
    delta = (v * (alpha * w)).sum(axis=1)  # (3,)
    x2 = x + delta[None, None, :]
    x2 = np.clip(x2, 0.0, 1.0)
    return (x2 * 255.0).round().astype(np.uint8)


def apply_limited_color_ops(
    rgb: np.ndarray,
    cfg: LimitedAugConfig,
    applied: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if applied is None:
        applied = []

    if not cfg.enable:
        return rgb, applied

    candidates: list[tuple[str, callable]] = []

    if cfg.bc_enable and random.random() < float(cfg.bc_p):
        candidates.append(
            ("BC", lambda im: brightness_contrast(im, cfg.b_delta, cfg.c_delta))
        )

    if cfg.fancypca_enable and random.random() < float(cfg.fancypca_p):
        candidates.append(("FancyPCA", lambda im: fancy_pca(im, cfg.fancypca_std)))

    if cfg.hsv_enable and random.random() < float(cfg.hsv_p):
        candidates.append(
            ("HSV", lambda im: hsv_shift(im, cfg.h_shift, cfg.s_shift, cfg.v_shift))
        )

    if not candidates:
        return rgb, applied

    k = int(max(0, cfg.max_ops))
    if k <= 0:
        return rgb, applied

    if len(candidates) > k:
        candidates = random.sample(candidates, k)

    if cfg.random_order:
        random.shuffle(candidates)

    out = rgb
    for name, fn in candidates:
        out = fn(out)
        applied.append(name)

    return out, applied
