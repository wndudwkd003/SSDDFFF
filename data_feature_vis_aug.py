# /workspace/SSDDFF/data_feature_vis_aug.py

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


# =========================
# paths
# =========================
ROOT = Path("/workspace/SSDDFF/datasets/KoDF")
TRAIN_JSONL = ROOT / "train.jsonl"
OUT_DIR = ROOT / "vis_features_aug"

# =========================
# knobs
# =========================
NUM_PAIRS = 20
SEED = 42
SIZE = 224

# =========================
# augmentation toggles
# =========================
AUG_ENABLE = True

AUG_ROTATE_ENABLE = True
AUG_ROTATE_P = 0.5
AUG_ROTATE_DEG = 10.0

AUG_ISO_RESIZE_ENABLE = True
AUG_ISO_RESIZE_P = 1.0
AUG_ISO_MAX_SIDE = 380

AUG_BC_ENABLE = True
AUG_BC_P = 0.5
AUG_B_DELTA = 0.10
AUG_C_DELTA = 0.10

AUG_FANCYPCA_ENABLE = True
AUG_FANCYPCA_P = 0.5
AUG_FANCYPCA_STD = 0.10

AUG_HSV_ENABLE = True
AUG_HSV_P = 1.0
AUG_H_SHIFT = 5
AUG_S_SHIFT = 7
AUG_V_SHIFT = 5

AUG_JPEG_ENABLE = True
AUG_JPEG_P = 0.5
AUG_JPEG_QMIN = 80
AUG_JPEG_QMAX = 95

# =========================
# NEW: keypoint-based face occlusion
# =========================
AUG_OCC_ENABLE = True
AUG_OCC_P = 0.5

# occlusion rectangle area bounds (as fraction of image area)
AUG_OCC_AREA_FRAC_MIN = 0.015
AUG_OCC_AREA_FRAC_MAX = 0.06

# aspect ratio range (w/h)
AUG_OCC_AR_MIN = 0.4
AUG_OCC_AR_MAX = 2.0

# how many rectangles to draw per image when applied
AUG_OCC_NUM_RECTS = 1

# fill color (RGB)
AUG_OCC_FILL_RGB = (0, 0, 0)


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _to_uint8_rgb01(x01):
    x01 = x01.clamp(0.0, 1.0)
    y = (x01 * 255.0).round().to(torch.uint8)
    y = y.permute(1, 2, 0).contiguous().cpu().numpy()
    return y


def _to_uint8_from_gray01(g01):
    g01 = g01.clamp(0.0, 1.0)
    y = (g01 * 255.0).round().to(torch.uint8).cpu().numpy()
    return y


def _npr_from_rgb01(rgb01):
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
    return (x - up).squeeze(0)


def _vis_signed_3ch(x3):
    a = x3.abs()
    m = torch.quantile(a.flatten(), 0.99).clamp(min=1e-6)
    v = (x3 / (2.0 * m)) + 0.5
    return _to_uint8_rgb01(v)


def _vis_signed_1ch(x1):
    a = x1.abs()
    m = torch.quantile(a.flatten(), 0.99).clamp(min=1e-6)
    v = (x1 / (2.0 * m)) + 0.5
    return _to_uint8_from_gray01(v)


def _put_text_rgb(img_rgb, text, x=6, y=18, scale=0.55, thickness=1):
    img = img_rgb.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + 6), (0, 0, 0), -1)
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return img


def _hstack_with_gap(images, gap=10):
    h = images[0].shape[0]
    gap_block = np.zeros((h, gap, 3), dtype=np.uint8)
    out = images[0]
    for im in images[1:]:
        out = np.concatenate([out, gap_block, im], axis=1)
    return out


def _vstack_with_gap_rgb(images, gap=14):
    w = images[0].shape[1]
    gap_block = np.zeros((gap, w, 3), dtype=np.uint8)
    out = images[0]
    for im in images[1:]:
        out = np.concatenate([out, gap_block, im], axis=0)
    return out


def save_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )


def _rgb01_to_gray01(rgb01):
    r, g, b = rgb01[0], rgb01[1], rgb01[2]
    return (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0)


def _residual_gray(gray01):
    k = torch.tensor(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=gray01.dtype, device=gray01.device
    )
    k = (k / k.sum()).view(1, 1, 3, 3)
    x = gray01.view(1, 1, gray01.shape[-2], gray01.shape[-1])
    blur = F.conv2d(x, k, padding=1)
    res = x - blur
    return res.squeeze(0).squeeze(0)


def _haar_wavelet_1level(gray01):
    x = gray01
    H, W = x.shape[-2], x.shape[-1]
    if H % 2 == 1:
        x = x[:-1, :]
    if W % 2 == 1:
        x = x[:, :-1]

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

    return up(ll), up(lh), up(hl), up(hh)


# =========================
# helpers: affine / keypoints
# =========================
def _mat2x3_to_hom(M2x3: np.ndarray) -> np.ndarray:
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = M2x3.astype(np.float32)
    return H


def _apply_hom_to_points(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    # pts_xy: (N,2)
    pts = np.concatenate(
        [pts_xy.astype(np.float32), np.ones((pts_xy.shape[0], 1), dtype=np.float32)],
        axis=1,
    )  # (N,3)
    out = (H @ pts.T).T  # (N,3)
    return out[:, :2]


def _get_kps5(face) -> np.ndarray | None:
    if face is None:
        return None
    kps = face.get("kps_5", None)
    if kps is None:
        return None
    kps = np.asarray(kps, dtype=np.float32)
    if kps.shape != (5, 2):
        return None
    return kps


# =========================
# augmentation ops (uint8 RGB) + return affine for coords
# =========================
def _iso_resize_and_pad(rgb, max_side):
    # returns (rgb_out, H_affine), where H maps original coords -> new coords
    h, w = rgb.shape[:2]
    m = max(h, w)
    size = int(max_side)

    # scale
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

    # pad to square size x size (center)
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    rgb_out = cv2.copyMakeBorder(
        rgb_rs, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # affine: [x';y'] = s*[x;y] + [left;top]
    H = np.array(
        [[s, 0.0, float(left)], [0.0, s, float(top)], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    return rgb_out, H


def _rotate(rgb, deg):
    # returns (rgb_out, H_rot) where H_rot maps input coords -> rotated coords
    h, w = rgb.shape[:2]
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    M = cv2.getRotationMatrix2D((cx, cy), float(deg), 1.0)  # 2x3
    out = cv2.warpAffine(
        rgb,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    H = _mat2x3_to_hom(M)
    return out, H


def _brightness_contrast(rgb, b_delta, c_delta):
    x = rgb.astype(np.float32) / 255.0
    b = random.uniform(-float(b_delta), float(b_delta))
    c = random.uniform(-float(c_delta), float(c_delta))
    x = x * (1.0 + c) + b
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)


def _hsv_shift(rgb, h_lim, s_lim, v_lim):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
    dh = int(random.randint(-int(h_lim), int(h_lim)))
    ds = int(random.randint(-int(s_lim), int(s_lim)))
    dv = int(random.randint(-int(v_lim), int(v_lim)))
    hsv[..., 0] = (hsv[..., 0] + dh) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + ds, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + dv, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _fancy_pca(rgb, alpha_std):
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


def _jpeg_compress(rgb, qmin, qmax):
    q = int(random.randint(int(qmin), int(qmax)))
    ok, enc = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), q],
    )
    if not ok:
        return rgb
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    out = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return out


def _occlude_rect_from_two_kps(
    rgb: np.ndarray,
    kp1_xy: tuple[float, float],
    kp2_xy: tuple[float, float],
    area_frac_min: float,
    area_frac_max: float,
    ar_min: float,
    ar_max: float,
    fill_rgb=(0, 0, 0),
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    두 키포인트를 포함하는 축정렬 직사각형을 만들고,
    면적 제약(area_frac_min~max)과 종횡비 제약(ar_min~max)을 만족하도록 확장한 뒤 fill.
    """
    H, W = rgb.shape[:2]
    x1f, y1f = float(kp1_xy[0]), float(kp1_xy[1])
    x2f, y2f = float(kp2_xy[0]), float(kp2_xy[1])

    # 초기 bbox (두 점 포함)
    xmin = min(x1f, x2f)
    xmax = max(x1f, x2f)
    ymin = min(y1f, y2f)
    ymax = max(y1f, y2f)

    # 너무 얇아지는 것을 방지하기 위한 최소 폭/높이(픽셀)
    bw = max(2.0, xmax - xmin)
    bh = max(2.0, ymax - ymin)

    # 이미지 전체 면적 기준 목표 area(랜덤)
    area_frac_min = float(area_frac_min)
    area_frac_max = float(area_frac_max)
    ar_min = float(ar_min)
    ar_max = float(ar_max)

    area_frac_min = max(0.0001, min(area_frac_min, 1.0))
    area_frac_max = max(area_frac_min, min(area_frac_max, 1.0))
    ar_min = max(0.05, ar_min)
    ar_max = max(ar_min, ar_max)

    target_area = random.uniform(area_frac_min, area_frac_max) * float(H * W)

    # 종횡비 목표(ar)를 뽑고, 그에 맞는 목표 폭/높이 계산
    # (target_w * target_h = target_area, target_w/target_h = ar)
    ar = random.uniform(ar_min, ar_max)
    target_w = np.sqrt(target_area * ar)
    target_h = np.sqrt(target_area / ar)

    # 두 점을 포함해야 하므로 목표 크기는 최소 (bw,bh) 이상
    target_w = max(target_w, bw)
    target_h = max(target_h, bh)

    # 그래도 이미지 밖으로 나갈 수 있으니 최대는 이미지 크기
    target_w = max(2.0, min(float(W), target_w))
    target_h = max(2.0, min(float(H), target_h))

    # 중심은 두 점 bbox의 중심을 기본으로 하되, 약간 랜덤 이동(선택)
    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5

    # bbox를 target_w/target_h로 확장
    x1 = int(round(cx - target_w * 0.5))
    x2 = int(round(cx + target_w * 0.5))
    y1 = int(round(cy - target_h * 0.5))
    y2 = int(round(cy + target_h * 0.5))

    # clip
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))

    # 유효성 보정
    if x2 <= x1 + 1:
        x2 = min(W, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(H, y1 + 2)

    # 최종적으로 "두 점이 포함"되는지 보정 (클리핑 때문에 빠질 수 있음)
    # 포함 안되면 가능한 범위에서 조금 확장/이동 시도
    def _ensure_in(x, y, x1, y1, x2, y2):
        if x < x1:
            shift = int(round(x1 - x))
            x1 = max(0, x1 - shift)
            x2 = min(W, x2 - shift)
        elif x >= x2:
            shift = int(round(x - (x2 - 1)))
            x1 = max(0, x1 + shift)
            x2 = min(W, x2 + shift)
        if y < y1:
            shift = int(round(y1 - y))
            y1 = max(0, y1 - shift)
            y2 = min(H, y2 - shift)
        elif y >= y2:
            shift = int(round(y - (y2 - 1)))
            y1 = max(0, y1 + shift)
            y2 = min(H, y2 + shift)
        return x1, y1, x2, y2

    x1, y1, x2, y2 = _ensure_in(x1f, y1f, x1, y1, x2, y2)
    x1, y1, x2, y2 = _ensure_in(x2f, y2f, x1, y1, x2, y2)

    out = rgb.copy()
    out[y1:y2, x1:x2, :] = np.array(fill_rgb, dtype=np.uint8)[None, None, :]
    return out, (x1, y1, x2, y2)


def augment_rgb_uint8(rgb: np.ndarray, face: dict | None):
    """
    rgb: uint8 RGB
    face: row.get("face") (may contain kps_5)
    returns: (rgb_aug, applied_list)
    """
    applied: list[str] = []

    # 누적 좌표 변환 (원본 좌표 -> 현재 rgb 좌표)
    H_acc = np.eye(3, dtype=np.float32)

    # 1) ISO resize/pad (geometric)
    if AUG_ISO_RESIZE_ENABLE and random.random() < float(AUG_ISO_RESIZE_P):
        rgb, H_iso = _iso_resize_and_pad(rgb, AUG_ISO_MAX_SIDE)
        H_acc = H_iso @ H_acc
        applied.append(f"ISO({AUG_ISO_MAX_SIDE})")

    # 2) Rotate (geometric)
    if AUG_ROTATE_ENABLE and random.random() < float(AUG_ROTATE_P):
        deg = random.uniform(-float(AUG_ROTATE_DEG), float(AUG_ROTATE_DEG))
        rgb, H_rot = _rotate(rgb, deg)
        H_acc = H_rot @ H_acc
        applied.append(f"Rot({deg:+.1f})")

    # 3) Keypoint-based occlusion (geometric, needs transformed kps)
    if AUG_OCC_ENABLE and random.random() < float(AUG_OCC_P):
        kps = _get_kps5(face)
        if kps is not None:
            kps_t = _apply_hom_to_points(H_acc, kps)  # (5,2) in current rgb coords
            Hh, Ww = rgb.shape[:2]

            for _ in range(int(max(1, AUG_OCC_NUM_RECTS))):
                # 서로 다른 두 키포인트 선택
                idx1, idx2 = random.sample(range(5), 2)
                kx1, ky1 = float(kps_t[idx1, 0]), float(kps_t[idx1, 1])
                kx2, ky2 = float(kps_t[idx2, 0]), float(kps_t[idx2, 1])

                # 변환 후 이미지 밖이면 스킵(둘 중 하나라도 밖이면 스킵)
                if not (0.0 <= kx1 < float(Ww) and 0.0 <= ky1 < float(Hh)):
                    continue
                if not (0.0 <= kx2 < float(Ww) and 0.0 <= ky2 < float(Hh)):
                    continue

                rgb, (x1, y1, x2, y2) = _occlude_rect_from_two_kps(
                    rgb,
                    (kx1, ky1),
                    (kx2, ky2),
                    AUG_OCC_AREA_FRAC_MIN,
                    AUG_OCC_AREA_FRAC_MAX,
                    AUG_OCC_AR_MIN,
                    AUG_OCC_AR_MAX,
                    fill_rgb=AUG_OCC_FILL_RGB,
                )
                applied.append(f"OCC(kp{idx1+1}&kp{idx2+1},{x2-x1}x{y2-y1})")

    # 4) Brightness/Contrast (photometric)
    if AUG_BC_ENABLE and random.random() < float(AUG_BC_P):
        rgb = _brightness_contrast(rgb, AUG_B_DELTA, AUG_C_DELTA)
        applied.append("BC")

    # 5) FancyPCA (photometric)
    if AUG_FANCYPCA_ENABLE and random.random() < float(AUG_FANCYPCA_P):
        rgb = _fancy_pca(rgb, AUG_FANCYPCA_STD)
        applied.append("FancyPCA")

    # 6) HSV shift (photometric)
    if AUG_HSV_ENABLE and random.random() < float(AUG_HSV_P):
        rgb = _hsv_shift(rgb, AUG_H_SHIFT, AUG_S_SHIFT, AUG_V_SHIFT)
        applied.append("HSV")

    # 7) JPEG compression (photometric / codec)
    if AUG_JPEG_ENABLE and random.random() < float(AUG_JPEG_P):
        rgb = _jpeg_compress(rgb, AUG_JPEG_QMIN, AUG_JPEG_QMAX)
        applied.append("JPEG")

    return rgb, applied


# =========================
# feature visualization rows
# =========================
def _make_feature_tiles_from_pil(pil, tfm, prefix_text):
    rgb01 = tfm(pil.convert("RGB"))

    rgb = _to_uint8_rgb01(rgb01)
    r = _to_uint8_rgb01(rgb01[[0, 0, 0], :, :])
    g = _to_uint8_rgb01(rgb01[[1, 1, 1], :, :])
    b = _to_uint8_rgb01(rgb01[[2, 2, 2], :, :])

    npr = _npr_from_rgb01(rgb01)
    npr_vis = _vis_signed_3ch(npr)

    gray01 = _rgb01_to_gray01(rgb01)
    res = _residual_gray(gray01)
    res_u8 = _vis_signed_1ch(res)
    res_u8_3 = np.repeat(res_u8[:, :, None], 3, axis=2)

    ll, lh, hl, hh = _haar_wavelet_1level(gray01)
    ll_u8 = _to_uint8_from_gray01(ll.clamp(0.0, 1.0))
    lh_u8 = _vis_signed_1ch(lh)
    hl_u8 = _vis_signed_1ch(hl)
    hh_u8 = _vis_signed_1ch(hh)

    ll_u8 = np.repeat(ll_u8[:, :, None], 3, axis=2)
    lh_u8 = np.repeat(lh_u8[:, :, None], 3, axis=2)
    hl_u8 = np.repeat(hl_u8[:, :, None], 3, axis=2)
    hh_u8 = np.repeat(hh_u8[:, :, None], 3, axis=2)

    tiles = [
        _put_text_rgb(rgb, f"{prefix_text} RGB"),
        _put_text_rgb(r, f"{prefix_text} R"),
        _put_text_rgb(g, f"{prefix_text} G"),
        _put_text_rgb(b, f"{prefix_text} B"),
        _put_text_rgb(npr_vis, f"{prefix_text} NPR"),
        _put_text_rgb(res_u8_3, f"{prefix_text} Residual"),
        _put_text_rgb(ll_u8, f"{prefix_text} Wav LL"),
        _put_text_rgb(lh_u8, f"{prefix_text} Wav LH"),
        _put_text_rgb(hl_u8, f"{prefix_text} Wav HL"),
        _put_text_rgb(hh_u8, f"{prefix_text} Wav HH"),
    ]
    return tiles


def _make_view_block(pil_orig, pil_aug, tfm, title_left, aug_desc):
    tiles_o = _make_feature_tiles_from_pil(pil_orig, tfm, "ORIG")
    tiles_a = _make_feature_tiles_from_pil(pil_aug, tfm, "AUG")

    row_o = _hstack_with_gap(tiles_o, gap=10)
    row_a = _hstack_with_gap(tiles_a, gap=10)

    row_o = _put_text_rgb(
        row_o, f"{title_left}  |  ORIG", x=8, y=26, scale=0.9, thickness=2
    )
    row_a = _put_text_rgb(
        row_a, f"{title_left}  |  AUG: {aug_desc}", x=8, y=26, scale=0.9, thickness=2
    )

    blk = _vstack_with_gap_rgb([row_o, row_a], gap=14)
    return blk


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    rows = read_jsonl(TRAIN_JSONL)
    real_rows = [r for r in rows if int(r["label"]) == 0]
    fake_rows = [r for r in rows if int(r["label"]) == 1]

    rr = real_rows[:]
    fr = fake_rows[:]
    random.shuffle(rr)
    random.shuffle(fr)
    n = min(NUM_PAIRS, len(rr), len(fr))

    tfm = T.Compose(
        [
            T.Resize(SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(SIZE),
            T.ToTensor(),
        ]
    )

    for i in range(n):
        rrow = rr[i]
        frow = fr[i]

        pil_r0 = Image.open(str(Path(rrow["path"]))).convert("RGB")
        pil_f0 = Image.open(str(Path(frow["path"]))).convert("RGB")

        rgb_r = np.asarray(pil_r0)
        rgb_f = np.asarray(pil_f0)

        face_r = rrow.get("face", None)
        face_f = frow.get("face", None)

        if AUG_ENABLE:
            rgb_r_aug, augs_r = augment_rgb_uint8(rgb_r.copy(), face_r)
            rgb_f_aug, augs_f = augment_rgb_uint8(rgb_f.copy(), face_f)
        else:
            rgb_r_aug, augs_r = rgb_r.copy(), []
            rgb_f_aug, augs_f = rgb_f.copy(), []

        pil_r_aug = Image.fromarray(rgb_r_aug, mode="RGB")
        pil_f_aug = Image.fromarray(rgb_f_aug, mode="RGB")

        blk_r = _make_view_block(
            pil_r0,
            pil_r_aug,
            tfm,
            "REAL (label=0)",
            " + ".join(augs_r) if augs_r else "None",
        )
        blk_f = _make_view_block(
            pil_f0,
            pil_f_aug,
            tfm,
            "FAKE (label=1)",
            " + ".join(augs_f) if augs_f else "None",
        )

        panel = _vstack_with_gap_rgb([blk_r, blk_f], gap=18)

        out_name = f"{i:05d}_real_vs_fake_orig_vs_aug_features.jpg"
        save_rgb(OUT_DIR / out_name, panel)


if __name__ == "__main__":
    main()
