# /workspace/SSDDFF/data_feature_vis.py

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import matplotlib.pyplot as plt


# =========================
# paths
# =========================
ROOT = Path("/workspace/SSDDFF/datasets/KoDF")
TRAIN_JSONL = ROOT / "train.jsonl"
OUT_DIR = ROOT / "vis_features"
STAT_DIR = ROOT / "vis_features_stats"

# =========================
# knobs
# =========================
NUM_PAIRS = 20
SEED = 42
SIZE = 224

# parts
USE_FACE_ZOOM = True
FACE_ZOOM_PAD = 0.25
FACE_ZOOM_SCALE = 2.2

USE_KP_PARTS = True
KP_PART_SIZE = 224
KP_PAD_FRAC = 0.18  # around each keypoint, relative to face bbox size

# distribution plots
DO_HIST = False
MAX_DIST_SAMPLES = 2000
HIST_BINS = 60


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
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        dtype=gray01.dtype,
        device=gray01.device,
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


def _face_box_from_meta(face, w, h):
    if face is None:
        return (0, 0, w, h)

    if "bbox_xyxy" in face:
        x1, y1, x2, y2 = face["bbox_xyxy"]
    else:
        kps = face["kps_5"]
        xs = [p[0] for p in kps]
        ys = [p[1] for p in kps]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

    x1 = max(0, int(round(float(x1))))
    y1 = max(0, int(round(float(y1))))
    x2 = min(w, int(round(float(x2))))
    y2 = min(h, int(round(float(y2))))
    return x1, y1, x2, y2


def _crop_zoom_pil(pil0, face):
    if face is None:
        return pil0

    w, h = pil0.size
    x1, y1, x2, y2 = _face_box_from_meta(face, w, h)

    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    pad = float(FACE_ZOOM_PAD)

    x1 = int(max(0, round(x1 - bw * pad)))
    y1 = int(max(0, round(y1 - bh * pad)))
    x2 = int(min(w, round(x2 + bw * pad)))
    y2 = int(min(h, round(y2 + bh * pad)))

    crop = pil0.crop((x1, y1, x2, y2))

    zoom_side = int(round(SIZE * float(FACE_ZOOM_SCALE)))
    if zoom_side < SIZE:
        zoom_side = SIZE
    crop = crop.resize((zoom_side, zoom_side), resample=Image.BICUBIC)

    cx = zoom_side // 2
    cy = zoom_side // 2
    half = SIZE // 2
    crop = crop.crop((cx - half, cy - half, cx - half + SIZE, cy - half + SIZE))
    return crop


def _crop_kp_part_pil(pil0, face, kp_idx):
    if face is None:
        return pil0.resize((KP_PART_SIZE, KP_PART_SIZE), resample=Image.BICUBIC)

    w, h = pil0.size
    x1, y1, x2, y2 = _face_box_from_meta(face, w, h)
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    base = max(bw, bh)

    kps = face["kps_5"]
    px, py = kps[kp_idx]
    px = float(px)
    py = float(py)

    half = 0.5 * base * float(KP_PAD_FRAC)
    cx1 = int(max(0, round(px - half)))
    cy1 = int(max(0, round(py - half)))
    cx2 = int(min(w, round(px + half)))
    cy2 = int(min(h, round(py + half)))

    if cx2 <= cx1 + 1 or cy2 <= cy1 + 1:
        return pil0.resize((KP_PART_SIZE, KP_PART_SIZE), resample=Image.BICUBIC)

    crop = pil0.crop((cx1, cy1, cx2, cy2))
    crop = crop.resize((KP_PART_SIZE, KP_PART_SIZE), resample=Image.BICUBIC)
    return crop


def make_feature_row(pil, tag, tfm, prefix_text):
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

    row = _hstack_with_gap(tiles, gap=10)
    row = _put_text_rgb(row, tag, x=8, y=26, scale=0.9, thickness=2)
    return row


def _add_stats(stats, prefix, x):
    if x.ndim == 2:
        stats.setdefault(prefix + "_mean", []).append(x.mean().item())
        stats.setdefault(prefix + "_std", []).append(x.std(unbiased=False).item())
        return

    for ci in range(x.shape[0]):
        ch = x[ci]
        stats.setdefault(f"{prefix}_c{ci}_mean", []).append(ch.mean().item())
        stats.setdefault(f"{prefix}_c{ci}_std", []).append(
            ch.std(unbiased=False).item()
        )


def _plot_hist_pair(real_vals, fake_vals, title, out_path, bins=60):
    plt.figure(figsize=(7.2, 4.2))
    plt.hist(real_vals, bins=bins, alpha=0.6, label="real", density=True)
    plt.hist(fake_vals, bins=bins, alpha=0.6, label="fake", density=True)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200)
    plt.close()


def compute_and_save_distributions(real_rows, fake_rows, tfm):
    random.shuffle(real_rows)
    random.shuffle(fake_rows)

    real_rows = real_rows[:MAX_DIST_SAMPLES]
    fake_rows = fake_rows[:MAX_DIST_SAMPLES]

    real_stats = {}
    fake_stats = {}

    def process(rows, stats):
        for row in rows:
            pil = Image.open(str(Path(row["path"]))).convert("RGB")
            rgb01 = tfm(pil)

            gray01 = _rgb01_to_gray01(rgb01)
            npr = _npr_from_rgb01(rgb01)
            res = _residual_gray(gray01)
            ll, lh, hl, hh = _haar_wavelet_1level(gray01)

            _add_stats(stats, "RGB", rgb01)
            _add_stats(stats, "GRAY", gray01)

            _add_stats(stats, "NPR", npr)
            _add_stats(stats, "RES", res)
            _add_stats(stats, "WLL", ll)
            _add_stats(stats, "WLH", lh)
            _add_stats(stats, "WHL", hl)
            _add_stats(stats, "WHH", hh)

    process(real_rows, real_stats)
    process(fake_rows, fake_stats)

    keys = sorted(set(real_stats.keys()) & set(fake_stats.keys()))
    for k in keys:
        out = STAT_DIR / f"hist_{k}.png"
        _plot_hist_pair(
            real_stats[k], fake_stats[k], f"{k} (real vs fake)", out, bins=HIST_BINS
        )


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    rows = read_jsonl(TRAIN_JSONL)
    real_rows = [r for r in rows if int(r["label"]) == 0]
    fake_rows = [r for r in rows if int(r["label"]) == 1]

    tfm = T.Compose(
        [
            T.Resize(SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(SIZE),
            T.ToTensor(),
        ]
    )

    rr = real_rows[:]
    fr = fake_rows[:]
    random.shuffle(rr)
    random.shuffle(fr)
    n = min(NUM_PAIRS, len(rr), len(fr))

    for i in range(n):
        rrow = rr[i]
        frow = fr[i]

        pil_r0 = Image.open(str(Path(rrow["path"]))).convert("RGB")
        pil_f0 = Image.open(str(Path(frow["path"]))).convert("RGB")

        face_r = rrow.get("face", None)
        face_f = frow.get("face", None)

        pil_rz = _crop_zoom_pil(pil_r0, face_r) if USE_FACE_ZOOM else pil_r0
        pil_fz = _crop_zoom_pil(pil_f0, face_f) if USE_FACE_ZOOM else pil_f0

        parts_r = [pil_r0, pil_rz]
        parts_f = [pil_f0, pil_fz]

        if USE_KP_PARTS and face_r is not None:
            for k in range(5):
                parts_r.append(_crop_kp_part_pil(pil_r0, face_r, k))
        if USE_KP_PARTS and face_f is not None:
            for k in range(5):
                parts_f.append(_crop_kp_part_pil(pil_f0, face_f, k))

        rows_real = []
        rows_fake = []

        rows_real.append(
            make_feature_row(parts_r[0], "REAL (label=0) - FULL", tfm, "FULL")
        )
        rows_fake.append(
            make_feature_row(parts_f[0], "FAKE (label=1) - FULL", tfm, "FULL")
        )

        rows_real.append(
            make_feature_row(parts_r[1], "REAL (label=0) - ZOOM", tfm, "ZOOM")
        )
        rows_fake.append(
            make_feature_row(parts_f[1], "FAKE (label=1) - ZOOM", tfm, "ZOOM")
        )

        if USE_KP_PARTS:
            for k in range(5):
                if len(parts_r) >= 2 + 5:
                    rows_real.append(
                        make_feature_row(
                            parts_r[2 + k], f"REAL (label=0) - KP{k+1}", tfm, f"KP{k+1}"
                        )
                    )
                if len(parts_f) >= 2 + 5:
                    rows_fake.append(
                        make_feature_row(
                            parts_f[2 + k], f"FAKE (label=1) - KP{k+1}", tfm, f"KP{k+1}"
                        )
                    )

        panel_blocks = []
        for j in range(min(len(rows_real), len(rows_fake))):
            block = _vstack_with_gap_rgb([rows_real[j], rows_fake[j]], gap=14)
            panel_blocks.append(block)

        panel = _vstack_with_gap_rgb(panel_blocks, gap=22)

        out_name = f"{i:05d}_real_vs_fake_features.jpg"
        save_rgb(OUT_DIR / out_name, panel)

    if DO_HIST:
        compute_and_save_distributions(real_rows, fake_rows, tfm)


if __name__ == "__main__":
    main()
