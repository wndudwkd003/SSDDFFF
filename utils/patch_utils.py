# utils/patch_utils.py

from PIL import Image
import torch

from config.config import InputPatchType
from config.config import Config
from utils.feature_utils import build_input_tensor


from torchvision import transforms as T


def _face_box_from_meta(face: dict | None, w: int, h: int):
    if face is None:
        return (0, 0, w, h)

    if "bbox_xyxy" in face:
        x1, y1, x2, y2 = face["bbox_xyxy"]
    else:
        kps = face.get("kps_5", None)
        if kps is None:
            return (0, 0, w, h)
        xs = [p[0] for p in kps]
        ys = [p[1] for p in kps]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

    x1 = max(0, int(round(float(x1))))
    y1 = max(0, int(round(float(y1))))
    x2 = min(w, int(round(float(x2))))
    y2 = min(h, int(round(float(y2))))
    return x1, y1, x2, y2


def _crop_center_zoom_pil(
    pil0: Image.Image,
    face: dict | None,
    *,
    out_size: int,
    pad_frac: float = 0.25,
    zoom_scale: float = 2.2,
) -> Image.Image:
    """
    질문에서 말한 CENTER_ZOOM: bbox 바깥을 잘라내고 bbox 내부를 확대해서 out_size 정사각형으로 만듦.
    """
    if face is None:
        # face 정보 없으면 그냥 중앙 crop로 대체
        return T.Compose(
            [
                T.Resize(out_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(out_size),
            ]
        )(pil0.convert("RGB"))

    w, h = pil0.size
    x1, y1, x2, y2 = _face_box_from_meta(face, w, h)

    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    pad = float(pad_frac)

    x1 = int(max(0, round(x1 - bw * pad)))
    y1 = int(max(0, round(y1 - bh * pad)))
    x2 = int(min(w, round(x2 + bw * pad)))
    y2 = int(min(h, round(y2 + bh * pad)))

    crop = pil0.crop((x1, y1, x2, y2))

    zoom_side = int(round(out_size * float(zoom_scale)))
    zoom_side = max(zoom_side, out_size)
    crop = crop.resize((zoom_side, zoom_side), resample=Image.BICUBIC)

    cx = zoom_side // 2
    cy = zoom_side // 2
    half = out_size // 2
    crop = crop.crop((cx - half, cy - half, cx - half + out_size, cy - half + out_size))
    return crop


def _crop_kp_part_pil(
    pil0: Image.Image,
    face: dict | None,
    kp_idx: int,
    *,
    out_size: int,
    kp_pad_frac: float = 0.18,
) -> Image.Image:
    """
    keypoint 주변 패치 crop.
    - face bbox 크기 기반으로 keypoint 주변을 잘라 out_size 정사각형으로 리사이즈
    """
    if face is None:
        return T.Compose(
            [
                T.Resize(out_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(out_size),
            ]
        )(pil0.convert("RGB"))

    kps = face.get("kps_5", None)
    if kps is None or len(kps) != 5:
        return T.Compose(
            [
                T.Resize(out_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(out_size),
            ]
        )(pil0.convert("RGB"))

    w, h = pil0.size
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

    if cx2 <= cx1 + 1 or cy2 <= cy1 + 1:
        return T.Compose(
            [
                T.Resize(out_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(out_size),
            ]
        )(pil0.convert("RGB"))

    crop = pil0.crop((cx1, cy1, cx2, cy2))
    crop = crop.resize((out_size, out_size), resample=Image.BICUBIC)
    return crop


def _make_outter_feature_zoom_pil(
    pil0: Image.Image,
    face: dict | None,
    *,
    out_size: int,
    seed: int,
    kp_pad_frac: float = 0.18,
    center_pad_frac: float = 0.25,
    center_zoom_scale: float = 2.2,
) -> Image.Image:
    """
    OUTTER_FEATURE_ZOOM:
    - 3x3 모자이크(정사각형)
      * center(1칸): CENTER_ZOOM
      * outer ring(8칸): 5 keypoint crop을 복사/반복해 채움 (8칸 필요 -> 3칸 중복)
    """
    # 3x3 tile
    grid = 3
    tile = out_size // grid
    canvas = Image.new("RGB", (tile * grid, tile * grid))

    # center
    center_pil = _crop_center_zoom_pil(
        pil0,
        face,
        out_size=tile,
        pad_frac=center_pad_frac,
        zoom_scale=center_zoom_scale,
    )

    # outer 8 tiles: kp(0~4) + 3 repeats
    g = torch.Generator()
    g.manual_seed(int(seed))

    # 기본 5개 kp
    kp_tiles = []
    for k in range(5):
        kp_tiles.append(
            _crop_kp_part_pil(pil0, face, k, out_size=tile, kp_pad_frac=kp_pad_frac)
        )

    # 8칸 채우기 위해 3개 더 뽑기(중복 허용)
    # face가 없으면 kp_tiles가 사실상 center-crop 기반이므로 그래도 동작은 함.
    extra_idx = torch.randint(low=0, high=5, size=(3,), generator=g).tolist()
    for ei in extra_idx:
        kp_tiles.append(kp_tiles[ei].copy())

    assert len(kp_tiles) == 8

    # placement: outer ring positions (row, col)
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

    # paste outer
    for t, (r, c) in zip(kp_tiles, outer_pos):
        canvas.paste(t, (c * tile, r * tile))

    # paste center
    canvas.paste(center_pil, (1 * tile, 1 * tile))

    # out_size가 3으로 나누어 떨어지지 않으면 마지막에 정확히 out_size로 맞춤
    if canvas.size != (out_size, out_size):
        canvas = canvas.resize((out_size, out_size), resample=Image.Resampling.BICUBIC)

    return canvas


def _apply_patch_policy(
    img_pil: Image.Image, face: dict | None, config: Config, patch: InputPatchType
) -> Image.Image:
    if patch == InputPatchType.CENTER_ZOOM:
        return _crop_center_zoom_pil(
            img_pil,
            face,
            out_size=config.image_size,
            pad_frac=0.25,
            zoom_scale=2.2,
        )

    if patch == InputPatchType.OUTTER_FEATURE_ZOOM:
        return _make_outter_feature_zoom_pil(
            img_pil,
            face,
            out_size=config.image_size,
            seed=config.seed,
            kp_pad_frac=0.18,
            center_pad_frac=0.25,
            center_zoom_scale=2.2,
        )

    raise ValueError(patch)


def select_input_pil(
    img_pil: Image.Image,
    face: dict | None,
    config: Config,
    *,
    split: str,
) -> tuple[Image.Image, str]:
    """
    모델에 들어가는 '최종 입력 PIL'과 그 이름을 반환.
    - INPUT_PATCHES 비어있으면: 원본 img_pil 그대로 ("ORIGINAL")
    - 있으면: 기존 정책대로 선택된 patch PIL ("center_zoom" 등)
    """
    if len(config.input_patches) <= 0:
        return img_pil, "ORIGINAL"

    if split == "train":
        g = torch.Generator()
        g.manual_seed(int(config.seed))
        idx = int(
            torch.randint(
                low=0, high=len(config.input_patches), size=(1,), generator=g
            ).item()
        )
        patch = config.input_patches[idx]
    else:
        patch = config.input_patches[0]

    pil_patch = _apply_patch_policy(img_pil, face, config, patch)
    return pil_patch, patch.value


def build_input_tensor_with_patches(
    img_pil: Image.Image,
    face: dict | None,
    config: Config,
    *,
    split: str,
) -> torch.Tensor:
    pil_in, _name = select_input_pil(img_pil, face, config, split=split)
    return build_input_tensor(pil_in, config)


def make_patch_pils(
    img_pil: Image.Image,
    face: dict | None,
    config: Config,
) -> list[tuple[str, Image.Image]]:
    """
    config.input_patches 순서대로 패치 PIL들을 생성해서 반환.
    반환: [(patch_name, patch_pil), ...]
    """
    out: list[tuple[str, Image.Image]] = []
    for patch in config.input_patches:
        pil_patch = _apply_patch_policy(img_pil, face, config, patch)
        out.append((patch.value, pil_patch))
    return out
