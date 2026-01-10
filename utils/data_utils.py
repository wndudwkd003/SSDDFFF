# utils/data_utils.py

from __future__ import annotations

import os
import json
from typing import List

from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from config.config import Config, InputFeature, DatasetName


def read_jsonl(jsonl_path: str):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_mean_std(config: Config):
    m = str(config.model_name).lower()
    if "clip" in m:
        return _CLIP_MEAN, _CLIP_STD
    return _IMAGENET_MEAN, _IMAGENET_STD


def _normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.tensor(mean, dtype=x.dtype, device=x.device)[:, None, None]
    s = torch.tensor(std, dtype=x.dtype, device=x.device)[:, None, None]
    return (x - m) / s


def _npr_from_rgb01(rgb01: torch.Tensor) -> torch.Tensor:
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


def build_input_tensor(img_pil: Image.Image, config: Config) -> torch.Tensor:
    rgb01 = T.Compose(
        [
            T.Resize(config.image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(config.image_size),
            T.ToTensor(),
        ]
    )(img_pil.convert("RGB"))

    mean, std = _get_mean_std(config)
    feats: List[torch.Tensor] = []

    for f in config.input_features:
        if f == InputFeature.RGB:
            feats.append(_normalize(rgb01, mean, std))
        elif f == InputFeature.NPR:
            npr = _npr_from_rgb01(rgb01)
            feats.append(_normalize(npr, mean, std))
        else:
            raise ValueError(f)

    return torch.cat(feats, dim=0)


def _subsample_rows(rows: list[dict], ratio: float, seed: int) -> list[dict]:
    if ratio >= 1.0:
        return rows
    n = len(rows)
    k = int(n * ratio)
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:k].tolist()
    return [rows[i] for i in idx]


def _dataset_dir(config: Config, ds: DatasetName) -> str:
    return os.path.join(config.datasets_path, ds.value)


def _ratio_in(datasets: list[tuple[DatasetName, float]], ds: DatasetName) -> float:
    for d, r in datasets:
        if d == ds:
            return float(r)
    return 1.0


def _load_split_rows(
    config: Config,
    datasets: list[tuple[DatasetName, float]],
    ds: DatasetName,
    split: str,
) -> list[dict]:
    ratio = _ratio_in(datasets, ds)
    p = os.path.join(_dataset_dir(config, ds), f"{split}.jsonl")
    rows = read_jsonl(p)
    return _subsample_rows(rows, ratio, config.seed)


def _merge_split_rows(
    config: Config,
    datasets: list[tuple[DatasetName, float]],
    split: str,
) -> list[dict]:
    all_rows: list[dict] = []
    for ds, _ratio in datasets:
        all_rows.extend(_load_split_rows(config, datasets, ds, split))

    g = torch.Generator()
    g.manual_seed(config.seed)
    perm = torch.randperm(len(all_rows), generator=g).tolist()
    return [all_rows[i] for i in perm]


class DF_Dataset_JSON(Dataset):
    def __init__(self, rows: list[dict], config: Config):
        self.rows = rows
        self.config = config

    def __len__(self):
        if self.config.DEBUG_MODE:
            return min(10, len(self.rows))
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["path"]).convert("RGB")
        x = build_input_tensor(img, self.config)
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return {"pixel_values": x, "labels": y}


class DF_Dataset_CSV(Dataset):
    def __init__(self, csv_path: str, config: Config):
        import pandas as pd

        self.df = pd.read_csv(csv_path)
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        crop_path = row["crop_path"]
        filename = row["filename"]
        media_type = row["media_type"]
        frame_index = int(row["frame_index"])
        rank_in_video = int(row["rank_in_video"])

        img = Image.open(crop_path).convert("RGB")
        x = build_input_tensor(img, self.config)

        return {
            "pixel_values": x,
            "filename": filename,
            "media_type": media_type,
            "frame_index": frame_index,
            "rank_in_video": rank_in_video,
            "index": idx,
        }


def get_train_loader(config: Config, split: str, dataset_name: DatasetName):
    train_sets = config.selected_datasets

    if config.use_dataset_sum:
        rows = _merge_split_rows(config, train_sets, split)
    else:
        rows = _load_split_rows(config, train_sets, dataset_name, split)

    ds = DF_Dataset_JSON(rows, config)
    return DataLoader(ds, batch_size=config.batch_size, shuffle=(split == "train"))


def get_test_loader_jsonl(config: Config, dataset_name: DatasetName | None):
    eval_sets = config.evaluate_datasets

    if dataset_name is None:
        if config.use_dataset_sum:
            rows = _merge_split_rows(config, eval_sets, "test")
        else:
            dataset_name = eval_sets[0][0]
            rows = _load_split_rows(config, eval_sets, dataset_name, "test")
    else:
        rows = _load_split_rows(config, eval_sets, dataset_name, "test")

    ds = DF_Dataset_JSON(rows, config)
    return DataLoader(ds, batch_size=config.batch_size, shuffle=False)


def get_test_loader_submission(config: Config):
    ds = DF_Dataset_CSV(config.test_meta_csv_path, config)
    return DataLoader(ds, batch_size=config.batch_size, shuffle=False)


def get_data_loader(config: Config, split: str, dataset_name: DatasetName):
    if config.do_mode == "train":
        if split in ["train", "valid"]:
            return get_train_loader(config, split, dataset_name)

        elif split == "test":
            return get_test_loader_jsonl(config, dataset_name)

    if config.do_mode == "test":
        return get_test_loader_submission(config)
