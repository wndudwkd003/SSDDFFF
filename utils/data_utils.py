# utils/data_utils.py


import os
import json

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from config.config import Config, DatasetName
from utils.aug.augmenter import ImageAugmenter, AugmentConfig
from utils.patch_utils import apply_input_patch

from utils.feature_utils import build_input_tensor


def read_jsonl(jsonl_path: str):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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
    def __init__(self, rows: list[dict], config: Config, split: str):
        self.rows = rows
        self.config = config
        self.split = split

        self.augmenter = None

        print(f"Total samples in {split} set: {len(rows)}")
        print(f"Label 0 count: {sum(1 for r in rows if r['label'] == 0)}")
        print(f"Label 1 count: {sum(1 for r in rows if r['label'] == 1)}")

        print(
            f"Current split is {split}, User augmentation: {config.use_augmentation}."
        )
        if split == "train" and config.use_augmentation:
            self.augmenter = ImageAugmenter(AugmentConfig(), seed=config.seed)

        self._dbg_tfm = T.Compose(
            [
                T.Resize(config.image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(config.image_size),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        if self.config.DEBUG_MODE:
            return min(10, len(self.rows))
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        path = row["path"]
        y = row["label"]

        if self.config.data_label_invert:
            y = 1 - y

        img = Image.open(path).convert("RGB")
        face = row["face"]

        if self.augmenter is not None:
            img, face = self.augmenter(img, face)

        prev_x = apply_input_patch(img, face, self.config)
        x = build_input_tensor(prev_x, self.config)

        out = {
            "pixel_values": x,
            "labels": torch.tensor(y, dtype=torch.long),
            "path": str(path),
            "face": face,
        }

        return out


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


def get_train_loader(
    config: Config, split: str, dataset_name: DatasetName | None = None
):
    train_sets = config.selected_datasets

    if dataset_name == DatasetName.SUM:
        dataset_name = None

    if config.use_dataset_sum:
        rows = _merge_split_rows(config, train_sets, split)
    else:
        assert dataset_name is not None
        rows = _load_split_rows(config, train_sets, dataset_name, split)

    ds = DF_Dataset_JSON(rows, config, split=split)
    return DataLoader(ds, batch_size=config.batch_size, shuffle=(split == "train"))


def get_test_loader_jsonl(config: Config, dataset_name: DatasetName | None):
    eval_sets = config.evaluate_datasets

    if dataset_name == DatasetName.SUM:
        dataset_name = None

    if dataset_name is None:
        if config.use_dataset_sum:
            rows = _merge_split_rows(config, eval_sets, "test")
        else:
            dataset_name = eval_sets[0][0]
            rows = _load_split_rows(config, eval_sets, dataset_name, "test")
    else:
        rows = _load_split_rows(config, eval_sets, dataset_name, "test")

    ds = DF_Dataset_JSON(rows, config, split="test")
    return DataLoader(ds, batch_size=config.batch_size, shuffle=False)


def get_test_loader_submission(config: Config):
    ds = DF_Dataset_CSV(config.test_meta_csv_path, config)
    return DataLoader(ds, batch_size=config.batch_size, shuffle=False)


def get_data_loader(
    config: Config, split: str | None = None, dataset_name: DatasetName | None = None
):
    if config.do_mode in ["train", "test"]:
        if split in ["train", "valid"]:
            return get_train_loader(config, split, dataset_name)

        elif split == "test":
            return get_test_loader_jsonl(config, dataset_name)

    if config.do_mode == "test_submission":
        return get_test_loader_submission(config)
