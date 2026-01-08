# utils/data_utils.py

import os
import json
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from config.config import Config


def read_jsonl(jsonl_path: str):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# CLIP normalize values
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class DF_Dataset_JSON(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        image_size: int,
        input_features: list[str],
        use_augmentation: bool = False,
        split: str = "train",
        debug_mode: bool = False,
    ):
        self.rows = read_jsonl(jsonl_path)

        self.transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
            ]
        )

        self.debug_mode = debug_mode

    def __len__(self):
        if self.debug_mode:
            return min(10, len(self.rows))
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        img = Image.open(row["path"]).convert("RGB")
        x = self.transform(img)  # [3, H, W]

        y = torch.tensor(row["label"], dtype=torch.long)

        return {
            "pixel_values": x,
            "labels": y,
        }


class DF_Dataset_CSV(Dataset):
    def __init__(self, csv_path: str, image_size: int):
        import pandas as pd

        self.df = pd.read_csv(csv_path)

        self.transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # meta.csv 기준
        crop_path = row["crop_path"]
        filename = row["filename"]  # 원본 샘플 id (video면 mp4, image면 jpg)
        media_type = row["media_type"]  # "video" | "image"
        frame_index = int(row["frame_index"])
        rank_in_video = int(row["rank_in_video"])

        img = Image.open(crop_path).convert("RGB")
        x = self.transform(img)

        return {
            "pixel_values": x,
            "filename": filename,
            "media_type": media_type,
            "frame_index": frame_index,
            "rank_in_video": rank_in_video,
            "index": idx,  # 필요하면 디버그/정렬용
        }


def get_train_loader(config: Config, split: str):
    jsonl_path = os.path.join(config.datasets_path, f"{split}.jsonl")
    ds = DF_Dataset_JSON(
        jsonl_path=jsonl_path,
        image_size=config.image_size,
        input_features=config.input_features,
        use_augmentation=config.use_augmentation,
        split=split,
        debug_mode=config.DEBUG_MODE,
    )
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
    )


def get_test_loader(config: Config):
    ds = DF_Dataset_CSV(
        csv_path=config.test_meta_csv_path,
        image_size=config.image_size,
    )
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=False,
    )


def get_data_loader(config: Config, split: str):
    if config.do_mode == "train":
        return get_train_loader(config, split)
    if config.do_mode == "test":
        return get_test_loader(config)
