# /workspace/SSDDFF/worker/train.py
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline

from config.config import Config, DatasetName
from utils.data_utils import get_data_loader
from utils.date_utils import get_current_timestamp
from utils.model_utils import build_model
from utils.losses import ReconLoss

from worker.pipelines import (
    train_ce,
    train_ae,
    eval_all_ce,
    eval_all_ae,
    collect_meta_probs_from_model,
    collect_meta_scores_from_ae,
    aggregate_per_file,
)


class Trainer:
    def __init__(
        self,
        config: Config,
        *,
        train_dataset: DatasetName,
        run_dir: Path | None = None,
    ):
        self.config = config
        self.train_dataset = train_dataset

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module = build_model(config).to(self.device)

        self.svm: Pipeline | None = None
        self.recon_loss: ReconLoss | None = None
        self.threshold: float | None = None

        if config.do_mode == "test":
            self.run_dir = Path(config.test_dir)
        else:
            if run_dir is None:
                ts = get_current_timestamp()
                name = f"{ts}_{config.run_name}_{config.image_size}"
                self.run_dir = Path(config.run_dir) / name
            else:
                self.run_dir = run_dir
            self.run_dir.mkdir(parents=True, exist_ok=True)

        if config.pretrained and config.pretrained_ckpt_path is not None:
            ckpt = torch.load(config.pretrained_ckpt_path, map_location="cpu")
            self.model.load_state_dict(ckpt, strict=False)

        if config.model_mode == "ae":
            self.recon_loss = ReconLoss(w_l1=1.0, w_mse=0.0, w_ssim=0.0)

        if config.do_mode == "test":
            self._load_for_test()

        self.model.to(self.device)

    def _load_for_test(self) -> None:
        if self.config.model_mode == "ae":
            ckpt = torch.load(self.run_dir / "best_ae.pth", map_location="cpu")
            self.model.load_state_dict(ckpt, strict=True)
            self.threshold = float(np.load(self.run_dir / "threshold.npy").item())
            return

        ckpt = torch.load(self.run_dir / "best_stage1.pth", map_location="cpu")
        self.model.load_state_dict(ckpt, strict=True)

    def train(self):
        train_loader = get_data_loader(
            self.config, "train", dataset_name=self.train_dataset
        )
        valid_loader = get_data_loader(
            self.config, "valid", dataset_name=self.train_dataset
        )

        if self.config.model_mode == "ae":
            out, thr = train_ae(
                model=self.model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                config=self.config,
                device=self.device,
                run_dir=self.run_dir,
                recon_loss=self.recon_loss,
            )
            self.threshold = float(thr)
            return out

        return train_ce(
            model=self.model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            config=self.config,
            device=self.device,
            run_dir=self.run_dir,
        )

    def test(self):
        if self.config.model_mode == "ae":
            out = eval_all_ae(
                model=self.model,
                config=self.config,
                device=self.device,
                recon_loss=self.recon_loss,
                threshold=self.threshold,
            )
            return {
                "run_dir": str(self.run_dir),
                "train_dataset": self.train_dataset.value,
                **out,
            }

        out = eval_all_ce(model=self.model, config=self.config, device=self.device)
        return {
            "run_dir": str(self.run_dir),
            "train_dataset": self.train_dataset.value,
            **out,
        }

    def test_for_submission(self):
        loader = get_data_loader(self.config, "test", dataset_name=self.train_dataset)

        if self.config.model_mode == "ae":
            scores, filenames, media_types = collect_meta_scores_from_ae(
                self.model, loader, self.device
            )
            out_fns, out_scores, per_file_scores = aggregate_per_file(
                scores, filenames, media_types, kind="scores", t_video=0.8
            )
            preds = [int(s >= float(self.threshold)) for s in out_scores]

            return {
                "filenames": out_fns,
                "scores": out_scores,
                "preds": preds,
                "per_file_scores": per_file_scores,
                "threshold": float(self.threshold),
                "run_dir": str(self.run_dir),
            }

        probs, filenames, media_types = collect_meta_probs_from_model(
            self.model, loader, self.device
        )

        out_fns, out_probs, per_file_probs = aggregate_per_file(
            probs, filenames, media_types, kind="probs", t_video=0.8
        )
        preds = [int(p >= 0.5) for p in out_probs]

        return {
            "filenames": out_fns,
            "probs": out_probs,
            "preds": preds,
            "per_file_probs": per_file_probs,
            "run_dir": str(self.run_dir),
        }
