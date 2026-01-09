# worker/train.py

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from config.config import Config
from utils.data_utils import get_data_loader
from utils.model_utils import build_model
from utils.date_utils import get_current_timestamp
from utils.strategy import sigmoid, confident_strategy


class Trainer:
    def __init__(self, config: Config):
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(config).to(self.device)
        self.probs_threshold = config.probs_threshold

        if config.pretrained and config.pretrained_ckpt_path is not None:
            ckpt = torch.load(config.pretrained_ckpt_path, map_location="cpu")
            self.model.load_state_dict(ckpt, strict=False)
            print(
                f"[trainer.py] Loaded pretrained model from {config.pretrained_ckpt_path}"
            )

        self.svm: Pipeline | None = None

        # =========================
        # run_dir 결정
        # =========================
        if config.do_mode == "test":
            # test_dir을 기존 학습 결과 폴더로 사용
            self.run_dir = Path(config.test_dir)
            print(f"[trainer.py] Test dir(run_dir): {self.run_dir}")
        else:
            # train 모드: 새 run_dir 만들기
            ts = get_current_timestamp()
            ad = f"{ts}_{config.run_name}_{config.image_size}"
            run_dir = Path(config.run_dir) / ad
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[trainer.py] Run dir: {run_dir}")
            self.run_dir = run_dir

        # =========================
        # test 모드면 필요한 것 로드
        # =========================
        if config.do_mode == "test":
            if config.head == "svm":
                svm_path = self.run_dir / "svm.joblib"
                self.svm = joblib.load(svm_path)
                print(f"[trainer.py] Loaded SVM from {svm_path}")
            else:
                ckpt_path = self.run_dir / "best_stage1.pth"
                ckpt = torch.load(ckpt_path, map_location="cpu")
                self.model.load_state_dict(ckpt, strict=True)
                print(f"[trainer.py] Loaded stage1 ckpt from {ckpt_path}")

        self.model.to(self.device)

    # -------------------------
    # Embedding collect (SVM용)
    # -------------------------
    @torch.no_grad()
    def _collect_embeddings(self, loader):
        self.model.eval()

        X_list = []
        y_list = []

        for batch in tqdm(loader, desc="collect_emb"):
            x = batch["pixel_values"].to(self.device, non_blocking=True)
            y = batch["labels"].cpu().numpy()

            out = self.model(x)
            emb = out["embedding"].detach().cpu().numpy()  # [B, D]

            X_list.append(emb)
            y_list.append(y)

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return X, y

    # -------------------------
    # Train
    # -------------------------
    def train(self):
        train_loader = get_data_loader(self.config, "train")
        valid_loader = get_data_loader(self.config, "valid")

        if self.config.head == "svm":
            # SVM 학습(=stage2)만
            return self.train_stage2(train_loader, valid_loader)

        # head != svm이면 stage1(CE) 학습
        if self.config.skip_stage1:
            raise NotImplementedError(
                "head!=svm인데 skip_stage1=True는 현재 흐름과 맞지 않습니다."
            )
        return self.train_stage1_ce(train_loader, valid_loader)

    def train_stage2(self, train_loader, valid_loader):
        # 1) embedding 수집
        X_train, y_train = self._collect_embeddings(train_loader)
        X_valid, y_valid = self._collect_embeddings(valid_loader)

        # 2) SVM 학습
        self.svm = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", LinearSVC()),
            ]
        )
        self.svm.fit(X_train, y_train)

        # 3) 검증 성능
        pred = self.svm.predict(X_valid)
        acc = float((pred == y_valid).mean())

        # 4) 저장
        svm_path = self.run_dir / "svm.joblib"
        joblib.dump(self.svm, svm_path)
        print(f"[stage2] valid_acc={acc:.6f} | saved: {svm_path}")

        return {
            "valid_acc": acc,
            "svm_path": str(svm_path),
            "run_dir": str(self.run_dir),
        }

    def run_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        split: str,
    ):
        is_train = split == "train"

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0

        all_probs = []
        all_preds = []
        all_labels = []

        loss_fn = nn.CrossEntropyLoss()

        for batch in tqdm(loader, desc=f"Epoch {split}"):
            x = batch["pixel_values"].to(self.device, non_blocking=True)
            y = batch["labels"].to(self.device, non_blocking=True)

            out = self.model(x)
            logits = out["logits"]  # [B, C]

            loss = loss_fn(logits, y)

            if is_train:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            total_loss += loss.item() * x.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= self.probs_threshold).long()

            all_probs.extend(probs.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(y.detach().cpu().tolist())

        avg_loss = total_loss / len(loader.dataset)

        return {
            "loss": avg_loss,
            "probs": all_probs,
            "preds": all_preds,
            "labels": all_labels,
        }

    def test(self):
        test_loader = get_data_loader(self.config, "test")
        test_results = self.run_epoch(test_loader, split="test")
        return {
            "run_dir": str(self.run_dir),
            "test": test_results,
        }

    def train_stage1_ce(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
    ):
        self.model.train()

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(
            params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=self.config.num_epochs,
        )

        best_valid_loss = float("inf")
        best_path = self.run_dir / "best_stage1.pth"

        for epoch in range(self.config.num_epochs):
            train_results = self.run_epoch(train_loader, split="train")
            valid_results = self.run_epoch(valid_loader, split="valid")

            self.scheduler.step()

            valid_loss = valid_results["loss"]

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    self.model.state_dict(),
                    best_path,
                )

        return {
            "run_dir": str(self.run_dir),
            "best_valid_loss": float(best_valid_loss),
            "best_ckpt_path": str(best_path),
            "train": train_results,
            "valid": valid_results,
        }

    # -------------------------
    # Test for submission (meta.csv 기반 집계)
    # -------------------------
    @torch.no_grad()
    def _collect_probs_test_meta_from_model(self, loader):
        self.model.eval()
        print("Model eval mode:", self.model.training)

        prob_list = []
        filename_list = []
        media_type_list = []

        for batch in tqdm(loader, desc="Collect prob test meta (model)"):
            print("batch keys:", batch.keys())
            x = batch["pixel_values"].to(self.device)

            out = self.model(x)
            logits = out["logits"]  # [B, 2] 가정
            probs = torch.softmax(logits, dim=1)[:, 1]  # fake prob
            prob_list.extend(probs.detach().cpu().numpy().tolist())

            filename_list.extend(list(batch["filename"]))
            media_type_list.extend(list(batch["media_type"]))

        return prob_list, filename_list, media_type_list

    @torch.no_grad()
    def _collect_probs_test_meta_from_svm(self, loader):
        self.model.eval()

        X_list = []
        filename_list = []
        media_type_list = []

        for batch in tqdm(loader, desc="Collect emb test meta (svm)"):
            x = batch["pixel_values"].to(self.device)

            out = self.model(x)
            emb = out["embedding"].detach().cpu().numpy()  # [B, D]
            X_list.append(emb)

            filename_list.extend(list(batch["filename"]))
            media_type_list.extend(list(batch["media_type"]))

        X = np.concatenate(X_list, axis=0)
        scores = self.svm.decision_function(X)  # [N]
        probs = sigmoid(scores).tolist()
        return probs, filename_list, media_type_list

    def test_for_submission(self):
        test_loader = get_data_loader(self.config, "test")

        if self.config.head == "svm":
            probs, filenames, media_types = self._collect_probs_test_meta_from_svm(
                test_loader
            )
        else:
            probs, filenames, media_types = self._collect_probs_test_meta_from_model(
                test_loader
            )

        # filename별로 확률 리스트 모으기 + media_type 보관
        per_file = {}
        order = []
        for fn, mt, p in zip(filenames, media_types, probs):
            if fn not in per_file:
                per_file[fn] = {"media_type": mt, "probs": []}
                order.append(fn)
            per_file[fn]["probs"].append(float(p))

        # 집계
        out_filenames = []
        out_probs = []
        out_labels = []

        for fn in order:
            mt = per_file[fn]["media_type"]
            ps = per_file[fn]["probs"]

            if mt == "image":
                final_p = float(ps[0])
            else:
                final_p = confident_strategy(ps, t=0.8)

            out_filenames.append(fn)
            out_probs.append(final_p)
            out_labels.append(int(final_p >= 0.5))

        return {
            "filenames": out_filenames,
            "probs": out_probs,
            "preds": out_labels,
            "per_file_probs": {k: v["probs"] for k, v in per_file.items()},
            "run_dir": str(self.run_dir),
        }
