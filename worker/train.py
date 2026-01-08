# worker/train.py

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
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

        # load pretrained model
        if config.pretrained:
            sd = torch.load(config.pretrained_ckpt_path, map_location="cpu")
            self.model.load_state_dict(sd, strict=False)
            print(
                f"[trainer.py] Loaded pretrained model from {config.pretrained_ckpt_path}"
            )

        # run_dir
        ts = get_current_timestamp()
        ad = f"{ts}_{config.run_name}_{config.image_size}"
        run_dir = Path(config.run_dir) / ad
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[trainer.py] Run dir: {run_dir}")
        self.run_dir = run_dir

        self.svm: Pipeline | None = None

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

    @torch.no_grad()
    def _collect_embeddings_test(self, loader):
        self.model.eval()

        X_list = []
        idx_list = []

        for batch in tqdm(loader, desc="collect_emb_test"):
            x = batch["pixel_values"].to(self.device, non_blocking=True)
            idx = batch["index"].cpu().numpy()

            out = self.model(x)
            emb = out["embedding"].detach().cpu().numpy()

            X_list.append(emb)
            idx_list.append(idx)

        X = np.concatenate(X_list, axis=0)
        idx = np.concatenate(idx_list, axis=0)
        return X, idx

    def train(self):
        train_loader = get_data_loader(self.config, "train")
        valid_loader = get_data_loader(self.config, "valid")

        # Stage1을 생략하고 Stage2(SVM)만 학습
        if self.config.skip_stage1:
            return self.train_stage2(train_loader, valid_loader)

        # Stage1(CE/contrastive) 학습을 붙일 계획이면 여기에 구현
        raise NotImplementedError(
            "skip_stage1=False인 Stage1 학습은 아직 구현되지 않았습니다."
        )

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

    def test(self):
        test_loader = get_data_loader(self.config, "test")
        X_test, y_test = self._collect_embeddings(test_loader)

        pred = self.svm.predict(X_test)
        acc = float((pred == y_test).mean())
        print(f"[test] valid_acc={acc:.6f}")
        return {"valid_acc": acc}

    @torch.no_grad()
    def _collect_embeddings_test_meta(self, loader):
        self.model.eval()

        X_list = []
        filename_list = []
        media_type_list = []

        for batch in tqdm(loader, desc="Collect emb test meta"):
            x = batch["pixel_values"].to(self.device)

            out = self.model(x)
            emb = out["embedding"].detach().cpu().numpy()  # [B, D]
            X_list.append(emb)

            # batch["filename"] / batch["media_type"]는 DataLoader가 list로 묶어줍니다.
            filename_list.extend(list(batch["filename"]))
            media_type_list.extend(list(batch["media_type"]))

        X = np.concatenate(X_list, axis=0)
        return X, filename_list, media_type_list

    def test_for_submission(self):

        test_loader = get_data_loader(self.config, "test")

        X, filenames, media_types = self._collect_embeddings_test_meta(test_loader)

        # LinearSVC는 predict_proba가 없으므로 decision_function -> sigmoid로 확률화
        scores = self.svm.decision_function(X)  # [N]
        probs = sigmoid(scores)  # [N], 0~1

        # filename별로 확률 리스트 모으기 + media_type 보관
        per_file = {}
        order = []  # meta.csv에 처음 등장한 filename 순서 유지
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
                final_p = float(ps[0])  # 이미지 샘플은 1개 row라고 가정
            else:
                final_p = confident_strategy(ps, t=0.8)

            out_filenames.append(fn)
            out_probs.append(final_p)
            out_labels.append(int(final_p >= 0.5))

        return {
            "filenames": out_filenames,  # 제출 단위(원본 샘플) 리스트
            "probs": out_probs,  # 집계된 확률
            "preds": out_labels,  # 0/1 예측 (필요 없으면 submit에서 무시 가능)
            "per_file_probs": {k: v["probs"] for k, v in per_file.items()},  # 디버그용
            "run_dir": str(self.run_dir),
        }
