# worker/analysis.py


import json
from pathlib import Path
from typing import Any, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def _compute_metrics(res):
    y_true = res["labels"]
    y_pred = res["preds"]
    y_prob = res["probs"]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


class Analyzer:
    def analyze(self, results: Dict[str, Any], name: str):
        """
        results: Trainer에서 반환한 dict (train/test 결과)
        name: "train" | "test" | ...
        """
        run_dir = Path(results["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1) raw json 저장 (재현/추적용)
        json_path = run_dir / f"analysis_{name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 2) 요약 txt 저장
        lines = []
        lines.append(f"[analysis] name={name}")
        lines.append(f"[analysis] run_dir={run_dir}")

        # ───────────────────────────────
        # ① SVM 유효성 정확도, 저장 경로
        # ───────────────────────────────
        if "valid_acc" in results:
            lines.append(f"valid_acc: {results['valid_acc']:.6f}")

        if "svm_path" in results:
            lines.append(f"svm_path: {results['svm_path']}")

        # ───────────────────────────────
        # ② Classification 평가 지표 (label이 있는 경우에만)
        # ───────────────────────────────
        if "labels" in results and "preds" in results and "probs" in results:
            metrics = _compute_metrics(results)
            lines.append("--- metrics ---")
            for k, v in metrics.items():
                lines.append(f"{k}: {v:.6f}")

        # ───────────────────────────────
        # ③ 확률 집계 통계
        # ───────────────────────────────
        if "probs" in results:
            probs = results["probs"]
            n = len(probs)
            mean_p = sum(probs) / max(n, 1)
            fake_ratio = sum(1 for p in probs if p >= 0.5) / max(n, 1)

            lines.append("--- prob stats ---")
            lines.append(f"num_samples: {n}")
            lines.append(f"mean_prob: {mean_p:.6f}")
            lines.append(f"fake_ratio(>=0.5): {fake_ratio:.6f}")

        # ───────────────────────────────
        # ④ per_file frame 수 통계 (video인 경우)
        # ───────────────────────────────
        if "per_file_probs" in results:
            per_file = results["per_file_probs"]
            lens = [len(v) for v in per_file.values()]
            if lens:
                lines.append("--- per_file frame stats ---")
                lines.append(f"num_files_with_frames: {len(lens)}")
                lines.append(f"min_frames: {min(lens)}")
                lines.append(f"max_frames: {max(lens)}")
                lines.append(f"mean_frames: {sum(lens) / len(lens):.3f}")

        # 3) 저장
        txt_path = run_dir / f"analysis_{name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"[analysis] saved: {json_path}")
        print(f"[analysis] saved: {txt_path}")
