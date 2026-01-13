# worker/analysis.py

import json
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(res: dict):
    y_true = res["labels"]
    y_pred = res["preds"]

    # CE: probs 기반
    if "probs" in res:
        y_score = res["probs"]
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_score)),
        }

    # AE: scores 기반 (높을수록 fake/anomaly)
    if "scores" in res:
        y_score = res["scores"]
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_score)),
        }

    raise KeyError("compute_metrics: res must contain either 'probs' or 'scores'.")


def prob_stats(probs, thr: float = 0.5):
    n = len(probs)
    mean_p = float(sum(probs) / max(n, 1))
    fake_ratio = float(sum(1 for p in probs if p >= thr) / max(n, 1))
    return {
        "num_samples": int(n),
        "mean_prob": mean_p,
        "fake_ratio": fake_ratio,
    }


def score_stats(scores, thr: float):
    n = len(scores)
    mean_s = float(sum(scores) / max(n, 1))
    fake_ratio = float(sum(1 for s in scores if s >= thr) / max(n, 1))
    return {
        "num_samples": int(n),
        "mean_score": mean_s,
        "fake_ratio": fake_ratio,
    }


def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_txt(path: Path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_csv(path: Path, header, rows):
    out_lines = [",".join(header)]
    for r in rows:
        out_lines.append(",".join("" if v is None else str(v) for v in r))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")


def plot_heatmap(mat, row_labels, col_labels, out_path: Path, title: str):
    fig, ax = plt.subplots(
        figsize=(max(6, 0.9 * len(col_labels)), max(4, 0.6 * len(row_labels)))
    )
    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_cross_dataset_grids(runs_root: str, out_dir=None, metric_keys=None):
    if metric_keys is None:
        metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    runs_root_p = Path(runs_root)
    out_p = Path(out_dir) if out_dir is not None else (runs_root_p / "_cross_grids")
    out_p.mkdir(parents=True, exist_ok=True)

    json_paths = glob(str(runs_root_p / "*" / "cross_test_*.json"))
    json_paths = sorted(json_paths, key=lambda p: Path(p).stat().st_mtime, reverse=True)

    cross_map = {}
    train_list = []
    seen = set()

    for jp in json_paths:
        p = Path(jp)
        train_ds = p.stem.replace("cross_test_", "", 1)
        if train_ds in seen:
            continue
        seen.add(train_ds)

        with open(p, "r", encoding="utf-8") as f:
            per_ds = json.load(f)

        cross_map[train_ds] = per_ds
        train_list.append(train_ds)

    test_set = set()
    for per_ds in cross_map.values():
        test_set.update(per_ds.keys())
    test_list = sorted(test_set)
    train_list = sorted(train_list)

    for mk in metric_keys:
        mat = np.full((len(train_list), len(test_list)), np.nan, dtype=float)

        for i, tr in enumerate(train_list):
            per_ds = cross_map.get(tr, {})
            for j, te in enumerate(test_list):
                cell = per_ds.get(te)
                if cell is None:
                    continue
                v = cell.get(mk)
                if v is None:
                    continue
                mat[i, j] = float(v)

        out_path = out_p / f"cross_grid_{mk}.png"
        plot_heatmap(
            mat=mat,
            row_labels=train_list,
            col_labels=test_list,
            out_path=out_path,
            title=f"Cross-Dataset Grid ({mk}) [row=train, col=test]",
        )

        csv_path = out_p / f"cross_grid_{mk}.csv"
        header = ["train\\test"] + test_list
        rows = []
        for i, tr in enumerate(train_list):
            r = [tr]
            for j in range(len(test_list)):
                v = mat[i, j]
                r.append("" if not np.isfinite(v) else f"{v:.6f}")
            rows.append(r)
        save_csv(csv_path, header=header, rows=rows)

    return str(out_p)


class Analyzer:
    def analyze(self, results: dict, name: str):
        run_dir = Path(results["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)

        json_path = run_dir / f"analysis_{name}.json"
        save_json(json_path, results)

        lines = []
        lines.append(f"[analysis] name={name}")
        lines.append(f"[analysis] run_dir={run_dir}")

        if "train_dataset" in results:
            lines.append(f"train_dataset: {results['train_dataset']}")

        if "valid_acc" in results:
            lines.append(f"valid_acc: {results['valid_acc']:.6f}")
        if "svm_path" in results:
            lines.append(f"svm_path: {results['svm_path']}")

        # flat (단일 결과) 지원: CE(probs) / AE(scores)
        if (
            "labels" in results
            and "preds" in results
            and ("probs" in results or "scores" in results)
        ):
            m = compute_metrics(results)
            lines.append("--- metrics (flat) ---")
            for k, v in m.items():
                lines.append(f"{k}: {v}")

        # split별 metrics: preds가 있어야 계산 가능
        for split_key in ["train", "valid"]:
            if split_key in results and isinstance(results[split_key], dict):
                sub = results[split_key]
                if (
                    "labels" in sub
                    and "preds" in sub
                    and ("probs" in sub or "scores" in sub)
                ):
                    m = compute_metrics(sub)
                    lines.append(f"--- metrics ({split_key}) ---")
                    for k, v in m.items():
                        lines.append(f"{split_key}.{k}: {v}")

        if "tests_all" in results and isinstance(results["tests_all"], dict):
            tests_all = results["tests_all"]
            order = results.get("test_order", list(tests_all.keys()))
            train_ds = results.get("train_dataset", "unknown_train")

            per_ds = {}
            csv_rows = []

            for ds_name in order:
                res = tests_all[ds_name]
                row = {"dataset": ds_name, "loss": res.get("loss")}

                m = compute_metrics(res)
                row.update(m)

                if "probs" in res:
                    row.update(prob_stats(res["probs"], thr=0.5))
                    csv_rows.append(
                        [
                            row["dataset"],
                            row["loss"],
                            row["accuracy"],
                            row["precision"],
                            row["recall"],
                            row["f1"],
                            row["roc_auc"],
                            row["num_samples"],
                            row["mean_prob"],
                            row["fake_ratio"],
                        ]
                    )
                else:
                    thr = float(res.get("threshold", results.get("threshold", 0.0)))
                    row.update(score_stats(res["scores"], thr=thr))
                    csv_rows.append(
                        [
                            row["dataset"],
                            row["loss"],
                            row["accuracy"],
                            row["precision"],
                            row["recall"],
                            row["f1"],
                            row["roc_auc"],
                            row["num_samples"],
                            row["mean_score"],
                            row["fake_ratio"],
                        ]
                    )

                per_ds[ds_name] = row

            cross_json = run_dir / f"cross_test_{train_ds}.json"
            save_json(cross_json, per_ds)

            cross_csv = run_dir / f"cross_test_{train_ds}.csv"
            if len(csv_rows) > 0 and len(csv_rows[0]) == 10:
                # CE
                save_csv(
                    cross_csv,
                    header=[
                        "dataset",
                        "loss",
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "roc_auc",
                        "num_samples",
                        "mean_prob",
                        "fake_ratio",
                    ],
                    rows=csv_rows,
                )
            else:
                # AE
                save_csv(
                    cross_csv,
                    header=[
                        "dataset",
                        "loss",
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "roc_auc",
                        "num_samples",
                        "mean_score",
                        "fake_ratio",
                    ],
                    rows=csv_rows,
                )

            metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            col_labels = [ds for ds in order if ds in per_ds]

            mat = np.full((len(metric_keys), len(col_labels)), np.nan, dtype=float)
            for j, ds in enumerate(col_labels):
                for i, mk in enumerate(metric_keys):
                    mat[i, j] = float(per_ds[ds][mk])

            heatmap_path = run_dir / f"cross_test_heatmap_{train_ds}.png"
            plot_heatmap(
                mat=mat,
                row_labels=metric_keys,
                col_labels=col_labels,
                out_path=heatmap_path,
                title=f"Cross-Dataset Test Heatmap (train={train_ds})",
            )

        else:
            cross_grids_dir = None

        if "probs" in results:
            ps = prob_stats(results["probs"], thr=0.5)
            lines.append("--- prob stats ---")
            lines.append(f"num_samples: {ps['num_samples']}")
            lines.append(f"mean_prob: {ps['mean_prob']:.6f}")
            lines.append(f"fake_ratio(>=0.5): {ps['fake_ratio']:.6f}")

        if "scores" in results:
            thr = float(results.get("threshold", 0.0))
            ss = score_stats(results["scores"], thr=thr)
            lines.append("--- score stats ---")
            lines.append(f"num_samples: {ss['num_samples']}")
            lines.append(f"mean_score: {ss['mean_score']:.6f}")
            lines.append(f"fake_ratio(>=thr): {ss['fake_ratio']:.6f}")
            lines.append(f"threshold: {thr:.6f}")

        if "per_file_probs" in results:
            per_file = results["per_file_probs"]
            lens = [len(v) for v in per_file.values()]
            lines.append("--- per_file frame stats ---")
            lines.append(f"num_files_with_frames: {len(lens)}")
            lines.append(f"min_frames: {min(lens)}")
            lines.append(f"max_frames: {max(lens)}")
            lines.append(f"mean_frames: {sum(lens) / len(lens):.3f}")

        if "per_file_scores" in results:
            per_file = results["per_file_scores"]
            lens = [len(v) for v in per_file.values()]
            lines.append("--- per_file frame stats ---")
            lines.append(f"num_files_with_frames: {len(lens)}")
            lines.append(f"min_frames: {min(lens)}")
            lines.append(f"max_frames: {max(lens)}")
            lines.append(f"mean_frames: {sum(lens) / len(lens):.3f}")

        txt_path = run_dir / f"analysis_{name}.txt"
        save_txt(txt_path, lines)

        print(f"[analysis] saved: {json_path}")
        print(f"[analysis] saved: {txt_path}")

        return {
            "run_dir": str(run_dir),
            "analysis_json": str(json_path),
            "analysis_txt": str(txt_path),
        }
