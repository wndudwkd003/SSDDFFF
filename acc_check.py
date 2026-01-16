# labelwise_acc_with_plots.py
from __future__ import annotations

import json
import math
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt


# ======= 여기만 바꾸면 됩니다 =======
RUN_ROOT = Path("runs/20260114_172322_CLIP_VIT_LARGE_224_224")
GLOB_PATTERN = "analysis_test_*.json"
PRED_THRESHOLD = 0.5  # preds가 없을 때 probs로부터 preds 생성에 사용
OUT_PLOT_DIRNAME = "labelwise_plots"
# ====================================


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _is_nan(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _ensure_preds(obj: Dict[str, Any], threshold: float) -> List[int]:
    # 1) preds가 있으면 사용
    if "preds" in obj and obj["preds"] is not None:
        return [int(x) for x in obj["preds"]]

    # 2) preds가 없으면 probs로 생성
    probs = obj.get("probs", None)
    if probs is None:
        raise KeyError(
            "preds도 없고 probs도 없습니다. (analysis_test json 포맷 확인 필요)"
        )
    return [1 if _safe_float(p) >= threshold else 0 for p in probs]


def _compute_labelwise_acc(preds: List[int], labels: List[int]) -> Dict[str, Any]:
    if len(preds) != len(labels):
        raise ValueError(f"len(preds)={len(preds)} != len(labels)={len(labels)}")

    n = len(labels)
    correct = [1 if int(p) == int(y) else 0 for p, y in zip(preds, labels)]

    idx1 = [i for i, y in enumerate(labels) if int(y) == 1]
    idx0 = [i for i, y in enumerate(labels) if int(y) == 0]

    def acc_on(idxs: List[int]) -> float:
        if not idxs:
            return float("nan")
        return sum(correct[i] for i in idxs) / len(idxs)

    return {
        "n": n,
        "acc": (sum(correct) / n) if n > 0 else float("nan"),
        "n1": len(idx1),
        "acc_label1": acc_on(idx1),
        "n0": len(idx0),
        "acc_label0": acc_on(idx0),
    }


def analyze_one_file(path: Path, threshold: float) -> Dict[str, Any]:
    j = json.loads(path.read_text(encoding="utf-8"))
    tests_all: Dict[str, Dict[str, Any]] = j.get("tests_all", {})

    per_test: Dict[str, Any] = {}
    for test_name, obj in tests_all.items():
        labels = obj.get("labels", None)
        if labels is None:
            raise KeyError(f"[{path}] tests_all['{test_name}']에 labels가 없습니다.")

        preds = _ensure_preds(obj, threshold=threshold)
        metrics = _compute_labelwise_acc(preds=preds, labels=[int(x) for x in labels])
        metrics["loss"] = _safe_float(obj.get("loss", float("nan")))
        per_test[test_name] = metrics

    out = {
        "source_file": str(path),
        "run_dir": j.get("run_dir"),
        "train_dataset": j.get("train_dataset"),
        "test_order": j.get("test_order"),
        "threshold": threshold,
        "tests": per_test,
    }
    return out


def _format_acc(x: float) -> str:
    if _is_nan(x):
        return "nan"
    return f"{x:.4f}"


def save_plot_filewise(
    *,
    out_dir: Path,
    rel_key: str,
    test_name: str,
    metrics: Dict[str, Any],
) -> Path:
    """
    파일 1개 + 테스트셋 1개에 대해
    overall / label0 / label1 정확도 막대그래프 저장
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = float(metrics.get("acc", float("nan")))
    acc0 = float(metrics.get("acc_label0", float("nan")))
    acc1 = float(metrics.get("acc_label1", float("nan")))
    n = int(metrics.get("n", 0))
    n0 = int(metrics.get("n0", 0))
    n1 = int(metrics.get("n1", 0))

    labels = ["overall", "label0", "label1"]
    values = [acc, acc0, acc1]

    fig = plt.figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    bars = ax.bar(labels, [0.0 if _is_nan(v) else v for v in values])

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{test_name} | {rel_key}\n(n={n}, n0={n0}, n1={n1})")

    # 막대 위 숫자 표시
    for b, v in zip(bars, values):
        h = b.get_height()
        txt = _format_acc(v)
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            min(1.0, h + 0.02),
            txt,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    safe_rel = rel_key.replace("/", "__").replace("\\", "__")
    out_path = out_dir / f"filewise__{safe_rel}__{test_name}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_plot_aggregate_testwise(
    *,
    out_dir: Path,
    test_name: str,
    items: List[Dict[str, Any]],
) -> Path:
    """
    RUN_ROOT 아래 모든 파일을 모아서 test_name 기준으로
    실험(파일)별 overall/label0/label1 그룹 막대그래프 저장
    items: [{"rel":..., "acc":..., "acc0":..., "acc1":..., "n":..., "n0":..., "n1":...}, ...]
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rels = [it["rel"] for it in items]
    acc = [it["acc"] for it in items]
    acc0 = [it["acc0"] for it in items]
    acc1 = [it["acc1"] for it in items]

    # x축 라벨이 길어질 수 있으니 줄바꿈/축 회전
    x = list(range(len(rels)))
    width = 0.25

    fig = plt.figure(figsize=(max(8.0, 0.6 * len(rels)), 4.5))
    ax = fig.add_subplot(111)

    # NaN은 0으로 그리되 텍스트는 nan으로 표시
    def _nan_to_zero(arr: List[float]) -> List[float]:
        return [0.0 if _is_nan(v) else float(v) for v in arr]

    b1 = ax.bar([i - width for i in x], _nan_to_zero(acc), width=width, label="overall")
    b2 = ax.bar(x, _nan_to_zero(acc0), width=width, label="label0")
    b3 = ax.bar([i + width for i in x], _nan_to_zero(acc1), width=width, label="label1")

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Aggregate label-wise accuracy | {test_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(rels, rotation=30, ha="right", fontsize=8)
    ax.legend()

    # 각 막대에 값 표시
    def _annotate(bars, values: List[float]):
        for b, v in zip(bars, values):
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                min(1.0, h + 0.02),
                _format_acc(v),
                ha="center",
                va="bottom",
                fontsize=7,
            )

    _annotate(b1, acc)
    _annotate(b2, acc0)
    _annotate(b3, acc1)

    fig.tight_layout()
    out_path = out_dir / f"aggregate__{test_name}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    run_root = RUN_ROOT.resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"RUN_ROOT가 존재하지 않습니다: {run_root}")

    files = sorted(run_root.rglob(GLOB_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"{run_root} 아래에서 '{GLOB_PATTERN}' 파일을 찾지 못했습니다."
        )

    plot_dir = run_root / OUT_PLOT_DIRNAME
    generated_at = datetime.now().isoformat(timespec="seconds")

    summary: Dict[str, Any] = {
        "run_root": str(run_root),
        "generated_at": generated_at,
        "glob": GLOB_PATTERN,
        "threshold": PRED_THRESHOLD,
        "files": {},
        "plots": {
            "plot_dir": str(plot_dir),
            "filewise": [],
            "aggregate": [],
        },
    }

    # aggregate용 수집: test_name별 list
    agg_bucket: Dict[str, List[Dict[str, Any]]] = {}

    for f in files:
        one = analyze_one_file(f, threshold=PRED_THRESHOLD)

        # (1) 각 파일 옆에 개별 결과 저장
        out_json = f.with_name(
            "analysis_labelwise_"
            + f.name.replace("analysis_test_", "").replace(".json", "")
            + ".json"
        )
        out_json.write_text(
            json.dumps(one, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        rel = str(f.relative_to(run_root))

        # (2) 파일별 plot 저장 + aggregate 수집
        for test_name, m in one["tests"].items():
            acc = float(m.get("acc", float("nan")))
            acc0 = float(m.get("acc_label0", float("nan")))
            acc1 = float(m.get("acc_label1", float("nan")))
            n = int(m.get("n", 0))
            n0 = int(m.get("n0", 0))
            n1 = int(m.get("n1", 0))

            p = save_plot_filewise(
                out_dir=plot_dir, rel_key=rel, test_name=test_name, metrics=m
            )
            summary["plots"]["filewise"].append(str(p))

            agg_bucket.setdefault(test_name, []).append(
                {
                    "rel": rel,
                    "acc": acc,
                    "acc0": acc0,
                    "acc1": acc1,
                    "n": n,
                    "n0": n0,
                    "n1": n1,
                }
            )

        # (3) summary 누적
        summary["files"][rel] = {
            "saved_json": str(out_json),
            "run_dir": one.get("run_dir"),
            "train_dataset": one.get("train_dataset"),
            "tests": one["tests"],
        }

    # (4) test_name별 aggregate plot 저장
    for test_name, items in agg_bucket.items():
        # rel 기준 정렬 (일관성)
        items = sorted(items, key=lambda d: d["rel"])
        p = save_plot_aggregate_testwise(
            out_dir=plot_dir, test_name=test_name, items=items
        )
        summary["plots"]["aggregate"].append(str(p))

    # (5) RUN_ROOT에 전체 요약 저장
    summary_path = run_root / "labelwise_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] found={len(files)}")
    print(f"[OK] summary saved: {summary_path}")
    print(f"[OK] plots saved under: {plot_dir}")


if __name__ == "__main__":
    main()
