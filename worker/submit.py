# worker/submit.py


from pathlib import Path
import pandas as pd

from config.config import Config
from dacon_submit_api import dacon_submit_api
import json
import os


def read_dacon_info(config: Config):
    with open(config.dacon_json_path, "r") as f:
        dacon_info = json.load(f)
    return dacon_info


def submit_predictions(config: Config, results: dict):
    """
    results expected keys (from Trainer.test_for_submission):
      - filenames: list[str]   (원본 샘플 단위)
      - probs:     list[float] (0~1)
      - preds:     list[int]   (0/1)  (없으면 probs로 생성 가능)
      - run_dir:   str         (저장 경로)
    저장 포맷:
      - submission.csv : filename, label
      - submission_prob.csv : filename, prob
    """
    run_dir = Path(results["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    filenames = results["filenames"]
    probs = results["probs"]

    if "preds" in results:
        preds = results["preds"]
    else:
        preds = [int(p >= 0.5) for p in probs]

    sub_path = run_dir / "submission.csv"
    df = pd.DataFrame({"filename": filenames, "label": preds})
    df.to_csv(sub_path, index=False)

    prob_path = run_dir / "submission_prob.csv"
    dfp = pd.DataFrame({"filename": filenames, "prob": probs})
    dfp.to_csv(prob_path, index=False)

    print(f"[submit] saved: {sub_path}")
    print(f"[submit] saved: {prob_path}")

    # dacon
    dacon_info = read_dacon_info(config)

    dacon_key = os.getenv(dacon_info["key_env"])
    competition_id = dacon_info["id"]
    team_name = dacon_info["team"]
    submission_memo = dacon_info["submission_memo"]

    if config.dacon_submit:
        result = dacon_submit_api.post_submission_file(
            str(sub_path), dacon_key, competition_id, team_name, submission_memo
        )

        print(f"[submit] dacon submission result: {result}")

    print("[submit] done.")
    return str(sub_path)
