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
    run_dir = Path(results["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    filenames = results["filenames"]
    preds = results["preds"]

    # CE: probs, AE: scores
    if "probs" in results:
        values = results["probs"]
        value_col = "prob"
        value_path = run_dir / "submission_prob.csv"
    else:
        values = results["scores"]
        value_col = "score"
        value_path = run_dir / "submission_score.csv"

    sub_path = run_dir / "submission.csv"
    df = pd.DataFrame({"filename": filenames, "label": preds})
    df.to_csv(sub_path, index=False)

    dfv = pd.DataFrame({"filename": filenames, value_col: values})
    dfv.to_csv(value_path, index=False, float_format="%.5f")

    print(f"[submit] saved: {sub_path}")
    print(f"[submit] saved: {value_path}")

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
