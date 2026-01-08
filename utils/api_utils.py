import json
import os


def initialize_api(key_json_path: str | None = None):
    if key_json_path is not None:
        with open(key_json_path, "r") as f:
            keys = json.load(f)

        for k, v in keys.items():
            os.environ[k] = v
            print(f"[api_utils.py] Set environment variable: {k}")

    else:
        print("[api_utils.py] No key_json_path provided, skipping API initialization.")
