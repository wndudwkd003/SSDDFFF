# utils/save_config.py
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml  # pyyaml 설치되어 있다고 가정

from config.config import Config


def _convert(obj: Any) -> Any:
    # JSON/YAML 직렬화를 위해 최소 변환만 수행
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    # Enum -> value 우선
    if hasattr(obj, "value"):
        return _convert(obj.value)

    if is_dataclass(obj):
        return _convert(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_convert(x) for x in obj]

    # 나머지는 그대로 문자열화하지 말고 그대로 두지 않으면 JSON에서 터질 수 있음
    # (방어적 코드 제거 요청이므로, 여기서는 str로 변환만 하겠습니다.)
    return str(obj)


def save_config(config: Config, run_dir: Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    data = asdict(config) if is_dataclass(config) else vars(config)
    data = _convert(data)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
