from pathlib import Path
import json
import shutil

MODE = 2  # 1 또는 2
INPUTS = [
    "datasets",
]
ROOT = None  # Path("/workspace") 같은 기준 루트. None이면 jsonl 부모 디렉토리 기준
BACKUP_DIR = Path("backup_jsonl")


def collect_jsonl_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.jsonl")))
        else:
            files.append(p)
    return files


def resolve_image_path(path_str: str, jsonl_path: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    base = ROOT if ROOT is not None else jsonl_path.parent
    return base / p


def check_jsonl(jsonl_path: Path) -> tuple[int, int]:
    total = 0
    missing = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            obj = json.loads(line)
            img_path = resolve_image_path(obj["path"], jsonl_path)
            total += 1
            if not img_path.exists():
                missing += 1
    return total, missing


def backup_file(src: Path) -> Path:
    src = src.resolve()
    rel = (
        src.relative_to(Path.cwd().resolve())
        if src.is_relative_to(Path.cwd().resolve())
        else src.as_posix().lstrip("/")
    )
    dst = (BACKUP_DIR / rel).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def clean_and_overwrite_jsonl(jsonl_path: Path) -> tuple[int, int, int, Path]:
    total = 0
    kept = 0
    dropped = 0

    tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")

    with jsonl_path.open("r", encoding="utf-8") as fin, tmp_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            obj = json.loads(line)
            img_path = resolve_image_path(obj["path"], jsonl_path)
            total += 1
            if img_path.exists():
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1

    backup_path = backup_file(jsonl_path)
    tmp_path.replace(jsonl_path)

    return total, kept, dropped, backup_path


def main() -> None:
    jsonl_files = collect_jsonl_files(INPUTS)

    grand_total = 0
    grand_missing = 0

    for jp in jsonl_files:
        if MODE == 1:
            total, missing = check_jsonl(jp)
            grand_total += total
            grand_missing += missing
            print(f"[MODE1] {jp}  total={total}  missing={missing}")
        else:
            total, kept, dropped, backup_path = clean_and_overwrite_jsonl(jp)
            grand_total += total
            grand_missing += dropped
            print(
                f"[MODE2] {jp}  total={total}  kept={kept}  dropped_missing={dropped}  backup={backup_path}"
            )

    print(
        f"\n[SUMMARY] files={len(jsonl_files)}  total={grand_total}  missing={grand_missing}"
    )


if __name__ == "__main__":
    main()
