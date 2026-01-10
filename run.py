# /workspace/SSDDFF/run.py
from __future__ import annotations

from pathlib import Path

from config.config import Config, InputFeature, DatasetName
from utils.seed_utils import set_seeds
from utils.api_utils import initialize_api
from worker.train import Trainer
from worker.submit import submit_predictions
from worker.analysis import Analyzer, build_cross_dataset_grids
from utils.date_utils import get_current_timestamp


def get_input_ch(input_features: list[InputFeature]) -> int:
    ch = 0
    for f in input_features:
        if f == InputFeature.RGB:
            ch += 3
        elif f == InputFeature.NPR:
            ch += 3
    return ch


def main(config: Config):
    if config.do_mode == "train":
        analyzer = Analyzer()

        ts = get_current_timestamp()
        base_name = f"{ts}_{config.run_name}_{config.image_size}"
        base_run_dir = Path(config.run_dir) / base_name
        base_run_dir.mkdir(parents=True, exist_ok=True)

        if config.use_dataset_sum:
            run_dir = base_run_dir / DatasetName.SUM.value
            trainer = Trainer(config, train_dataset=DatasetName.SUM, run_dir=run_dir)

            train_results = trainer.train()
            test_results = trainer.test()

            analyzer.analyze(train_results, "train_SUM")
            analyzer.analyze(test_results, "test_SUM")

            build_cross_dataset_grids(
                runs_root=str(base_run_dir),
                out_dir=str(base_run_dir / "cross_grids"),
            )
            return

        for ds, _ in config.selected_datasets:
            run_dir = base_run_dir / ds.value
            trainer = Trainer(config, train_dataset=ds, run_dir=run_dir)

            train_results = trainer.train()
            test_results = trainer.test()

            analyzer.analyze(train_results, f"train_{ds.value}")
            analyzer.analyze(test_results, f"test_{ds.value}")

        build_cross_dataset_grids(
            runs_root=str(base_run_dir),
            out_dir=str(base_run_dir / "cross_grids"),
        )
        return

    if config.do_mode == "test":
        trainer = Trainer(config, train_dataset=DatasetName.SUM)
        results = trainer.test_for_submission()
        submit_predictions(config, results)
        return


if __name__ == "__main__":
    config = Config()
    config.input_channels = get_input_ch(config.input_features)
    set_seeds(config.seed)
    initialize_api(config.key_json_path)
    main(config)
