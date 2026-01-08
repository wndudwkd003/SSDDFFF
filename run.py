# run.py

from config.config import Config
from utils.seed_utils import set_seeds
from utils.api_utils import initialize_api
from worker.train import Trainer
from worker.submit import submit_predictions
from worker.analysis import Analyzer


def main(config: Config):

    if config.do_mode == "train":
        trainer = Trainer(config)
        train_results = trainer.train()
        test_results = trainer.test()

        analyzer = Analyzer()
        analyzer.analyze(train_results, "train")
        analyzer.analyze(test_results, "test")

    elif config.do_mode == "test":
        trainer = Trainer(config)
        results = trainer.test_for_submission()
        submit_predictions(config, results)


if __name__ == "__main__":
    config = Config()
    set_seeds(config.seed)
    initialize_api(config.key_json_path)
    main(config)
