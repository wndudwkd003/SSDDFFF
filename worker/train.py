# worker/train.py

from config.config import Config


class Trainer:
    def __init__(
        self,
        config: Config,
    ):
        self.config = config

    def train(self):
        pass

    def test(self):
        pass

    def test_for_submission(self):
        pass
