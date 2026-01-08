# utils/model_utils.py

from config.config import Config

from core.detector.clip_large import CLIPLargeDetector


def build_model(config: Config):
    if config.model_name == "openai/clip-vit-large-patch14":
        return CLIPLargeDetector(config)
    else:
        raise ValueError(f"Unknown model name: {config.model_name}")
