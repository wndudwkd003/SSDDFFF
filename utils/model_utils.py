# utils/model_utils.py

from config.config import Config

from core.detector.clip_large import CLIPLargeDetector
from core.detector.convnextv2 import ConvNeXtV2LargeDetector


def build_model(config: Config):
    if config.model_name == "openai/clip-vit-large-patch14":
        return CLIPLargeDetector(config)
    elif config.model_name == "facebook/convnextv2-large-22k-384":
        return ConvNeXtV2LargeDetector(config)
    else:
        raise ValueError(f"Unknown model name: {config.model_name}")
