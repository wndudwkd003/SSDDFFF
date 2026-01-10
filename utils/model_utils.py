# utils/model_utils.py

from config.config import Config, ModelName

from core.detector.clip_large import CLIPLargeDetector
from core.detector.convnextv2 import ConvNeXtV2LargeDetector
from core.detector.xception import XceptionDetector
from core.detector.xception_ae import XceptionAE


def build_model(config: Config):
    if config.model_name == ModelName.CLIP_VIT_LARGE_224:
        return CLIPLargeDetector(config)
    elif config.model_name == ModelName.CONVNEXTV2_LARGE_384:
        return ConvNeXtV2LargeDetector(config)
    elif config.model_name == ModelName.XCEPTION:
        return XceptionDetector(config)
    elif config.model_name == ModelName.XCEPTION_AE:
        return XceptionAE(config)
