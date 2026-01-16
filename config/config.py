from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from regex import T


class ModelName(Enum):
    CLIP_VIT_LARGE_224 = "openai/clip-vit-large-patch14"
    CONVNEXTV2_LARGE_384 = "facebook/convnextv2-large-22k-384"
    CONVNEXTV2_BASE_224 = "facebook/convnextv2-base-22k-224"
    XCEPTION = "xception"
    XCEPTION_AE = "xception_ae"


class InputFeature(Enum):
    RGB = "rgb"
    WAVELET = "wavelet"
    NPR = "npr"
    RESIDUAL = "residual"


class DatasetName(Enum):
    SUM = "SUM"
    KODF = "KoDF"
    RVF = "RVF"
    CELEB_A_FAKE = "CelebAFake"
    TIMIT_RVF = "TIMIT_RVF"
    REAL = "REAL"
    DF40 = "DF40"
    KODF_FULL = "KoDF_Full"
    VFBR = "VFBR"
    RVF_FULL = "RVF_FULL"
    FFPP_C23 = "FFPP_C23"
    DF40_EFS = "DF40_EFS"
    DF40_FE = "DF40_FE"
    DF40_FR = "DF40_FR"
    DF40_FS = "DF40_FS"


SELECTED_DATASES = [
    # (DatasetName.DF40_EFS, 0.5),
    # (DatasetName.DF40_FE, 0.5),
    # (DatasetName.DF40_FR, 0.5),
    (DatasetName.DF40_FS, 0.5),
]


EVALUATE_DATASES = [
    # (DatasetName.DF40_EFS, 0.5),
    # (DatasetName.DF40_FE, 0.5),
    # (DatasetName.DF40_FR, 0.5),
    (DatasetName.DF40_FS, 0.5),
]


class InputPatchType(Enum):
    CENTER_ZOOM = "center_zoom"
    OUTTER_FEATURE_ZOOM = "outter_feature_zoom"


INPUT_PATCHES = [
    InputPatchType.CENTER_ZOOM,
    # InputPatchType.OUTTER_FEATURE_ZOOM,
]

USE_MODEL = ModelName.XCEPTION  # <<<<<<<<<<<<<<<<<<<<<<<< use model


@dataclass
class Config:
    DEBUG_MODE: bool = False  # True | False
    model_mode: str = "ce"  # "ce" | "ae"
    ae_normal: Literal["real", "fake"] = "real"
    input_patches: list[InputPatchType] = field(default_factory=lambda: INPUT_PATCHES)
    data_label_invert: bool = False  # True | False
    logits_invert: bool = False  # True | False
    input_patch_preview_max_items: int = 8
    do_mode: str = (
        "train"  # train | test | test_submission <<<<<<<<<<<<<<<<<<<<< do mode
    )
    use_dataset_sum: bool = False  # True | False
    test_dir: str | None = (
        "runs/20260116_092727_XCEPTION_224"  # "" | None  <<<<<<<<<<<<<<<<<<<<<<< test dir
    )
    pad_fraction: float = 0.0  # 0.25
    pretrained: bool = (
        False  # True | False # <<<<<<<<<<<<<<<<<<<<<< pretrained False | True
    )
    pretrained_ckpt_path: (
        str | None
    ) = (  # pretrained_ckpt_path "/workspace/df40/df40_weights/train_on_df40/clip_large.pth"
        "runs/20260116_092727_XCEPTION_224/DF40_FE/best_stage1.pth"
    )
    seed: int = 42
    datasets_path: str = "datasets"
    selected_datasets: list[tuple[DatasetName, float]] = field(
        default_factory=lambda: SELECTED_DATASES
    )
    evaluate_datasets: list[tuple[DatasetName, float]] = field(
        default_factory=lambda: EVALUATE_DATASES
    )
    test_meta_csv_path: str = (
        "/workspace/preproc_runs/20260116_053754__primarydlib__croptight__pad0.2__vf6__keep5/meta.csv"
    )
    key_json_path: str = "keys.json"
    dacon_json_path: str = "dacon_info.json"
    dacon_submit: bool = False
    input_features: list[InputFeature] = field(
        default_factory=lambda: [
            InputFeature.RGB,
            # InputFeature.NPR,
            # InputFeature.WAVELET,
        ]
    )
    input_channels: int = -1
    use_augmentation: bool = False  # <<<<<<<<<<<<<<<< augmentation False | True
    SSDDFF: bool = False
    SSDDFF_mode: str = "stage1_convnextv2_train"
    run_name: str = USE_MODEL.name
    model_name: ModelName = USE_MODEL
    num_classes: int = 2
    image_size: int = 224
    probs_threshold: float = 0.5
    skip_stage1: bool = False
    head: str = "linear"
    freeze_backbone: bool = False

    num_epochs: int = 30
    batch_size: int = 64  # <<<<<<<<<<<<<<<<<<<<<<<<<<<< batch size 16
    lr: float = 0.0001
    weight_decay: float = 1e-2
    scheduler: str = "cosine"
    early_stopping_patience: int | None = 5
    early_stopping_delta: float = 1e-4
    out_dir: str = "out"
    run_dir: str = "runs"
    recon_preview_max_items: int = 16
    correct_wrong_max_total: int = 20
