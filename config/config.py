from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from regex import T


class ModelName(Enum):
    CLIP_VIT_LARGE_224 = "openai/clip-vit-large-patch14"
    CONVNEXTV2_LARGE_384 = "facebook/convnextv2-large-22k-384"
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
    RVF_FULL = "RVF_Full"
    FFPP_C23 = "FFPP_C23"
    DF40_EFS = "DF40_EFS"
    DF40_FE = "DF40_FE"
    DF40_FR = "DF40_FR"
    DF40_FS = "DF40_FS"


SELECTED_DATASES = [
    (DatasetName.DF40_EFS, 1.0),
    (DatasetName.DF40_FE, 1.0),
    (DatasetName.DF40_FR, 1.0),
    (DatasetName.DF40_FS, 1.0),
]


EVALUATE_DATASES = [
    (DatasetName.DF40_EFS, 1.0),
    (DatasetName.DF40_FE, 1.0),
    (DatasetName.DF40_FR, 1.0),
    (DatasetName.DF40_FS, 1.0),
]


# SELECTED_DATASES = [
#     (DatasetName.FFPP_C23, 1.0),
# ]

# EVALUATE_DATASES = [
#     (DatasetName.REAL, 1.0),
#     (DatasetName.DF40, 1.0),
#     (DatasetName.KODF, 1.0),
#     (DatasetName.CELEB_A_FAKE, 1.0),
#     (DatasetName.TIMIT_RVF, 1.0),
#     (DatasetName.VFBR, 1.0),
#     (DatasetName.FFPP_C23, 1.0),
# ]


class InputPatchType(Enum):
    CENTER_ZOOM = "center_zoom"
    OUTTER_FEATURE_ZOOM = "outter_feature_zoom"


INPUT_PATCHES = [
    # InputPatchType.CENTER_ZOOM,
    # InputPatchType.OUTTER_FEATURE_ZOOM,
]

# /workspace/df40/df40_weights/train_on_df40/clip_large.pth


@dataclass
class Config:
    DEBUG_MODE: bool = False  # True | False
    model_mode: str = "ce"  # "ce" | "ae"
    ae_normal: Literal["real", "fake"] = "fake"
    input_patches: list[InputPatchType] = field(default_factory=lambda: INPUT_PATCHES)

    input_patch_preview_max_items: int = 8
    do_mode: str = "train"  # train | test
    use_dataset_sum: bool = False  # True | False
    test_dir: str | None = None
    seed: int = 42
    datasets_path: str = "datasets"
    selected_datasets: list[tuple[DatasetName, float]] = field(
        default_factory=lambda: SELECTED_DATASES
    )
    evaluate_datasets: list[tuple[DatasetName, float]] = field(
        default_factory=lambda: EVALUATE_DATASES
    )
    test_meta_csv_path: str = (
        "/workspace/preproc_runs/20260103_095554__pad0.5__vf6__qf1__keep5__landmark1__scrfd_faceonly__force1/meta.csv"
    )
    key_json_path: str = "keys.json"
    dacon_json_path: str = "dacon_info.json"
    dacon_submit: bool = False
    input_features: list[InputFeature] = field(
        default_factory=lambda: [
            InputFeature.RGB,
            InputFeature.NPR,
            InputFeature.WAVELET,
        ]
    )
    input_channels: int = -1
    # use_augmentation: bool = False
    SSDDFF: bool = False
    SSDDFF_mode: str = "stage1_convnextv2_train"
    run_name: str = ModelName.CLIP_VIT_LARGE_224.name
    model_name: ModelName = ModelName.CLIP_VIT_LARGE_224
    num_classes: int = 2
    image_size: int = 224
    probs_threshold: float = 0.5
    skip_stage1: bool = False
    head: str = "linear"
    freeze_backbone: bool = False
    pretrained: bool = True
    pretrained_ckpt_path: str | None = (
        "/workspace/df40/df40_weights/train_on_df40/clip_large.pth"
    )
    num_epochs: int = 5
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    early_stopping_patience: int | None = 2
    early_stopping_delta: float = 1e-4
    out_dir: str = "out"
    run_dir: str = "runs"
    recon_preview_max_items: int = 16
    correct_wrong_max_total: int = 20
