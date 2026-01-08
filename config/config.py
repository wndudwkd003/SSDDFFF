from dataclasses import dataclass, field
import sched


"""
# model name
    - openai/clip-vit-large-patch14
    - facebook/convnextv2-large-22k-384
"""


@dataclass
class Config:
    DEBUG_MODE: bool = False
    model_mode: str = "ce"  # ce | contrastive

    do_mode: str = "train"  # train | test

    test_meta_csv_path: str = (
        "/workspace/preproc_runs/20260103_095554__pad0.5__vf6__qf1__keep5__landmark1__scrfd_faceonly__force1/meta.csv"
    )

    use_augmentation: bool = False  # True | False

    SSDDFFF: bool = False  # True | False

    input_features: list[str] = field(  # rgb | wavelet | npr | residual
        default_factory=lambda: [
            "rgb",
        ]
    )

    out_dir: str = "out"
    key_json_path: str = "keys.json"

    model_name: str = "openai/clip-vit-large-patch14"

    head: str = "linear"  # linear | svm
    just_train_head: bool = True  # True | False

    pretrained: bool = True  # True | False
    pretrained_ckpt_path: str | None = ""  # "" | None

    image_size: int = 224

    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 10

    datasets_path: str = "/workspace/total_datasets/datasets"

    run_dir: str = "runs"

    seed: int = 42

    early_stopping_patience: int | None = 5  # 5 | None
    early_stopping_delta: float = 1e-4

    scheduler: str = "cosine"  # cosine | linear | step
