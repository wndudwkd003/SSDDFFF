from dataclasses import dataclass, field

"""
# model name
    - openai/clip-vit-large-patch14
    - facebook/convnextv2-large-22k-384
"""


@dataclass
class Config:
    # =========================
    # 1) 기본 실행/디버그 모드
    # =========================
    DEBUG_MODE: bool = False
    model_mode: str = "ce"  # ce | contrastive
    do_mode: str = "train"  # train | test
    seed: int = 42

    # =========================
    # 2) 데이터/경로 관련
    # =========================
    datasets_path: str = "/workspace/total_datasets/datasets"
    test_meta_csv_path: str = (
        "/workspace/preproc_runs/20260103_095554__pad0.5__vf6__qf1__keep5__landmark1__scrfd_faceonly__force1/meta.csv"
    )
    key_json_path: str = "keys.json"
    dacon_json_path: str = "dacon_info.json"

    # =========================
    # 3) 입력/특징(모달리티) 설정
    # =========================
    input_features: list[str] = field(  # rgb | wavelet | npr | residual
        default_factory=lambda: [
            "rgb",
        ]
    )
    use_augmentation: bool = False  # True | False
    SSDDFFF: bool = False  # True | False

    # =========================
    # 4) 모델/아키텍처 설정
    # =========================
    run_name: str = "clip_large"
    model_name: str = "openai/clip-vit-large-patch14"
    num_classes: int = 2
    image_size: int = 224

    # --- stage/head/backbone 동결 등 ---
    skip_stage1: bool = True  # True | False
    head: str = "svm"  # mlp | linear | svm
    freeze_backbone: bool = True  # True | False

    # --- 사전학습/체크포인트 ---
    pretrained: bool = True  # True | False
    pretrained_ckpt_path: str | None = (
        "/workspace/df40/df40_weights/train_on_df40/clip_large.pth"
    )

    # =========================
    # 5) 학습 하이퍼파라미터
    # =========================
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 10

    # =========================
    # 6) 스케줄러/조기종료
    # =========================
    scheduler: str = "cosine"  # cosine | linear | step
    early_stopping_patience: int | None = 5  # 5 | None
    early_stopping_delta: float = 1e-4

    # =========================
    # 7) 출력/실험 관리
    # =========================
    out_dir: str = "out"
    run_dir: str = "runs"
