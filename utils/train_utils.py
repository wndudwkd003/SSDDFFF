import torch
import torch.nn as nn
from config.config import Config
import matplotlib.pyplot as plt
from pathlib import Path
import os


def build_optim_and_scheduler(model: nn.Module, config: Config):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    if config.scheduler == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.num_epochs)
    elif config.scheduler == "linear":
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda e: max(0.0, 1.0 - (e / float(max(1, config.num_epochs)))),
        )
    elif config.scheduler == "step":
        sch = torch.optim.lr_scheduler.StepLR(
            opt, step_size=max(1, config.num_epochs // 3), gamma=0.1
        )
    else:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.num_epochs)

    return opt, sch


def update_loss_curve_image(
    train_losses: list[float], valid_losses: list[float], out_path: Path
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))

    fig = plt.figure()
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, valid_losses, label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()

    tmp = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    fig.savefig(tmp, dpi=160)
    plt.close(fig)
    os.replace(tmp, out_path)
