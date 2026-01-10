from __future__ import annotations

from config.config import Config

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from core.projector.head import get_head


class XceptionDetector(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.backbone = timm.create_model(
            config.model_name.value,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
            in_chans=config.input_channels,
        )

        self.emb_dim = int(getattr(self.backbone, "num_features"))

        self.head: nn.Module | None = None
        if config.head != "svm":
            self.head = get_head(
                head=config.head,
                input_dim=self.emb_dim,
                output_dim=config.num_classes,
            )

        if getattr(config, "freeze_backbone", False):
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        emb = self.backbone(x)

        if emb.dim() > 2:
            emb = torch.flatten(emb, 1)

        emb = F.normalize(emb, dim=-1)

        if self.head is None:
            return {"embedding": emb}

        logits = self.head(emb)
        return {"logits": logits, "embedding": emb}
