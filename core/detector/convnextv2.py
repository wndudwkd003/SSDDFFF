from __future__ import annotations

from config.config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ConvNextV2Model

from core.projector.head import get_head


class ConvNeXtV2LargeDetector(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # feature extractor backbone (no classifier head)
        self.backbone = ConvNextV2Model.from_pretrained(config.model_name)

        self.emb_dim = int(self.backbone.config.hidden_sizes[-1])

        self.head: nn.Module | None = None
        if config.head != "svm":
            self.head = get_head(
                head=config.head,
                input_dim=self.emb_dim,
                output_dim=config.num_classes,
            )

        if config.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.backbone(pixel_values=x, return_dict=True)

        emb = out.pooler_output

        emb = F.normalize(emb, dim=-1)

        if self.head is None:
            return {"embedding": emb}

        logits = self.head(emb)
        return {"logits": logits, "embedding": emb}
