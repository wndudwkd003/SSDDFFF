# core/detector/clip_large.py


from config.config import Config
import torch.nn as nn
from transformers import CLIPVisionModel
from core.projector.mlp import MLP

from core.projector.head import get_head
import torch

import torch.nn.functional as F


class CLIPLargeDetector(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()

        self.config = config

        self.backbone = CLIPVisionModel.from_pretrained(
            config.model_name,
        )

        self.emb_dim = self.backbone.config.hidden_size

        self.head = None
        if config.head != "svm":
            self.head = get_head(
                head=config.head,
                input_dim=self.emb_dim,
                output_dim=config.num_classes,
            )

        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x, return_dict=True)

        emb = out.pooler_output
        emb = F.normalize(emb, dim=-1)

        if self.head is None:
            return {"embedding": emb}

        logits = self.head(emb)
        return {"logits": logits, "embedding": emb}
