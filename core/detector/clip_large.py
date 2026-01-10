# core/detector/clip_large.py

from __future__ import annotations

from config.config import Config

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPVisionModel

from core.projector.head import get_head


class CLIPLargeDetector(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.backbone = CLIPVisionModel.from_pretrained(config.model_name.value)

        in_ch = int(getattr(config, "input_channels", 3))
        if in_ch != 3:
            w = self.backbone.vision_model.embeddings.patch_embedding.weight
            out_ch, _, kh, kw = w.shape
            new = nn.Conv2d(
                in_ch, out_ch, kernel_size=(kh, kw), stride=(kh, kw), bias=False
            )

            with torch.no_grad():
                if in_ch > 3:
                    new.weight[:, :3] = w
                    for c in range(3, in_ch):
                        new.weight[:, c : c + 1] = w[:, :1]
                else:
                    new.weight.copy_(w[:, :in_ch])

            self.backbone.vision_model.embeddings.patch_embedding = new

        self.emb_dim = int(self.backbone.config.hidden_size)

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
        out = self.backbone(pixel_values=x, return_dict=True)

        emb = out.pooler_output
        emb = F.normalize(emb, dim=-1)

        if self.head is None:
            return {"embedding": emb}

        logits = self.head(emb)
        return {"logits": logits, "embedding": emb}
