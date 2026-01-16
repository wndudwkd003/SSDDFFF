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

        self.backbone = ConvNextV2Model.from_pretrained(config.model_name.value)

        in_ch = int(getattr(config, "input_channels", 3))
        if in_ch != 3:
            pe = self.backbone.embeddings.patch_embeddings.projection
            out_ch = pe.out_channels
            kh, kw = pe.kernel_size
            sh, sw = pe.stride
            ph, pw = pe.padding
            new = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=(kh, kw),
                stride=(sh, sw),
                padding=(ph, pw),
                bias=(pe.bias is not None),
            )

            with torch.no_grad():
                w = pe.weight
                if in_ch > 3:
                    new.weight[:, :3] = w
                    for c in range(3, in_ch):
                        new.weight[:, c : c + 1] = w[:, :1]
                else:
                    new.weight.copy_(w[:, :in_ch])
                if pe.bias is not None:
                    new.bias.copy_(pe.bias)

            self.backbone.embeddings.patch_embeddings.projection = new

        self.emb_dim = int(self.backbone.config.hidden_sizes[-1])

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


class ConvNeXtV2LargeDetector224(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.backbone = ConvNextV2Model.from_pretrained(config.model_name.value)

        in_ch = int(getattr(config, "input_channels", 3))
        if in_ch != 3:
            pe = self.backbone.embeddings.patch_embeddings.projection
            out_ch = pe.out_channels
            kh, kw = pe.kernel_size
            sh, sw = pe.stride
            ph, pw = pe.padding
            new = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=(kh, kw),
                stride=(sh, sw),
                padding=(ph, pw),
                bias=(pe.bias is not None),
            )

            with torch.no_grad():
                w = pe.weight
                if in_ch > 3:
                    new.weight[:, :3] = w
                    for c in range(3, in_ch):
                        new.weight[:, c : c + 1] = w[:, :1]
                else:
                    new.weight.copy_(w[:, :in_ch])
                if pe.bias is not None:
                    new.bias.copy_(pe.bias)

            self.backbone.embeddings.patch_embeddings.projection = new

        self.emb_dim = int(self.backbone.config.hidden_sizes[-1])

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
