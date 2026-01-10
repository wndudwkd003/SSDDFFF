# core/detector/xception_ae.py
from __future__ import annotations

from config.config import Config, ModelName
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvTBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class XceptionAE(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # 백본은 항상 xception 사용
        self.encoder = timm.create_model(
            ModelName.XCEPTION.value,  # "xception"
            pretrained=True,
            num_classes=0,
            global_pool="avg",
            in_chans=config.input_channels,
        )

        self.emb_dim = int(getattr(self.encoder, "num_features"))

        if getattr(config, "freeze_backbone", False):
            for p in self.encoder.parameters():
                p.requires_grad = False

        s0 = int(config.image_size) // 32
        self.s0 = s0
        base_ch = 512

        self.fc = nn.Linear(self.emb_dim, base_ch * s0 * s0)

        self.up1 = ConvTBlock(base_ch, base_ch // 2)
        self.up2 = ConvTBlock(base_ch // 2, base_ch // 4)
        self.up3 = ConvTBlock(base_ch // 4, base_ch // 8)
        self.up4 = ConvTBlock(base_ch // 8, base_ch // 16)
        self.up5 = ConvTBlock(base_ch // 16, base_ch // 16)

        self.to_img = nn.Conv2d(
            base_ch // 16, int(config.input_channels), kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        emb = self.encoder(x)
        if emb.dim() > 2:
            emb = torch.flatten(emb, 1)
        emb = F.normalize(emb, dim=-1)

        z = self.fc(emb)
        z = z.view(z.size(0), -1, self.s0, self.s0)
        z = self.up1(z)
        z = self.up2(z)
        z = self.up3(z)
        z = self.up4(z)
        z = self.up5(z)
        recon = self.to_img(z)

        return {"embedding": emb, "recon": recon}
