"""ResNet-50 + CBAM backbone for Run 9.

Wraps torchvision `resnet50` (optionally with ImageNet pretrained
weights) and inserts a CBAM block after each of the four residual
stages — same placement as `ResNet18CBAM` in `model.py`, just on the
deeper backbone.

Stages output 256 / 512 / 1024 / 2048 channels; pooled feature vector
is 2048-d (vs 512-d for the Run 6/8 ResNet-18 backbone).

Exposes the same surface as ResNet18CBAM so it slots into
`model_twostream.PoseFusionCBAM` without further changes:
    - `features(x)`            -> (B, 2048)
    - `forward(x)`             -> (B, num_classes)
    - `last_spatial_attention()` -> (B, 1, H/32, W/32) SAM map for demo
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model import CBAM


class ResNet50CBAM(nn.Module):
    feat_dim: int = 2048

    def __init__(self, num_classes: int = 10, use_cbam: bool = True,
                 pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = resnet50(weights=weights)

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avgpool = net.avgpool

        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam1 = CBAM(256)
            self.cbam2 = CBAM(512)
            self.cbam3 = CBAM(1024)
            self.cbam4 = CBAM(2048)
        else:
            self.cbam1 = nn.Identity()
            self.cbam2 = nn.Identity()
            self.cbam3 = nn.Identity()
            self.cbam4 = nn.Identity()

        self.fc = nn.Linear(self.feat_dim, num_classes)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        return self.avgpool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))

    def last_spatial_attention(self) -> torch.Tensor | None:
        if not self.use_cbam:
            return None
        return self.cbam4.last_sam


def build_resnet50_cbam(num_classes: int = 10, use_cbam: bool = True,
                        pretrained: bool = True) -> ResNet50CBAM:
    return ResNet50CBAM(num_classes=num_classes, use_cbam=use_cbam,
                        pretrained=pretrained)
