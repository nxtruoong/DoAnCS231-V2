"""Two-stream ResNet-18 + CBAM for Run 7.

Architecture:
    full image  -> ResNet18CBAM.features() -> 512-d
    top crop    -> ResNet18CBAM.features() -> 512-d
    concat -> Linear(1024 -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 10)

Two streams have *separate* weights (not shared) so each can specialise
on its own field of view: full stream learns hand+steering+cabin
context, face stream learns gaze + head pose + hand-to-head proximity.

No face detector — top stream takes a fixed top-N% crop of the frame.
Works because State Farm dashcam is fixed and the driver's head is
always in the upper portion. See augment_twostream.TopCrop.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model import build_model
from model_resnet50 import build_resnet50_cbam


def _build_backbone(backbone: str, num_classes: int, use_cbam: bool,
                    pretrained: bool) -> tuple[nn.Module, int]:
    """Return (backbone_module, feature_dim) for the requested arch."""
    if backbone == "resnet18":
        return build_model(num_classes=num_classes, use_cbam=use_cbam), 512
    if backbone == "resnet50":
        return (build_resnet50_cbam(num_classes=num_classes, use_cbam=use_cbam,
                                    pretrained=pretrained), 2048)
    raise ValueError(f"unknown backbone: {backbone}")


class TwoStreamCBAM(nn.Module):
    def __init__(self, num_classes: int = 10, use_cbam: bool = True,
                 hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.full_stream = build_model(num_classes=num_classes, use_cbam=use_cbam)
        self.face_stream = build_model(num_classes=num_classes, use_cbam=use_cbam)
        # final fc on each backbone is unused; we read .features() instead.
        # Keep it (frees no memory) but it never sees gradient via this path.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x_full: torch.Tensor, x_face: torch.Tensor) -> torch.Tensor:
        f_full = self.full_stream.features(x_full)   # (B, 512)
        f_face = self.face_stream.features(x_face)   # (B, 512)
        fused = torch.cat([f_full, f_face], dim=1)   # (B, 1024)
        return self.classifier(fused)

    def last_spatial_attention(self) -> torch.Tensor | None:
        """Return SAM map from the full stream (for demo heatmap).

        Face stream attention is intentionally not exposed — top crop
        already restricts the spatial field, so its SAM map is less
        informative.
        """
        return self.full_stream.last_spatial_attention()


def build_twostream(num_classes: int = 10, use_cbam: bool = True) -> TwoStreamCBAM:
    return TwoStreamCBAM(num_classes=num_classes, use_cbam=use_cbam)


# --- Run 8 pose-fusion (full CNN + rich pose vector) -----------------------

class PoseFusionCBAM(nn.Module):
    """Single ResNet18+CBAM + rich-pose MLP fusion.

    Pose vector is a 36-d body-pose descriptor precomputed via
    MediaPipe Pose: head (8) + wrists (8) + elbows (4) + fingers (4) +
    hips (4) + derived (4) + visibility gates (4). MLP projects it to
    128-d before concatenation with the 512-d CNN features.

    The dedicated hand-crop stream from the original Run 8 plan was
    dropped: wrist + finger landmarks already encode hand position,
    making a second CNN backbone redundant and overfit-prone.
    """

    def __init__(self, num_classes: int = 10, use_cbam: bool = True,
                 pose_dim: int = 36, pose_hidden: int = 128,
                 fusion_hidden: int = 256, dropout: float = 0.3,
                 backbone: str = "resnet18", pretrained: bool = False):
        super().__init__()
        self.full_stream, feat_dim = _build_backbone(
            backbone, num_classes=num_classes, use_cbam=use_cbam,
            pretrained=pretrained,
        )
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, pose_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(pose_hidden, pose_hidden),
        )
        fused_dim = feat_dim + pose_hidden
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, x_full: torch.Tensor,
                x_pose: torch.Tensor) -> torch.Tensor:
        f_full = self.full_stream.features(x_full)
        f_pose = self.pose_proj(x_pose)
        fused = torch.cat([f_full, f_pose], dim=1)
        return self.classifier(fused)

    def last_spatial_attention(self) -> torch.Tensor | None:
        return self.full_stream.last_spatial_attention()


def build_posefusion(num_classes: int = 10, use_cbam: bool = True,
                     pose_dim: int = 36, backbone: str = "resnet18",
                     pretrained: bool = False) -> PoseFusionCBAM:
    return PoseFusionCBAM(num_classes=num_classes, use_cbam=use_cbam,
                          pose_dim=pose_dim, backbone=backbone,
                          pretrained=pretrained)
