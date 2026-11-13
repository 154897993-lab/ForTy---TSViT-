"""Lightweight U-Net and optional decoder refinement for A/B/D/B+D.

No data reader or cloud augmentation is included. Outputs are raw logits.
"""

import torch
from torch import nn
from torch.nn import functional as F


MODEL_VERSION = "unet_v0.1"


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((x, skip), dim=1))


class DecoderRefinement(nn.Module):
    """Fuse decoder features at 1/4 and 1/2 resolution into residual logits."""

    def __init__(self, base_channels, num_classes):
        super().__init__()
        self.quarter_projection = nn.Conv2d(4 * base_channels, base_channels, 1)
        self.half_projection = nn.Conv2d(2 * base_channels, base_channels, 1)
        self.fuse = DoubleConv(2 * base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, quarter, half, output_size):
        low = self.quarter_projection(quarter)
        low = F.interpolate(low, size=half.shape[-2:], mode="bilinear", align_corners=False)
        high = self.half_projection(half)
        residual = self.head(self.fuse(torch.cat((low, high), dim=1)))
        return F.interpolate(residual, size=output_size, mode="bilinear", align_corners=False)


class ForestUNet(nn.Module):
    """Input [B,40,H,W], output [B,8,H,W]; H/W must be divisible by 16.

    Create separate instances with use_edge=False (A/B) or True (D/B+D).
    Refinement has no auxiliary edge labels, edge loss, or cloud logic.
    """

    def __init__(self, in_channels=40, num_classes=8, base_channels=16, use_edge=False):
        super().__init__()
        if min(in_channels, num_classes, base_channels) < 1:
            raise ValueError("Channel counts must be positive")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.use_edge = use_edge
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, 2 * c)
        self.enc3 = DoubleConv(2 * c, 4 * c)
        self.enc4 = DoubleConv(4 * c, 8 * c)
        self.bottleneck = DoubleConv(8 * c, 16 * c)
        self.pool = nn.MaxPool2d(2)
        self.up4 = Up(16 * c, 8 * c, 8 * c)
        self.up3 = Up(8 * c, 4 * c, 4 * c)
        self.up2 = Up(4 * c, 2 * c, 2 * c)
        self.up1 = Up(2 * c, c, c)
        self.head = nn.Conv2d(c, num_classes, 1)
        # Construct after the shared trunk so a reset seed gives identical trunk initialization.
        self.refinement = DecoderRefinement(c, num_classes) if use_edge else None

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected [B,{self.in_channels},H,W], got {tuple(x.shape)}")
        if min(x.shape[-2:]) < 16 or any(s % 16 for s in x.shape[-2:]):
            raise ValueError("Spatial dimensions must be at least 16 and divisible by 16")
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        z = self.bottleneck(self.pool(e4))
        d4 = self.up4(z, e4)
        quarter = self.up3(d4, e3)
        half = self.up2(quarter, e2)
        full = self.up1(half, e1)
        logits = self.head(full)
        if self.refinement is not None:
            logits = logits + self.refinement(quarter, half, logits.shape[-2:])
        return logits

    def config(self):
        return {
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "base_channels": self.base_channels,
            "use_edge": self.use_edge,
        }

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        refinement = sum(p.numel() for p in self.refinement.parameters()) if self.refinement else 0
        trunk = total - refinement
        return {
            "trunk": trunk,
            "refinement": refinement,
            "total": total,
            "refinement_over_trunk": refinement / trunk,
        }
