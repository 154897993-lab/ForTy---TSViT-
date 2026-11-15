"""Independent model factory; no historical training/Sobel code is imported."""
from forest_unet import ForestUNet

GROUPS = {
    "A": ("clean", False), "D": ("clean", True),
    "B_ind": ("independent", False), "B_ind+D": ("independent", True),
    "B_ovl": ("overlap", False), "B_ovl+D": ("overlap", True),
}


def build(group, base_channels=16):
    if group not in GROUPS:
        raise ValueError(f"Unknown group {group}; choose from {tuple(GROUPS)}")
    return ForestUNet(base_channels=base_channels, use_edge=GROUPS[group][1])
