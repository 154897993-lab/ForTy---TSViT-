"""Independent model factory; no historical training/Sobel code is imported."""
import hashlib

from forest_unet import ForestUNet

GROUPS = {
    "A": ("clean", False), "D": ("clean", True),
    "B_ind": ("independent", False), "B_ind+D": ("independent", True),
    "B_ovl": ("overlap", False), "B_ovl+D": ("overlap", True),
    # New groups use the same U-Net as A (no decoder refinement).
    "G": ("erase", False),
    "B_mul": ("independent", False),
}

# Ruan's shared evaluator uses these exact group keys in checkpoint provenance.
DELIVERY_GROUPS = {
    "A": "A", "D": "D", "B_ind": "B_ind", "B_ind+D": "B_ind_D",
    "B_ovl": "B_ovl", "B_ovl+D": "B_ovl_D",
    "G": "G", "B_mul": "B_mul",
}

# P2 core matrix and the single-seed variant are kept separate.
CORE_GROUPS = ("A", "D", "B_ind", "B_ind+D", "B_ovl", "B_ovl+D", "G")
VARIANT_GROUPS = ("B_mul",)

GROUP_VARIANTS = {"B_mul": "multiplicative", "G": "rect_erase"}


def delivery_group(group):
    if group not in DELIVERY_GROUPS:
        raise ValueError(f"Unknown group {group}")
    return DELIVERY_GROUPS[group]


def training_variant(group):
    """Distinguish clean, additive cloud, multiplicative cloud and G erasing."""
    if group not in GROUPS:
        raise ValueError(f"Unknown group {group}")
    mode = GROUPS[group][0]
    if mode == "clean":
        return "none"
    if mode == "erase":
        return "rect_erase"
    return GROUP_VARIANTS.get(group, "additive")


def build(group, base_channels=16):
    if group not in GROUPS:
        raise ValueError(f"Unknown group {group}; choose from {tuple(GROUPS)}")
    return ForestUNet(base_channels=base_channels, use_edge=GROUPS[group][1])


def trunk_hash(model):
    """Hash shared-trunk initialization exactly like Ruan's evaluator contract."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith("refinement."):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
