"""Candidate ForTy-to-U-Net interface, pending joint verification with the trainer."""

import torch


IGNORE_INDEX = -100
SEASONS = ("DJF", "MAM", "JJA", "SON")
BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
CLASS_NAMES = ("natural", "planted", "tree_crops", "other_veg", "built", "water", "ice", "bare")


def prepare_s2(s2_dn):
    """Raw DN [B,4,128,128,10] -> float32 [B,40,128,128], divide by 10000 once.

    Channel index = season_index * 10 + band_index. No clipping or masking.
    Apply cloud augmentation before flattening, under the team's shared protocol.
    """
    s2 = torch.as_tensor(s2_dn)
    if s2.ndim != 5 or tuple(s2.shape[1:]) != (4, 128, 128, 10):
        raise ValueError(f"Expected raw S2 [B,4,128,128,10], got {tuple(s2.shape)}")
    s2 = s2.to(dtype=torch.float32) / 10000.0
    return s2.permute(0, 1, 4, 2, 3).contiguous().reshape(s2.shape[0], 40, 128, 128)


def encode_labels(raw_labels):
    """Raw IDs 1..8 -> 0..7; unknown 0 -> -100. No input mutation."""
    raw = torch.as_tensor(raw_labels)
    if raw.is_floating_point() or raw.is_complex() or raw.dtype == torch.bool:
        raise ValueError("Raw labels must use an integer dtype")
    if bool(((raw < 0) | (raw > 8)).any()):
        raise ValueError("Raw labels must be in 0..8")
    raw = raw.long()
    return torch.where(raw == 0, IGNORE_INDEX, raw - 1)


def decode_predictions(class_indices):
    """U-Net indices 0..7 -> original ForTy IDs 1..8 for shared evaluation/export."""
    pred = torch.as_tensor(class_indices)
    if pred.is_floating_point() or pred.is_complex() or pred.dtype == torch.bool:
        raise ValueError("Predictions must use an integer dtype")
    if bool(((pred < 0) | (pred > 7)).any()):
        raise ValueError("U-Net prediction indices must be in 0..7")
    return pred.long() + 1
