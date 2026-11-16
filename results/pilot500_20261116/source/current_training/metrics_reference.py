"""Reference metric for trainer review, accepting original ForTy IDs 0..8.

For U-Net predictions call decode_predictions first. A 9-output TSViT can pass
argmax IDs directly. Ignore unknown targets, but keep unknown predictions as FN.
"""

import torch


def confusion_matrix(pred_raw, target_raw):
    pred = torch.as_tensor(pred_raw)
    target = torch.as_tensor(target_raw, device=pred.device)
    if pred.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    for values in (pred, target):
        if values.is_floating_point() or values.is_complex() or values.dtype == torch.bool:
            raise ValueError("Raw IDs must use an integer dtype")
        if bool(((values < 0) | (values > 8)).any()):
            raise ValueError("Raw IDs must be in 0..8")
    valid = target != 0
    bins = target[valid].long() * 9 + pred[valid].long()
    return torch.bincount(bins, minlength=81).reshape(9, 9)


def summarize(cm):
    """Return ratios in [0,1]; absent classes use 0, with support also returned.

    Sum confusion matrices across batches before calling this function.
    Never average per-batch F1 scores as the dataset-level score.
    """
    cm = torch.as_tensor(cm, dtype=torch.float64)
    if cm.shape != (9, 9) or not bool(torch.isfinite(cm).all()) or bool((cm < 0).any()):
        raise ValueError("Expected a finite nonnegative 9x9 confusion matrix")
    cm = cm.clone()
    cm[0, :] = 0  # ignored target pixels cannot contribute FP
    tp = cm.diagonal()[1:]
    support = cm[1:, :].sum(1)  # includes predictions into unknown column 0
    predicted = cm[:, 1:].sum(0)
    fn, fp = support - tp, predicted - tp
    iou = tp / (tp + fn + fp).clamp_min(1)
    f1 = 2 * tp / (2 * tp + fn + fp).clamp_min(1)
    return {
        "iou": iou,
        "f1": f1,
        "support": support,
        "miou": iou.mean(),
        "macro_f1": f1.mean(),
        "forest_macro_f1": f1[:3].mean(),
    }
