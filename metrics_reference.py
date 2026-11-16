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
    """Return dataset-level metrics with NA for zero-support classes.

    Sum confusion matrices across batches before calling this function.
    Never average per-batch F1 scores as the dataset-level score.

    NA policy (shared with the P2 contract):
    - a class with GT support 0 has per-class IoU/F1 = None;
    - eight-class macro means use only support>0 classes and report
      ``valid_classes``;
    - forest macro F1 is None unless natural/planted/tree_crops all have
      support>0; it never silently averages two classes;
    - planted F1 is None when planted support is 0.
    ``None`` serializes to JSON null and is not a substitute for a NaN/Inf
    training error.
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
    iou_den = (tp + fn + fp).clamp_min(1)
    f1_den = (2 * tp + fn + fp).clamp_min(1)
    iou = tp / iou_den
    f1 = 2 * tp / f1_den
    has_support = support > 0
    iou_values = [float(iou[i]) if bool(has_support[i]) else None for i in range(8)]
    f1_values = [float(f1[i]) if bool(has_support[i]) else None for i in range(8)]
    valid_iou = [v for v in iou_values if v is not None]
    valid_f1 = [v for v in f1_values if v is not None]
    forest_ok = all(bool(has_support[i]) for i in range(3))
    return {
        "iou": iou_values,
        "f1": f1_values,
        "support": [int(v) for v in support.tolist()],
        "valid_classes": int(has_support.sum()),
        "miou": float(sum(valid_iou) / len(valid_iou)) if valid_iou else None,
        "macro_f1": float(sum(valid_f1) / len(valid_f1)) if valid_f1 else None,
        "forest_macro_f1": (float(sum(f1_values[:3]) / 3.0) if forest_ok else None),
        "planted_f1": f1_values[1],
    }
