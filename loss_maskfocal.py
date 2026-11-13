# -*- coding: utf-8 -*-
"""Masked Focal-Dice Loss（ForTy 长尾 + unknown 掩码）

    L = mean_{i∈valid}[ w_{c_i} · (1 - p_{c_i})^γ · ( -log p_{c_i} ) ]
        + λ · ( 1 - mean_c Dice_c )

    w_c ∝ f_c^(-1/2)，f_c 为「训练集」类别像素频率，下限 0.02，再按 9 类均值归一；
    valid = 标签 != unknown(0) 的像素，Dice 同样只在 valid 上累计。

注意：freq 必须来自训练集统计（class_stats.py 产出的 CSV），
      用 val/test 参与统计会把评估分布写进先验，属于泄漏。

用法：
    crit = MaskedFocalDice(read_freq("class_distribution_train.csv"))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedFocalDice(nn.Module):
    def __init__(self, freq, gamma=2.0, lam=0.5, ignore=0, eps=1.0, floor=0.02):
        super().__init__()
        f = torch.as_tensor(freq, dtype=torch.float32).clamp(min=floor)
        w = 1.0 / f ** 0.5
        self.register_buffer("w", w / w.mean())
        self.gamma, self.lam, self.ignore, self.eps = gamma, lam, ignore, eps

    def forward(self, logits, target):
        valid = target != self.ignore
        if not bool(valid.any()):
            return logits.sum() * 0.0
        tgt = target.clamp(min=0)
        logpt = F.log_softmax(logits, 1).gather(1, tgt.unsqueeze(1)).squeeze(1)
        w = self.w.gather(0, tgt.reshape(-1)).view_as(tgt)
        focal = (-(1 - logpt.exp().detach()) ** self.gamma * w * logpt)[valid].mean()
        prob = F.softmax(logits, 1) * valid.unsqueeze(1)
        onehot = F.one_hot(tgt, logits.shape[1]).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
        dice = (2 * (prob * onehot).sum((0, 2, 3)) + self.eps) / \
               (prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3)) + self.eps)
        return focal + self.lam * (1 - dice).mean()


def read_freq(csv_path):
    import csv
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    pct = [float(r["pixel_pct"]) for r in rows]
    total = sum(pct)
    return [p / total for p in pct]
