#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ForTy × TSViT 基线复现：训练 + 验证 + 封存测试（单文件可运行版）。

设计口径（与组会交付文档一致）：
  1. 数据 96 片 = train 32 / validation 32 / test 32，划分只由文件名前缀决定；
     三集合两两无交集，test 在训练全程不读取，只在最后评一次。
  2. 输入 4 期 S2，10 个波段，反射率除以 10000 归一；第 11 通道是 4 个 DOY 标量。
  3. 标签 9 类，0 为 unknown，只作掩码：不参与梯度，也不进指标分母。
  4. 指标：mIoU / F1 只在 8 个真实类上算，overall F1 = 8 类 macro 平均；
     预测成 unknown 记为该类漏检（FN），不再额外记误检。
  5. 权重选择：验证集 overall F1 最好的那一轮，训练结束才加载去评 test。

用法（P40 实例 /data/forty 下，数据与仓库同级）：
    # 基线（CE），20 轮，batch 32，seed 0
    python3 -u forty_tsvit_baseline.py \
        --data /data/forty --repo /data/forty/DeepSatModels \
        --train-pat 'forty_v1-train.tfrecord-*' \
        --val-pat   'forty_v1-validation.tfrecord-*' \
        --test-pat  'forty_v1-test.tfrecord-*' \
        --epochs 20 --batch 32 --seed 0 --lr 1e-3 \
        --ckpt best_ce_b32.pt --test --log train_log.txt

    # 冒烟（只跑 3 步、1 轮，确认依赖与数据解析）
    python3 -u forty_tsvit_baseline.py --epochs 1 --limit-steps 3 --batch 8
"""

import argparse
import glob
import os
import random
import sys
import time
import types

import numpy as np
import tensorflow as tf

# TensorFlow 只用于读 TFRecord，算力全部留给 PyTorch
tf.config.set_visible_devices([], "GPU")

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = 128                    # ForTy 图块边长
BANDS = 10                   # S2 波段数
NCLS = 9                     # 含 unknown 的类别数
IGNORE = 0                   # unknown，只作掩码
CLASS_NAMES = ["unknown", "natural", "planted", "tree_crops",
               "other_veg", "built", "water", "ice", "bare"]

# TSViT 结构（与 PASTIS24/fold1 官方配置同源，按 ForTy 改成 128 / 4 期 / 9 类）
MODEL_CFG = {
    "architecture": "TSViT", "img_res": IMG, "patch_size": 16,
    "num_channels": BANDS + 1, "num_classes": NCLS, "max_seq_len": 4,
    "dim": 192, "temporal_depth": 2, "spatial_depth": 2,
    "heads": 6, "dim_head": 32, "pool": "cls",
    "emb_dropout": 0.0, "dropout": 0.0, "scale_dim": 4,
}


# --------------------------------------------------------------------------
# 划分：只认文件名前缀，任何混串、重叠、缺片都直接退出
# --------------------------------------------------------------------------
def resolve(pattern, data_dir):
    files = []
    for part in pattern.split(","):
        files += glob.glob(os.path.join(data_dir, part.strip()))
    return sorted(files)


def check_split(files, token, expect):
    """文件名必须含自己 split 的标记，片数默认必须等于 expect。"""
    if not files:
        sys.exit("划分失败：%s 一个文件都没匹配到，检查 --*-pat 与数据目录" % token)
    bad = [os.path.basename(f) for f in files if ("-" + token + ".") not in os.path.basename(f)]
    if bad:
        sys.exit("划分串味：以下文件不属于 %s：%s" % (token, bad[:5]))
    if expect and len(files) != expect:
        sys.exit("片数不符：%s 期望 %d 片，实际 %d 片" % (token, expect, len(files)))


def check_disjoint(*groups):
    names = [set(g) for g in groups]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = names[i] & names[j]
            if inter:
                sys.exit("划分重叠：出现同一文件 %s" % sorted(inter)[:3])


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
LABEL_SPEC = {"s2": tf.io.FixedLenFeature([4, IMG, IMG, BANDS], tf.float32),
              "segmentation_labels": tf.io.FixedLenFeature([IMG, IMG], tf.int64)}


def parse_record(record, clip_s2):
    e = tf.io.parse_single_example(record, LABEL_SPEC)
    s2 = e["s2"] / 10000.0            # DN -> 反射率
    if clip_s2:
        s2 = tf.clip_by_value(s2, 0.0, 1.0)
    e["s2"] = s2
    return e


def make_dataset(files, batch, shuffle, seed, cache, clip_s2, num_parallel=8):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=num_parallel)
    ds = ds.map(lambda r: parse_record(r, clip_s2), num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(512, seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def to_input(batch, device, doy):
    """[B,4,H,W,10] + 4 个 DOY 标量 -> [B,4,H,W,11]（TSViT 要 NTHWC）。"""
    s2 = torch.from_numpy(batch["s2"].numpy()).float().to(device)
    doy_map = doy.expand(s2.shape[0], 4, IMG, IMG).unsqueeze(-1).to(s2.dtype)
    return torch.cat([s2, doy_map], dim=-1)


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def confusion(pred, target):
    """在标签 != unknown 的像素上累计 9×9 混淆矩阵。"""
    m = target != IGNORE
    idx = (target[m] * NCLS + pred[m]).numpy()
    return np.bincount(idx, minlength=NCLS * NCLS).reshape(NCLS, NCLS)


def metrics(cm):
    """丢掉 unknown 行列，在 8 个真实类上算 IoU / F1。"""
    cm = cm[1:, 1:].astype(np.float64)
    tp = np.diag(cm)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    iou = tp / np.maximum(tp + fp + fn, 1e-9)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    return iou, f1


# --------------------------------------------------------------------------
# 可选：Masked Focal-Dice（改进分支，与 loss_maskfocal.py 等价）
# --------------------------------------------------------------------------
class MaskedFocalDice(nn.Module):
    def __init__(self, freq, gamma=2.0, lam=0.5, ignore=IGNORE, eps=1.0, floor=0.02):
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
    """类别频率只允许来自训练集统计；文件必须 9 行、含 pixel_pct 列。"""
    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path, encoding="utf-8")))
    if len(rows) != NCLS:
        sys.exit("类别表有 %d 行，期望 %d 行（含 unknown）" % (len(rows), NCLS))
    pct = np.array([float(r["pixel_pct"]) for r in rows], dtype=np.float64)
    return (pct / pct.sum()).tolist()


class Tee:
    """同时写终端与日志文件，保证交付的日志逐字可复现。"""

    def __init__(self, path):
        self.term = sys.stdout
        self.file = open(path, "w", encoding="utf-8") if path else None

    def write(self, text):
        self.term.write(text)
        if self.file:
            self.file.write(text)

    def flush(self):
        self.term.flush()
        if self.file:
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/forty")
    ap.add_argument("--repo", default="", help="DeepSatModels 仓库根目录")
    ap.add_argument("--train-pat", default="forty_v1-train.tfrecord-*")
    ap.add_argument("--val-pat", default="forty_v1-validation.tfrecord-*")
    ap.add_argument("--test-pat", default="forty_v1-test.tfrecord-*")
    ap.add_argument("--expect", type=int, default=32, help="每个 split 的片数，0 表示不检查")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--class-csv", default="", help="focal 分支的类别频率表（必须来自 train）")
    ap.add_argument("--cache", action="store_true", help="缓存训练集（约 13 GB 内存，谨慎开启）")
    ap.add_argument("--clip-s2", action="store_true", help="把反射率裁到 [0,1]（默认不裁，与基线一致）")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--test", action="store_true", help="训练结束后评一次封存 test")
    ap.add_argument("--limit-steps", type=int, default=0, help="每轮最多走多少步，冒烟用")
    ap.add_argument("--log", default="", help="日志文件路径")
    args = ap.parse_args()

    log = Tee(args.log)
    sys.stdout = log

    # 1) 划分与隔离
    train = resolve(args.train_pat, args.data)
    val = resolve(args.val_pat, args.data)
    test = resolve(args.test_pat, args.data)
    check_split(train, "train", args.expect)
    check_split(val, "validation", args.expect)
    check_split(test, "test", args.expect)
    check_disjoint(train, val, test)

    # 2) 可复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 3) 模型（仓库导入，绕开 models/__init__ 的无关依赖）
    repo = args.repo or os.path.join(args.data, "DeepSatModels")
    sys.path.insert(0, repo)
    try:
        from models.TSViT.TSViTdense import TSViT
    except Exception:
        for name, sub in (("models", "models"), ("models.TSViT", "models/TSViT"),
                          ("utils", "utils")):
            mod = types.ModuleType(name)
            mod.__path__ = [os.path.join(repo, sub)]
            sys.modules[name] = mod
        from models.TSViT.TSViTdense import TSViT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TSViT(MODEL_CFG).to(device)

    # 4) 损失
    if args.loss == "focal":
        if not args.class_csv or not os.path.exists(args.class_csv):
            sys.exit("focal 分支需要 --class-csv，且该表必须只统计训练集")
        crit = MaskedFocalDice(read_freq(args.class_csv)).to(device)
        print("频率来源 %s | 权重 %s" % (os.path.basename(args.class_csv),
                                       " ".join("%.2f" % x for x in crit.w.tolist())))
    else:
        crit = nn.CrossEntropyLoss(ignore_index=IGNORE).to(device)

    opt = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.9999), 1e-8)
    doy = torch.tensor([0.25, 0.5, 0.75, 1.0], device=device).view(1, 4, 1, 1)

    print("设备 %s | 参数 %.2fM | 训练 %d 片 | 验证 %d 片 | test 封存 %d 片 | 损失 %s | batch %d | seed %d"
          % (device, sum(p.numel() for p in model.parameters()) / 1e6,
             len(train), len(val), len(test), args.loss, args.batch, args.seed))

    train_ds = make_dataset(train, args.batch, True, args.seed, args.cache, args.clip_s2)

    def evaluate(files, tag):
        model.eval()
        cm = np.zeros((NCLS, NCLS), dtype=np.int64)
        with torch.no_grad():
            for batch in make_dataset(files, args.eval_batch, False, args.seed, False, args.clip_s2):
                y = torch.from_numpy(batch["segmentation_labels"].numpy()).long()
                pred = model(to_input(batch, device, doy)).argmax(1).cpu()
                cm += confusion(pred, y)
        iou, f1 = metrics(cm)
        print("%s mIoU %.2f | overall F1 %.2f | forest F1 %.2f"
              % (tag, 100 * iou.mean(), 100 * f1.mean(), 100 * f1[0:3].mean()))
        print("逐类 F1  : " + " ".join("%s %.1f" % (CLASS_NAMES[i + 1], 100 * f1[i]) for i in range(8)))
        print("逐类 IoU : " + " ".join("%s %.1f" % (CLASS_NAMES[i + 1], 100 * iou[i]) for i in range(8)))
        return iou, f1

    # 5) 训练：每轮验证选一个最优权重，test 全程不碰
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses, steps = [], 0
        for batch in train_ds:
            steps += 1
            y = torch.from_numpy(batch["segmentation_labels"].numpy()).long().to(device)
            loss = crit(model(to_input(batch, device, doy)), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            losses.append(loss.item())
            if args.limit_steps and steps >= args.limit_steps:
                break

        model.eval()
        cm = np.zeros((NCLS, NCLS), dtype=np.int64)
        with torch.no_grad():
            for batch in make_dataset(val, args.eval_batch, False, args.seed, False, args.clip_s2):
                y = torch.from_numpy(batch["segmentation_labels"].numpy()).long()
                pred = model(to_input(batch, device, doy)).argmax(1).cpu()
                cm += confusion(pred, y)
        iou, f1 = metrics(cm)

        score = 100 * f1.mean()
        if score > best:
            best = score
            torch.save(model.state_dict(), args.ckpt)

        print("epoch %d | 平均 loss %.4f | %.1fs | %d 步 | val mIoU %.2f | overall F1 %.2f | forest F1 %.2f"
              % (epoch, float(np.mean(losses)), time.time() - t0, steps,
                 100 * iou.mean(), score, 100 * f1[0:3].mean()))

    print("口径：只在标签 != unknown(0) 的像素上统计，8 类 macro 平均；"
          "预测成 unknown 记为漏检，不计额外误检")

    # 6) 封存 test：整个流程只在这里评一次
    if args.test:
        if not os.path.exists(args.ckpt):
            sys.exit("找不到权重 %s" % args.ckpt)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print("test | %d 片 | 权重来自验证集 overall F1 最好的一轮" % len(test))
        evaluate(test, "test")

    log.close()


if __name__ == "__main__":
    main()
