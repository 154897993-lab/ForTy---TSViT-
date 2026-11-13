#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ForTy 类别分布统计：只统计训练集分片，输出可直接喂给 Focal 分支的 CSV。

为什么必须只统计 train：
    早期版本抽了 20 片（4 片 train + 16 片 validation/test）算频率，等于把评估集
    的类别分布当成了先验，属于分布泄漏。基线 CE 不受影响，但 Focal 权重会被污染。
    本脚本默认只读训练集分片，并在文件名前缀对不上时直接退出。

用法：
    python3 class_stats.py --data /data/forty \
        --pat 'forty_v1-train.tfrecord-*' --token train \
        --out class_distribution_train.csv
"""

import argparse
import glob
import os
import sys

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

IMG = 128
NCLS = 9
NAMES = ["unknown", "natural", "planted", "tree_crops",
         "other_veg", "built", "water", "ice", "bare"]

SPEC = {"segmentation_labels": tf.io.FixedLenFeature([IMG, IMG], tf.int64)}


def parse_label(record):
    e = tf.io.parse_single_example(record, SPEC)
    return e["segmentation_labels"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/forty")
    ap.add_argument("--pat", default="forty_v1-train.tfrecord-*")
    ap.add_argument("--token", default="train", help="该 split 的文件名标记，如 train / validation / test")
    ap.add_argument("--out", default="class_distribution_train.csv")
    args = ap.parse_args()

    files = []
    for part in args.pat.split(","):
        files += glob.glob(os.path.join(args.data, part.strip()))
    files = sorted(files)
    if not files:
        sys.exit("没匹配到分片：%s" % args.pat)

    bad = [f for f in files if ("-" + args.token + ".") not in os.path.basename(f)]
    if bad:
        sys.exit("前缀不符：%d 个文件不属于 %s，例如 %s" % (len(bad), args.token, os.path.basename(bad[0])))

    pixel = np.zeros(NCLS, dtype=np.int64)
    tile_present = np.zeros(NCLS, dtype=np.int64)
    tile_dominant = np.zeros(NCLS, dtype=np.int64)
    tiles = 0

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=8).map(
        parse_label, num_parallel_calls=tf.data.AUTOTUNE).batch(8)
    for labels in ds:
        arr = labels.numpy()
        for t in range(arr.shape[0]):
            lab = arr[t].reshape(-1)
            counts = np.bincount(lab, minlength=NCLS)[:NCLS]
            pixel += counts
            tile_present += (counts > 0)
            tile_dominant[int(counts.argmax())] += 1
            tiles += 1
        print("已处理图块 %d" % tiles, end="\r", flush=True)

    total = int(pixel.sum())
    ex_unknown = int(pixel[1:].sum())

    rows = []
    for c in range(NCLS):
        pct = 100.0 * pixel[c] / max(total, 1)
        pct_ex = 0.0 if c == 0 else 100.0 * pixel[c] / max(ex_unknown, 1)
        rows.append("%d,%s,%d,%.3f,%.3f,%d,%.2f,%d,%.2f" % (
            c, NAMES[c], int(pixel[c]), pct, pct_ex,
            int(tile_present[c]), 100.0 * tile_present[c] / max(tiles, 1),
            int(tile_dominant[c]), 100.0 * tile_dominant[c] / max(tiles, 1)))

    header = ("class_id,class_name,pixels,pixel_pct,pixel_pct_ex_unknown,"
              "tiles_present,tile_present_pct,tiles_dominant,tile_dominant_pct")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(header + "\n" + "\n".join(rows) + "\n")

    print("\n分片 %d 片 | 图块 %d | 像素 %d" % (len(files), tiles, total))
    print("已写出 %s" % args.out)
    ratio = pixel[4] / max(pixel[7], 1)
    print("other_veg %d 像素 / ice %d 像素，相差 %.1f 倍" % (pixel[4], pixel[7], ratio))


if __name__ == "__main__":
    main()
