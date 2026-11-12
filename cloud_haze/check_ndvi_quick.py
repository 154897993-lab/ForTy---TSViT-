# -*- coding: utf-8 -*-
"""快速核查：各档位加性雾在全图（含非云区）与云区的 NDVI 偏差"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloud_haze import CloudHazeEngine, ndvi_bias_stats  # noqa: E402

# 复用自测脚本的合成样本生成（只取函数定义）
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "selftest_cloud_haze.py"), encoding="utf-8").read()
ns = {"__file__": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "selftest_cloud_haze.py"),
      "np": np}
exec(src.split("# ------------------------------------------------ 开始自测")[0], ns)
make_synthetic = ns["make_synthetic"]

s2, _ = make_synthetic()
eng = CloudHazeEngine()
for lev in ("light", "medium", "heavy"):
    deg, m = eng.degrade(s2, seed=11, level=lev, variant="additive", seasons=[0])
    full = ndvi_bias_stats(s2[[0]], deg[[0]], mask=None)
    cloud = ndvi_bias_stats(s2[[0]], deg[[0]], mask=m["binary_masks"][0])
    print(f"{lev:7s} 全图 median={full['median']:.4f} p95={full['p95']:.4f} | "
          f"云区 median={cloud['median']:.4f} p95={cloud['p95']:.4f}")
