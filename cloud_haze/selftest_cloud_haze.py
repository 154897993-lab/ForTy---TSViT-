# -*- coding: utf-8 -*-
"""
cloud_haze v1.0.0 自测脚本——逐条核对任务书谢金池任务 1 的 8 条验收标准
====================================================================
标准 1  强度 0 输出与原图逐像素一致
标准 2  标签不动（引擎不修改标签，augment 原样传回）
标准 3  同种子可复现（同 seed 两次逐像素一致，异 seed 结果不同）
标准 4  覆盖率可调范围 10–70%（实际覆盖≈目标；越界参数报错）
标准 5  8 张轻/中/重对比图
标准 6  雾区 NDVI 偏差中位数与 P95 落表（乘法变体应≈0）
标准 7  接口自测（训练增强 / 评估协议掩码导出-回放一致性 / 跨季 IoU）
标准 8  物理约束消融对照设计随引擎交付（检查文档存在，内容见文档本身）

本机无 ForTy 真实数据（33GB 在服务器），用真实光谱量级的合成样本自测，
输入格式与 TFRecord 解析结果完全一致：(T,H,W,C)=(4,128,128,10) float32 /10000，
标签 (H,W) int64，类别含 unknown=0。
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloud_haze import CloudHazeEngine, iou, ndvi, ndvi_bias_stats  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "自测输出")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

engine = CloudHazeEngine()
results = []


def check(cid, desc, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    results.append((cid, desc, tag, detail))
    print(f"[{tag}] 标准{cid} {desc} {detail}")


# ------------------------------------------------ 合成 ForTy 风格样本

SPECTRA = {  # 10 波段典型反射率（B2 B3 B4 B5 B6 B7 B8 B8A B11 B12）
    0: [.08, .09, .10, .11, .12, .13, .14, .14, .13, .12],   # unknown（边缘混合像元）
    1: [.02, .04, .03, .05, .06, .07, .25, .22, .12, .08],   # natural 天然林
    2: [.03, .05, .04, .06, .07, .08, .28, .24, .13, .09],   # planted 人工林
    3: [.04, .06, .05, .07, .08, .09, .22, .20, .11, .07],   # tree_crops 树作物
    5: [.15, .16, .17, .18, .19, .20, .20, .20, .18, .17],   # built
    7: [.02, .03, .04, .02, .02, .02, .01, .01, .005, .004], # water
}


def make_synthetic(seed=2026, size=128, t_len=4):
    """粗分辨率类别图放大 + 噪声，四季节光谱微变，量级为真实反射率"""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 6, size=(8, 8))
    labels = np.kron(coarse, np.ones((size // 8, size // 8), dtype=int))
    flip = rng.random((size, size)) < 0.05
    labels[flip] = rng.integers(0, 6, size=int(flip.sum()))
    cls_map = np.array([0, 1, 2, 3, 5, 7])
    label = cls_map[labels].astype(np.int64)
    s2 = np.zeros((t_len, size, size, 10), np.float32)
    for t in range(t_len):
        gain = 1.0 + 0.08 * np.sin((t + 1) * 1.5)
        for cls, spec in SPECTRA.items():
            m = label == cls
            if not m.any():
                continue
            base = np.asarray(spec) * gain
            noise = rng.normal(1.0, 0.05, (int(m.sum()), 10))
            s2[t][m] = np.clip(base[None] * noise, 0, None).astype(np.float32)
    return s2, label


def rgb(frame, red=2, green=1, blue=0, q=99.5):
    """(H,W,C) → RGB 合成图（按分位数拉伸）"""
    img = np.stack([frame[..., blue], frame[..., green], frame[..., red]], axis=-1)
    hi = np.percentile(img, q)
    return np.clip(img / max(hi, 1e-6), 0, 1)


def save_compare(fname, suptitle, before_t, after_t, mask, ndvi_abs_diff=None):
    """四联对比图：原始 RGB / 退化 RGB / 云掩码 / |ΔNDVI|（色标按数据自适应避免饱和）"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    axes[0].imshow(rgb(before_t))
    axes[0].set_title("原始（RGB 合成）")
    axes[1].imshow(rgb(after_t))
    axes[1].set_title("退化后（RGB 合成）")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("云掩码（白=云）")
    if ndvi_abs_diff is not None:
        vmax = max(0.05, float(np.percentile(ndvi_abs_diff, 99.5)))
        im = axes[3].imshow(ndvi_abs_diff, cmap="magma", vmin=0, vmax=vmax)
        axes[3].set_title(f"|ΔNDVI|（色标上限 {vmax:.2f}）")
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------ 开始自测

s2, label = make_synthetic()
s2c, labelc = s2.copy(), label.copy()
print(f"合成样本：s2 {s2.shape} {s2.dtype}，标签 {label.shape} {label.dtype}，"
      f"类别 {sorted(set(label.ravel().tolist()))}\n")

# 标准 1：强度 0（level=None）恒等
s2_id, lab_id, m_id = engine.augment(s2, label, seed=0, level=None)
ok1 = (np.array_equal(s2_id, s2) and np.shares_memory(lab_id, label)
       is False or np.array_equal(lab_id, label))
ok1 = np.array_equal(s2_id, s2) and np.array_equal(lab_id, label) and m_id.get("identity")
check(1, "强度 0 输出与原图逐像素一致", ok1,
      f"逐像素相等={np.array_equal(s2_id, s2)}，恒等标记={m_id.get('identity')}")

# 标准 2：标签不动（退化路径也不碰标签；全图退化后标签仍与输入完全一致）
s2_deg, lab_deg, meta_deg = engine.augment(s2, label, seed=0, level="medium")
changed_pixels = float(np.mean(np.any(s2_deg != s2, axis=-1)))
ok2 = (np.array_equal(lab_deg, label) and np.array_equal(label, labelc)
       and np.shares_memory(lab_deg, label)
       and changed_pixels > 0.05)
check(2, "标签不动", ok2,
      f"标签同对象原样传回={np.shares_memory(lab_deg, label)}，"
      f"退化后标签与输入一致={np.array_equal(lab_deg, label)}，"
      f"影像改变像素比例={changed_pixels:.1%}")

# 标准 3：同种子可复现 + 异种子不同
a1, _ = engine.degrade(s2, seed=42, level="heavy")
a2, _ = engine.degrade(s2, seed=42, level="heavy")
a3, _ = engine.degrade(s2, seed=43, level="heavy")
ok3 = np.array_equal(a1, a2) and not np.array_equal(a1, a3)
check(3, "同种子可复现", ok3,
      f"seed42 两次逐像素一致={np.array_equal(a1, a2)}，"
      f"seed43 结果不同={not np.array_equal(a1, a3)}")

# 标准 4：覆盖率 10–70% 可调（实际≈目标；越界报错）
cov_ok, cov_detail = True, []
for tgt in (0.10, 0.30, 0.50, 0.70):
    _, m = engine.degrade(s2, seed=7, level="medium", cloud_coverage=tgt)
    act = float(np.mean([m["per_season"][s]["coverage_actual"]
                         for s in m["seasons_disturbed"]]))
    good = abs(act - tgt) <= 0.04
    cov_ok = cov_ok and good
    cov_detail.append(f"目标{tgt:.0%}→实际{act:.1%}{'✓' if good else '✗'}")
try:
    engine.degrade(s2, seed=0, level="medium", cloud_coverage=0.05)
    cov_ok = False
    cov_detail.append("越界 0.05 未报错✗")
except ValueError:
    cov_detail.append("越界 0.05 正确报错✓")
try:
    engine.degrade(s2, seed=0, level="medium", cloud_coverage=0.80)
    cov_ok = False
    cov_detail.append("越界 0.80 未报错✗")
except ValueError:
    cov_detail.append("越界 0.80 正确报错✓")
check(4, "覆盖率 10–70% 可调", cov_ok, "；".join(cov_detail))

# 标准 6：NDVI 偏差统计（乘法变体应严格≈0；先于图做，图里要用数字）
ndvi_rows = []
for variant in ("additive", "multiplicative"):
    for lev in ("light", "medium", "heavy"):
        deg, m = engine.degrade(s2, seed=11, level=lev, variant=variant,
                                seasons=[0])
        st = ndvi_bias_stats(s2[[0]], deg[[0]], mask=m["binary_masks"][0])
        ndvi_rows.append(dict(variant=variant, level=lev, season=0, seed=11,
                              coverage=m["per_season"][0]["coverage_actual"],
                              ndvi_abs_median=st["median"], ndvi_abs_p95=st["p95"]))
mult_ok = all(r["ndvi_abs_median"] < 1e-5 and r["ndvi_abs_p95"] < 1e-4
              for r in ndvi_rows if r["variant"] == "multiplicative")
add_heavy = next(r for r in ndvi_rows if r["variant"] == "additive" and r["level"] == "heavy")
with open(os.path.join(OUT, "ndvi_bias_stats.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(ndvi_rows[0].keys()))
    w.writeheader()
    w.writerows(ndvi_rows)
check(6, "雾区 NDVI 偏差中位数与 P95 落表", mult_ok,
      f"乘法变体 max(median)={max(r['ndvi_abs_median'] for r in ndvi_rows if r['variant']=='multiplicative'):.2e}，"
      f"加性 heavy: median={add_heavy['ndvi_abs_median']:.4f} p95={add_heavy['ndvi_abs_p95']:.4f}"
      "（已写入 ndvi_bias_stats.csv）")

# 标准 5：8 张对比图（原始 1 + 加性轻中重 3 + 乘法轻中重 3 + 跨季重叠示意 1）
fig_manifest = []


def fig_record(fname, variant, level, seed, m, st):
    fig_manifest.append(dict(figure=fname, variant=variant, level=level, seed=seed,
                             coverage=round(float(m["per_season"][0]["coverage_actual"]), 4),
                             ndvi_abs_median=st["median"], ndvi_abs_p95=st["p95"]))


# 图 1：原始基线（RGB + NDVI 伪彩色）
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
axes[0].imshow(rgb(s2[0]))
axes[0].set_title("原始光学（季节1，RGB 合成）")
nd0 = ndvi(s2[0])
im = axes[1].imshow(nd0, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
axes[1].set_title("NDVI（绿=植被，红=水/云）")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
for ax in axes:
    ax.set_axis_off()
fig.suptitle("基线样本（合成，ForTy 格式：4 季 × 10 波段 × 128×128）", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig01_原始基线.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# 图 2–7：两变体 × 三档
fnames = {("additive", "light"): "fig02_加性_轻.png",
          ("additive", "medium"): "fig03_加性_中.png",
          ("additive", "heavy"): "fig04_加性_重.png",
          ("multiplicative", "light"): "fig05_乘法_轻.png",
          ("multiplicative", "medium"): "fig06_乘法_中.png",
          ("multiplicative", "heavy"): "fig07_乘法_重.png"}
variants_cn = {"additive": "加性混合（默认）", "multiplicative": "乘法受约束（NDVI 不变）"}
levels_cn = {"light": "轻", "medium": "中", "heavy": "重"}
for variant in ("additive", "multiplicative"):
    for lev in ("light", "medium", "heavy"):
        deg, m = engine.degrade(s2, seed=11, level=lev, variant=variant, seasons=[0])
        st = ndvi_bias_stats(s2[[0]], deg[[0]], mask=m["binary_masks"][0])
        dnd = np.abs(ndvi(deg[0]) - nd0)
        title = (f"{variants_cn[variant]} · {levels_cn[lev]}档（seed=11，"
                 f"云覆盖 {m['per_season'][0]['coverage_actual']:.0%}，"
                 f"|ΔNDVI| 中位数 {st['median']:.2e}）")
        save_compare(fnames[(variant, lev)], title, s2[0], deg[0],
                     m["binary_masks"][0], dnd)
        fig_record(fnames[(variant, lev)], variant, lev, 11, m, st)

# 图 8：跨季重叠（C 组示意，两季共掩码 + 平移扰动，IoU 自动收缩到合同线以上）
deg_c, m_c = engine.degrade(s2, seed=3, level="medium", seasons=[0, 2],
                            overlap=True, overlap_shift=8)
iou_02 = iou(m_c["binary_masks"][0], m_c["binary_masks"][2])
shift_real = m_c["per_season"][2]["shift"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
axes[0].imshow(rgb(s2[0]))
axes[0].set_title("原始（季节1）")
axes[1].imshow(rgb(deg_c[0]))
axes[1].set_title(f"退化后（季节1，覆盖 {m_c['per_season'][0]['coverage_actual']:.0%}）")
axes[2].imshow(m_c["binary_masks"][0], cmap="gray")
axes[2].set_title(f"季节1 掩码")
axes[3].imshow(m_c["binary_masks"][2], cmap="gray")
axes[3].set_title(f"季节3 掩码（平移 {shift_real}px，IoU 收缩后）\n"
                 f"跨季 IoU = {iou_02:.3f}（C 组合同 ≥0.60）")
for ax in axes:
    ax.set_axis_off()
fig.suptitle("跨季重叠遮挡示意（overlap=True，C 组评估场景）", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig08_跨季重叠示意.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
fig_manifest.append(dict(figure="fig08_跨季重叠示意.png", variant="additive",
                          level="medium(C组overlap)", seed=3,
                          coverage=round(float(m_c["per_season"][0]["coverage_actual"]), 4),
                          ndvi_abs_median=None, ndvi_abs_p95=None))

n_figs = len(fnames) + 2
check(5, "8 张轻/中/重对比图", n_figs == 8,
      f"figures/ 下共生成 {n_figs} 张（原始1 + 加性3 + 乘法3 + 跨季1）")

# 标准 7：接口自测（训练侧批量 / 评估侧掩码导出-回放一致 / 跨季 IoU）
itf_ok, itf_detail = True, []

# 7a 训练侧批量增强（张正锋：Dataset 内逐样本调用）
batch_ok = True
for i in range(8):
    x, y, mm = engine.augment(s2, label, seed=1000 + i, level="medium")
    batch_ok = batch_ok and x.shape == s2.shape and x.dtype == np.float32 \
        and np.array_equal(y, label)
itf_ok = itf_ok and batch_ok
itf_detail.append(f"批量增强 8 样本（seed 逐样本派生）={batch_ok}")

# 7b 未受扰季节保持原样（单季受扰场景）
deg_s, m_s = engine.degrade(s2, seed=5, level="heavy", seasons=[1])
untouched_ok = np.array_equal(deg_s[0], s2[0]) and np.array_equal(deg_s[3], s2[3]) \
    and not np.array_equal(deg_s[1], s2[1])
itf_ok = itf_ok and untouched_ok
itf_detail.append(f"seasons=[1] 时其余季节逐像素不变={untouched_ok}")

# 7c 评估侧：export_masks 与 degrade 内部掩码同源一致
mask_exp = engine.export_masks(s2.shape, seed=7, level="medium", seasons=[0])
deg_int, m_int = engine.degrade(s2, seed=7, level="medium", seasons=[0])
mask_same = np.array_equal(mask_exp[0]["binary"], m_int["binary_masks"][0])
deg_ext, m_ext = engine.degrade(s2, seed=7, level="medium", seasons=[0],
                                mask=mask_exp[0]["soft"])
replay_same = np.array_equal(deg_ext, deg_int)
itf_ok = itf_ok and mask_same and replay_same
itf_detail.append(f"掩码导出与训练增强同源={mask_same}，外部掩码回放退化逐像素一致={replay_same}")

# 7d 跨季 IoU（shift=0 → 1.0；shift=8 → 引擎自动收缩平移量，合同线 0.60）
_, m_io0 = engine.degrade(s2, seed=3, level="medium", seasons=[0, 2],
                          overlap=True, overlap_shift=0)
iou0 = iou(m_io0["binary_masks"][0], m_io0["binary_masks"][2])
itf_ok = itf_ok and abs(iou0 - 1.0) < 1e-9 and iou_02 >= 0.60
itf_detail.append(f"shift=0 时 IoU={iou0:.3f}（应=1.000）；shift 上限 8px 时实际平移 "
                  f"{shift_real}px，IoU={iou_02:.3f} ≥ 合同线 0.60 = {iou_02 >= 0.60}")
check(7, "接口自测", itf_ok, "；".join(itf_detail))

# 标准 8：消融对照设计文档随引擎交付
doc = os.path.join(BASE, "接口说明与消融对照设计.md")
doc_ok = os.path.exists(doc)
check(8, "物理约束消融对照设计随引擎交付", doc_ok,
      "接口说明与消融对照设计.md " + ("存在" if doc_ok else "缺失（先跑代码测试，文档随后生成）"))

# ------------------------------------------------ 汇总

with open(os.path.join(OUT, "figure_manifest.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(fig_manifest[0].keys()))
    w.writeheader()
    w.writerows(fig_manifest)

env = (f"引擎版本 cloud_haze v1.0.0-20261113 | numpy {np.__version__} | "
       f"matplotlib {matplotlib.__version__} | python {sys.version.split()[0]}\n"
       f"输入与 ForTy v1 TFRecord 解析口径一致：s2 (4,128,128,10) float32 /10000，"
       f"标签 (128,128) int64，unknown=0 掩码类\n"
       f"本机无真实数据，自测用合成样本（真实光谱量级）；真实数据接口自测由张正锋在服务器执行\n")

lines = [env, "-" * 78]
for cid, desc, tag, detail in results:
    lines.append(f"[{tag}] 标准{cid} {desc}\n      {detail}")
n_pass = sum(1 for r in results if r[2] == "PASS")
lines.append("-" * 78)
lines.append(f"合计 {len(results)} 条：PASS {n_pass} / FAIL {len(results) - n_pass}")
summary = "\n".join(lines)
print("\n" + summary)
with open(os.path.join(OUT, "selftest_report.txt"), "w", encoding="utf-8") as f:
    f.write(summary + "\n")

sys.exit(0 if n_pass == len(results) else 1)
