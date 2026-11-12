# cloud_haze——物理仿真退化引擎（云雾）

AI Earth-X 森林方向第一篇论文的退化引擎，谢金池任务 1 交付物。
版本：cloud_haze v1.0.0-20261113（改动升版本号，见版本纪律）

## 功能

对 ForTy v1 光学时序（(4,128,128,10) float32，/10000 反射率口径）施加物理仿真云雾退化：

| 变体 | 公式 | NDVI | 用途 |
| --- | --- | --- | --- |
| additive（默认） | x' = (1-β)·x + β·c，c 为波段相关云亮度 | 会偏移（统计落表） | B 组主实验 |
| multiplicative | x' = a·x，逐像素全波段同乘，a ∈ [0.25,1] | 严格不变（~3e-08） | 物理约束消融组 |

- 档位：light / medium / heavy（云覆盖率 20/35/55%，雾幅度 0.02/0.04/0.06）
- 云覆盖率可调范围 0.10–0.70，越界报错
- 只退化光学，标签不碰；同 seed 逐像素可复现
- C 组跨季重叠接口：overlap=True + overlap_shift，IoU 自动收缩保证 ≥0.60
- 训练增强（augment）与评估协议（export_masks + mask 回放）掩码同源

## 快速开始

```bash
# 自测（8 条验收标准，需 numpy + matplotlib）
python selftest_cloud_haze.py
```

自测通过后接训练管线（张正锋侧）：

```python
from cloud_haze import CloudHazeEngine

engine = CloudHazeEngine()
s2_deg, label, meta = engine.augment(s2, label, seed=0, level="medium",
                                     variant="additive")
```

详细接口（训练调用 / 评估协议 / 消融对照设计）见 [接口说明与消融对照设计.md](接口说明与消融对照设计.md)。

## 目录

```
cloud_haze.py                  引擎本体（单文件，仅依赖 numpy）
selftest_cloud_haze.py         自测脚本（8 条验收标准）
check_ndvi_quick.py            各档位 NDVI 偏差快速核查
接口说明与消融对照设计.md        张正锋/阮业诚接口文档 + B/B-mult 消融设计
自测输出/selftest_report.txt    自测报告（8/8 PASS）
自测输出/ndvi_bias_stats.csv    NDVI 偏差统计落表
自测输出/figure_manifest.csv   对比图清单（图与参数的对应关系）
自测输出/figures/              8 张轻/中/重对比图（可由自测脚本重新生成）
```

## 验收状态

8 条验收标准全部 PASS（2026-11-13，本机合成样本自测）：
恒等 / 标签不动 / 同种子可复现 / 覆盖率 10–70% / 8 张对比图 / NDVI 统计落表 / 接口自测 / 消融设计交付。

真实数据冒烟（服务器端）由张正锋执行；若波段顺序与 DeepSatModels/TSViT 惯例不符，
改 `CloudHazeEngine(red_idx=, nir_idx=)` 构造参数即可。

## 版本纪律

任何参数/行为改动升版本号并全组周知；meta.engine 自带版本号可追溯。
