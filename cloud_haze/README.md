# cloud_haze——物理仿真退化引擎（云雾）

AI Earth-X 森林方向的物理仿真退化引擎。
版本：cloud_haze v1.0.0-20261113（改动升版本号，见版本纪律）
对标：第一篇论文项目清单 v2_20261116，git blob fe65c15f

## 功能

对 ForTy v1 光学时序（(4,128,128,10) float32，/10000 反射率口径）施加物理仿真云雾退化：

| 变体 | 公式 | NDVI | 用途 |
| --- | --- | --- | --- |
| additive（默认） | x' = (1-β)·x + β·c，c 为波段相关云亮度 | 会偏移（统计落表） | 六组主表全部使用 |
| multiplicative | x' = a·x，逐像素全波段同乘，a ∈ [0.25,1] | 严格不变（~3e-08） | B_ind 变体消融臂 |

- 档位：light / medium / heavy（云覆盖率 20/35/55%，雾幅度 0.02/0.04/0.06）
- 云覆盖率可调范围 0.10–0.70，越界报错
- 只退化光学，标签不碰；同 seed 逐像素可复现
- S3 跨季重叠场景接口（B_ovl 训练模式）：overlap=True + overlap_shift，IoU 自动收缩保证 ≥0.60
- 训练增强（augment）与评估协议（export_masks + mask 回放）掩码同源

## 快速开始

```bash
# 自测（8 条验收标准，需 numpy + matplotlib）
python selftest_cloud_haze.py
```

自测通过后接训练管线：

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
接口说明与消融对照设计.md        训练/评估接口 + 六组框架下的变体消融设计
自测输出/selftest_report.txt    运行时生成的自测报告
自测输出/ndvi_bias_stats.csv    NDVI 偏差统计落表
自测输出/figure_manifest.csv   对比图清单（图与参数的对应关系）
自测输出/figures/              8 张轻/中/重对比图（可由自测脚本重新生成）
```

自测报告、统计和图片属于运行产物，不提交到源码仓库；可由脚本重新生成并移至项目工作区。真实数据使用前需完成冒烟；若波段顺序与 DeepSatModels/TSViT 惯例不符，
改 `CloudHazeEngine(red_idx=, nir_idx=)` 构造参数即可。

## 版本纪律

任何参数/行为改动升版本号并全组周知；meta.engine 自带版本号可追溯。
