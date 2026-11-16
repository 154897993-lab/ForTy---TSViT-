# 500 样本 Pilot 证据包

本目录归档 2026-11-16 的 500 train / 500 validation pilot 实验事实，用于回应审阅意见，不替代论文主表或正式 test。

## 内容

- `protocols/`：冻结协议候选、评估清单、交付哈希、逐运行公开配置。
- `runs/`：13 个配置的 run.json、history、config_public、checkpoint SHA256、训练云审计（压缩）和审计摘要；`runs/handoff_checkpoints.json` 汇总交接哈希。
- `evaluation/`：阮固定入口的 S0–S3 场景指标、逐类指标、边界指标、混淆矩阵、样本误差、预测清单及掩膜覆盖/IoU 审计。主表见 `evaluation/comparison.csv` 与 `comparison.md`。
- `fit32_acceptance/`：32 样本干净 edge 组 50 epoch 正式验收记录和权重哈希；train mIoU = 0.99113，accepted = true。
- `performance/`：GPU 吞吐/显存/适配器开销与跨机确定性记录。

## 快速结论（pilot 单种子，只作方向）

- batch32 下 B_ind/B_ind+D/B_ovl+D 通过 B−A 的 S1–S3 ≥+2 点与 S0 降幅 ≤1 点；B_ovl 的 S0 降幅 -1.68 点，未通过 clean 代价门槛。
- B_ovl+D batch32 的 S0–S3 mIoU 为 0.5333/0.5332/0.5256/0.5179，S2/S3 为六组最高。
- D−A 边界 mIoU 为 -0.99/-1.48/-2.61/-0.80 点，D 单独未过 +1 点门槛；B+D−B 边界为 +1.5 到 +2.7 点，提示交互效应。
- 云训练约 1.05 h/100 epoch，clean 约 0.12 h；云增广占单步 97.5%–98.5%。

## 边界

- 权重文件和逐像素预测 NPZ 不进入 git；其 SHA256 见 checkpoint 与 prediction_manifest。
- 本证据包不包含官方 test；不包含全量 3000 训练结果。
- 时间为实验记录，不代表平台计费承诺。
