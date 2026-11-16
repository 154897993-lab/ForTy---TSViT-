# 500+500 pilot 结果与核验

范围：500 train / 500 validation，13 个单种子配置；不包含正式三种子训练或官方 test。模型为 ForestUNet unet_v0.1。主指标为森林三类宏 F1；八类 mIoU 单独报告。

best 按干净验证集八类宏 F1 选择，last 为第100轮。全部26个权重使用同一冻结评估代码、数据清单、PyTorch/CUDA及推理设置，产生52,000份四场景预测；逐份哈希和全图/边界累计已独立核对，另重放104份预测逐像素一致。

## batch32 六组 best：森林三类宏 F1（%）

| 组 | S0 clean | S1 单季 | S2 双季独立 | S3 双季重叠 |
| --- | ---: | ---: | ---: | ---: |
| A | 51.40 | 44.44 | 36.79 | 36.30 |
| D | 53.04 | 44.04 | 32.54 | 34.11 |
| B_ind | 52.08 | 50.45 | 49.44 | 49.48 |
| B_ind+D | 51.01 | 49.68 | 46.19 | 46.02 |
| B_ovl | 50.25 | 48.89 | 48.39 | 48.37 |
| B_ovl+D | 51.05 | 50.66 | 48.89 | 48.15 |

B_ind 在 S2/S3 的森林宏 F1 最高；B_ovl+D 在这两个场景的八类 mIoU 最高。B_ovl 的 S0 森林宏 F1 比 A 低约1.14点，超出1点代价门槛；D−A 的 S0 边界 mIoU 约为−0.99点，未达到+1点要求。单种子只作方向参考，不能据此宣称多种子有效性或统计显著性。

## 32样本验收与性能

冻结 D/50 epoch/batch32/类均衡训练集拟合 mIoU 为99.1132%，阈值95%，已通过；这不是验证集精度。

独立云、重叠云的串行/4进程各完整运行三轮，每种实现处理同一批500训练样本，合计48次更新，模型、BN、Adam、除耗时外的历史和云审计完全一致。整轮加速2.24×、1.87×。三轮中首轮用于启动，后两轮计时包括验证、审计和权重写入。3,000+500、18组、100epoch外推约24.5卡时，预留32卡时；3元/卡时仅为费用示例，详见预算JSON。

## 文件和复算

- [best/last 全表](evaluation/comparison_best_last.csv)、[batch32 主表](evaluation/comparison_batch32_best.csv)。CSV为0–1比例，表格乘100。
- [逐类指标和support](evaluation/per_class_all.csv)、[全图和边界指标](evaluation/scene_metrics_all.csv)、`evaluation/per_run/` 混淆矩阵与预测文件哈希。
- [训练曲线](training_curves.csv)、[冻结配置与源码哈希](protocols/execution_lock.json)、[权重交接哈希](handoff_checkpoints.json)。
- `source/` 保存与字节哈希对应的训练/评估源码快照；实际运行源文件保留原字节格式，避免跨平台换行转换破坏SHA256。
- [拟合验收记录](fit32_acceptance/fit_acceptance.json)、[整轮性能与等价性](performance/epoch_benchmark.json)、[预算](performance/budget_updated.json)。
- 本目录提供必要公共证据；完整逐像素预测、验证标签和拟合权重在项目交付包中。个人计划、交接说明及执行过程记录留在仓库外。

在仓库环境运行以下命令，核对本目录全部文件哈希，并独立从26份权重的混淆矩阵复算全图/边界指标和逐类support：

```text
python results/pilot500_20261116/verify_results.py
```

原固定评估器快照依赖交付包中的协议文件及prepared数据；数据或评估契约改变时应另起版本。
