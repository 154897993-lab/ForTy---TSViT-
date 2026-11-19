# D2 开发实验源码

本目录归档论文一 D2 开发实验在服务器实际执行的源码与配置快照。父提交为 ForTy 训练代码 494ebd32aa07a2c527bfa1bb80ff0cb306b5c7f0。正式 P2 的训练、模型、指标和评估入口仍由原路径提供。

| 目录 | 内容 |
|---|---|
| D2_direction_screen_20261118_v1 | 原 A/D 对照、四种改造分支、边界加权、30 轮初筛与三种子长程训练调度 |
| D2_followup_20261119_v1 | 轻量 depthwise 的同损失矩阵补齐、首轮桥接和 S0–S3 重放 |
| D2_local_refine_20261119_v1 | 预测边界覆盖诊断、冻结主干后的全图与局部残差对照 |
| D2_class_balanced_refine_20261119_v1 | 在相同残差对照下加入只从训练集计算的类别权重 |
| D2_bias_and_B_pair_20261119_v1 | 八类常数偏置和 D2 在同一冻结 A、B_ind 上的收尾验证，以及配对统计复算 |

## 使用入口与复现依赖

训练及评估脚本按现有服务器目录布局运行。把上述阶段目录放在项目根的 development/ 下。目录名用于定位相邻阶段。原训练入口位于 ruan_delivery_20261117/training_494ebd3，接受的固定评估器位于 contracts/P2_formal_eval_20261118_v2/code。各 JSON 保留原始绑定路径、源码和输入哈希，可用于核对已完成运行。

收尾验证脚本的入口是 D2_bias_and_B_pair_20261119_v1/bias_and_b_pair.py，提供 --prepare、--preflight 和 --run。原运行依赖 Python 3.12、PyTorch 2.8 CUDA、冻结 train3000/validation500、S0–S3 清单与掩码、A/B 检查点、此前的构建结果和三个 A+D2 小头。复跑需使用新的输出位置，并在运行计划中记录实际路径和哈希。仓库外的资产由团队的现有评估环境或实验交接包提供。

最后的统计复算可以脱离 GPU 和原始数据。在临时目录放入 bias_pair_reanalyze.py、修正前的 bias_pair_completion_original.json，并将该原件复制为 bias_pair_completion.json。调用 Python 执行脚本，结果写回临时目录的 bias_pair_completion.json。脚本根据已有混淆矩阵汇总记录重新计算配对及参考门槛字段。

## 版本和结果边界

SOURCE_SHA256.json 记录 16 个源码与配置文件的服务器快照哈希。代码和配置保留原始字节，以便对应检查点的运行身份。旧机器的定时关机及暂停脚本不属于实验实现。

本批结果来自开发验证集。冻结 B_ind 后拟合的小头使用原始训练输入。B+D2 的 S2/S3 边界均值接近但低于 +1 点的预设参考门槛。正式 P2 数据、模型与结果文件按原项目保留，官方 test 仍封存。
