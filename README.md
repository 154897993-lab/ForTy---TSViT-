# ForTy 森林分割实验

独立实现的训练、保存重载、本地诊断评估与六组调度链路，使用本项目的 40 通道、8 类 U-Net 和内置解码特征细化支路；正式四场景评估交给阮的固定入口。

实验设计以《AI Earth-X 森林 第一篇论文项目清单 v2（2026-11-16）》为准：六组 A / D / B_ind / B_ind+D / B_ovl / B_ovl+D，budget-matched 云雾主口径（固定严口径作稳健性附录），三种子 20261113 / 20261114 / 20261115，主干为 unet_v0.1 单轨（TSViT 对照轨道关闭）。

新链路不导入或调用张原有训练代码；`forty_tsvit_baseline.py`、`class_stats.py`、`loss_maskfocal.py` 三个历史文件已按清单 v2 决策从当前版本移除（git 历史可查）。此前复制的模型适配层已改为独立工厂，Sobel 旁路副本已移除。

## 入口

- `train.py`：单组训练、每轮干净训练集/验证集指标、best/last checkpoint 与断点恢复。
- `evaluate.py`：本地诊断评估，导出逐类/边界指标、混淆矩阵、场景审计与原始预测数组；不生成官方四场景指标。
- `run_experiments.py`：六组及种子矩阵顺序训练，遇错停止；训练完成后写 checkpoint 交接清单，正式评估由阮的固定入口执行。
- `summarize_runs.py`：均值、样本标准差和配对差值/交互量；不混合不同协议或数据划分。
- `data.py`、`record_io.py`：按固定清单和 byte_offset 读取 TFRecord，核对 ID 和 CRC；不依赖 TensorFlow。NPZ 用于软件闭环核验或已明确的缓存数据。
- `check_data.py`：训练前逐条预检清单、分片、标签和类别支持，不使用 GPU、不接触 test。
- `pipeline.py`：冻结协议、云生成器接口、数据加载、指标、边界带与结果记录。
- `model.py`、`forest_unet.py`、`forty_adapter.py`：六组模型设置、正式 U-Net、输入和标签转换。
- `ruan_cloud_provider.py`：训练侧云适配器；逐字封装 `ruan_vendor/` 中冻结的阮云引擎和训练包装器，导入时校验源码哈希。正式评估仍由阮的固定入口执行。

## 开始使用

先阅读 [全链路使用说明](docs/pipeline.md)，按阮确认的协议填写 `configs/protocol.template.json`。模板是未冻结草案，不能直接训练。缺少云生成器或参数时，受扰训练/评估会报错，不自动套用默认云。

模型结构和标签说明见 [U-Net 说明](docs/unet.md)。通用依赖见 `requirements.txt`；`environment.yml` 提供可选的环境定义。具体运行的 Python、PyTorch 和设备信息写入运行记录。

## 科学验收状态

软件闭环检查不等同于模型有效性证据。`fit_train32.py` 支持已核验 prepared 数据的训练集拟合；实际验收结论以独立运行记录为准，未达到门槛前不进入后续阶段。完整 ForTy、阮的云协议和正式六组仍需相应数据与协议验收。官方 test 默认封存，只有冻结 checkpoint 矩阵及独立批准文件后才能统一评一次。

500+500 pilot 的 13 个配置、阮固定入口的 S0–S3 评估、边界/逐类/掩膜审计、32 样本正式验收与 GPU 性能记录已归档在 [`evidence/20261116_pilot500/`](evidence/20261116_pilot500/)。主表见该目录的 `evaluation/comparison.csv`，配置对比与门槛核对见 `evidence/20261116_pilot500/evaluation/comparison.md` 与 [`docs/反馈闭环_20261116.md`](docs/反馈闭环_20261116.md)。单种子 pilot 只作方向性参考，不替代全量三种子主表。

个人计划、文献工作稿和详细核验记录保留在仓库外；运行输出建议放在已忽略的 `runs/` 中。
