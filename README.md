# ForTy 森林分割实验

独立实现的训练、保存重载、本地诊断评估与六组调度链路，使用本项目的 40 通道、8 类 U-Net 和内置解码特征细化支路；正式四场景评估交给阮的固定入口。

实验设计包含六组 A / D / B_ind / B_ind+D / B_ovl / B_ovl+D，使用 budget-matched 云雾主口径，并可另行冻结固定强度的稳健性评估。主干为 unet_v0.1；实际训练种子、批次与数据规模以每次共同冻结的协议为准。第一指标为森林三类宏 F1，mIoU 与八类宏 F1 分别报告。

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
- `fit_train32.py`：训练集拟合与冻结配置验收；诊断运行不会自动成为正式验收。
- `benchmark_training.py`：按实际确定性训练步骤测量 GPU 性能，并核验串行/并行云适配器的一致性。
- `benchmark_epoch.py`：测量完整训练轮次，包含验证、审计和 checkpoint 写入；比较串行/并行训练状态。

## 开始使用

先阅读 [全链路使用说明](docs/pipeline.md)，按阮确认的协议填写 `configs/protocol.template.json`。模板是未冻结草案，不能直接训练。缺少云生成器或参数时，受扰训练/评估会报错，不自动套用默认云。

模型结构和标签说明见 [U-Net 说明](docs/unet.md)。通用依赖见 `requirements.txt`；`environment.yml` 提供可选的环境定义。具体运行的 Python、PyTorch 和设备信息写入运行记录。

## 科学验收状态

软件闭环检查不等同于模型有效性证据。`fit_train32.py` 支持已核验 prepared 数据的训练集拟合；实际验收结论以独立运行记录为准，未达到门槛前不进入后续阶段。完整 ForTy、阮的云协议和正式六组仍需相应数据与协议验收。官方 test 默认封存，只有冻结 checkpoint 矩阵及独立批准文件后才能统一评一次。

必要的公共实验结果与配置放在 [`results/pilot500_20261116/`](results/pilot500_20261116/)，包含指标表、混淆矩阵、逐类 support、训练曲线、源码/权重哈希、拟合验收和性能记录，可用随附脚本独立复算。单种子 pilot 只作方向参考。

仓库维护公共代码、模板、技术说明和支撑文档数字的必要结果。个人任务计划、交接文档、执行过程记录、完整逐像素预测、数据与大权重保存在仓库外的项目工作区，通过完整交付包提供。
