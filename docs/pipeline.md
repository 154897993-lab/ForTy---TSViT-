# 独立训练与统一评估链路

本链路独立编写，不执行旧训练、评估或 Sobel 模块。保留本项目自己的 U-Net、标签转换和指标参考。先完成软件闭环，真实科学结论等待阮的数据与冻结协议。

## 环境与冻结配置

Python 3.10+，依赖 PyTorch、NumPy、protobuf；无需 TensorFlow。`requirements.txt` 列出依赖，不会自动安装；`environment.yml` 提供可选的环境定义。`google-crc32c` 为可选加速；标准库后备实现也校验 CRC，但真实大数据吞吐必须实测。

环境是否可用 CUDA，需以 `torch.cuda.is_available()` 和模型前反向核验为准；本文不绑定具体机器或环境名。

对于已核验的 32 样本 prepared 包，使用独立的训练集拟合入口：

```text
python fit_train32.py --package PREPARED_PACKAGE_DIR --output runs/fit32_50 --device cuda --epochs 50 --batch 8 --lr 0.001 --seed 20261113 --threads 4
```

它只读取 train，不构造 validation/test，不按最佳中间轮挑选结果；预先核对包、index 和每份 NPZ 的哈希，prepared 反射率不再次缩放，统一使用 -100 忽略标签。默认开启内置支路，Adam betas=.9/.9999、clip norm=10、float32。完成后输出最后一轮 fit50.pt、可恢复的 last.pt、逐轮日志和严格重载后的训练集验收结果。实际 mIoU 以最后一轮为准，未达标不自动延长或改参数重跑。

复制 `configs/protocol.template.json` 到独立配置文件，填完再确认 frozen=true。模板中的种子、学习率、batch 等只是候选值，云配置和边界宽度故意留空，不代表已获阮确认。配置文件相对路径以该文件所在目录解析。

必须确认版本、用途 research/synthetic、季节波段顺序、清单 SHA256、数据位置/格式、预期样本数、归一化尺度、训练配置、训练种子列表、评估种子、边界宽度和云生成器。数据加载严格按 sample_id 和 split，校验清单/记录 ID、数值、形状及 TFRecord CRC，不默认接受缺片或重复 ID。

TFRecord 清单字段包括 sample_id、split（train/validation）、shard、byte_offset；索引到未压缩 TFRecord 的头部。按官方 Example float_list 解码 s2、int64_list 解码 segmentation_labels 和 id。数据源必须实际核验；压缩 TFRecord 不支持随机 byte_offset。NPZ 每样本一文件，含 s2、segmentation_labels、id，仍走同样的 manifest 和预处理。

训练配置不得混入 test 行。可选 data.limits 为显式小范围验收使用，例如 train:32、validation:32；加载器先核对完整清单的 expected_counts，再取所声明的前 N 个样本，运行记录保留实际数量。正式小样本实验清空 limits。cache_samples 会缓存解码后的样本；实际 RAM 与 worker 数需先实测，不自动开启大缓存。

冻结配置后，先运行 `python check_data.py --config protocol.json --output runs/data_check`，逐条检查 train/validation，输出样本数、原始类别像素数、全 unknown 样本数及归一化值域。验证集三类森林必须均有像素支持。该预检不接触 test，不替代阮的空间配准和物理云验证。

## 云生成器交接接口

阮提供可导入的 Python 模块和函数，配置 clouds.provider 为 `模块名:函数名`、sha256 为函数所在源码文件的 SHA256、settings 为冻结参数字典。模块及依赖需在运行环境的 Python 路径内。生成器不接收标签，契约如下：

```python
def augment(*, image, sample_ids, kind, scope, seed, epoch, settings):
    # image: 已归一化的浮点 Tensor [B,4,128,128,10]
    # kind: independent/overlap（训练）或 single/dual_indep/dual_overlap（评估）
    # scope: training/evaluation；评估 epoch 固定 0
    # seed: 训练或独立评估种子；按 ID 稳定生成，不能依赖 batch 位置
    # 不修改输入原图；不接触标签。实际算法和参数由阮冻结。
    return affected_image, cloud_mask, {"seasons": affected_season_indices}
```

输出图像保持形状、dtype、设备且有限；mask 为 `[B,4,128,128]` bool。single 声明一个季节，其他类型声明两个不同季节，未声明季节不得有 mask。metadata 可补充版本或强度等 JSON 信息。新引擎独立记录每样本每季实际覆盖率和双季 IoU。

独立组和重叠组匹配每季覆盖率、强度、启用率及季节数量，差别为跨季空间关系。生成器应确保同一评估 ID/场景/种子在不同模型和 batch 划分下相同，训练和评估图样独立。引擎检查结构和来源哈希；真实物理、独立性及 IoU 门槛仍须阮的生成器自测和协议验收，不会由名字推断。

没有 provider 时可以训练/评价 clean；受扰组及四场景矩阵在开始前拒绝运行。这里不附任何可被误当科学默认值的云生成器；临时验证用的生成器不随项目提交。

## 训练、冒烟和续训

在仓库根目录执行。以下假定文件路径、种子和 frozen 状态均已按实际协议确认，不可直接套用草案开跑。

```text
python train.py --config protocol.smoke.json --group A --seed 20261113 --output runs/smoke_A --device cuda
python train.py --config protocol.smoke.json --group D --seed 20261113 --output runs/smoke_D --device cuda
```

冒烟配置显式设置 epochs=2、32 样本限制和对应版本。干净 50 epoch 验收用另一个冻结配置，事前约定样本规模、epochs=50、measure_train=true；从完整干净训练集 eval 指标核验 mIoU，而非从验证集或在线 loss 推测。

输出目录必须全新；历史运行不自动覆盖。输出 run.json、protocol.json、history.json、cloud_audit.jsonl、best.pt、last.pt。模型选择只使用协议规定的干净验证指标。每轮记训练损失、实际更新数、全忽略批数、干净训练和验证分数、耗时；整批 unknown 不进行任何优化器更新。

```text
python train.py --config protocol.json --group D --seed 20261113 --output runs/D_seed20261113 --device cuda --resume
```

续训从 last.pt 恢复模型、优化器、轮次及历史；协议、源码、组名和种子不符时拒绝。显式逐轮种子使 epoch 边界恢复可重现。`--stop-after-epoch N` 用于有记录的边界停止，不改变冻结训练总轮数。此实现不承诺逐 step 恢复；未保存完的 epoch 从头重跑。完整模型/生成器在 CUDA 上的确定性仍须实测。

CPU 调试将 --device 设为 cpu。float32/bfloat16、batch、worker 和线程数由协议显式确定；未实现多卡和 float16/GradScaler，不自动改变精度或阈值。

## 统一评估和预测

```text
python evaluate.py --config protocol.smoke.json --checkpoint runs/smoke_A/best.pt --output runs/smoke_eval --scene clean --export-predictions --device cuda
```

可重复 --checkpoint 一次传入多个模型，默认在 validation 的四场景统一评价。只有 --scene clean 时不要求云生成器。评估只加载 checkpoint 记录的模型配置，核验训练协议/源码哈希，拒绝重复的组/种子权重。

eval_manifest.csv 中 train cloud、训练种子、评估场景、场景种子分列，分数统一 0–1；乘 100 后才是百分数或百分点。metrics.json 保留八类 F1/IoU/support、全图与边界指标和混淆矩阵。scene_audit.jsonl 保留实际覆盖率及 IoU。按整集混淆矩阵计算分数，不平均每批 F1。

--export-predictions 导出每组/种子/场景/ID 对应 NPZ，含原始预测 ID 1–8、标签 0–8、干净/受扰 S2（反射率）及受扰 mask，供高辉后续制图。图例与论文排版不由此命令自动生成。

## 六组与三种子调度

```text
python run_experiments.py --config protocol.pilot.json --seeds 20261113 --output runs/pilot --device cuda --export-predictions
```

默认六组 A、D、B_ind、B_ind+D、B_ovl、B_ovl+D。可用 --groups 选择有决策依据的子集。所有组按同一配置训练，每种子四场景共 24 个评价单元；任一异常立即停止，不改参数、不自动重试或扩量。阮未冻结生成器时，该命令在开始矩阵前拒绝运行。

调度完成后产生统一评价和 summary/summary.json，给出各组场景均值、样本标准差（ddof=1）和配对差值/交互。单种子标准差记 null，并标记仅供方向判断。缺少配对组/种子时列为缺失，不拼不匹配的样本。独立使用：

```text
python summarize_runs.py --metrics runs/pilot/evaluation/metrics.json --output runs/pilot_summary
```

三种子需要在冻结 training_seeds 中列明并显式传入 --seeds；科学阶段是否扩量和哪些组入选仍由项目决策控制。程序不会自动作出“通过科研门槛”结论。

## 官方 test 保持封存

训练从不打开 test；普通评价默认 validation。最终决策通过后，另提供批准 JSON，不能修改训练配置来解锁：

```json
{
  "approved": true,
  "decision_id": "REPLACE_WITH_FINAL_DECISION",
  "approved_by": "REPLACE_WITH_APPROVER",
  "protocol_sha256": "REPLACE_WITH_TRAINING_PROTOCOL_HASH",
  "checkpoint_sha256": ["REPLACE_WITH_EVERY_APPROVED_CHECKPOINT_HASH"],
  "test_manifest": "REPLACE_WITH_SEALED_TEST_MANIFEST",
  "test_manifest_sha256": "REPLACE_WITH_TEST_MANIFEST_HASH",
  "expected_test_count": 0
}
```

该示例仅定义字段，不是批准或可用配置。实际 test 清单相对批准文件解析，分片位于冻结数据根目录。明确传 --split test --confirm-test --test-approval approval.json，一次传完批准的所有 checkpoint。

开始前在训练协议目录以排他方式创建该协议的 official_test_*.started 回执；同一协议不能靠换输出目录反复评 test。异常中断也保留回执，须人工决策处理，不自动清除。这是工作流约束，不是文件权限隔离。批准文件必须覆盖准确的协议和全部权重哈希。

## 核验边界

临时合成样本检查覆盖软件行为；真实 ForTy 编码、阮生成器、运行环境依赖、CUDA/吞吐/内存和科学门槛仍需真实小样本核验。运行目录应保留日志及权重，详细个人工作记录不进入源码仓库。全链路核验临时文件通过后清理，仅在仓库外保留结果记录。

TFRecord 与 Example 格式依据 [TensorFlow 官方说明](https://www.tensorflow.org/tutorials/load_data/tfrecord)及[官方 feature.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto)。
