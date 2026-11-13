# U-Net 使用说明

模型版本：`unet_v0.1`。模块只依赖 PyTorch，不包含 TFRecord 读取、云雾生成或完整训练循环。训练端需使用项目环境并统一数据和评估协议。

## 配置

```python
from forest_unet import ForestUNet

model = ForestUNet(in_channels=40, num_classes=8, base_channels=16, use_edge=False)
```

A/B 组使用 `use_edge=False`，D/B+D 组使用 `use_edge=True`。云雾增强由数据端独立控制。按实验组构造模型，不在一次训练中途切换结构。

模型接收 `[B,40,128,128]`，输出 `[B,8,128,128]` 的原始 logits。默认编码通道为 16/32/64/128/256，四次最大池化，解码器双线性插值并拼接同尺度编码特征。空间尺寸需至少为 16 且能被 16 整除。

细化支路投影并融合解码器 1/4 和 1/2 尺度的特征，输出残差 logits 加到基础分类头。它没有额外边缘标签或边缘损失，也不接收 NDVI。基础模型参数为 1,969,544，支路新增 8,680，相对增量为 0.4407%。

## 输入与标签

`prepare_s2` 接收原始 DN 的 NumPy 数组或 PyTorch Tensor，形状为 `[B,4,128,128,10]`；TensorFlow EagerTensor 先调用 `.numpy()`。函数只除以 10000 一次，不裁剪、不插补缺测，并转换为 `[B,40,128,128]`。

候选季节顺序为 DJF/MAM/JJA/SON，波段顺序为 B2/B3/B4/B5/B6/B7/B8/B8A/B11/B12。通道索引为 `季节索引*10+波段索引`。不拼 SAR、DEM、气候或 DOY。

实际通道顺序、缺测处理、数值域须与数据加载器联合核验后冻结。现有 TSViT 加载器已经除以 10000，不能将其输出再传入 `prepare_s2` 重复缩放；应在归一化前调用此函数，或仅对已归一化数据按上述顺序转换通道。

原始标签 0 表示 unknown；1–8 依次为天然林、人工林、树作物、其他植被、建筑、水体、冰雪、裸地。`encode_labels` 将原始 0 映射为 -100，将有效标签减一到 0–7。训练损失必须使用 `ignore_index=-100`，不能忽略训练索引 0 的天然林。

```python
import torch
from forty_adapter import prepare_s2, encode_labels, decode_predictions, IGNORE_INDEX

# s2_dn 和 raw_labels 由数据加载器交付，分别为原始 DN 和原始标签 ID。
x = prepare_s2(s2_dn).to(device)
y = encode_labels(raw_labels).to(device)
logits = model(x)

if bool((y != IGNORE_INDEX).any()):
    loss = torch.nn.functional.cross_entropy(logits, y, ignore_index=IGNORE_INDEX)
    # 接入训练循环的清梯度、反向和优化器更新。
# 整批无有效标签时跳过 backward 和 optimizer.step。

pred_raw = decode_predictions(logits.detach().argmax(1))  # 原始 ID 1..8
```

模型在训练循环外创建一次。云雾变换应在通道展平前施加；变换接收 DN 还是反射率、调用先后和强度须与引擎接口统一，不由模型模块推定。

## 指标参考

```python
from metrics_reference import confusion_matrix, summarize

cm_batch = confusion_matrix(pred_raw, raw_labels)
# 整个评估集累加所有 cm_batch，再调用 summarize(total_cm)。
```

参考函数接收原始 ID 0–8：忽略真实标签为 0 的像素，但保留真实标签有效、预测为 0 导致的漏检。8 输出 U-Net 先还原预测 ID；9 输出 TSViT 可直接传原始 argmax ID。不得将 encode 后的 -100 标签传入参考函数。

输出比率为 0–1；`forest_macro_f1` 对原始类 1–3 求均值，`macro_f1` 对八个有效类求均值，`miou` 对八类 IoU 求均值。无分母类别暂按 0 处理并返回 support，正式使用前需确认该口径。不要平均各批 F1 来代替整集 F1。

## 保存与重载

```python
from forest_unet import MODEL_VERSION

torch.save({
    "model_version": MODEL_VERSION,
    "model_config": model.config(),
    "state_dict": model.state_dict(),
}, "checkpoint.pt")

checkpoint = torch.load("checkpoint.pt", map_location=device, weights_only=True)
restored = ForestUNet(**checkpoint["model_config"]).to(device)
restored.load_state_dict(checkpoint["state_dict"], strict=True)
restored.eval()
```

正式 checkpoint 另附数据清单版本/哈希、种子、训练配置及引擎版本。8 类 U-Net 和 9 类 TSViT 权重不能互相直接加载。

## 核验范围

已完成合成输入的输出尺寸、反向传播、参数量、共享主干初始化、输入排列、标签映射、指标数值与保存重载检查。尚未完成真实数据的 32 样本 × 2 epochs 完整冒烟、边缘版干净小样本 50 epoch 训练 mIoU ≥95%、边界收益和 GPU 性能验收。

接入数据时按固定 manifest 的 sample_id 过滤；现有小样本涉及训练 20 片、验证 28 片，不沿用每个划分 32 片的检查默认值。当前模块不改动既有 TSViT 数据集、指标或 test 入口；接入方需统一相关口径，官方 test 仍按项目阶段限制使用。
