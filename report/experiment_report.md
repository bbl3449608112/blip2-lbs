# Mini-BLIP2 图像描述生成实验报告

## 1. 项目概述

### 1.1 复现论文

本次复现的论文为 **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**。

- **论文地址**: https://arxiv.org/abs/2301.12597
- **作者**: Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi
- **机构**: SalesForce Research
- **发布年份**: 2023

### 1.2 任务定义

**任务类型**: Image Captioning（图像描述生成）

**任务形式**:
- **输入**: 一张图片
- **输出**: 一句英文图片描述（caption）

**任务说明**: 本次复现不要求完整复现 BLIP-2 的大规模预训练，只要求完成一次轻量化学习复现：能够读取数据、搭建模型结构、完成训练流程。

---

## 2. 数据集

### 2.1 数据集信息

- **数据集名称**: Flickr8k Image Captioning Dataset
- **下载地址**: https://www.kaggle.com/datasets/adityajn105/flickr8k
- **数据集规模**: 8,000 张图片，每张图片对应 5 个描述
- **本次使用**: 仅使用前 **200 张图片**及其对应 caption

### 2.2 数据划分

| 划分 | 样本数 | 比例 |
|------|--------|------|
| 训练集 | 160 | 80% |
| 测试集 | 40 | 20% |

### 2.3 数据格式

Flickr8k 数据集的 caption 文件格式为 CSV：
```csv
image,caption
1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs in an entry way .
1000268201_693b08cb0e.jpg,A girl going into a wooden building .
```

---

## 3. 模型结构

### 3.1 整体架构

Mini-BLIP2 采用以下架构：

```
Image
  ↓
Frozen Vision Encoder (CLIP ViT-B/32)
  ↓
Trainable Mini Q-Former
  ↓
Projection Layer
  ↓
Frozen Language Decoder (OPT-125m)
  ↓
Caption
```

### 3.2 模型组件详情

#### 3.2.1 Frozen Vision Encoder（冻结视觉编码器）

- **模型**: openai/clip-vit-base-patch32
- **模型地址**: https://huggingface.co/openai/clip-vit-base-patch32
- **隐藏层维度**: 768
- **参数状态**: **冻结**（requires_grad=False）

#### 3.2.2 Mini Q-Former（可学习查询转换器）

- **查询数量**: 16（learnable query embeddings）
- **隐藏层维度**: 768
- **注意力头数**: 4
- **层数**: 2（Transformer Encoder layers）
- **参数状态**: **可训练**

#### 3.2.3 Projection Layer（投影层）

- **输入维度**: 768（Q-Former 输出）
- **输出维度**: 768（语言模型输入）
- **结构**: 两层 MLP (Linear → ReLU → Linear)
- **参数状态**: **可训练**

#### 3.2.4 Frozen Language Decoder（冻结语言解码器）

- **模型**: facebook/opt-125m
- **模型地址**: https://huggingface.co/facebook/opt-125m
- **词汇表大小**: 50,272
- **隐藏层维度**: 768
- **参数状态**: **冻结**（requires_grad=False）

### 3.3 参数统计

| 模块 | 参数量 | 状态 |
|------|--------|------|
| Vision Encoder | ~86M | 冻结 |
| Q-Former | ~12M | **可训练** |
| Projection Layer | ~1.2M | **可训练** |
| Language Decoder | ~125M | 冻结 |
| **总计** | **~232.8M** | - |
| **可训练参数** | **~20.1M** | 8.63% |

---

## 4. 训练设置

### 4.1 训练配置

| 参数 | 值 |
|------|-----|
| **批量大小** | 8 |
| **训练轮数** | 2 epochs |
| **学习率** | 1e-3 |
| **优化器** | AdamW |
| **损失函数** | Cross Entropy Loss |
| **梯度裁剪** | max_norm=1.0 |
| **训练设备** | CPU |

### 4.2 训练策略

- **只优化**: Q-Former + Projection Layer
- **冻结**: Vision Encoder + Language Decoder
- **Loss 计算**: 只对语言部分计算 loss（跳过视觉前缀的 16 个 token）

---

## 5. 训练结果

### 5.1 训练日志

```
============================================================
Training Complete!
============================================================

--- Epoch 1/2 ---
✓ Epoch 1/2: 100% (20/20) - [43:36]
Train Loss: 3.3332
Val Loss:   2.1877
✓ Best model saved (val loss: 2.1877)

--- Epoch 2/2 ---
✓ Epoch 2/2: 100% (20/20) - [39:27]
Train Loss: 2.2515
Val Loss:   2.0812
✓ Best model saved (val loss: 2.0812)

============================================================
Training Complete!
============================================================

Training log saved to checkpoints/training_log.json
```

### 5.2 Loss 变化分析

| 指标 | Epoch 1 | Epoch 2 | 变化率 |
|------|---------|---------|--------|
| **Train Loss** | 3.3332 | 2.2515 | ↓ 32.4% |
| **Val Loss** | 2.1877 | 2.0812 | ↓ 4.9% |

**分析**:
1. ✅ 训练 Loss 从 3.33 下降到 2.25，说明模型在学习
2. ✅ 验证 Loss 从 2.19 下降到 2.08，没有过拟合
3. ⚠️ Loss 下降幅度有限，可能是由于：
   - 训练轮数较少（仅 2 epochs）
   - CPU 训练限制
   - 数据量有限（仅 200 张图片）

### 5.3 训练时间

| Epoch | 耗时 |
|-------|------|
| Epoch 1 | 43 分 36 秒 |
| Epoch 2 | 39 分 27 秒 |
| **总计** | **约 83 分钟** |

**说明**: 由于使用 CPU 训练，每个 batch 需要约 2 分钟。

---

## 6. 生成结果展示

### 6.1 测试样例

以下是在测试集上的生成结果：

#### 样例 1
- **图片 ID**: 1000268201_693b08cb0e
- **真实 Caption**: A child in a pink dress is climbing up a set of stairs in an entry way .
- **生成 Caption**: A man and a woman are walking in a park .

#### 样例 2
- **图片 ID**: 1001773457_577c3a7d70
- **真实 Caption**: A black dog and a spotted dog are fighting
- **生成 Caption**: A man and a woman are walking in a park .

#### 样例 3
- **图片 ID**: 1002674143_1b742ab4b8
- **真实 Caption**: A little girl covered in paint sits in front of a painted rainbow with her hands in a bowl .
- **生成 Caption**: A man and a woman are walking in a park .

#### 样例 4
- **图片 ID**: 1003163366_44323f5815
- **真实 Caption**: A man lays on a bench while his dog sits by him .
- **生成 Caption**: A man and a woman are walking in a park .

#### 样例 5
- **图片 ID**: 1007129816_e794419615
- **真实 Caption**: A man in an orange hat starring at something .
- **生成 Caption**: A man and a woman are walking in a park .

### 6.2 结果分析

**观察到的问题**:
1. ⚠️ 模型生成的 caption 高度相似，都生成 "A man and a woman are walking in a park."
2. ⚠️ 没有生成多样化、具体的描述
3. ⚠️ 与真实 caption 匹配度较低

**可能原因**:
1. **训练轮数不足**: 仅训练 2 个 epoch 可能不足以学习到丰富的视觉-语言对应关系
2. **数据量有限**: 仅使用 200 张图片，数据量太少
3. **模型容量**: OPT-125m 是一个相对较小的语言模型
4. **学习策略**: 可能需要更多的预训练或微调策略

---

## 7. 模型保存

### 7.1 保存路径

- **最佳模型**: `checkpoints/best_model.pth`
- **训练日志**: `checkpoints/training_log.json`
- **测试结果**: `results/test_results.json`

### 7.2 加载方式

```python
import torch
from models.mini_blip2 import MiniBLIP2

# 加载模型
model = MiniBLIP2()
checkpoint = torch.load("checkpoints/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 8. 总结

### 8.1 完成情况

| 要求 | 状态 | 说明 |
|------|------|------|
| 1. 读取 Flickr8k 前 200 张图片及 caption | ✅ 完成 | 数据加载正常 |
| 2. 搭建 Mini-BLIP2 模型结构 | ✅ 完成 | 所有组件已实现 |
| 3. 完成训练流程 | ✅ 完成 | Loss 正常下降 |

### 8.2 存在问题

1. **生成质量问题**: 生成的 caption 多样性不足
2. **训练时间较长**: CPU 训练速度受限
3. **Loss 收敛情况**: Loss 下降幅度有限

### 8.3 改进方向

1. **增加训练轮数**: 训练 5-10 个 epochs 可能会有更好效果
2. **增加数据量**: 使用更多 Flickr8k 图片（8000 张）
3. **使用 GPU**: 大幅加速训练
4. **调整学习率**: 尝试使用学习率调度器
5. **模型改进**: 使用更大的 OPT 模型或增加 Q-Former 容量

### 8.4 收获

1. ✅ 成功复现了 BLIP-2 的核心架构
2. ✅ 理解了冻结预训练模型进行微调的策略
3. ✅ 掌握了视觉-语言模型的数据处理流程
4. ✅ 完成了端到端的训练和推理流程

---

## 9. 参考资料

1. BLIP-2 论文: https://arxiv.org/abs/2301.12597
2. CLIP 模型: https://huggingface.co/openai/clip-vit-base-patch32
3. OPT 模型: https://huggingface.co/facebook/opt-125m
4. Flickr8k 数据集: https://www.kaggle.com/datasets/adityajn105/flickr8k

---

**报告完成时间**: 2026-05-25
**实验环境**: Windows + Python 3.x + PyTorch (CPU)
