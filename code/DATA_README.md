# 数据准备模块说明

## 📁 文件结构

```
code/
├── data_loader.py      # 数据加载核心模块
├── download_data.py     # 数据下载和管理工具
├── test_data.py        # 数据加载测试脚本
└── setup_data.py       # 数据环境初始化脚本
```

## 🚀 快速开始

### 1. 环境初始化

```powershell
# 在虚拟环境中运行
python code/setup_data.py
```

这将：
- 创建必要的目录结构
- 检查 Python 依赖包是否安装
- 显示数据集下载说明

### 2. 下载 Flickr8k 数据集

**方法 1：手动下载（推荐）**
1. 访问 https://www.kaggle.com/datasets/adityajn105/flickr8k
2. 点击 Download 按钮
3. 解压 ZIP 文件
4. 将 `Images` 文件夹复制到 `data/Images`
5. 将 `captions.txt` 复制到 `data/captions.txt`

**方法 2：使用 Kaggle CLI**
```powershell
pip install kaggle
kaggle datasets download -d adityajn105/flickr8k
```

### 3. 验证数据加载

```powershell
python code/test_data.py
```

如果看到以下输出，说明数据准备成功：
```
✓ All tests passed!
Data preparation is ready for training!
```

## 📊 数据格式

### 目录结构
```
data/
├── Images/
│   ├── 1000268201_693b08cb0e.jpg
│   ├── 1001773457_577c3a7d35.jpg
│   └── ...
└── captions.txt
```

### Caption 文件格式
```
image#caption_number	caption_text
1000268201_693b08cb0e#0	A child in a pink dress is climbing into a wooden popcar...
1000268201_693b08cb0e#1	A girl going into a wooden building...
...
```

## 🔧 核心组件

### Flickr8kDataset
自定义 PyTorch Dataset，用于加载前 200 张图片及其 captions。

**主要功能：**
- 自动加载图片和 captions
- 使用 CLIP processor 进行图片预处理
- 支持数据集划分

**使用方法：**
```python
from data_loader import Flickr8kDataset
from transformers import CLIPProcessor

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

dataset = Flickr8kDataset(
    images_dir="data/Images",
    captions_file="data/captions.txt",
    clip_processor=clip_processor,
    max_samples=200
)

sample = dataset[0]
print(sample['image'].shape)  # (3, 224, 224)
print(sample['captions'])     # List of captions
```

### MiniBLIP2DataCollator
数据整理器，用于批处理数据。

**功能：**
- 批量整理图片和文本数据
- Tokenize captions
- 返回适合模型训练的格式

### prepare_flickr8k_data
一键准备数据的便捷函数。

**返回值：**
```python
train_loader, test_loader, clip_processor = prepare_flickr8k_data(
    data_root="data",
    max_samples=200,
    train_ratio=0.8,
    batch_size=4
)
```

## 🧪 测试

运行测试脚本验证所有功能：

```powershell
python code/test_data.py
```

测试内容：
1. ✓ 数据目录结构检查
2. ✓ CLIP 模型加载
3. ✓ 数据集创建
4. ✓ DataLoader 批处理

## 📝 注意事项

1. **数据量**：本项目使用 Flickr8k 的前 200 张图片（不是全部 8000 张）
2. **每张图片 5 个 caption**：Flickr8k 每张图片有 5 个人工标注的 caption
3. **数据划分**：默认 80% 训练集，20% 测试集
4. **CLIP 预处理**：所有图片会被 resize 到 224x224，并进行标准化

## 🔍 故障排除

### 问题 1：找不到 Images 目录
```
FileNotFoundError: Images directory not found: data/Images
```
**解决**：确保将 Flickr8k 的 Images 文件夹放在 `data/` 目录下

### 问题 2：找不到 captions.txt
```
FileNotFoundError: Captions file not found: data/captions.txt
```
**解决**：确保 captions.txt 文件在 `data/` 目录下

### 问题 3：CLIP 模型下载失败
**解决**：检查网络连接，可能需要设置代理或使用镜像

### 问题 4：内存不足
**解决**：减小 batch_size，从 4 降到 2 或 1

## 📚 相关资源

- [Flickr8k Dataset - Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [CLIP - OpenAI](https://openai.com/blog/clip/)
- [BLIP-2 Paper](https://arxiv.org/abs/2301.12597)
