# Git 提交指南

## 前置条件

### 1. 安装 Git

如果还没安装 Git，请先下载安装：
- Windows: https://git-scm.com/download/win

安装完成后，重启终端或命令提示符。

### 2. 验证 Git 安装

在终端中运行：
```powershell
git --version
```

如果看到版本号，说明安装成功。

### 3. 配置 Git 用户信息

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

（替换成您的名字和邮箱）

---

## 按照 requirements.md 要求进行小步提交

### 步骤 1: 初始化 Git 仓库

```powershell
cd d:\HDUhomework\blip2-main
git init
```

### 步骤 2: 提交 1 - 数据准备模块

```powershell
git add requirements.txt
git add code/data_loader.py
git add code/download_data.py
git add code/test_data.py
git add code/setup_data.py
git add code/debug_data.py
git add code/DATA_README.md
git add .gitignore
git commit -m "feat: 加载 Flickr8k 前 200 张图片与 caption"
```

### 步骤 3: 提交 2 - Frozen Vision Encoder

```powershell
git add code/models/__init__.py
git add code/models/vision_encoder.py
git commit -m "feat: 接入 CLIP ViT-B/32 作为 frozen vision encoder"
```

### 步骤 4: 提交 3 - Mini Q-Former

```powershell
git add code/models/q_former.py
git commit -m "feat: 实现 Mini Q-Former 模块（含 learnable queries）"
```

### 步骤 5: 提交 4 - Projection Layer

```powershell
git add code/models/projection_layer.py
git commit -m "feat: 添加 projection layer 对齐到 OPT 词向量空间"
```

### 步骤 6: 提交 5 - Frozen Language Decoder

```powershell
git add code/models/language_decoder.py
git commit -m "feat: 接入 frozen OPT-125m 作为语言解码器"
```

### 步骤 7: 提交 6 - Mini-BLIP2 完整模型

```powershell
git add code/models/mini_blip2.py
git add code/models/__init__.py
git commit -m "feat: 组装 Mini-BLIP2 完整模型"
```

### 步骤 8: 提交 7 - 训练脚本

```powershell
git add code/train.py
git commit -m "feat: 实现训练 loop 与 cross entropy loss"
```

### 步骤 9: 提交 8 - 生成脚本

```powershell
git add code/generate.py
git commit -m "feat: 添加 caption 生成脚本（greedy search）"
```

### 步骤 10: 提交 9 - 实验报告

```powershell
git add report/experiment_report.md
git add GIT_GUIDE.md
git commit -m "docs: 补充实验报告与文档"
```

---

## 创建 GitHub 仓库并推送

### 1. 在 GitHub 上创建新仓库

1. 访问: https://github.com/new
2. 仓库名称: `blip2-mini`（或其他您喜欢的名称）
3. 设置为 Public 或 Private
4. **不要**勾选 "Initialize this repository with a README"
5. 点击 "Create repository"

### 2. 关联远程仓库并推送

```powershell
# 替换下面的 URL 为您的 GitHub 仓库地址
git remote add origin https://github.com/您的用户名/blip2-mini.git
git branch -M main
git push -u origin main
```

### 3. 查看提交历史

```powershell
git log --oneline --graph --all
```

您应该能看到类似这样的提交历史：
```
* abc1234 (HEAD -> main) docs: 补充实验报告与文档
* def5678 feat: 添加 caption 生成脚本（greedy search）
* ghi90ab feat: 实现训练 loop 与 cross entropy loss
* jklcdef feat: 组装 Mini-BLIP2 完整模型
* mno0123 feat: 接入 frozen OPT-125m 作为语言解码器
* pqr4567 feat: 添加 projection layer 对齐到 OPT 词向量空间
* stu89ab feat: 实现 Mini Q-Former 模块（含 learnable queries）
* vwxcde feat: 接入 CLIP ViT-B/32 作为 frozen vision encoder
* yz12345 feat: 加载 Flickr8k 前 200 张图片与 caption
```

---

## 快速检查命令

```powershell
# 查看当前状态
git status

# 查看已暂存的更改
git diff --cached

# 查看提交历史
git log --oneline
```

---

## AI 对话记录要求

根据 requirements.md，您需要使用 **entir.io**（或同类工具）记录与 AI 的全部开发对话。

### 步骤：
1. 访问 https://entir.io
2. 开始录制会话
3. 复制粘贴您与我的对话（从最开始到现在）
4. 保存并获取可分享链接
5. 在实验报告中记录该链接

### 对话记录示例：
```
AI 对话记录：https://entir.io/s/xxxxxx
使用模型：Claude / ChatGPT
对话时长：累计约 X 小时，分 X 次会话
```
