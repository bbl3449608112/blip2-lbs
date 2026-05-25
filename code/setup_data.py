"""
数据初始化脚本

用于创建必要的目录结构和示例数据
运行此脚本前，请先下载 Flickr8k 数据集
"""

import os
import sys
from pathlib import Path


def create_directory_structure():
    """创建必要的目录结构"""
    print("Creating directory structure...")

    directories = [
        "data/Images",
        "data/captions",
        "code/models",
        "code/utils",
        "report/images",
        "report/logs"
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {dir_path}")

    print("Directory structure created!\n")


def check_python_packages():
    """检查必要的 Python 包"""
    print("Checking Python packages...")

    required_packages = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers",
        "PIL": "Pillow",
        "numpy": "NumPy",
        "tqdm": "tqdm"
    }

    missing_packages = []

    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False

    return True


def download_instructions():
    """打印下载说明"""
    print("\n" + "=" * 60)
    print("📥 Flickr8k Dataset Download Instructions")
    print("=" * 60)
    print("""
1. Kaggle Download (Recommended):
   - Visit: https://www.kaggle.com/datasets/adityajn105/flickr8k
   - Click the 'Download' button
   - Extract the ZIP file

2. Alternative Sources:
   - Search for 'Flickr8k dataset download' online
   - Many academic mirrors are available

3. After Download:
   - Extract the ZIP file
   - You should have:
     📁 Flickr8k/
        📁 Images/
           🖼️ *.jpg (8000 images)
        📄 captions.txt

   - Copy 'Images' folder to: data/Images/
   - Copy 'captions.txt' to: data/captions.txt

4. Verify:
   - Run: python code/test_data.py
   - Should see "✓ All tests passed!"
""")
    print("=" * 60 + "\n")


def verify_installation():
    """验证安装"""
    print("Verifying installation...")

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")

        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")

        from transformers import CLIPProcessor, CLIPModel
        print("  ✓ CLIP models available")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🔧 Mini-BLIP2 Data Setup")
    print("=" * 60 + "\n")

    print("Step 1: Creating directory structure...")
    create_directory_structure()

    print("Step 2: Checking Python packages...")
    if not check_python_packages():
        print("\n⚠ Please install missing packages first!")
        return

    print("\nStep 3: Verifying installation...")
    if not verify_installation():
        print("\n⚠ Installation verification failed!")
        return

    download_instructions()

    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Download the Flickr8k dataset")
    print("2. Organize the data as shown above")
    print("3. Run: python code/test_data.py")


if __name__ == "__main__":
    main()
