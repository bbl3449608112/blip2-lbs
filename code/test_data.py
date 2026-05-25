"""
测试数据加载模块

使用说明：
1. 首先下载 Flickr8k 数据集到 data/ 目录
2. 运行此脚本验证数据加载是否正常
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    Flickr8kDataset,
    MiniBLIP2DataCollator,
    prepare_flickr8k_data
)
from download_data import (
    Flickr8kDownloader,
    verify_data_structure
)


def test_data_structure():
    """测试数据结构"""
    print("=" * 60)
    print("Testing Data Structure...")
    print("=" * 60)

    result = verify_data_structure("data")

    if result["valid"]:
        print("✓ Data structure is valid!")
        print(f"  - Images found: {result['images_found']}")
        print(f"  - Captions found: {result['captions_found']}")
        return True
    else:
        print("✗ Data structure has issues:")
        for error in result["errors"]:
            print(f"  - {error}")
        return False


def test_clip_processor():
    """测试 CLIP 处理器"""
    print("\n" + "=" * 60)
    print("Testing CLIP Processor...")
    print("=" * 60)

    try:
        from transformers import CLIPProcessor, CLIPModel

        print("Loading CLIP model (openai/clip-vit-base-patch32)...")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("✓ CLIP processor loaded successfully!")

        return clip_processor

    except Exception as e:
        print(f"✗ Error loading CLIP: {e}")
        return None


def test_dataset_loading(clip_processor):
    """测试数据集加载"""
    print("\n" + "=" * 60)
    print("Testing Dataset Loading...")
    print("=" * 60)

    try:
        dataset = Flickr8kDataset(
            images_dir="data/Images",
            captions_file="data/captions.txt",
            clip_processor=clip_processor,
            max_samples=200
        )

        print(f"✓ Dataset created successfully!")
        print(f"  - Total samples: {len(dataset)}")

        sample = dataset[0]
        print(f"  - Sample image shape: {sample['image'].shape}")
        print(f"  - Number of captions: {len(sample['captions'])}")
        print(f"  - Sample caption: {sample['captions'][0][:50]}...")

        return dataset

    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None


def test_dataloader(dataset):
    """测试数据加载器"""
    print("\n" + "=" * 60)
    print("Testing DataLoader...")
    print("=" * 60)

    try:
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )

        batch = next(iter(dataloader))
        print(f"✓ DataLoader working!")
        print(f"  - Batch keys: {list(batch.keys())}")
        print(f"  - Batch image shape: {batch['image'].shape}")
        print(f"  - Number of captions in batch: {len(batch['captions'])}")

        return dataloader

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"✗ Error with DataLoader: {e}")
        return None


def main():
    """主测试流程"""
    print("\n" + "=" * 60)
    print("Mini-BLIP2 Data Loading Test Suite")
    print("=" * 60)

    print("\nStep 1: Checking data structure...")
    if not test_data_structure():
        print("\n⚠ Data structure check failed!")
        print("Please download and organize the Flickr8k dataset first.")
        print("Run 'python code/download_data.py --help' for instructions.")

        downloader = Flickr8kDownloader("data")
        downloader.print_setup_instructions()
        return False

    print("\nStep 2: Loading CLIP processor...")
    clip_processor = test_clip_processor()
    if not clip_processor:
        return False

    print("\nStep 3: Loading dataset...")
    dataset = test_dataset_loading(clip_processor)
    if not dataset:
        return False

    print("\nStep 4: Testing DataLoader...")
    dataloader = test_dataloader(dataset)
    if not dataloader:
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nData preparation is ready for training!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
