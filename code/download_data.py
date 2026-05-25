import os
import zipfile
import requests
from pathlib import Path
from typing import Optional
import sys


class Flickr8kDownloader:
    """
    Flickr8k 数据集下载器

    由于 Kaggle 数据集需要认证才能下载，
    这里提供手动下载指南和自动检查功能
    """

    DATASET_URLS = {
        "kaggle": "https://www.kaggle.com/datasets/adityajn105/flickr8k/download",
        "github_mirror": "https://github.com/CadikMF/dataset/archive/refs/heads/main.zip"
    }

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "Images"
        self.captions_file = self.output_dir / "captions.txt"

    def check_dataset_exists(self) -> bool:
        """
        检查数据集是否已存在

        Returns:
            bool: 数据集是否存在
        """
        if not self.images_dir.exists():
            return False

        if not self.captions_file.exists():
            return False

        image_count = len(list(self.images_dir.glob("*.jpg")))
        print(f"Found {image_count} images in {self.images_dir}")

        if image_count == 0:
            return False

        return True

    def download_from_kaggle(self, kaggle_dataset: str = "adityajn105/flickr8k") -> bool:
        """
        从 Kaggle 下载数据集（需要 kaggle CLI）

        Args:
            kaggle_dataset: Kaggle 数据集名称

        Returns:
            bool: 下载是否成功
        """
        try:
            import kaggle
        except ImportError:
            print("Kaggle library not installed. Installing...")
            os.system("pip install kaggle")
            import kaggle

        print(f"Downloading {kaggle_dataset} from Kaggle...")
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        print("Downloading dataset...")
        api.dataset_download_files(
            kaggle_dataset,
            path=self.output_dir,
            unzip=True
        )

        self._organize_dataset()
        return True

    def _organize_dataset(self):
        """整理下载的数据集"""
        print("Organizing dataset...")

        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                if file.endswith('.zip'):
                    zip_path = Path(root) / file
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(self.output_dir)
                        zip_path.unlink()
                        print(f"Extracted and removed: {file}")
                    except:
                        pass

        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name != "Images":
                for sub_item in item.iterdir():
                    if sub_item.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                        sub_item.rename(self.images_dir / sub_item.name)

    def create_sample_captions(self, num_samples: int = 200):
        """
        创建示例 caption 文件（用于测试）

        Args:
            num_samples: 样本数量
        """
        self.images_dir.mkdir(parents=True, exist_ok=True)

        sample_captions = []

        for i in range(num_samples):
            image_id = f"sample_{i:04d}.jpg"
            for j in range(5):
                caption = f"A sample image description for image {i}, caption {j}."
                sample_captions.append(f"{image_id}#{j}\t{caption}")

        with open(self.captions_file, 'w', encoding='utf-8') as f:
            f.write("image\tcaption\n")
            f.write("\n".join(sample_captions))

        print(f"Created sample captions file with {num_samples} images")

    def print_setup_instructions(self):
        """打印数据集设置说明"""
        print("\n" + "="*60)
        print("Flickr8k Dataset Setup Instructions")
        print("="*60)
        print("\nOption 1: Manual Download from Kaggle")
        print("-" * 40)
        print("1. Go to: https://www.kaggle.com/datasets/adityajn105/flickr8k")
        print("2. Click 'Download' button")
        print("3. Extract the ZIP file")
        print(f"4. Move 'Images' folder to: {self.output_dir / 'Images'}")
        print(f"5. Move 'captions.txt' to: {self.captions_file}")
        print("\nOption 2: Using Kaggle CLI")
        print("-" * 40)
        print("1. Install kaggle: pip install kaggle")
        print("2. Set up Kaggle API credentials")
        print("3. Run: kaggle datasets download -d adityajn105/flickr8k")
        print("4. Extract and organize the files")
        print("\nOption 3: Download from Alternative Sources")
        print("-" * 40)
        print("1. Search for 'Flickr8k dataset download'")
        print("2. Download from a mirror site")
        print("3. Extract and organize the files")
        print("\n" + "="*60)
        print("\nExpected dataset structure:")
        print(f"  {self.output_dir}/")
        print(f"    Images/")
        print(f"      image1.jpg")
        print(f"      image2.jpg")
        print(f"      ...")
        print(f"    captions.txt")
        print("="*60 + "\n")


def verify_data_structure(data_dir: str) -> dict:
    """
    验证数据目录结构

    Args:
        data_dir: 数据目录路径

    Returns:
        dict: 验证结果
    """
    data_path = Path(data_dir)
    result = {
        "valid": False,
        "images_found": 0,
        "captions_found": False,
        "errors": []
    }

    images_dir = data_path / "Images"
    captions_file = data_path / "captions.txt"

    if not images_dir.exists():
        result["errors"].append(f"Images directory not found: {images_dir}")
    else:
        image_files = list(images_dir.glob("*.jpg")) + \
                      list(images_dir.glob("*.jpeg")) + \
                      list(images_dir.glob("*.png"))
        result["images_found"] = len(image_files)
        if result["images_found"] == 0:
            result["errors"].append("No images found in Images directory")

    if not captions_file.exists():
        result["errors"].append(f"Captions file not found: {captions_file}")
    else:
        result["captions_found"] = True
        with open(captions_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            result["caption_lines"] = len(lines)

    result["valid"] = len(result["errors"]) == 0

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flickr8k Dataset Setup")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data directory path"
    )
    parser.add_argument(
        "--create_sample",
        action="store_true",
        help="Create sample data for testing"
    )

    args = parser.parse_args()

    downloader = Flickr8kDownloader(args.data_dir)

    if args.create_sample:
        print("Creating sample dataset...")
        downloader.create_sample_captions()
    else:
        if downloader.check_dataset_exists():
            print("✓ Dataset found!")
            result = verify_data_structure(args.data_dir)
            print(f"  Images: {result['images_found']}")
            print(f"  Captions file: {'Yes' if result['captions_found'] else 'No'}")
        else:
            print("✗ Dataset not found")
            downloader.print_setup_instructions()
