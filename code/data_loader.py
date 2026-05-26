import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import pandas as pd


class Flickr8kDataset(Dataset):
    """
    Flickr8k 数据集加载器
    用于加载 Flickr8k 的前 200 张图片及其对应的 caption
    """

    def __init__(
        self,
        images_dir: str,
        captions_file: str,
        clip_processor: CLIPProcessor,
        max_samples: int = 200,
        split: str = "train"
    ):
        """
        初始化 Flickr8k 数据集

        Args:
            images_dir: 图片目录路径
            captions_file: caption 文件路径
            clip_processor: CLIP 模型的处理器
            max_samples: 最大样本数，默认 200
            split: 数据集划分 ('train' 或 'test')
        """
        self.images_dir = Path(images_dir)
        self.clip_processor = clip_processor
        self.max_samples = max_samples
        self.split = split

        self.captions_dict = self._load_captions(captions_file)
        self.image_files = self._get_image_files()

        self.image_ids = list(self.captions_dict.keys())[:max_samples]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def _load_captions(self, captions_file: str) -> Dict[str, List[str]]:
        """
        加载 caption 文件

        Flickr8k 的 caption 文件格式 (CSV):
        image.jpg,caption
        """
        import csv
        captions_dict = {}
        line_count = 0

        with open(captions_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)

            for row in reader:
                line_count += 1

                if len(row) >= 2:
                    image_filename = row[0].strip()
                    caption = ','.join(row[1:]).strip()

                    if image_filename.endswith('.jpg'):
                        image_id = image_filename[:-4]
                    else:
                        image_id = image_filename

                    if image_id not in captions_dict:
                        captions_dict[image_id] = []
                    captions_dict[image_id].append(caption)

        print(f"Loaded {len(captions_dict)} unique images from {line_count} lines")
        return captions_dict

    def _get_image_files(self) -> List[str]:
        """获取所有可用的图片文件名"""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        image_files = []

        for file in self.images_dir.iterdir():
            if file.suffix.lower() in supported_formats:
                image_files.append(file.name)

        return image_files

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个样本

        Returns:
            dict: 包含以下键的字典
                - image: 处理后的图片张量
                - captions: 该图片的所有 captions 列表
                - image_id: 图片 ID
                - image_path: 图片路径
        """
        if idx >= len(self.image_ids):
            raise IndexError("Index out of range")

        image_id = self.image_ids[idx]

        image_filename = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            if (self.images_dir / (image_id + ext)).exists():
                image_filename = image_id + ext
                break

        if image_filename is None:
            for file in self.image_files:
                if file.startswith(image_id) or image_id in file:
                    image_filename = file
                    break

        if image_filename is None:
            image_filename = image_id if image_id in self.image_files else f"{image_id}.jpg"

        image_path = self.images_dir / image_filename

        try:
            image = Image.open(image_path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Could not open image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='white')

        processed = self.clip_processor(
            images=image,
            return_tensors="pt"
        )

        captions = self.captions_dict.get(image_id, ["No caption available"])

        return {
            'image': processed['pixel_values'].squeeze(0),
            'caption': captions[0],  # 只返回第一个 caption
            'image_id': image_id,
            'image_path': str(image_path)
        }


class MiniBLIP2DataCollator:
    """
    数据整理器，用于批处理数据
    """

    def __init__(self, tokenizer, max_length: int = 77):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict:
        """
        整理批次数据

        Args:
            batch: 样本列表

        Returns:
            dict: 整理后的批次数据
        """
        images = torch.stack([item['image'] for item in batch])

        all_captions = []
        for item in batch:
            all_captions.extend(item['captions'])

        encoded = self.tokenizer(
            all_captions,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'images': images,
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'captions_per_image': [len(item['captions']) for item in batch]
        }


def prepare_flickr8k_data(
    data_root: str,
    max_samples: int = 200,
    train_ratio: float = 0.8,
    batch_size: int = 4
) -> Tuple[DataLoader, DataLoader, CLIPProcessor]:
    """
    准备 Flickr8k 数据集

    Args:
        data_root: 数据根目录
        max_samples: 最大样本数
        train_ratio: 训练集比例
        batch_size: 批处理大小

    Returns:
        tuple: (train_loader, test_loader, clip_processor)
    """
    print("Loading CLIP model and processor...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)

    images_dir = os.path.join(data_root, "Images")
    captions_file = os.path.join(data_root, "captions.txt")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Captions file not found: {captions_file}")

    print(f"Loading dataset with {max_samples} samples...")
    dataset = Flickr8kDataset(
        images_dir=images_dir,
        captions_file=captions_file,
        clip_processor=clip_processor,
        max_samples=max_samples
    )

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader, clip_processor


def create_sample_data(output_dir: str, num_samples: int = 200):
    """
    创建示例数据文件（用于测试）
    当没有真实 Flickr8k 数据时，创建模拟数据
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sample_data = {
        "sample_count": num_samples,
        "description": "Mini dataset for testing Mini-BLIP2 model"
    }

    with open(output_path / "dataset_info.json", 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    print(f"Sample data created at: {output_path}")


if __name__ == "__main__":
    print("Testing data loading module...")
    print("Please ensure you have downloaded the Flickr8k dataset first.")
    print("Dataset should be organized as:")
    print("  data/")
    print("    Images/")
    print("      image1.jpg")
    print("      image2.jpg")
    print("      ...")
    print("    captions.txt")
