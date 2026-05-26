"""
生成脚本
测试 Mini-BLIP2 并生成图像描述
"""

import os
import sys
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import Flickr8kDataset
from models.mini_blip2 import MiniBLIP2
from transformers import CLIPImageProcessor


class Generator:
    """
    生成器
    """

    def __init__(
        self,
        model: MiniBLIP2,
        image_processor: CLIPImageProcessor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.model.eval()
        self.image_processor = image_processor
        self.device = device

    def generate_from_path(self, image_path: str, **kwargs) -> str:
        """
        从图像路径生成描述
        """
        image = Image.open(image_path).convert('RGB')
        return self.generate_from_image(image, **kwargs)

    def generate_from_image(self, image: Image.Image, **kwargs) -> str:
        """
        从 PIL 图像生成描述
        """
        inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)

        captions = self.model.generate_caption(
            pixel_values=pixel_values,
            **kwargs
        )

        return captions[0]


def test_on_dataset():
    """
    在测试集上测试模型
    """
    print("=" * 60)
    print("Testing Mini-BLIP2 on Flickr8k Dataset")
    print("=" * 60 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = MiniBLIP2(device=device)

    checkpoint_path = Path("checkpoints/best_model.pth")
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Checkpoint loaded!")

    from transformers import CLIPImageProcessor
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("\nLoading dataset...")
    dataset = Flickr8kDataset(
        images_dir="data/Images",
        captions_file="data/captions.txt",
        clip_processor=image_processor,
        max_samples=200
    )

    print(f"Dataset size: {len(dataset)}")

    generator = Generator(model, image_processor, device)

    num_test_samples = min(5, len(dataset))
    indices = list(range(len(dataset)))[:num_test_samples]

    results = []

    print("\n" + "=" * 60)
    print("Generating Captions")
    print("=" * 60 + "\n")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image_id = sample['image_id']
        image_path = sample['image_path']
        ground_truth_caption = sample['caption']

        print(f"\n--- Sample {i+1}/{num_test_samples} ---")
        print(f"Image ID: {image_id}")
        print(f"Image: {image_path}")

        generated_caption = generator.generate_from_path(
            image_path,
            max_length=50,
            do_sample=False
        )

        print(f"\nGround Truth Caption:")
        print(f"  {ground_truth_caption}")

        print(f"\nGenerated Caption:")
        print(f"  {generated_caption}")

        results.append({
            'image_id': image_id,
            'image_path': image_path,
            'ground_truth': ground_truth_caption,
            'generated': generated_caption
        })

    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "test_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to {output_dir / 'test_results.json'}")

    return results


def generate_single_image(image_path: str):
    """
    为单张图片生成描述
    """
    print("=" * 60)
    print("Single Image Caption Generation")
    print("=" * 60 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = MiniBLIP2(device=device)

    from transformers import CLIPImageProcessor
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    generator = Generator(model, image_processor, device)

    print(f"\nGenerating caption for: {image_path}")

    caption = generator.generate_from_path(
        image_path,
        max_length=50,
        do_sample=False
    )

    print("\n" + "=" * 60)
    print("Generated Caption:")
    print(caption)
    print("=" * 60)


def main():
    """
    主函数
    """
    import argparse

    parser = argparse.ArgumentParser(description="Mini-BLIP2 Caption Generation")
    parser.add_argument(
        "--mode",
        type=str,
        default="dataset",
        choices=["dataset", "single"],
        help="Mode: 'dataset' (test on Flickr8k) or 'single' (single image)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to single image (for 'single' mode)"
    )

    args = parser.parse_args()

    if args.mode == "dataset":
        test_on_dataset()
    elif args.mode == "single":
        if not args.image_path:
            print("Error: Please specify --image_path for 'single' mode")
            return
        generate_single_image(args.image_path)


if __name__ == "__main__":
    main()
