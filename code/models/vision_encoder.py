"""
Frozen Vision Encoder
使用 CLIP ViT-B/32 作为冻结的视觉编码器
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor


class FrozenVisionEncoder(nn.Module):
    """
    冻结的视觉编码器
    使用 CLIP ViT-B/32 提取视觉特征
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()

        self.device = device
        self.model_name = model_name

        print(f"Loading CLIP vision encoder: {model_name}")
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)

        for param in self.vision_model.parameters():
            param.requires_grad = False

        self.vision_model.eval()
        self.vision_model.to(device)

        self.hidden_size = self.vision_model.config.hidden_size

        print(f"✓ Frozen vision encoder loaded!")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Device: {device}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            pixel_values: [batch_size, 3, 224, 224]

        Returns:
            visual_features: [batch_size, num_patches, hidden_size]
        """
        with torch.no_grad():
            outputs = self.vision_model(pixel_values=pixel_values)
            last_hidden_state = outputs.last_hidden_state

        return last_hidden_state

    def get_visual_features(self, images):
        """
        获取视觉特征（备用方法）
        """
        if isinstance(images, list):
            inputs = self.image_processor(images, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)
        else:
            pixel_values = images.to(self.device)

        return self.forward(pixel_values)


if __name__ == "__main__":
    print("Testing FrozenVisionEncoder...")

    encoder = FrozenVisionEncoder()

    dummy_input = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = encoder(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Test passed!")
