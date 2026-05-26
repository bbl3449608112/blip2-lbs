"""
Mini-BLIP2
完整的图像描述生成模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .vision_encoder import FrozenVisionEncoder
from .q_former import MiniQFormer
from .projection_layer import ProjectionLayer
from .language_decoder import FrozenLanguageDecoder


class MiniBLIP2(nn.Module):
    """
    Mini-BLIP2 模型
    Image -> Frozen Vision Encoder -> Mini Q-Former -> Projection -> Frozen Language Decoder -> Caption
    """

    def __init__(
        self,
        vision_encoder_name: str = "openai/clip-vit-base-patch32",
        language_decoder_name: str = "facebook/opt-125m",
        num_queries: int = 16,
        q_former_layers: int = 2,
        q_former_heads: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()

        self.device = device

        print("=" * 60)
        print("Building Mini-BLIP2 Model")
        print("=" * 60)

        self.vision_encoder = FrozenVisionEncoder(
            model_name=vision_encoder_name,
            device=device
        )

        self.q_former = MiniQFormer(
            num_queries=num_queries,
            visual_dim=self.vision_encoder.hidden_size,
            d_model=768,
            n_heads=q_former_heads,
            n_layers=q_former_layers
        )

        self.projection = ProjectionLayer(
            q_former_dim=768,
            language_dim=768,
            num_queries=num_queries
        )

        self.language_decoder = FrozenLanguageDecoder(
            model_name=language_decoder_name,
            device=device
        )

        self._freeze_and_unfreeze()

        print("=" * 60)
        print("✓ Mini-BLIP2 model built successfully!")
        print("=" * 60)

    def _freeze_and_unfreeze(self):
        """
        设置需要训练和冻结的参数
        """
        total_params = 0
        trainable_params = 0

        for name, param in self.named_parameters():
            total_params += param.numel()

            if 'vision_encoder' in name or 'language_decoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                trainable_params += param.numel()

        print(f"\n  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable ratio: {trainable_params / total_params * 100:.2f}%")
        print(f"\n  Frozen: Vision Encoder + Language Decoder")
        print(f"  Trainable: Mini Q-Former + Projection Layer")

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        前向传播（训练时使用）

        Args:
            pixel_values: [batch_size, 3, 224, 224]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            outputs: 语言模型的输出
        """
        visual_features = self.vision_encoder(pixel_values)

        query_features = self.q_former(visual_features)

        projected_features = self.projection(query_features)

        # 确保数据类型和语言模型一致
        projected_features = projected_features.to(self.language_decoder.model.dtype)

        outputs = self.language_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_prefix=projected_features,
            use_grad=True  # 训练时需要梯度
        )

        return outputs

    def generate_caption(
        self,
        pixel_values: torch.Tensor,
        max_length: int = 50,
        temperature: float = 0.7,
        do_sample: bool = False
    ) -> List[str]:
        """
        生成图像描述

        Args:
            pixel_values: [batch_size, 3, 224, 224]
            max_length: 最大生成长度
            temperature: 温度参数
            do_sample: 是否采样

        Returns:
            captions: 生成的文本描述列表
        """
        visual_features = self.vision_encoder(pixel_values)

        query_features = self.q_former(visual_features)

        projected_features = self.projection(query_features)

        generated_ids = self.language_decoder.generate(
            visual_prefix=projected_features,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample
        )

        captions = []
        for i in range(len(generated_ids)):
            caption = self.language_decoder.decode(generated_ids[i])
            captions.append(caption)

        return captions

    def get_trainable_parameters(self):
        """
        获取需要训练的参数
        """
        return [p for p in self.parameters() if p.requires_grad]


if __name__ == "__main__":
    print("Testing MiniBLIP2...")

    model = MiniBLIP2()

    dummy_images = torch.randn(2, 3, 224, 224)

    print("\nTesting forward pass...")
    dummy_input_ids = torch.randint(0, 50000, (2, 10))
    outputs = model(dummy_images, input_ids=dummy_input_ids)
    print(f"  Output logits shape: {outputs.logits.shape}")

    print("\nTesting caption generation...")
    captions = model.generate_caption(
        dummy_images,
        max_length=20,
        do_sample=False
    )
    for i, cap in enumerate(captions):
        print(f"  Caption {i}: {cap}")

    print("\n✓ All tests passed!")
