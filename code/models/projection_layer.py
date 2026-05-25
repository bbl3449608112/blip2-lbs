"""
Projection Layer
将 Q-Former 的输出投影到语言模型的词向量空间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionLayer(nn.Module):
    """
    投影层
    将视觉特征投影到语言模型的词向量空间
    """

    def __init__(
        self,
        q_former_dim: int = 768,
        language_dim: int = 768,
        num_queries: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.q_former_dim = q_former_dim
        self.language_dim = language_dim

        self.projection = nn.Sequential(
            nn.Linear(q_former_dim, language_dim),
            nn.LayerNorm(language_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(language_dim, language_dim)
        )

        print(f"✓ Projection layer created!")
        print(f"  - Q-Former dim: {q_former_dim}")
        print(f"  - Language dim: {language_dim}")

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            query_features: [batch_size, num_queries, q_former_dim]

        Returns:
            projected_features: [batch_size, num_queries, language_dim]
        """
        return self.projection(query_features)


if __name__ == "__main__":
    print("Testing ProjectionLayer...")

    proj_layer = ProjectionLayer(
        q_former_dim=768,
        language_dim=768,
        num_queries=32
    )

    dummy_input = torch.randn(2, 32, 768)

    output = proj_layer(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Test passed!")
