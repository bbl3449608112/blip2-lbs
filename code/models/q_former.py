"""
Mini Q-Former
查询式 Transformer，用于从冻结的视觉编码器提取特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = q.size(0)

        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(output)


class FeedForward(nn.Module):
    """
    前馈网络
    """

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class TransformerLayer(nn.Module):
    """
    Transformer 层
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            queries: [batch_size, num_queries, d_model]
            visual_features: [batch_size, num_patches, d_model]

        Returns:
            output: [batch_size, num_queries, d_model]
        """
        queries2 = self.norm1(queries)
        queries = queries + self.dropout1(self.self_attn(queries2, queries2, queries2))

        queries2 = self.norm2(queries)
        queries = queries + self.dropout2(self.cross_attn(queries2, visual_features, visual_features))

        queries2 = self.norm3(queries)
        queries = queries + self.dropout3(self.ff(queries2))

        return queries


class MiniQFormer(nn.Module):
    """
    Mini Q-Former
    简化版的查询式 Transformer
    """

    def __init__(
        self,
        num_queries: int = 32,
        visual_dim: int = 768,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 6,
        d_ff: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_queries = num_queries
        self.d_model = d_model

        self.query_embeddings = nn.Parameter(torch.randn(num_queries, d_model))

        self.visual_proj = nn.Linear(visual_dim, d_model) if visual_dim != d_model else nn.Identity()

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        print(f"✓ Mini Q-Former created!")
        print(f"  - Number of queries: {num_queries}")
        print(f"  - Hidden size: {d_model}")
        print(f"  - Number of layers: {n_layers}")
        print(f"  - Number of heads: {n_heads}")

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            visual_features: [batch_size, num_patches, visual_dim]

        Returns:
            query_outputs: [batch_size, num_queries, d_model]
        """
        batch_size = visual_features.size(0)

        visual_features = self.visual_proj(visual_features)

        queries = self.query_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)

        for layer in self.layers:
            queries = layer(queries, visual_features)

        queries = self.norm(queries)

        return queries


if __name__ == "__main__":
    print("Testing MiniQFormer...")

    q_former = MiniQFormer(
        num_queries=32,
        visual_dim=768,
        d_model=768,
        n_heads=12,
        n_layers=6
    )

    dummy_visual = torch.randn(2, 50, 768)

    output = q_former(dummy_visual)

    print(f"Input visual features shape: {dummy_visual.shape}")
    print(f"Output query features shape: {output.shape}")
    print("✓ Test passed!")
