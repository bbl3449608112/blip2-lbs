"""
Frozen Language Decoder
使用 OPT-125m 作为冻结的语言解码器
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import OPTForCausalLM, AutoTokenizer


class FrozenLanguageDecoder(nn.Module):
    """
    冻结的语言解码器
    使用 OPT-125m 进行文本生成
    """

    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()

        self.device = device
        self.model_name = model_name

        print(f"Loading OPT language decoder: {model_name}")
        self.model = OPTForCausalLM.from_pretrained(model_name, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model.to(device)

        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

        print(f"✓ Frozen language decoder loaded!")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Vocab size: {self.vocab_size}")
        print(f"  - Device: {device}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        visual_prefix: Optional[torch.Tensor] = None,
        use_grad: bool = False
    ):
        """
        前向传播

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            visual_prefix: [batch_size, num_queries, hidden_size]
                          视觉前缀特征
            use_grad: 是否允许梯度传播（训练时为True，推理时为False）

        Returns:
            outputs: 模型输出
        """
        # 只有在推理时使用 no_grad
        context_manager = torch.no_grad() if not use_grad else torch.enable_grad()
        
        with context_manager:
            if visual_prefix is not None:
                inputs_embeds = self.model.model.decoder.embed_tokens(input_ids)
                
                # 确保数据类型一致
                visual_prefix = visual_prefix.to(inputs_embeds.dtype)
                
                # 简单版本：假设 visual_prefix 和 inputs_embeds 的 batch size 已经一致
                inputs_embeds = torch.cat([visual_prefix, inputs_embeds], dim=1)

                if attention_mask is not None:
                    visual_mask = torch.ones(
                        visual_prefix.size(0),
                        visual_prefix.size(1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )

        return outputs

    def generate(
        self,
        visual_prefix: torch.Tensor,
        max_length: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        do_sample: bool = True
    ):
        """
        生成文本

        Args:
            visual_prefix: [batch_size, num_queries, hidden_size]
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: top-k 采样
            do_sample: 是否采样

        Returns:
            generated_ids: [batch_size, seq_len]
        """
        batch_size = visual_prefix.size(0)

        input_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=self.device
        )

        generated = []

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(
                    input_ids=input_ids,
                    visual_prefix=visual_prefix
                )

                next_token_logits = outputs.logits[:, -1, :]

                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, next_token], dim=-1)
                generated.append(next_token)

                if (next_token == self.tokenizer.eos_token_id).all():
                    break

        return input_ids

    def decode(self, token_ids):
        """
        解码 token ids 为文本
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


if __name__ == "__main__":
    print("Testing FrozenLanguageDecoder...")

    decoder = FrozenLanguageDecoder()

    dummy_visual = torch.randn(2, 32, 768)

    generated_ids = decoder.generate(
        visual_prefix=dummy_visual,
        max_length=20,
        do_sample=False
    )

    print(f"Generated ids shape: {generated_ids.shape}")
    print("Generated text:")
    for i in range(len(generated_ids)):
        text = decoder.decode(generated_ids[i])
        print(f"  {i}: {text}")
    print("✓ Test passed!")
