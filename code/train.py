"""
训练脚本
训练 Mini-BLIP2 进行图像描述生成
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import Flickr8kDataset, prepare_flickr8k_data
from models.mini_blip2 import MiniBLIP2


class Trainer:
    """
    训练器
    """

    def __init__(
        self,
        model: MiniBLIP2,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir

        self.optimizer = optim.AdamW(
            model.get_trainable_parameters(),
            lr=learning_rate
        )

        self.criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []

        print(f"✓ Trainer initialized!")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Number of epochs: {num_epochs}")
        print(f"  - Device: {device}")

    def train_epoch(self, epoch: int):
        """
        训练一个 epoch
        """
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch in pbar:
            self.optimizer.zero_grad()

            pixel_values = batch['image'].to(self.device)

            captions = batch['caption']  # 现在是单个 caption，不是列表了

            tokenizer = self.model.language_decoder.tokenizer
            encoded = tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=50,
                return_tensors="pt"
            )

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids[:, :-1],
                attention_mask=attention_mask[:, :-1]
            )

            logits = outputs.logits

            # 只对语言部分（去掉视觉前缀）计算 loss
            num_visual_tokens = 16
            language_logits = logits[:, num_visual_tokens:, :]

            labels = input_ids[:, 1:].contiguous()
            loss = self.criterion(language_logits.reshape(-1, language_logits.size(-1)), labels.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self):
        """
        验证
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                pixel_values = batch['image'].to(self.device)

                captions = batch['caption']  # 现在是单个 caption，不是列表了

                tokenizer = self.model.language_decoder.tokenizer
                encoded = tokenizer(
                    captions,
                    padding=True,
                    truncation=True,
                    max_length=50,
                    return_tensors="pt"
                )

                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids[:, :-1],
                    attention_mask=attention_mask[:, :-1]
                )

                logits = outputs.logits

                # 只对语言部分（去掉视觉前缀）计算 loss
                num_visual_tokens = 16
                language_logits = logits[:, num_visual_tokens:, :]

                labels = input_ids[:, 1:].contiguous()
                loss = self.criterion(language_logits.reshape(-1, language_logits.size(-1)), labels.reshape(-1))

                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)

        return avg_loss

    def train(self):
        """
        完整训练流程
        """
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60 + "\n")

        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.num_epochs} ---")

            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f"{self.save_dir}/best_model.pth")
                print(f"✓ Best model saved (val loss: {best_val_loss:.4f})")

            self.save_checkpoint(f"{self.save_dir}/last_model.pth")

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        self.save_training_log()

    def save_checkpoint(self, path: str):
        """
        保存模型
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)

    def save_training_log(self):
        """
        保存训练日志
        """
        log = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        with open(f"{self.save_dir}/training_log.json", 'w') as f:
            json.dump(log, f, indent=2)

        print(f"\nTraining log saved to {self.save_dir}/training_log.json")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("Mini-BLIP2 Training")
    print("=" * 60 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nLoading data...")
    train_loader, val_loader, clip_processor = prepare_flickr8k_data(
        data_root="data",
        max_samples=200,
        train_ratio=0.8,
        batch_size=8
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    print("\nBuilding model...")
    model = MiniBLIP2(device=device)

    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        num_epochs=2,
        device=device,
        save_dir="checkpoints"
    )

    trainer.train()

    print("\nDone!")


if __name__ == "__main__":
    main()
