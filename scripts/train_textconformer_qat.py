#!/usr/bin/env python3
"""TextConformer QAT 학습 — FakeQuantize로 양자화 내성 학습"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.quantization import QuantStub, DeQuantStub
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import json, os

from train_textconformer import TextConformer, IntentDataset

class TextConformerQAT(nn.Module):
    """TextConformer + FakeQuantize"""
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        # FakeQuantize at 3 positions (Conformer STT QAT과 동일)
        self.quant_input = torch.quantization.FakeQuantize(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            quant_min=0, quant_max=255, dtype=torch.quint8)
        self.quant_mid = torch.quantization.FakeQuantize(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            quant_min=0, quant_max=255, dtype=torch.quint8)
        self.quant_output = torch.quantization.FakeQuantize(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            quant_min=0, quant_max=255, dtype=torch.quint8)

    def forward(self, x):
        # 입력 양자화
        seq_len = min(x.shape[1], self.model.max_len)
        emb = self.model.embedding(x[:, :seq_len].long())
        pos = self.model.pos_embedding(torch.arange(seq_len, device=x.device))
        x = emb + pos
        x = self.quant_input(x)

        # Conformer blocks
        for i, block in enumerate(self.model.blocks):
            x = block(x)
            if i == len(self.model.blocks) // 2:
                x = self.quant_mid(x)  # 중간 양자화

        # 출력 양자화
        cls = x[:, 0, :]
        logits = self.model.intent_fc(cls)
        logits = self.quant_output(logits)
        return logits


def train_qat(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("tokenizer/")

    # 기존 학습된 모델 로드
    ckpt = torch.load("checkpoints/textconformer_small_2L_best.pt",
                       map_location="cpu", weights_only=False)
    base_config = ckpt["config"]
    label2id = ckpt["label2id"]
    id2label = ckpt["id2label"]

    base_model = TextConformer(
        vocab_size=tokenizer.vocab_size,
        d_model=base_config["d_model"], num_classes=len(label2id),
        num_layers=base_config["num_layers"], nhead=base_config["nhead"],
        conv_kernel=base_config["conv_kernel"], ff_dim=base_config["ff_dim"],
        max_len=base_config["max_len"])
    base_model.load_state_dict(ckpt["model_state_dict"])

    # QAT wrapper
    model = TextConformerQAT(base_model).to(device)
    print(f"QAT Model loaded (base val_acc: {ckpt['val_acc']:.4f})")

    # Data
    smarthome = pd.read_csv("data/smarthome_intent.csv")
    kochat = pd.read_csv("data/kochat_intent.csv")
    all_texts = smarthome["question"].tolist() + kochat["question"].tolist()
    all_labels = [label2id[l] for l in smarthome["label"].tolist() + kochat["label"].tolist()]

    indices = np.random.RandomState(42).permutation(len(all_texts))
    split = int(len(indices) * 0.9)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = IntentDataset([all_texts[i] for i in train_idx],
                              [all_labels[i] for i in train_idx], tokenizer, 32)
    val_ds = IntentDataset([all_texts[i] for i in val_idx],
                            [all_labels[i] for i in val_idx], tokenizer, 32)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=64, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Margin loss (Conformer STT QAT과 동일)
    margin_target = config.get("margin_target", 0.3)
    margin_lambda = config.get("margin_lambda", 0.1)

    best_val_acc = 0
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_dl:
            input_ids = batch["input_ids"].to(device).float()
            labels = torch.tensor(batch["label"]).to(device)

            logits = model(input_ids)
            ce_loss = criterion(logits, labels)

            # Margin loss
            sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
            margin = sorted_logits[:, 0] - sorted_logits[:, 1]
            margin_loss = torch.clamp(margin_target - margin, min=0).mean()

            loss = ce_loss + margin_lambda * margin_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device).float()
                labels = torch.tensor(batch["label"]).to(device)
                logits = model(input_ids)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"QAT Epoch {epoch+1}/{config['epochs']} | loss={total_loss/len(train_dl):.4f} | train={correct/total:.4f} | val={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # base model의 state_dict 저장 (QAT wrapper 없이)
            torch.save({
                "model_state_dict": model.model.state_dict(),
                "config": base_config,
                "label2id": label2id,
                "id2label": id2label,
                "val_acc": val_acc,
                "qat": True,
            }, "checkpoints/textconformer_small_2L_qat_best.pt")
            print(f"  → Saved QAT best (val_acc={val_acc:.4f})")

    print(f"\nQAT Best val_acc: {best_val_acc:.4f}")
    return model

if __name__ == "__main__":
    config = {
        "lr": 1e-4,
        "epochs": 20,
        "margin_target": 0.3,
        "margin_lambda": 0.1,
    }

    print("=" * 60)
    print("=== TextConformer QAT (FakeQuantize + MarginLoss) ===")
    print("=" * 60)
    train_qat(config)
