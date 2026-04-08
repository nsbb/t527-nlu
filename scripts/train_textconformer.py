#!/usr/bin/env python3
"""TextConformer 학습 스크립트 — 한국어 의도 분류"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import json, os, time

# ============================================================
# Model: TextConformer
# ============================================================
class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead=4, conv_kernel=15, ff_dim=512, dropout=0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, ff_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model), nn.Dropout(dropout))
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, conv_kernel, padding=conv_kernel//2, groups=d_model),
            nn.ReLU(), nn.Conv1d(d_model, d_model, 1), nn.Dropout(dropout))
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, ff_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model), nn.Dropout(dropout))
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        attn_in = self.attn_norm(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + self.attn_drop(attn_out)
        conv_in = self.conv_norm(x).permute(0, 2, 1)
        x = x + self.conv(conv_in).permute(0, 2, 1)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)

class TextConformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_classes, num_layers=18,
                 nhead=8, conv_kernel=15, ff_dim=2048, max_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, nhead, conv_kernel, ff_dim) for _ in range(num_layers)])
        self.intent_fc = nn.Linear(d_model, num_classes)
        self.max_len = max_len

    def forward(self, x):
        seq_len = min(x.shape[1], self.max_len)
        emb = self.embedding(x[:, :seq_len].long())
        pos = self.pos_embedding(torch.arange(seq_len, device=x.device))
        x = emb + pos
        for block in self.blocks:
            x = block(x)
        return self.intent_fc(x[:, 0, :])

# ============================================================
# Dataset
# ============================================================
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx], padding="max_length", truncation=True,
                                max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "label": self.labels[idx],
        }

# ============================================================
# Training
# ============================================================
def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("tokenizer/")
    vocab_size = tokenizer.vocab_size

    # Data
    print("Loading data...")
    smarthome = pd.read_csv("data/smarthome_intent.csv")
    kochat = pd.read_csv("data/kochat_intent.csv")

    # label mapping
    all_labels = sorted(set(smarthome["label"].tolist() + kochat["label"].tolist()))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_classes = len(label2id)

    print(f"  Classes: {num_classes}")
    print(f"  Smarthome: {len(smarthome)}, Kochat: {len(kochat)}")

    # 합치기
    all_texts = smarthome["question"].tolist() + kochat["question"].tolist()
    all_labels_id = [label2id[l] for l in smarthome["label"].tolist() + kochat["label"].tolist()]

    # train/val split (90/10)
    indices = np.random.RandomState(42).permutation(len(all_texts))
    split = int(len(indices) * 0.9)
    train_idx, val_idx = indices[:split], indices[split:]

    train_texts = [all_texts[i] for i in train_idx]
    train_labels = [all_labels_id[i] for i in train_idx]
    val_texts = [all_texts[i] for i in val_idx]
    val_labels = [all_labels_id[i] for i in val_idx]

    print(f"  Train: {len(train_texts)}, Val: {len(val_texts)}")

    train_ds = IntentDataset(train_texts, train_labels, tokenizer, config["max_len"])
    val_ds = IntentDataset(val_texts, val_labels, tokenizer, config["max_len"])
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=config["batch_size"], num_workers=4)

    # Model
    model = TextConformer(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        num_classes=num_classes,
        num_layers=config["num_layers"],
        nhead=config["nhead"],
        conv_kernel=config["conv_kernel"],
        ff_dim=config["ff_dim"],
        max_len=config["max_len"],
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {params/1e6:.2f}M params, {config['num_layers']}L {config['d_model']}d")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training loop
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
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device).float()
                labels = torch.tensor(batch["label"]).to(device)
                logits = model(input_ids)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{config['epochs']} | loss={total_loss/len(train_dl):.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "label2id": label2id,
                "id2label": id2label,
                "val_acc": val_acc,
            }, f"checkpoints/textconformer_{config['tag']}_best.pt")
            print(f"  → Saved best (val_acc={val_acc:.4f})")

    print(f"\nBest val_acc: {best_val_acc:.4f}")

    # Save label mapping
    with open("checkpoints/label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    return model, label2id

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)

    # 빠른 검증: small (2L)
    config_small = {
        "tag": "small_2L",
        "d_model": 128,
        "num_layers": 2,
        "nhead": 4,
        "conv_kernel": 7,
        "ff_dim": 256,
        "max_len": 32,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 30,
    }

    print("=" * 60)
    print("=== TextConformer Small (2L, 128d) — 빠른 검증 ===")
    print("=" * 60)
    train(config_small)
