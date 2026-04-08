#!/usr/bin/env python3
"""PureCNN v6 학습 — 50개 intent, 확장 데이터"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import json, os

class PureCNN(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_classes=50, max_len=32,
                 kernels=[3,5,7,3]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        layers = []
        for k in kernels:
            layers += [
                nn.Conv1d(d_model, d_model, k, padding=k//2),
                nn.ReLU(),
                nn.BatchNorm1d(d_model),
            ]
        self.convs = nn.Sequential(*layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.max_len = max_len

    def forward(self, x):
        seq_len = min(x.shape[1], self.max_len)
        x = self.embedding(x[:, :seq_len].long())  # [B, T, D]
        x = x.permute(0, 2, 1)  # [B, D, T]
        x = self.convs(x)
        x = x.mean(dim=2)  # global avg pool
        return self.fc(x)

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx], padding="max_length", truncation=True,
                                max_length=self.max_len, return_tensors="pt")
        return {"input_ids": tokens["input_ids"].squeeze(0), "label": self.labels[idx]}

def train(tag="v6"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("tokenizer/")

    # Data
    smarthome = pd.read_csv(f"data/smarthome_intent_{tag}.csv")
    kochat = pd.read_csv(f"data/kochat_intent_{tag}.csv")
    print(f"Smarthome: {len(smarthome)}, Kochat: {len(kochat)}")

    all_labels = sorted(set(smarthome["label"].tolist() + kochat["label"].tolist()))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_classes = len(label2id)
    print(f"Classes: {num_classes}")

    all_texts = smarthome["question"].tolist() + kochat["question"].tolist()
    all_labels_id = [label2id[l] for l in smarthome["label"].tolist() + kochat["label"].tolist()]

    indices = np.random.RandomState(42).permutation(len(all_texts))
    split = int(len(indices) * 0.9)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = IntentDataset([all_texts[i] for i in train_idx],
                              [all_labels_id[i] for i in train_idx], tokenizer, 32)
    val_ds = IntentDataset([all_texts[i] for i in val_idx],
                            [all_labels_id[i] for i in val_idx], tokenizer, 32)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=64, num_workers=4)

    model = PureCNN(vocab_size=tokenizer.vocab_size, d_model=128,
                    num_classes=num_classes, max_len=32).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(30):
        model.train()
        total_loss = correct = total = 0
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
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device).float()
                labels = torch.tensor(batch["label"]).to(device)
                logits = model(input_ids)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/30 | loss={total_loss/len(train_dl):.4f} | train={correct/total:.4f} | val={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"state": model.state_dict(), "l2i": label2id, "i2l": id2label, "va": val_acc},
                       f"checkpoints/cnn_4L_{tag}_best.pt")
            print(f"  → Saved (val={val_acc:.4f})")

    # Save label_map
    with open("checkpoints/label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    print(f"\nBest val_acc: {best_val_acc:.4f}")

    # Test evaluation
    test = pd.read_csv(f"data/test_{tag}.csv")
    test_texts = test["question"].tolist()
    test_labels = test["label"].tolist()

    model.load_state_dict(torch.load(f"checkpoints/cnn_4L_{tag}_best.pt",
                                      map_location=device, weights_only=False)["state"])
    model.eval()

    correct = 0
    errors = []
    with torch.no_grad():
        for q, expected in zip(test_texts, test_labels):
            tokens = tokenizer(q, padding="max_length", truncation=True,
                              max_length=32, return_tensors="pt")
            input_ids = tokens["input_ids"].to(device).float()
            logits = model(input_ids)
            pred_id = logits.argmax(1).item()
            pred = id2label[pred_id]
            if pred == expected:
                correct += 1
            else:
                errors.append((q, expected, pred))

    acc = correct / len(test_texts)
    print(f"\nTest accuracy: {acc:.4f} ({correct}/{len(test_texts)})")
    print(f"Errors ({len(errors)}):")
    for q, exp, pred in errors[:20]:
        print(f"  '{q}' → {pred} (정답: {exp})")

    return model

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train("v6")
