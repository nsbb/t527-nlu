#!/usr/bin/env python3
"""5-Head CNN Training Script"""
import torch, torch.nn as nn, json, os, sys, time
from torch.utils.data import Dataset, DataLoader
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer

# ============================================================
# Dataset
# ============================================================
class MultiHeadDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        tk = self.tokenizer(d['utterance'], padding='max_length', truncation=True,
                           max_length=self.max_len, return_tensors='pt')
        input_ids = tk['input_ids'].squeeze(0)

        labels = {}
        for h in HEAD_NAMES:
            val = d['labels'].get(h, 'none')
            labels[h] = HEAD_L2I[h].get(val, 0)

        return input_ids, labels


def collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = {h: torch.tensor([b[1][h] for b in batch]) for h in HEAD_NAMES}
    return input_ids, labels


# ============================================================
# Training
# ============================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    with open('data/train_final.json', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('data/val_final.json', encoding='utf-8') as f:
        val_data = json.load(f)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('tokenizer/')

    # Embeddings from ko-sbert
    print("Loading ko-sbert embeddings...")
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach()
    del sbert
    print(f"Embedding: {pw.shape}")

    # Model
    model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total_params/1e6:.1f}M, Trainable: {train_params/1e6:.1f}M")

    # Dataloaders
    train_ds = MultiHeadDataset(train_data, tokenizer)
    val_ds = MultiHeadDataset(val_data, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Class weights for imbalanced heads
    fn_counts = Counter(d['labels']['fn'] for d in train_data)
    fn_total = sum(fn_counts.values())
    fn_weights = torch.tensor([fn_total / (len(fn_counts) * fn_counts.get(c, 1))
                               for c in HEAD_CLASSES['fn']], dtype=torch.float32).to(device)
    fn_weights = fn_weights.clamp(max=5.0)

    best_combo = 0
    best_fn = 0

    for epoch in range(40):
        # Train
        model.train()
        train_loss = 0
        train_n = 0

        for input_ids, labels in train_dl:
            input_ids = input_ids.to(device)
            labels = {h: v.to(device) for h, v in labels.items()}

            logits = model(input_ids)

            # Weighted loss for fn head
            loss = 0
            loss += 2.0 * F.cross_entropy(logits['fn'], labels['fn'], weight=fn_weights)
            loss += 2.0 * F.cross_entropy(logits['exec_type'], labels['exec_type'])
            loss += 1.5 * F.cross_entropy(logits['param_direction'], labels['param_direction'])
            loss += 1.0 * F.cross_entropy(logits['param_type'], labels['param_type'])
            loss += 1.5 * F.cross_entropy(logits['judge'], labels['judge'])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * input_ids.size(0)
            train_n += input_ids.size(0)

        scheduler.step()

        # Eval
        model.eval()
        head_correct = {h: 0 for h in HEAD_NAMES}
        head_total = {h: 0 for h in HEAD_NAMES}
        all_correct = 0
        all_total = 0

        with torch.no_grad():
            for input_ids, labels in val_dl:
                input_ids = input_ids.to(device)
                labels = {h: v.to(device) for h, v in labels.items()}

                logits = model(input_ids)
                B = input_ids.size(0)
                all_match = torch.ones(B, dtype=torch.bool, device=device)

                for h in HEAD_NAMES:
                    pred = logits[h].argmax(1)
                    correct = (pred == labels[h])
                    head_correct[h] += correct.sum().item()
                    head_total[h] += B
                    all_match &= correct

                all_correct += all_match.sum().item()
                all_total += B

        # Results
        fn_acc = head_correct['fn'] / head_total['fn'] * 100
        exec_acc = head_correct['exec_type'] / head_total['exec_type'] * 100
        dir_acc = head_correct['param_direction'] / head_total['param_direction'] * 100
        param_acc = head_correct['param_type'] / head_total['param_type'] * 100
        judge_acc = head_correct['judge'] / head_total['judge'] * 100
        combo_acc = all_correct / all_total * 100

        avg_loss = train_loss / train_n

        print(f"[{epoch+1:2d}] loss={avg_loss:.3f} | "
              f"fn={fn_acc:.1f} exec={exec_acc:.1f} dir={dir_acc:.1f} "
              f"param={param_acc:.1f} judge={judge_acc:.1f} | "
              f"combo={combo_acc:.1f}%")

        # Save best
        if combo_acc > best_combo or (combo_acc == best_combo and fn_acc > best_fn):
            best_combo = combo_acc
            best_fn = fn_acc
            torch.save({
                'epoch': epoch + 1,
                'state': model.state_dict(),
                'combo': combo_acc,
                'fn': fn_acc,
                'exec': exec_acc,
                'dir': dir_acc,
                'param': param_acc,
                'judge': judge_acc,
            }, 'checkpoints/cnn_multihead_best.pt')
            print(f"  ★ Best saved: combo={combo_acc:.1f}%, fn={fn_acc:.1f}%")

    print(f"\n=== Final Best: combo={best_combo:.1f}%, fn={best_fn:.1f}% ===")


if __name__ == '__main__':
    train()
