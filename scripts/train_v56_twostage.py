#!/usr/bin/env python3
"""v56 — Two-stage fine-tuning
Stage 1: Train from scratch like v46 (mixup, 30 epochs)
Stage 2: Freeze fn head, fine-tune exec/dir with low LR (10 epochs)

Goal: preserve v46's fn generalization (KE 97.8%) while improving exec/dir (TS 93→95%+)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import json, os, sys, time, random, copy
from torch.utils.data import Dataset, DataLoader
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer

class MixupDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=32, mixup_prob=0.3):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mixup_prob = mixup_prob
        self.fn_groups = {}
        for i, d in enumerate(data):
            fn = d['labels']['fn']
            self.fn_groups.setdefault(fn, []).append(i)

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

        if random.random() < self.mixup_prob:
            fn = d['labels']['fn']
            candidates = self.fn_groups.get(fn, [])
            if len(candidates) > 1:
                j = random.choice(candidates)
                while j == idx and len(candidates) > 1:
                    j = random.choice(candidates)
                d2 = self.data[j]
                tk2 = self.tokenizer(d2['utterance'], padding='max_length', truncation=True,
                                    max_length=self.max_len, return_tensors='pt')
                input_ids = tk2['input_ids'].squeeze(0)
        return input_ids, labels

def collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = {h: torch.tensor([b[1][h] for b in batch]) for h in HEAD_NAMES}
    return input_ids, labels


def eval_model(model, val_dl, device):
    model.eval()
    head_correct = {h: 0 for h in HEAD_NAMES}
    head_total = {h: 0 for h in HEAD_NAMES}
    all_correct = 0; all_total = 0
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
    return {h: head_correct[h] / head_total[h] * 100 for h in HEAD_NAMES}, all_correct / all_total * 100


def eval_koelectra(model, tok, device):
    """Evaluate on KoELECTRA val set"""
    ke_val = json.load(open('data/koelectra_converted_val.json'))
    fn_ok = 0
    for d in ke_val:
        t = tok(d['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        with torch.no_grad():
            l = model(t['input_ids'].to(device))
        pred_fn = HEAD_I2L['fn'][l['fn'].argmax(1).item()]
        if pred_fn == d['labels']['fn']:
            fn_ok += 1
    return fn_ok / len(ke_val) * 100


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Use same data as v46
    train_file = 'data/train_final_v43.json'
    val_file = 'data/val_final_v43.json'
    with open(train_file) as f: train_data = json.load(f)
    with open(val_file) as f: val_data = json.load(f)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    tok = AutoTokenizer.from_pretrained('tokenizer/')
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert

    model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15).to(device)

    fn_counts = Counter(d['labels']['fn'] for d in train_data)
    fn_total = sum(fn_counts.values())
    fn_weights = torch.tensor([fn_total / (len(fn_counts) * fn_counts.get(c, 1))
                               for c in HEAD_CLASSES['fn']], dtype=torch.float32).to(device)
    fn_weights = fn_weights.clamp(max=5.0)

    train_ds = MixupDataset(train_data, tok, mixup_prob=0.3)
    val_ds = MixupDataset(val_data, tok, mixup_prob=0.0)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # ============================================================
    # Stage 1: Full training with mixup (like v46), 30 epochs
    # ============================================================
    print("\n" + "="*60)
    print("STAGE 1: Full training with mixup (30 epochs)")
    print("="*60)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_combo = 0; best_fn = 0; best_state = None

    for epoch in range(30):
        model.train()
        train_loss = 0; train_n = 0

        for input_ids, labels in train_dl:
            input_ids = input_ids.to(device)
            labels = {h: v.to(device) for h, v in labels.items()}
            logits = model(input_ids)

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

        head_accs, combo_acc = eval_model(model, val_dl, device)
        fn_acc = head_accs['fn']

        print(f"S1[{epoch+1:2d}] loss={train_loss/train_n:.3f} | fn={fn_acc:.1f} "
              f"exec={head_accs['exec_type']:.1f} dir={head_accs['param_direction']:.1f} | combo={combo_acc:.1f}%")

        if combo_acc > best_combo or (combo_acc == best_combo and fn_acc > best_fn):
            best_combo = combo_acc; best_fn = fn_acc
            best_state = copy.deepcopy(model.state_dict())
            print(f"  ★ Best: combo={combo_acc:.1f}%, fn={fn_acc:.1f}%")

    # Load best stage 1 model
    model.load_state_dict(best_state)
    print(f"\nStage 1 best: combo={best_combo:.1f}%, fn={best_fn:.1f}%")

    # Check KoELECTRA at stage 1 end
    ke_fn = eval_koelectra(model, tok, device)
    print(f"Stage 1 KoELECTRA fn: {ke_fn:.1f}%")

    # ============================================================
    # Stage 2: Freeze fn head, fine-tune exec/dir with low LR
    # ============================================================
    print("\n" + "="*60)
    print("STAGE 2: Freeze fn head, fine-tune exec/dir (15 epochs)")
    print("="*60)

    # Freeze fn head
    for param in model.heads['fn'].parameters():
        param.requires_grad = False

    # Also freeze judge head (it's already good)
    for param in model.heads['judge'].parameters():
        param.requires_grad = False

    trainable_s2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 2 trainable: {trainable_s2/1e6:.1f}M (fn+judge frozen)")

    # Lower LR, higher exec/dir weights
    optimizer2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=3e-4, weight_decay=0.01)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=15)

    # Use less mixup in stage 2 (focus on exact patterns)
    train_ds2 = MixupDataset(train_data, tok, mixup_prob=0.15)
    train_dl2 = DataLoader(train_ds2, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)

    best_combo_s2 = best_combo
    best_state_s2 = copy.deepcopy(model.state_dict())

    for epoch in range(15):
        model.train()
        train_loss = 0; train_n = 0

        for input_ids, labels in train_dl2:
            input_ids = input_ids.to(device)
            labels = {h: v.to(device) for h, v in labels.items()}
            logits = model(input_ids)

            # Higher exec/dir weights, no fn loss (frozen)
            loss = 0
            loss += 3.0 * F.cross_entropy(logits['exec_type'], labels['exec_type'])
            loss += 2.5 * F.cross_entropy(logits['param_direction'], labels['param_direction'])
            loss += 1.0 * F.cross_entropy(logits['param_type'], labels['param_type'])

            optimizer2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer2.step()
            train_loss += loss.item() * input_ids.size(0)
            train_n += input_ids.size(0)

        scheduler2.step()

        head_accs, combo_acc = eval_model(model, val_dl, device)
        fn_acc = head_accs['fn']

        print(f"S2[{epoch+1:2d}] loss={train_loss/train_n:.3f} | fn={fn_acc:.1f} "
              f"exec={head_accs['exec_type']:.1f} dir={head_accs['param_direction']:.1f} | combo={combo_acc:.1f}%")

        if combo_acc > best_combo_s2:
            best_combo_s2 = combo_acc
            best_state_s2 = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': 30 + epoch + 1, 'state': model.state_dict(),
                'combo': combo_acc, 'fn': fn_acc,
                'exec': head_accs['exec_type'], 'dir': head_accs['param_direction'],
                'stage': 2,
            }, 'checkpoints/cnn_multihead_v56.pt')
            print(f"  ★ Best S2: combo={combo_acc:.1f}%, fn={fn_acc:.1f}%")

    # Load best stage 2
    model.load_state_dict(best_state_s2)
    ke_fn_s2 = eval_koelectra(model, tok, device)
    print(f"\nStage 2 best: combo={best_combo_s2:.1f}%")
    print(f"Stage 2 KoELECTRA fn: {ke_fn_s2:.1f}%")

    # Save final
    torch.save({
        'epoch': 45, 'state': best_state_s2,
        'combo': best_combo_s2,
        'ke_fn': ke_fn_s2,
        'stage': 'final',
    }, 'checkpoints/cnn_multihead_v56.pt')

    print(f"\n=== v56 Final: combo={best_combo_s2:.1f}%, KE fn={ke_fn_s2:.1f}% ===")


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    train()
