#!/usr/bin/env python3
"""v61b — Warm-start from v28, train on full v43 data with mixup
Key: v28 initialization + v46's data/recipe + lower LR
Hypothesis: v28's weight initialization gives better exec/dir starting point
"""
import torch, torch.nn as nn, torch.nn.functional as F
import json, os, sys, random, copy, re
from torch.utils.data import Dataset, DataLoader
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer


class MixupDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=32, mixup_prob=0.3):
        self.data = data; self.tokenizer = tokenizer
        self.max_len = max_len; self.mixup_prob = mixup_prob
        self.fn_groups = {}
        for i, d in enumerate(data):
            self.fn_groups.setdefault(d['labels']['fn'], []).append(i)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        tk = self.tokenizer(d['utterance'], padding='max_length', truncation=True,
                           max_length=self.max_len, return_tensors='pt')
        input_ids = tk['input_ids'].squeeze(0)
        labels = {h: HEAD_L2I[h].get(d['labels'].get(h, 'none'), 0) for h in HEAD_NAMES}
        if random.random() < self.mixup_prob:
            fn = d['labels']['fn']; cands = self.fn_groups.get(fn, [])
            if len(cands) > 1:
                j = random.choice(cands)
                while j == idx and len(cands) > 1: j = random.choice(cands)
                tk2 = self.tokenizer(self.data[j]['utterance'], padding='max_length',
                                    truncation=True, max_length=self.max_len, return_tensors='pt')
                input_ids = tk2['input_ids'].squeeze(0)
        return input_ids, labels

def collate_fn(batch):
    return (torch.stack([b[0] for b in batch]),
            {h: torch.tensor([b[1][h] for b in batch]) for h in HEAD_NAMES})


def eval_test_suite(model, tok, device):
    suite = json.load(open('data/test_suite.json'))
    fn_ok = exec_ok = dir_ok = all_ok = 0
    for t in suite:
        text = re.sub(r'\s+', ' ', ''.join(c if c.isprintable() or c == ' ' else ' ' for c in t['utterance'])).strip()
        tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        with torch.no_grad(): l = model(tk['input_ids'].to(device))
        p = {h: HEAD_I2L[h][l[h].argmax(1).item()] for h in HEAD_NAMES}
        if p['param_direction'] in ('open', 'close', 'stop'): p['param_type'] = 'none'
        if p['judge'] != 'none': p['param_type'] = 'none'
        if p['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'): p['param_type'] = 'none'
        f = p['fn'] == t['fn']; e = p['exec_type'] == t['exec']; d = p['param_direction'] == t['dir']
        if f: fn_ok += 1;
        if e: exec_ok += 1
        if d: dir_ok += 1
        if f and e and d: all_ok += 1
    n = len(suite)
    return fn_ok/n*100, exec_ok/n*100, dir_ok/n*100, all_ok/n*100

def eval_koelectra(model, tok, device):
    ke_val = json.load(open('data/koelectra_converted_val.json'))
    fn_ok = 0
    for d in ke_val:
        t = tok(d['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        with torch.no_grad(): l = model(t['input_ids'].to(device))
        if HEAD_I2L['fn'][l['fn'].argmax(1).item()] == d['labels']['fn']: fn_ok += 1
    return fn_ok / len(ke_val) * 100


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open('data/train_final_v43.json') as f: train_data = json.load(f)
    with open('data/val_final_v43.json') as f: val_data = json.load(f)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    tok = AutoTokenizer.from_pretrained('tokenizer/')
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert

    # Initialize from v28
    model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    ckpt28 = torch.load('checkpoints/cnn_multihead_v28.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt28['state'])
    model = model.to(device)
    print(f"Initialized from v28 (epoch {ckpt28['epoch']})")

    # Verify loaded correctly
    model.eval()
    ts_fn, ts_exec, ts_dir, ts_combo = eval_test_suite(model, tok, device)
    ke_fn = eval_koelectra(model, tok, device)
    print(f"v28 baseline — TS: fn={ts_fn:.1f} exec={ts_exec:.1f} dir={ts_dir:.1f} combo={ts_combo:.1f}% | KE: {ke_fn:.1f}%")

    train_ds = MixupDataset(train_data, tok, mixup_prob=0.3)
    val_ds = MixupDataset(val_data, tok, mixup_prob=0.0)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)

    fn_counts = Counter(d['labels']['fn'] for d in train_data)
    fn_total = sum(fn_counts.values())
    fn_weights = torch.tensor([fn_total / (len(fn_counts) * fn_counts.get(c, 1))
                               for c in HEAD_CLASSES['fn']], dtype=torch.float32).to(device)
    fn_weights = fn_weights.clamp(max=5.0)

    # Lower LR since we're fine-tuning, not training from scratch
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_balanced = (ts_combo * ke_fn) ** 0.5
    best_state = copy.deepcopy(model.state_dict())

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

        # Full eval every 3 epochs, quick eval otherwise
        model.eval()
        if (epoch + 1) % 3 == 0 or epoch == 0:
            ts_fn, ts_exec, ts_dir, ts_combo = eval_test_suite(model, tok, device)
            ke_fn = eval_koelectra(model, tok, device)
            balanced = (ts_combo * ke_fn) ** 0.5

            marker = ""
            if balanced > best_balanced:
                best_balanced = balanced
                best_state = copy.deepcopy(model.state_dict())
                marker = " ★"

            print(f"[{epoch+1:2d}] loss={train_loss/train_n:.3f} | TS={ts_combo:.1f}% KE={ke_fn:.1f}% bal={balanced:.1f}{marker}")
        else:
            # Quick val eval
            head_correct = {h: 0 for h in HEAD_NAMES}
            all_correct = 0; total = 0
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
                        all_match &= correct
                    all_correct += all_match.sum().item()
                    total += B
            combo = all_correct / total * 100
            fn_acc = head_correct['fn'] / total * 100
            print(f"[{epoch+1:2d}] loss={train_loss/train_n:.3f} | val_fn={fn_acc:.1f} val_combo={combo:.1f}%")

    # Save best
    model.load_state_dict(best_state); model.eval()
    ts_fn, ts_exec, ts_dir, ts_combo = eval_test_suite(model, tok, device)
    ke_fn = eval_koelectra(model, tok, device)

    torch.save({
        'epoch': 0, 'state': best_state,
        'ts_combo': ts_combo, 'ke_fn': ke_fn,
        'balanced': (ts_combo * ke_fn) ** 0.5,
        'warmstart': 'v28',
    }, 'checkpoints/cnn_multihead_v61.pt')

    print(f"\n{'='*60}")
    print(f"v61 (warm-start v28 → v43 data + mixup):")
    print(f"  Test Suite: fn={ts_fn:.1f} exec={ts_exec:.1f} dir={ts_dir:.1f} combo={ts_combo:.1f}%")
    print(f"  KoELECTRA fn: {ke_fn:.1f}%")
    print(f"  Balanced: {(ts_combo * ke_fn)**0.5:.1f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    train()
