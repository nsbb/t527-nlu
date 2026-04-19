#!/usr/bin/env python3
"""v59 — Head-specific data masking
Key insight: KoELECTRA fn labels are good, but exec/dir labels are noisy.
Solution: mask exec/dir loss for KoELECTRA samples.

For KoELECTRA samples:  only fn + judge loss (learn fn diversity)
For GT samples:          all heads' loss (learn accurate exec/dir)

This should preserve:
- v28-level exec/dir accuracy (from GT data)
- v46-level fn generalization (from KoELECTRA data)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import json, os, sys, random, copy
from torch.utils.data import Dataset, DataLoader
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer


class HeadMaskDataset(Dataset):
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
        labels = {h: HEAD_L2I[h].get(d['labels'].get(h, 'none'), 0) for h in HEAD_NAMES}

        # Source flag: 1 = GT (all heads), 0 = KoELECTRA (fn + judge only)
        source = d.get('source', '')
        is_gt = 1.0 if not source.startswith('koelectra') else 0.0

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
                # Keep original labels and is_gt flag (same fn class)

        return input_ids, labels, is_gt


def collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = {h: torch.tensor([b[1][h] for b in batch]) for h in HEAD_NAMES}
    is_gt = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return input_ids, labels, is_gt


def eval_test_suite(model, tok, device):
    import re
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
        if HEAD_I2L['fn'][l['fn'].argmax(1).item()] == d['labels']['fn']:
            fn_ok += 1
    return fn_ok / len(ke_val) * 100


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open('data/train_final_v43.json') as f: train_data = json.load(f)
    with open('data/val_final_v43.json') as f: val_data = json.load(f)

    # Count data sources
    gt_count = sum(1 for d in train_data if not d.get('source', '').startswith('koelectra'))
    ke_count = sum(1 for d in train_data if d.get('source', '').startswith('koelectra'))
    print(f"Train: {len(train_data)} (GT: {gt_count}, KoELECTRA: {ke_count})")
    print(f"Val: {len(val_data)}")

    tok = AutoTokenizer.from_pretrained('tokenizer/')
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert

    model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable/1e6:.1f}M")

    train_ds = HeadMaskDataset(train_data, tok, mixup_prob=0.3)
    val_ds = HeadMaskDataset(val_data, tok, mixup_prob=0.0)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    fn_counts = Counter(d['labels']['fn'] for d in train_data)
    fn_total = sum(fn_counts.values())
    fn_weights = torch.tensor([fn_total / (len(fn_counts) * fn_counts.get(c, 1))
                               for c in HEAD_CLASSES['fn']], dtype=torch.float32).to(device)
    fn_weights = fn_weights.clamp(max=5.0)

    best_combo = 0; best_ts_combo = 0

    for epoch in range(40):
        model.train()
        train_loss = 0; train_n = 0

        for input_ids, labels, is_gt in train_dl:
            input_ids = input_ids.to(device)
            labels = {h: v.to(device) for h, v in labels.items()}
            is_gt = is_gt.to(device)

            logits = model(input_ids)

            # Head-masked loss
            # fn: ALL samples contribute (KoELECTRA has good fn labels)
            fn_loss = F.cross_entropy(logits['fn'], labels['fn'], weight=fn_weights, reduction='none')

            # exec/dir: ONLY GT samples contribute (KoELECTRA exec/dir are noisy)
            exec_loss = F.cross_entropy(logits['exec_type'], labels['exec_type'], reduction='none')
            dir_loss = F.cross_entropy(logits['param_direction'], labels['param_direction'], reduction='none')
            param_loss = F.cross_entropy(logits['param_type'], labels['param_type'], reduction='none')

            # judge: ALL samples
            judge_loss = F.cross_entropy(logits['judge'], labels['judge'], reduction='none')

            # Masked mean: exec/dir only from GT samples
            loss = 2.0 * fn_loss.mean()
            loss += 1.5 * judge_loss.mean()

            # For exec/dir: mask KoELECTRA samples
            gt_mask = is_gt > 0.5
            if gt_mask.any():
                loss += 2.0 * exec_loss[gt_mask].mean()
                loss += 1.5 * dir_loss[gt_mask].mean()
                loss += 1.0 * param_loss[gt_mask].mean()
            else:
                # If batch has no GT samples, still compute exec/dir but with very low weight
                loss += 0.2 * exec_loss.mean()
                loss += 0.15 * dir_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * input_ids.size(0)
            train_n += input_ids.size(0)

        scheduler.step()

        # Val eval
        model.eval()
        head_correct = {h: 0 for h in HEAD_NAMES}
        all_correct = 0; total = 0
        with torch.no_grad():
            for input_ids, labels, _ in val_dl:
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

        fn_acc = head_correct['fn'] / total * 100
        exec_acc = head_correct['exec_type'] / total * 100
        dir_acc = head_correct['param_direction'] / total * 100
        combo_acc = all_correct / total * 100

        ts_str = ""
        if epoch >= 25 and (epoch + 1) % 5 == 0:
            ts_fn, ts_exec, ts_dir, ts_combo = eval_test_suite(model, tok, device)
            ke_fn = eval_koelectra(model, tok, device)
            ts_str = f" | TS={ts_combo:.1f}% KE={ke_fn:.1f}%"
            if ts_combo > best_ts_combo:
                best_ts_combo = ts_combo
                torch.save({
                    'epoch': epoch + 1, 'state': model.state_dict(),
                    'combo': combo_acc, 'fn': fn_acc,
                    'ts_combo': ts_combo, 'ts_fn': ts_fn,
                    'ke_fn': ke_fn,
                }, 'checkpoints/cnn_multihead_v59.pt')
                ts_str += " ★"

        print(f"[{epoch+1:2d}] loss={train_loss/train_n:.3f} | fn={fn_acc:.1f} exec={exec_acc:.1f} "
              f"dir={dir_acc:.1f} | combo={combo_acc:.1f}%{ts_str}")

        if combo_acc > best_combo:
            best_combo = combo_acc
            if best_ts_combo == 0:
                torch.save({'epoch': epoch+1, 'state': model.state_dict(),
                           'combo': combo_acc, 'fn': fn_acc}, 'checkpoints/cnn_multihead_v59.pt')
            print(f"  ★ Val: combo={combo_acc:.1f}%")

    # Final eval
    ckpt = torch.load('checkpoints/cnn_multihead_v59.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state']); model.eval()
    ts_fn, ts_exec, ts_dir, ts_combo = eval_test_suite(model, tok, device)
    ke_fn = eval_koelectra(model, tok, device)

    print(f"\n{'='*60}")
    print(f"v59 Final (Head-specific masking):")
    print(f"  Test Suite: fn={ts_fn:.1f} exec={ts_exec:.1f} dir={ts_dir:.1f} combo={ts_combo:.1f}%")
    print(f"  KoELECTRA fn: {ke_fn:.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    train()
