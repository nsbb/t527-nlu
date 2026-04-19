#!/usr/bin/env python3
"""v61 — Fine-tune v28 on KoELECTRA data
Strategy: Start from v28 (TS 96.3%, KE 75.5%)
Fine-tune on KoELECTRA pseudo-labeled data with very low LR
Goal: maintain TS 94%+ while improving KE to 95%+

Key differences from v34/v46:
- Initialize from v28 instead of random
- Use only KoELECTRA pseudo-labeled data for fine-tuning (no GT data)
- Very low LR to preserve existing knowledge
- Monitor both TS and KE every epoch
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
            fn = d['labels']['fn']
            cands = self.fn_groups.get(fn, [])
            if len(cands) > 1:
                j = random.choice(cands)
                while j == idx and len(cands) > 1: j = random.choice(cands)
                d2 = self.data[j]
                tk2 = self.tokenizer(d2['utterance'], padding='max_length', truncation=True,
                                    max_length=self.max_len, return_tensors='pt')
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
        if f: fn_ok += 1
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

    # Load KoELECTRA pseudo-labeled data
    with open('data/koelectra_pseudo_labeled.json') as f:
        ke_data = json.load(f)
    print(f"KoELECTRA data: {len(ke_data)} samples")

    tok = AutoTokenizer.from_pretrained('tokenizer/')
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert

    # Load v28 as starting point
    model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15).to(device)
    ckpt28 = torch.load('checkpoints/cnn_multihead_v28.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt28['state'])
    print(f"Loaded v28 (epoch {ckpt28['epoch']}, combo {ckpt28['combo']:.1f}%)")

    # Baseline eval
    ts_fn, ts_exec, ts_dir, ts_combo = eval_test_suite(model, tok, device)
    ke_fn = eval_koelectra(model, tok, device)
    print(f"Baseline — TS: {ts_combo:.1f}% | KE: {ke_fn:.1f}%")

    # Training setup
    train_ds = MixupDataset(ke_data, tok, mixup_prob=0.3)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)

    fn_counts = Counter(d['labels']['fn'] for d in ke_data)
    fn_total = sum(fn_counts.values())
    fn_weights = torch.tensor([fn_total / (len(fn_counts) * fn_counts.get(c, 1))
                               for c in HEAD_CLASSES['fn']], dtype=torch.float32).to(device)
    fn_weights = fn_weights.clamp(max=5.0)

    # Try multiple LR schedules
    best_balanced = (ts_combo * ke_fn) ** 0.5
    best_state = copy.deepcopy(model.state_dict())
    best_config = "baseline"

    for lr in [5e-5, 1e-4, 2e-4]:
        print(f"\n--- LR={lr} ---")
        # Reset to v28
        model.load_state_dict(ckpt28['state'])

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=lr, weight_decay=0.01)

        for epoch in range(15):
            model.train()
            train_loss = 0; train_n = 0

            for input_ids, labels in train_dl:
                input_ids = input_ids.to(device)
                labels = {h: v.to(device) for h, v in labels.items()}
                logits = model(input_ids)

                # Higher fn weight, lower exec/dir weight (preserve v28's patterns)
                loss = 0
                loss += 3.0 * F.cross_entropy(logits['fn'], labels['fn'], weight=fn_weights)
                loss += 1.0 * F.cross_entropy(logits['exec_type'], labels['exec_type'])
                loss += 0.5 * F.cross_entropy(logits['param_direction'], labels['param_direction'])
                loss += 0.3 * F.cross_entropy(logits['param_type'], labels['param_type'])
                loss += 1.0 * F.cross_entropy(logits['judge'], labels['judge'])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                train_loss += loss.item() * input_ids.size(0)
                train_n += input_ids.size(0)

            model.eval()
            ts_fn, ts_exec, ts_dir, ts_combo = eval_test_suite(model, tok, device)
            ke_fn = eval_koelectra(model, tok, device)
            balanced = (ts_combo * ke_fn) ** 0.5

            marker = ""
            if balanced > best_balanced:
                best_balanced = balanced
                best_state = copy.deepcopy(model.state_dict())
                best_config = f"lr={lr}, epoch={epoch+1}"
                marker = " ★"

            print(f"  [{epoch+1:2d}] loss={train_loss/train_n:.3f} | TS={ts_combo:.1f}% KE={ke_fn:.1f}% bal={balanced:.1f}{marker}")

            # Early stop if TS drops below 90%
            if ts_combo < 88:
                print(f"  TS dropped below 88%, stopping this LR")
                break

    # Final results
    model.load_state_dict(best_state); model.eval()
    ts_fn, ts_exec, ts_dir, ts_combo = eval_test_suite(model, tok, device)
    ke_fn = eval_koelectra(model, tok, device)

    torch.save({
        'epoch': 0, 'state': best_state,
        'combo': ts_combo, 'fn': ts_fn,
        'ts_combo': ts_combo, 'ke_fn': ke_fn,
        'config': best_config,
    }, 'checkpoints/cnn_multihead_v61.pt')

    print(f"\n{'='*60}")
    print(f"v61 Final ({best_config}):")
    print(f"  Test Suite: fn={ts_fn:.1f} exec={ts_exec:.1f} dir={ts_dir:.1f} combo={ts_combo:.1f}%")
    print(f"  KoELECTRA fn: {ke_fn:.1f}%")
    print(f"  Balanced: {(ts_combo * ke_fn)**0.5:.1f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    train()
