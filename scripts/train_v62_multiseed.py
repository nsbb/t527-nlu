#!/usr/bin/env python3
"""v62 — Multi-seed v46 ensemble
Train 2 more v46 variants (seeds 123, 999) and evaluate:
1. Individual models
2. 3-model average predictions (seed 42 + 123 + 999)

If averaging 3 identical-recipe models beats single v46,
it means variance reduction > architecture changes.
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


def train_one(seed, train_data, val_data, pw, tok, device, save_path):
    """Train one v46-recipe model with given seed"""
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    print(f"\n{'='*40} SEED {seed} {'='*40}")

    model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15).to(device)

    train_ds = MixupDataset(train_data, tok, mixup_prob=0.3)
    val_ds = MixupDataset(val_data, tok, mixup_prob=0.0)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)

    fn_counts = Counter(d['labels']['fn'] for d in train_data)
    fn_total = sum(fn_counts.values())
    fn_weights = torch.tensor([fn_total / (len(fn_counts) * fn_counts.get(c, 1))
                               for c in HEAD_CLASSES['fn']], dtype=torch.float32).to(device)
    fn_weights = fn_weights.clamp(max=5.0)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)

    best_combo = 0

    for epoch in range(35):
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

        model.eval()
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

        if (epoch + 1) % 5 == 0:
            print(f"  [{epoch+1:2d}] loss={train_loss/train_n:.3f} fn={fn_acc:.1f} combo={combo:.1f}%"
                  + (" ★" if combo > best_combo else ""))

        if combo > best_combo:
            best_combo = combo
            torch.save({'epoch': epoch+1, 'state': model.state_dict(),
                       'combo': combo, 'fn': fn_acc, 'seed': seed}, save_path)

    print(f"  Best combo: {best_combo:.1f}%")
    return save_path


def eval_ensemble(models, tok, device):
    """Evaluate ensemble by averaging logits"""
    # Test Suite
    suite = json.load(open('data/test_suite.json'))
    fn_ok = exec_ok = dir_ok = all_ok = 0
    for t in suite:
        text = re.sub(r'\s+', ' ', ''.join(c if c.isprintable() or c == ' ' else ' ' for c in t['utterance'])).strip()
        tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        input_ids = tk['input_ids'].to(device)

        # Average logits from all models
        avg_logits = {h: None for h in HEAD_NAMES}
        with torch.no_grad():
            for m in models:
                logits = m(input_ids)
                for h in HEAD_NAMES:
                    if avg_logits[h] is None:
                        avg_logits[h] = logits[h].clone()
                    else:
                        avg_logits[h] += logits[h]
        for h in HEAD_NAMES:
            avg_logits[h] /= len(models)

        p = {h: HEAD_I2L[h][avg_logits[h].argmax(1).item()] for h in HEAD_NAMES}
        if p['param_direction'] in ('open', 'close', 'stop'): p['param_type'] = 'none'
        if p['judge'] != 'none': p['param_type'] = 'none'
        if p['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'): p['param_type'] = 'none'

        f = p['fn'] == t['fn']; e = p['exec_type'] == t['exec']; d = p['param_direction'] == t['dir']
        if f: fn_ok += 1
        if e: exec_ok += 1
        if d: dir_ok += 1
        if f and e and d: all_ok += 1

    n = len(suite)
    ts_fn, ts_exec, ts_dir, ts_combo = fn_ok/n*100, exec_ok/n*100, dir_ok/n*100, all_ok/n*100

    # KoELECTRA
    ke_val = json.load(open('data/koelectra_converted_val.json'))
    ke_fn_ok = 0
    for d in ke_val:
        tk = tok(d['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        input_ids = tk['input_ids'].to(device)
        avg_fn = None
        with torch.no_grad():
            for m in models:
                logits = m(input_ids)
                if avg_fn is None:
                    avg_fn = logits['fn'].clone()
                else:
                    avg_fn += logits['fn']
        avg_fn /= len(models)
        if HEAD_I2L['fn'][avg_fn.argmax(1).item()] == d['labels']['fn']:
            ke_fn_ok += 1

    ke_fn = ke_fn_ok / len(ke_val) * 100
    return ts_fn, ts_exec, ts_dir, ts_combo, ke_fn


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open('data/train_final_v43.json') as f: train_data = json.load(f)
    with open('data/val_final_v43.json') as f: val_data = json.load(f)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    tok = AutoTokenizer.from_pretrained('tokenizer/')
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert

    # Train seed 123 and 999
    seeds = [123, 999]
    paths = {}

    for seed in seeds:
        path = f'checkpoints/cnn_multihead_v62_s{seed}.pt'
        train_one(seed, train_data, val_data, pw, tok, device, path)
        paths[seed] = path

    # Load all 3 models (v46=seed42, + 2 new seeds)
    print(f"\n{'='*60}")
    print(f"Multi-seed Ensemble Evaluation")
    print(f"{'='*60}")

    all_models = []
    model_names = []

    # v46 (seed 42)
    m46 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15).to(device)
    ckpt46 = torch.load('checkpoints/cnn_multihead_v46.pt', map_location='cpu', weights_only=False)
    m46.load_state_dict(ckpt46['state']); m46.eval()
    all_models.append(m46); model_names.append("v46(s42)")

    for seed in seeds:
        m = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15).to(device)
        ckpt = torch.load(paths[seed], map_location='cpu', weights_only=False)
        m.load_state_dict(ckpt['state']); m.eval()
        all_models.append(m); model_names.append(f"v62(s{seed})")

    # Evaluate individual models
    for name, model in zip(model_names, all_models):
        ts_fn, ts_exec, ts_dir, ts_combo, ke_fn = eval_ensemble([model], tok, device)
        print(f"  {name}: TS={ts_combo:.1f}% KE={ke_fn:.1f}%")

    # Evaluate 2-model and 3-model ensembles
    print()
    for i in range(len(all_models)):
        for j in range(i+1, len(all_models)):
            name = f"{model_names[i]}+{model_names[j]}"
            ts_fn, ts_exec, ts_dir, ts_combo, ke_fn = eval_ensemble(
                [all_models[i], all_models[j]], tok, device)
            print(f"  {name}: TS={ts_combo:.1f}% KE={ke_fn:.1f}% bal={(ts_combo*ke_fn)**0.5:.1f}")

    print()
    ts_fn, ts_exec, ts_dir, ts_combo, ke_fn = eval_ensemble(all_models, tok, device)
    print(f"  ALL 3 models: TS fn={ts_fn:.1f} exec={ts_exec:.1f} dir={ts_dir:.1f} combo={ts_combo:.1f}% | KE={ke_fn:.1f}%")
    print(f"  Balanced: {(ts_combo * ke_fn)**0.5:.1f}")

    # Compare with v28+v46 ensemble
    m28 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15).to(device)
    ckpt28 = torch.load('checkpoints/cnn_multihead_v28.pt', map_location='cpu', weights_only=False)
    m28.load_state_dict(ckpt28['state']); m28.eval()

    # v28+v46 strategy B (fn=v46, exec/dir=v28)
    suite = json.load(open('data/test_suite.json'))
    fn_ok = all_ok = 0
    for t in suite:
        text = re.sub(r'\s+', ' ', ''.join(c if c.isprintable() or c == ' ' else ' ' for c in t['utterance'])).strip()
        tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        input_ids = tk['input_ids'].to(device)
        with torch.no_grad():
            l28 = m28(input_ids); l46 = m46(input_ids)
        p = {
            'fn': HEAD_I2L['fn'][l46['fn'].argmax(1).item()],
            'exec_type': HEAD_I2L['exec_type'][l28['exec_type'].argmax(1).item()],
            'param_direction': HEAD_I2L['param_direction'][l28['param_direction'].argmax(1).item()],
        }
        f = p['fn'] == t['fn']; e = p['exec_type'] == t['exec']; d = p['param_direction'] == t['dir']
        if f: fn_ok += 1
        if f and e and d: all_ok += 1
    n = len(suite)
    print(f"\n  Ref: v28+v46 Strategy B: TS={all_ok/n*100:.1f}%")


if __name__ == '__main__':
    main()
