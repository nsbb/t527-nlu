#!/usr/bin/env python3
"""v71 — 간접/유사 표현 데이터 통합 재학습

변경점:
  1. train_final_v71.json: 29,015개
     - v70 기반 (28,763) + 어두침침 52개 fix (dir=set→on)
     - llm_paraphrases.jsonl 1,846개 추가 (param_direction 추론)
     - indirect_expressions.csv 52개 추가 (flat→multi-head 변환)
  2. v46 recipe 동일: CNN 4L + mixup, 30 epochs, batch=64, lr=1e-3
  3. 평가: TS combo + KE fn (balanced geometric mean)
"""
import os, sys, json, random, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer


class MixupDataset(Dataset):
    def __init__(self, data, tok, max_len=32, mixup_prob=0.3):
        self.data = data; self.tok = tok
        self.max_len = max_len; self.mixup_prob = mixup_prob
        self.fn_groups = {}
        for i, d in enumerate(data):
            self.fn_groups.setdefault(d['labels']['fn'], []).append(i)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        tk = self.tok(d['utterance'], padding='max_length', truncation=True,
                      max_length=self.max_len, return_tensors='pt')
        input_ids = tk['input_ids'].squeeze(0)
        labels = {h: HEAD_L2I[h].get(d['labels'].get(h, 'none'), 0) for h in HEAD_NAMES}
        if random.random() < self.mixup_prob:
            fn = d['labels']['fn']; cands = self.fn_groups.get(fn, [])
            if len(cands) > 1:
                j = random.choice(cands)
                while j == idx and len(cands) > 1: j = random.choice(cands)
                tk2 = self.tok(self.data[j]['utterance'], padding='max_length',
                               truncation=True, max_length=self.max_len, return_tensors='pt')
                input_ids = tk2['input_ids'].squeeze(0)
        return input_ids, labels


def collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = {h: torch.tensor([b[1][h] for b in batch]) for h in HEAD_NAMES}
    return input_ids, labels


def eval_ts(model, tok, device):
    import re
    suite = json.load(open('data/test_suite.json'))
    fn_ok = exec_ok = dir_ok = all_ok = 0
    for t in suite:
        text = re.sub(r'\s+', ' ', ''.join(c if c.isprintable() or c == ' ' else ' ' for c in t['utterance'])).strip()
        tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt').to(device)
        with torch.no_grad(): l = model(tk['input_ids'])
        p = {h: HEAD_I2L[h][l[h].argmax(1).item()] for h in HEAD_NAMES}
        if p['param_direction'] in ('open', 'close', 'stop'): p['param_type'] = 'none'
        f = p['fn'] == t['fn']; e = p['exec_type'] == t['exec']; d = p['param_direction'] == t['dir']
        if f: fn_ok += 1
        if e: exec_ok += 1
        if d: dir_ok += 1
        if f and e and d: all_ok += 1
    n = len(suite)
    return fn_ok/n*100, exec_ok/n*100, dir_ok/n*100, all_ok/n*100


def eval_ke(model, tok, device):
    ke = json.load(open('data/koelectra_converted_val.json'))
    ok = 0
    for d in ke:
        tk = tok(d['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='pt').to(device)
        with torch.no_grad(): l = model(tk['input_ids'])
        if HEAD_I2L['fn'][l['fn'].argmax(1).item()] == d['labels']['fn']:
            ok += 1
    return ok / len(ke) * 100


def main():
    random.seed(42); torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    train = json.load(open('data/train_final_v71.json'))
    val   = json.load(open('data/val_final_v33.json'))
    print(f"Train: {len(train)}, Val: {len(val)}")

    src_dist = Counter(x.get('source', '?') for x in train)
    print("소스 분포 (주요):")
    for k, v in src_dist.most_common(5):
        print(f"  {k}: {v}")

    tok = AutoTokenizer.from_pretrained('tokenizer/')
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert

    model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable\n")

    train_ds = MixupDataset(train, tok, mixup_prob=0.3)
    val_ds   = MixupDataset(val,   tok, mixup_prob=0.0)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    fn_counts = Counter(d['labels']['fn'] for d in train)
    fn_total  = sum(fn_counts.values())
    fn_weights = torch.tensor(
        [fn_total / (len(fn_counts) * fn_counts.get(c, 1)) for c in HEAD_CLASSES['fn']],
        dtype=torch.float32).to(device)
    fn_weights = fn_weights.clamp(max=5.0)

    best_balanced = 0; best_ts = 0; best_ke = 0

    print("=== v71 Training (30 epochs) ===")
    for epoch in range(30):
        t0 = time.time()
        model.train()
        t_loss = 0; t_n = 0
        for input_ids, lbl in train_dl:
            input_ids = input_ids.to(device)
            lbl = {h: v.to(device) for h, v in lbl.items()}
            logits = model(input_ids)
            loss  = 2.0 * F.cross_entropy(logits['fn'],             lbl['fn'],             weight=fn_weights)
            loss += 2.0 * F.cross_entropy(logits['exec_type'],      lbl['exec_type'])
            loss += 1.5 * F.cross_entropy(logits['param_direction'], lbl['param_direction'])
            loss += 1.0 * F.cross_entropy(logits['param_type'],     lbl['param_type'])
            loss += 1.5 * F.cross_entropy(logits['judge'],          lbl['judge'])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * input_ids.size(0); t_n += input_ids.size(0)
        scheduler.step()

        model.eval()
        val_fn = val_combo = val_n = 0
        with torch.no_grad():
            for input_ids, lbl in val_dl:
                input_ids = input_ids.to(device)
                lbl = {h: v.to(device) for h, v in lbl.items()}
                logits = model(input_ids)
                B = input_ids.size(0)
                all_match = torch.ones(B, dtype=torch.bool, device=device)
                for h in HEAD_NAMES:
                    pred = logits[h].argmax(1)
                    correct = (pred == lbl[h])
                    all_match &= correct
                    if h == 'fn': val_fn += correct.sum().item()
                val_combo += all_match.sum().item(); val_n += B

        elapsed = time.time() - t0

        if (epoch+1) % 5 == 0 or epoch == 29:
            ts_fn, ts_e, ts_d, ts_c = eval_ts(model, tok, device)
            ke_fn = eval_ke(model, tok, device)
            balanced = (ts_c * ke_fn) ** 0.5
            print(f"[{epoch+1:2d}] {elapsed:4.1f}s loss={t_loss/t_n:.3f} | "
                  f"val_fn={val_fn/val_n*100:.1f} val_combo={val_combo/val_n*100:.1f} | "
                  f"TS={ts_c:.2f} KE={ke_fn:.2f} bal={balanced:.2f}")
            if balanced > best_balanced:
                best_balanced = balanced; best_ts = ts_c; best_ke = ke_fn
                torch.save({
                    'epoch': epoch+1, 'state': model.state_dict(),
                    'combo': ts_c, 'fn': ts_fn, 'exec': ts_e, 'dir': ts_d,
                    'ke_fn': ke_fn, 'balanced': balanced,
                    'desc': 'v71 — llm_para + indirect_expr + 어두침침 fix',
                }, 'checkpoints/cnn_multihead_v71.pt')
                print(f"   ★ saved (balanced={balanced:.2f})")
        else:
            print(f"[{epoch+1:2d}] {elapsed:4.1f}s loss={t_loss/t_n:.3f} | "
                  f"val_fn={val_fn/val_n*100:.1f} val_combo={val_combo/val_n*100:.1f}")

    print(f"\n=== v71 최종 ===")
    print(f"  Best: TS combo {best_ts:.2f}% | KE fn {best_ke:.2f}% | balanced {best_balanced:.2f}")
    print(f"  Baseline v70: 이전 체크포인트 참고")


if __name__ == '__main__':
    main()
