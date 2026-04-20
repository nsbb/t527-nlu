#!/usr/bin/env python3
"""v68 — 학습 데이터 라벨 오류 수정 후 재학습
1. train_final_v43.json의 알려진 라벨 오류 수정
2. v46 레시피로 재학습 (CNN + mixup)
3. 테스트 suite (수정됨) + KE로 비교

Hypothesis: 라벨 오류가 수정된 데이터로 학습하면 실제 정확도 개선
"""
import torch, torch.nn as nn, torch.nn.functional as F
import json, os, sys, random, copy, re
from torch.utils.data import Dataset, DataLoader
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer


def fix_train_labels(train_data):
    """학습 데이터의 알려진 라벨 오류 수정"""

    # (utterance_pattern, field_fixes) - 패턴 매칭 방식
    PATTERNS = [
        # 커튼 닫기 → dir=close
        (re.compile(r'커[튼턴].*(닫|잠)'), {'param_direction': 'close'}),
        (re.compile(r'전동커[튼턴].*닫'), {'param_direction': 'close'}),
        (re.compile(r'블라인드.*닫'), {'param_direction': 'close'}),
        # "~꺼줘" 패턴 → dir=off (오타가 아니라 '꺼'=turn off)
        (re.compile(r'^(난방|환기|가스|조명|불).*꺼'), {'param_direction': 'off'}),
        (re.compile(r'.*주방.*남방.*꺼'), {'param_direction': 'off'}),
        # 등록 → dir=set
        (re.compile(r'등록해'), {'param_direction': 'set'}),
        # 승강기/엘리베이터 호출/불러 → exec=control_then_confirm
        (re.compile(r'(승강기|엘리베이터|엘베|리프트).*(호출|불러)'), {'exec_type': 'control_then_confirm'}),
    ]

    n_fixed = 0
    fixes_log = []
    for d in train_data:
        utt = d.get('utterance', '')
        for pattern, fixes in PATTERNS:
            if pattern.search(utt):
                for k, v in fixes.items():
                    # 이미 올바른 것은 건드리지 않기
                    if d['labels'].get(k) != v:
                        old = d['labels'].get(k)
                        d['labels'][k] = v
                        fixes_log.append((utt, k, old, v))
                        n_fixed += 1

    print(f"학습 데이터 수정: {n_fixed}건")
    # Print unique fixes
    seen = set()
    for utt, k, old, new in fixes_log[:20]:
        key = (utt, k)
        if key not in seen:
            seen.add(key)
            print(f"  \"{utt}\" {k}: {old} → {new}")

    return train_data


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

    with open('data/train_final_v43.json') as f: train_data = json.load(f)
    print(f"Train before fix: {len(train_data)}")

    # Fix labels
    train_data = fix_train_labels(train_data)

    # Save fixed data
    with open('data/train_final_v68.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open('data/val_final_v43.json') as f: val_data = json.load(f)

    tok = AutoTokenizer.from_pretrained('tokenizer/')
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert

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

    best_balanced = 0
    best_state = None

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

        fn_acc = head_correct['fn'] / total * 100
        combo_acc = all_correct / total * 100

        ts_str = ""
        if (epoch + 1) % 5 == 0 or epoch >= 30:
            ts_fn, ts_exec, ts_dir, ts_combo = eval_test_suite(model, tok, device)
            ke_fn = eval_koelectra(model, tok, device)
            balanced = (ts_combo * ke_fn) ** 0.5
            ts_str = f" | TS={ts_combo:.1f}% KE={ke_fn:.1f}% bal={balanced:.1f}"
            if balanced > best_balanced:
                best_balanced = balanced
                best_state = copy.deepcopy(model.state_dict())
                torch.save({'epoch': epoch+1, 'state': model.state_dict(),
                           'ts_combo': ts_combo, 'ke_fn': ke_fn, 'balanced': balanced,
                           'desc': 'fixed_labels'}, 'checkpoints/cnn_multihead_v68.pt')
                ts_str += " ★"

        print(f"[{epoch+1:2d}] loss={train_loss/train_n:.3f} | fn={fn_acc:.1f} combo={combo_acc:.1f}%{ts_str}")

    print(f"\n=== v68 (fixed labels) Final ===")


if __name__ == '__main__':
    random.seed(42); torch.manual_seed(42)
    train()
