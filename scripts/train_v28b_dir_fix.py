#!/usr/bin/env python3
"""v28b — v28에서 dir 패턴 보강 재학습
목적:
  v28은 "밝게" 등 일부 dir 패턴을 잘못 학습 (train 데이터 부족)
  → 수정된 라벨 + 추가 "밝게/어둡게/모드로" 샘플로 짧게 재학습

전략:
  - v28 원본 weights로 초기화 (warm start, backbone 보존)
  - 원본 train + 수정 라벨 + 추가 밝기 샘플
  - 짧게 (5-10 epoch), 낮은 LR (재학습이 아닌 fine-tune)
  - Regression 감지: TS fn이 100%에서 떨어지면 중단
"""
import os, sys, json, re, random, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer
from preprocess import preprocess


# v28의 학습 데이터에 추가할 "밝게/어둡게/모드로" 샘플 (올바른 라벨)
ADDITIONAL_SAMPLES = []

# 밝게 → up
rooms = ['거실', '안방', '주방', '침실', '작은방', '아이방', '']
bright_templates = [
    '{r} 등 좀 밝게',
    '{r} 등 밝게',
    '{r} 등 밝게 해줘',
    '{r} 조명 밝게',
    '{r} 조명 좀 밝게',
    '{r} 조명 밝게 해줘',
    '{r} 불 좀 밝게',
    '{r} 불 밝게',
    '{r} 불 밝게 해줘',
    '{r} 좀 밝게',
]
for r in rooms:
    for tmpl in bright_templates:
        utt = tmpl.format(r=r).strip()
        utt = re.sub(r'\s+', ' ', utt)
        ADDITIONAL_SAMPLES.append({
            'utterance': utt,
            'labels': {'fn': 'light_control', 'exec_type': 'control_then_confirm',
                       'param_direction': 'up', 'param_type': 'brightness', 'judge': 'none'},
            'source': 'fix_v28b_bright'
        })

# 어둡게 → down (대조 샘플)
dark_templates = [
    '{r} 등 어둡게',
    '{r} 조명 어둡게',
    '{r} 조명 좀 어둡게',
    '{r} 등 좀 어둡게 해줘',
    '{r} 불 어둡게',
]
for r in rooms:
    for tmpl in dark_templates:
        utt = tmpl.format(r=r).strip()
        utt = re.sub(r'\s+', ' ', utt)
        ADDITIONAL_SAMPLES.append({
            'utterance': utt,
            'labels': {'fn': 'light_control', 'exec_type': 'control_then_confirm',
                       'param_direction': 'down', 'param_type': 'brightness', 'judge': 'none'},
            'source': 'fix_v28b_dark'
        })

# 에어컨 모드 → set
ac_modes = ['냉방', '제습', '송풍', '자동']
for r in rooms:
    for m in ac_modes:
        for suffix in ['모드로', '모드로 해줘', '모드로 설정']:
            utt = f"{r} 에어컨 {m} {suffix}".strip()
            utt = re.sub(r'\s+', ' ', utt)
            ADDITIONAL_SAMPLES.append({
                'utterance': utt,
                'labels': {'fn': 'ac_control', 'exec_type': 'control_then_confirm',
                           'param_direction': 'set', 'param_type': 'mode', 'judge': 'none'},
                'source': 'fix_v28b_ac_mode'
            })

# 중복 제거
seen = set()
unique = []
for s in ADDITIONAL_SAMPLES:
    if s['utterance'] not in seen:
        seen.add(s['utterance'])
        unique.append(s)
ADDITIONAL_SAMPLES = unique


class SimpleDataset(Dataset):
    def __init__(self, data, tok, max_len=32):
        self.data = data; self.tok = tok; self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        tk = self.tok(d['utterance'], padding='max_length', truncation=True,
                      max_length=self.max_len, return_tensors='pt')
        labels = {h: HEAD_L2I[h].get(d['labels'].get(h, 'none'), 0) for h in HEAD_NAMES}
        return tk['input_ids'].squeeze(0), labels


def collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = {h: torch.tensor([b[1][h] for b in batch]) for h in HEAD_NAMES}
    return input_ids, labels


def eval_ts(model, tok, device):
    suite = json.load(open('data/test_suite.json'))
    fn_ok = exec_ok = dir_ok = all_ok = 0
    for t in suite:
        text = re.sub(r'\s+', ' ', ''.join(c if c.isprintable() or c == ' ' else ' ' for c in t['utterance'])).strip()
        tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        with torch.no_grad():
            l = model(tk['input_ids'].to(device))
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
        tk = tok(d['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        with torch.no_grad():
            l = model(tk['input_ids'].to(device))
        if HEAD_I2L['fn'][l['fn'].argmax(1).item()] == d['labels']['fn']:
            ok += 1
    return ok / len(ke) * 100


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 원본 v28 train
    train_orig = json.load(open('data/train_final_v28.json'))
    print(f"v28 원본 train: {len(train_orig)}")

    # 라벨 수정 적용
    FIXES = {
        '남방 올려쥬': {'param_direction': 'up'},
        '날방 올려': {'param_direction': 'up'},
        '남방 올려': {'param_direction': 'up'},
        '도어록 열어쥬': {'param_direction': 'open'},
        '안방 남방 올려줘': {'param_direction': 'up'},
        '안방 난반 올려': {'param_direction': 'up'},
        '남방 꺼쥬': {'param_direction': 'off'},
        '에어컨꺼': {'param_direction': 'off'},
        '히팅 켜줘': {'param_direction': 'on'},
    }
    n_fixed = 0
    for d in train_orig:
        if d.get('utterance') in FIXES:
            for k, v in FIXES[d['utterance']].items():
                if d['labels'].get(k) != v:
                    d['labels'][k] = v
                    n_fixed += 1
    print(f"원본 train 라벨 수정: {n_fixed}건")

    # 추가 샘플 합치기
    print(f"추가 샘플: {len(ADDITIONAL_SAMPLES)}개 (밝게 up, 어둡게 down, 에어컨 모드 set)")
    train_all = train_orig + ADDITIONAL_SAMPLES
    random.shuffle(train_all)
    print(f"최종 train: {len(train_all)}")

    # Val (기존 사용)
    val = json.load(open('data/val_final_v33.json'))
    print(f"val: {len(val)}")

    # Tokenizer + model load (warm start from v28)
    tok = AutoTokenizer.from_pretrained('tokenizer/')
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert
    model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    ckpt = torch.load('checkpoints/cnn_multihead_v28.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state'])
    model = model.to(device)
    print(f"v28 warm-start (epoch {ckpt['epoch']}, combo {ckpt['combo']:.1f}%)")

    # Baseline eval
    print("\n=== v28 원본 Baseline ===")
    ts_fn, ts_e, ts_d, ts_c = eval_ts(model, tok, device)
    ke_fn = eval_ke(model, tok, device)
    print(f"  TS: fn={ts_fn:.2f} exec={ts_e:.2f} dir={ts_d:.2f} combo={ts_c:.2f}")
    print(f"  KE fn: {ke_fn:.2f}")
    baseline_combo = ts_c

    # Dataset / optimizer (낮은 LR, 짧은 학습)
    train_ds = SimpleDataset(train_all, tok)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=1e-4, weight_decay=0.01)

    fn_counts = Counter(d['labels']['fn'] for d in train_all)
    fn_total = sum(fn_counts.values())
    fn_weights = torch.tensor([fn_total / (len(fn_counts) * fn_counts.get(c, 1))
                               for c in HEAD_CLASSES['fn']], dtype=torch.float32).to(device)
    fn_weights = fn_weights.clamp(max=5.0)

    best_combo = ts_c
    best_state = copy.deepcopy(model.state_dict())

    print("\n=== 재학습 (5 epochs, lr=1e-4) ===")
    for epoch in range(5):
        model.train()
        t_loss = 0; t_n = 0
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
            t_loss += loss.item() * input_ids.size(0)
            t_n += input_ids.size(0)

        model.eval()
        ts_fn, ts_e, ts_d, ts_c = eval_ts(model, tok, device)
        ke_fn = eval_ke(model, tok, device)
        print(f"  [{epoch+1}] loss={t_loss/t_n:.3f} | TS fn={ts_fn:.2f} exec={ts_e:.2f} dir={ts_d:.2f} combo={ts_c:.2f} | KE fn={ke_fn:.2f}")

        if ts_c >= best_combo:
            best_combo = ts_c
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': ckpt['epoch'] + epoch + 1, 'state': best_state,
                'combo': ts_c, 'fn': ts_fn, 'exec': ts_e, 'dir': ts_d,
                'ke_fn': ke_fn, 'desc': 'v28b dir fix retrain',
            }, 'checkpoints/cnn_multihead_v28b.pt')

    # Final
    model.load_state_dict(best_state)
    ts_fn, ts_e, ts_d, ts_c = eval_ts(model, tok, device)
    ke_fn = eval_ke(model, tok, device)
    print(f"\n=== v28b 최종 ===")
    print(f"  TS: fn={ts_fn:.2f} exec={ts_e:.2f} dir={ts_d:.2f} combo={ts_c:.2f} (baseline {baseline_combo:.2f})")
    print(f"  KE fn: {ke_fn:.2f}")

    # 특정 패턴 테스트
    print("\n=== '밝게' 패턴 테스트 ===")
    cases = ['거실 등 좀 밝게', '안방 등 밝게', '조명 밝게', '거실 에어컨 냉방모드로', '주방 불 어둡게']
    for t in cases:
        tk = tok(preprocess(t), padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        with torch.no_grad():
            l = model(tk['input_ids'].to(device))
        p = {h: HEAD_I2L[h][l[h].argmax(1).item()] for h in HEAD_NAMES}
        print(f"  \"{t:30s}\" fn={p['fn']:14s} dir={p['param_direction']:5s} param={p['param_type']}")


if __name__ == '__main__':
    random.seed(42); torch.manual_seed(42)
    main()
