#!/usr/bin/env python3
"""v28 + v68 앙상블 시도 (v46 대체)
v68은 라벨 수정된 데이터로 학습 → 일부 버그 수정
v28은 안정된 베이스 → 기존 패턴 정확
"""
import os, sys, json, re, torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer


def main():
    tok = AutoTokenizer.from_pretrained('tokenizer/')
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert

    m28 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    m28.load_state_dict(torch.load('checkpoints/cnn_multihead_v28.pt', map_location='cpu', weights_only=False)['state'])
    m28.eval()

    m68 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    m68.load_state_dict(torch.load('checkpoints/cnn_multihead_v68.pt', map_location='cpu', weights_only=False)['state'])
    m68.eval()

    # Strategies to try
    strategies = {
        'B (fn=v68, exec/dir=v28)': {
            'fn': 'v68', 'exec_type': 'v28', 'param_direction': 'v28',
            'param_type': 'v28', 'judge': 'v68',
        },
        'B2 (fn=v68, exec=v28, dir=v68)': {
            'fn': 'v68', 'exec_type': 'v28', 'param_direction': 'v68',
            'param_type': 'v28', 'judge': 'v68',
        },
        'C (all average)': 'average',
    }

    suite = json.load(open('data/test_suite.json'))
    ke_val = json.load(open('data/koelectra_converted_val.json'))

    def eval_strategy(strat):
        fn_ok = exec_ok = dir_ok = all_ok = 0
        for t in suite:
            tk = tok(t['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='pt')
            with torch.no_grad():
                l28 = m28(tk['input_ids']); l68 = m68(tk['input_ids'])

            if strat == 'average':
                logits = {h: (l28[h] + l68[h]) / 2 for h in HEAD_NAMES}
            else:
                logits = {h: l28[h] if strat[h] == 'v28' else l68[h] for h in HEAD_NAMES}

            p = {h: HEAD_I2L[h][logits[h].argmax(1).item()] for h in HEAD_NAMES}

            f = p['fn'] == t['fn']; e = p['exec_type'] == t['exec']; d = p['param_direction'] == t['dir']
            if f: fn_ok += 1
            if e: exec_ok += 1
            if d: dir_ok += 1
            if f and e and d: all_ok += 1

        # KE
        ke_fn_ok = 0
        for d in ke_val:
            tk = tok(d['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='pt')
            with torch.no_grad():
                l28 = m28(tk['input_ids']); l68 = m68(tk['input_ids'])
            if strat == 'average':
                fn_logit = (l28['fn'] + l68['fn']) / 2
            else:
                fn_logit = l28['fn'] if strat['fn'] == 'v28' else l68['fn']
            if HEAD_I2L['fn'][fn_logit.argmax(1).item()] == d['labels']['fn']:
                ke_fn_ok += 1

        n = len(suite)
        return {
            'fn': fn_ok/n*100, 'exec': exec_ok/n*100, 'dir': dir_ok/n*100,
            'combo': all_ok/n*100, 'ke_fn': ke_fn_ok/len(ke_val)*100,
        }

    # Baselines
    print("=== Baselines ===")
    for name, m in [('v28', m28), ('v68', m68)]:
        fn_ok = exec_ok = dir_ok = all_ok = 0
        for t in suite:
            tk = tok(t['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='pt')
            with torch.no_grad(): l = m(tk['input_ids'])
            p = {h: HEAD_I2L[h][l[h].argmax(1).item()] for h in HEAD_NAMES}
            f = p['fn'] == t['fn']; e = p['exec_type'] == t['exec']; d = p['param_direction'] == t['dir']
            if f: fn_ok += 1
            if e: exec_ok += 1
            if d: dir_ok += 1
            if f and e and d: all_ok += 1
        ke_ok = 0
        for d in ke_val:
            tk = tok(d['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='pt')
            with torch.no_grad(): l = m(tk['input_ids'])
            if HEAD_I2L['fn'][l['fn'].argmax(1).item()] == d['labels']['fn']:
                ke_ok += 1
        n = len(suite)
        print(f"  {name}: TS combo={all_ok/n*100:.2f}% KE fn={ke_ok/len(ke_val)*100:.2f}%")

    # Strategies
    print("\n=== Ensemble v28+v68 Strategies ===")
    for name, strat in strategies.items():
        r = eval_strategy(strat)
        balanced = (r['combo'] * r['ke_fn']) ** 0.5
        print(f"  [{name}]")
        print(f"    TS: fn={r['fn']:.2f} exec={r['exec']:.2f} dir={r['dir']:.2f} combo={r['combo']:.2f}")
        print(f"    KE fn={r['ke_fn']:.2f} | balanced={balanced:.2f}")


if __name__ == '__main__':
    main()
