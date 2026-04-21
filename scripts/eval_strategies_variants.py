#!/usr/bin/env python3
"""Strategy 변형 비교 — TS + KE 통합.

각 head별로 v28 / v46 / confidence 기반 선택 전략 테스트.
"""
import torch, torch.nn.functional as F, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, 'scripts')
from model_cnn_multihead import *
from preprocess import preprocess
from transformers import AutoModel, AutoTokenizer


def load_models():
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach()
    del sbert
    m28 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    m28.load_state_dict(torch.load('checkpoints/cnn_multihead_v28.pt', map_location='cpu', weights_only=False)['state'])
    m28.eval()
    m46 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    m46.load_state_dict(torch.load('checkpoints/cnn_multihead_v46.pt', map_location='cpu', weights_only=False)['state'])
    m46.eval()
    tok = AutoTokenizer.from_pretrained('tokenizer/')
    return m28, m46, tok


def logits_for(text, m28, m46, tok):
    text = preprocess(text)
    if not text:
        return None, None
    tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
    with torch.no_grad():
        l28 = m28(tk['input_ids'])
        l46 = m46(tk['input_ids'])
    return l28, l46


def strategy_pred(l28, l46, strategy):
    """Strategy에 따라 head별 소스 결정."""
    out = {}
    for h in HEAD_NAMES:
        src = strategy.get(h, 'v28')
        if src == 'v28':
            out[h] = HEAD_I2L[h][l28[h].argmax(1).item()]
        elif src == 'v46':
            out[h] = HEAD_I2L[h][l46[h].argmax(1).item()]
        elif src == 'avg':
            avg = (F.softmax(l28[h], 1) + F.softmax(l46[h], 1)) / 2
            out[h] = HEAD_I2L[h][avg.argmax(1).item()]
        elif src == 'maxconf':
            c28 = F.softmax(l28[h], 1).max().item()
            c46 = F.softmax(l46[h], 1).max().item()
            src_pick = 'v28' if c28 >= c46 else 'v46'
            li = l28 if src_pick == 'v28' else l46
            out[h] = HEAD_I2L[h][li[h].argmax(1).item()]
    return out


# test_suite format mapping: dir → param_direction, exec → exec_type, param → param_type
KEY_MAP = {'fn': 'fn', 'exec': 'exec_type', 'dir': 'param_direction', 'param': 'param_type', 'judge': 'judge'}


def eval_ts(m28, m46, tok, strategy):
    ts = json.load(open('data/test_suite.json'))
    n = 0
    exact = 0
    per_head = {h: 0 for h in HEAD_NAMES}
    for item in ts:
        utt = item['utterance']
        # flat 포맷 + optional keys
        gt = {}
        for short, long in KEY_MAP.items():
            if short in item:
                gt[long] = item[short]
            elif long in item:
                gt[long] = item[long]
        l28, l46 = logits_for(utt, m28, m46, tok)
        if l28 is None:
            continue
        pred = strategy_pred(l28, l46, strategy)
        n += 1
        head_ok = True
        for h in HEAD_NAMES:
            if h in gt:
                if pred.get(h) == gt[h]:
                    per_head[h] += 1
                else:
                    head_ok = False
        if head_ok:
            exact += 1
    return {
        'n': n,
        'combo': exact / n * 100,
        **{h: per_head[h] / n * 100 for h in HEAD_NAMES},
    }


def eval_ke(m28, m46, tok, strategy, max_n=None):
    ke_path = 'data/koelectra_converted_val.json'
    if not os.path.exists(ke_path):
        return None
    ke = json.load(open(ke_path))
    if max_n:
        ke = ke[:max_n]
    n = 0
    fn_ok = 0
    for item in ke:
        utt = item.get('utterance') or item.get('text')
        gt = item.get('labels', {}).get('fn')
        if not utt or not gt:
            continue
        l28, l46 = logits_for(utt, m28, m46, tok)
        if l28 is None:
            continue
        pred = strategy_pred(l28, l46, strategy)
        n += 1
        if pred['fn'] == gt:
            fn_ok += 1
    return fn_ok / n * 100 if n else 0


def main():
    print("=== Strategy 변형 비교 ===\n")
    m28, m46, tok = load_models()

    strategies = {
        'B (current)':           {'fn': 'v46', 'exec_type': 'v28', 'param_direction': 'v28', 'param_type': 'v28', 'judge': 'v46'},
        'B-mod (dir=v46)':       {'fn': 'v46', 'exec_type': 'v28', 'param_direction': 'v46', 'param_type': 'v28', 'judge': 'v46'},
        'B-param (param=v46)':   {'fn': 'v46', 'exec_type': 'v28', 'param_direction': 'v28', 'param_type': 'v46', 'judge': 'v46'},
        'B-both (dir+param=v46)':{'fn': 'v46', 'exec_type': 'v28', 'param_direction': 'v46', 'param_type': 'v46', 'judge': 'v46'},
        'MaxConf all':           {h: 'maxconf' for h in HEAD_NAMES},
        'MaxConf dir/param only':{'fn': 'v46', 'exec_type': 'v28', 'param_direction': 'maxconf', 'param_type': 'maxconf', 'judge': 'v46'},
        'All v46':               {h: 'v46' for h in HEAD_NAMES},
        'Avg (softmax mean)':    {h: 'avg' for h in HEAD_NAMES},
        'Avg ex-fn (fn=v46)':    {'fn': 'v46', 'exec_type': 'avg', 'param_direction': 'avg', 'param_type': 'avg', 'judge': 'v46'},
        'MaxConf ex-fn':         {'fn': 'v46', 'exec_type': 'maxconf', 'param_direction': 'maxconf', 'param_type': 'maxconf', 'judge': 'v46'},
        'Avg fn=v46, judge=avg': {'fn': 'v46', 'exec_type': 'avg', 'param_direction': 'avg', 'param_type': 'avg', 'judge': 'avg'},
    }

    results = {}
    for name, strat in strategies.items():
        print(f"  [{name}]  ...", end='', flush=True)
        ts_metrics = eval_ts(m28, m46, tok, strat)
        ke_fn = eval_ke(m28, m46, tok, strat, max_n=1536)
        results[name] = {
            'ts_combo': ts_metrics['combo'],
            'ts_fn': ts_metrics['fn'],
            'ts_exec': ts_metrics['exec_type'],
            'ts_dir': ts_metrics['param_direction'],
            'ts_param': ts_metrics['param_type'],
            'ke_fn_sample': ke_fn if ke_fn else 0,
        }
        print(f"  TS {ts_metrics['combo']:.2f}%  KE(500) {ke_fn:.2f}%")

    print()
    print(f"{'Strategy':<28} {'TS combo':>10} {'fn':>7} {'exec':>7} {'dir':>7} {'param':>7} {'KE(500)':>9}")
    print('-' * 80)
    for name, r in results.items():
        print(f"{name:<28} {r['ts_combo']:>9.2f}% {r['ts_fn']:>6.2f}% {r['ts_exec']:>6.2f}% {r['ts_dir']:>6.2f}% {r['ts_param']:>6.2f}% {r['ke_fn_sample']:>8.2f}%")

    os.makedirs('data', exist_ok=True)
    with open('data/strategy_variants_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n저장: data/strategy_variants_results.json")


if __name__ == '__main__':
    main()
