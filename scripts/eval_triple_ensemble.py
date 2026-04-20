#!/usr/bin/env python3
"""3-Model Ensemble (v28 + v34 + v46) 시도
- v28: GT 전용, 100% Test Suite fn
- v34: Pseudo-labeling만
- v46: Pseudo + Mixup

여러 전략 비교:
- Majority voting
- Weighted logit average
- Strategy: head별 최적 모델 선택
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

    models = {}
    for name in ['v28', 'v34', 'v46']:
        m = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
        ckpt = torch.load(f'checkpoints/cnn_multihead_{name}.pt', map_location='cpu', weights_only=False)
        m.load_state_dict(ckpt['state']); m.eval()
        models[name] = m
        print(f"Loaded {name}: combo={ckpt.get('combo', '-'):.1f}")

    suite = json.load(open('data/test_suite.json'))
    ke_val = json.load(open('data/koelectra_converted_val.json'))

    # 모든 모델의 logit 캐시
    print(f"\nCaching logits for {len(suite)} suite + {len(ke_val)} KE...")
    all_logits = {}  # {model_name: {'ts': [...], 'ke': [...]}}
    for name, m in models.items():
        ts_logits = []
        for t in suite:
            tk = tok(t['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='pt')
            with torch.no_grad(): l = m(tk['input_ids'])
            ts_logits.append({h: l[h][0].clone() for h in HEAD_NAMES})
        ke_logits = []
        for d in ke_val:
            tk = tok(d['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='pt')
            with torch.no_grad(): l = m(tk['input_ids'])
            ke_logits.append(l['fn'][0].clone())
        all_logits[name] = {'ts': ts_logits, 'ke': ke_logits}
        print(f"  {name} done")

    # Strategy evaluator
    def eval_strat(strat_name, get_logits):
        """strat_name: description, get_logits: fn(i, head) -> logit tensor"""
        fn_ok = exec_ok = dir_ok = all_ok = 0
        for i, t in enumerate(suite):
            preds = {}
            for h in HEAD_NAMES:
                logit = get_logits(i, h, 'ts')
                preds[h] = HEAD_I2L[h][logit.argmax().item()]
            if preds['param_direction'] in ('open', 'close', 'stop'): preds['param_type'] = 'none'
            if preds['judge'] != 'none': preds['param_type'] = 'none'
            if preds['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'):
                preds['param_type'] = 'none'
            f = preds['fn'] == t['fn']; e = preds['exec_type'] == t['exec']; d = preds['param_direction'] == t['dir']
            if f: fn_ok += 1
            if e: exec_ok += 1
            if d: dir_ok += 1
            if f and e and d: all_ok += 1

        # KE
        ke_fn = 0
        for i, d in enumerate(ke_val):
            logit = get_logits(i, 'fn', 'ke')
            if HEAD_I2L['fn'][logit.argmax().item()] == d['labels']['fn']:
                ke_fn += 1

        n = len(suite)
        ts_combo = all_ok/n*100
        ke = ke_fn/len(ke_val)*100
        bal = (ts_combo * ke) ** 0.5
        print(f"  [{strat_name:40s}] TS fn={fn_ok/n*100:.2f} exec={exec_ok/n*100:.2f} dir={dir_ok/n*100:.2f} combo={ts_combo:.2f} | KE={ke:.2f} | bal={bal:.2f}")
        return ts_combo, ke, bal

    def single_logit(model_name):
        def get(i, head, split):
            if split == 'ts': return all_logits[model_name]['ts'][i][head]
            else: return all_logits[model_name]['ke'][i]
        return get

    print(f"\n=== Baselines ===")
    eval_strat("v28 alone", single_logit('v28'))
    eval_strat("v34 alone", single_logit('v34'))
    eval_strat("v46 alone", single_logit('v46'))

    # 2-model ensembles
    print(f"\n=== 2-model Ensembles ===")
    # v28+v46 Strategy B (current prod)
    def get_v28v46_b(i, head, split):
        if split == 'ke':
            return all_logits['v46']['ke'][i]  # fn from v46
        if head == 'fn' or head == 'judge':
            return all_logits['v46']['ts'][i][head]
        return all_logits['v28']['ts'][i][head]
    eval_strat("v28+v46 Strategy B (current)", get_v28v46_b)

    # 3-model ensembles
    print(f"\n=== 3-model Ensembles ===")

    # Average all
    def get_avg(i, head, split):
        if split == 'ke':
            return (all_logits['v28']['ke'][i] + all_logits['v34']['ke'][i] + all_logits['v46']['ke'][i]) / 3
        return (all_logits['v28']['ts'][i][head] + all_logits['v34']['ts'][i][head] + all_logits['v46']['ts'][i][head]) / 3
    eval_strat("v28+v34+v46 avg", get_avg)

    # Weighted avg (v46 heaviest for fn, v28 for exec/dir)
    def get_weighted(i, head, split):
        if split == 'ke':
            return 0.2 * all_logits['v28']['ke'][i] + 0.3 * all_logits['v34']['ke'][i] + 0.5 * all_logits['v46']['ke'][i]
        if head in ('fn', 'judge'):
            return 0.1 * all_logits['v28']['ts'][i][head] + 0.3 * all_logits['v34']['ts'][i][head] + 0.6 * all_logits['v46']['ts'][i][head]
        else:
            return 0.6 * all_logits['v28']['ts'][i][head] + 0.2 * all_logits['v34']['ts'][i][head] + 0.2 * all_logits['v46']['ts'][i][head]
    eval_strat("v28+v34+v46 weighted by head", get_weighted)

    # Strategy: fn from v46, exec/dir from v28, param_type from v34
    def get_head_specific(i, head, split):
        if split == 'ke':
            return all_logits['v46']['ke'][i]
        if head == 'fn':
            return all_logits['v46']['ts'][i][head]
        if head == 'judge':
            return all_logits['v46']['ts'][i][head]
        if head == 'param_type':
            return all_logits['v34']['ts'][i][head]
        return all_logits['v28']['ts'][i][head]
    eval_strat("Head-specific (fn=v46, e/d=v28, pt=v34)", get_head_specific)

    # Soft voting (majority among argmax)
    def get_vote(i, head, split):
        if split == 'ke':
            # Majority vote on fn (single head)
            preds = [all_logits['v28']['ke'][i].argmax().item(),
                     all_logits['v34']['ke'][i].argmax().item(),
                     all_logits['v46']['ke'][i].argmax().item()]
            from collections import Counter
            top = Counter(preds).most_common(1)[0][0]
            # Return logit corresponding to the voted class
            logit = torch.zeros_like(all_logits['v28']['ke'][i])
            logit[top] = 100.0  # Winner takes all
            return logit
        preds = [all_logits['v28']['ts'][i][head].argmax().item(),
                 all_logits['v34']['ts'][i][head].argmax().item(),
                 all_logits['v46']['ts'][i][head].argmax().item()]
        from collections import Counter
        top = Counter(preds).most_common(1)[0][0]
        logit = torch.zeros_like(all_logits['v28']['ts'][i][head])
        logit[top] = 100.0
        return logit
    eval_strat("Majority vote", get_vote)


if __name__ == '__main__':
    main()
