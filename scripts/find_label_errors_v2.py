#!/usr/bin/env python3
"""라벨 오류 자동 탐지 v2 — 모델 high-conf disagreement + 규칙 결합

전략:
1. 앙상블 (v28+v46) 모두 같은 예측을 conf>=0.95로 하는데 라벨과 다르면 → 라벨 오류 후보
2. 규칙 기반 (밝게→up, 어둡게→down, N모드→set 등)
3. 두 증거 일치하는 것만 자동 수정 candidate로 분류

카테고리:
A. 두 방법 모두 다르다 말함 (매우 확실)
B. 규칙만 말함 (검토 필요)
C. 모델만 말함 (검토 필요)
"""
import os, sys, json, re
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from preprocess import preprocess
from transformers import AutoModel, AutoTokenizer


def load_ensemble():
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert
    tok = AutoTokenizer.from_pretrained('tokenizer/')
    m28 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    m28.load_state_dict(torch.load('checkpoints/cnn_multihead_v28.pt', map_location='cpu', weights_only=False)['state'])
    m28.eval()
    m46 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    m46.load_state_dict(torch.load('checkpoints/cnn_multihead_v46.pt', map_location='cpu', weights_only=False)['state'])
    m46.eval()
    return m28, m46, tok


def ensemble_predict(m28, m46, tok, text):
    tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
    with torch.no_grad():
        l28 = m28(tk['input_ids']); l46 = m46(tk['input_ids'])
    # Strategy B
    p = {
        'fn': HEAD_I2L['fn'][l46['fn'].argmax(1).item()],
        'exec_type': HEAD_I2L['exec_type'][l28['exec_type'].argmax(1).item()],
        'param_direction': HEAD_I2L['param_direction'][l28['param_direction'].argmax(1).item()],
    }
    # 각 head의 confidence
    conf = {
        'fn': F.softmax(l46['fn'], dim=1).max().item(),
        'exec': F.softmax(l28['exec_type'], dim=1).max().item(),
        'dir': F.softmax(l28['param_direction'], dim=1).max().item(),
    }
    # 두 모델 agreement 체크
    agree = {
        'fn': HEAD_I2L['fn'][l28['fn'].argmax(1).item()] == p['fn'] and HEAD_I2L['fn'][l46['fn'].argmax(1).item()] == p['fn'],
        'exec': HEAD_I2L['exec_type'][l28['exec_type'].argmax(1).item()] == HEAD_I2L['exec_type'][l46['exec_type'].argmax(1).item()],
        'dir': HEAD_I2L['param_direction'][l28['param_direction'].argmax(1).item()] == HEAD_I2L['param_direction'][l46['param_direction'].argmax(1).item()],
    }
    return p, conf, agree


# 규칙 기반 예상 라벨
RULES = [
    # (compiled_pattern, field, expected_value, description, require_not_contain)
    (re.compile(r'밝게|밝아지|밝혀|좀 환하게'), 'param_direction', 'up', '밝게=up', ['어둡게', '어둠']),
    (re.compile(r'어둡게|은은하게|어두워지|침침|희미'), 'param_direction', 'down', '어둡게=down', []),
    (re.compile(r'모드로|모드로 해|모드 설정'), 'param_direction', 'set', 'N모드로=set', []),
    (re.compile(r'닫아|닫자|잠가|잠궈|잠궜|잠금'), 'param_direction', 'close', '닫아=close', ['현관.*열', '연결']),
    (re.compile(r'^(?!.*(?:있|나|되|냐)).*(열어줘|열어봐|열어|열기)'), 'param_direction', 'open', '열어=open', []),
    (re.compile(r'올려|높여|최대로'), 'param_direction', 'up', '올려=up', ['올라와']),
    (re.compile(r'낮춰|줄여|최소로'), 'param_direction', 'down', '낮춰=down', []),
    (re.compile(r'등록해|설정해줘|맞춰줘'), 'param_direction', 'set', '등록/설정=set', []),
    (re.compile(r'(승강기|엘리베이터|엘베).*(호출|불러|오라고|올라와|내려와)'), 'exec_type', 'control_then_confirm', '승강기 호출=control', []),
    (re.compile(r'몇\s*도|몇\s*시|얼마|얼만큼'), 'exec_type', 'query_then_respond', '수치 질의=query', []),
]


def check_rules(utt, labels):
    """규칙 기반 예상 라벨"""
    suspects = []
    for pat, field, exp_val, desc, excludes in RULES:
        if pat.search(utt):
            skip = False
            for ex in excludes:
                if re.search(ex, utt):
                    skip = True; break
            if skip: continue

            actual = labels.get(field)
            if actual and actual != exp_val:
                suspects.append({
                    'field': field, 'actual': actual, 'expected': exp_val, 'reason': desc
                })
    return suspects


def main():
    print("=== 모델 로드 ===")
    m28, m46, tok = load_ensemble()

    print("\n=== Test Suite 로드 ===")
    suite = json.load(open('data/test_suite.json'))
    print(f"  {len(suite)}개")

    # Test Suite에 대해 각 케이스 분석
    cat_a = []  # 모델+규칙 둘 다 일치
    cat_b = []  # 규칙만
    cat_c = []  # 모델만

    for i, t in enumerate(suite):
        utt = t['utterance']
        labels = {'fn': t['fn'], 'exec_type': t['exec'], 'param_direction': t['dir']}

        # 규칙
        rule_suspects = check_rules(utt, labels)

        # 모델 예측
        pred, conf, agree = ensemble_predict(m28, m46, tok, preprocess(utt))

        # 모델이 라벨과 다른데 high-conf + agreement인 것
        model_suspects = []
        for field in ['fn', 'exec_type', 'param_direction']:
            pred_val = pred[field]
            actual = labels[field]
            conf_key = {'fn': 'fn', 'exec_type': 'exec', 'param_direction': 'dir'}[field]
            if pred_val != actual and conf[conf_key] >= 0.95 and agree[conf_key]:
                model_suspects.append({
                    'field': field, 'actual': actual, 'expected': pred_val,
                    'reason': f'모델 앙상블 conf={conf[conf_key]:.2f} 일치'
                })

        # 카테고리 분류
        for rs in rule_suspects:
            matching_model = [m for m in model_suspects if m['field'] == rs['field'] and m['expected'] == rs['expected']]
            if matching_model:
                cat_a.append({'idx': i, 'utterance': utt, **rs, 'model_conf': matching_model[0]['reason']})
            else:
                cat_b.append({'idx': i, 'utterance': utt, **rs})

        for ms in model_suspects:
            # A 카테고리 아닌 것만 C
            matching_rule = [r for r in rule_suspects if r['field'] == ms['field'] and r['expected'] == ms['expected']]
            if not matching_rule:
                cat_c.append({'idx': i, 'utterance': utt, **ms})

    print(f"\n=== 카테고리 A (모델 + 규칙 둘 다 일치) — {len(cat_a)}건 ===")
    for s in cat_a[:30]:
        print(f"  [{s['idx']}] \"{s['utterance']}\" {s['field']}: {s['actual']} → {s['expected']} ({s['reason']})")
    if len(cat_a) > 30: print(f"  ... 그 외 {len(cat_a)-30}건")

    print(f"\n=== 카테고리 B (규칙만 의심) — {len(cat_b)}건 — 검토 필요 ===")
    for s in cat_b[:10]:
        print(f"  [{s['idx']}] \"{s['utterance']}\" {s['field']}: {s['actual']} → {s['expected']} ({s['reason']})")
    if len(cat_b) > 10: print(f"  ... 그 외 {len(cat_b)-10}건")

    print(f"\n=== 카테고리 C (모델만 의심 high-conf) — {len(cat_c)}건 — 검토 필요 ===")
    for s in cat_c[:15]:
        print(f"  [{s['idx']}] \"{s['utterance']}\" {s['field']}: {s['actual']} → {s['expected']} ({s['reason']})")
    if len(cat_c) > 15: print(f"  ... 그 외 {len(cat_c)-15}건")

    # 저장
    with open('data/label_errors_v2.json', 'w', encoding='utf-8') as f:
        json.dump({
            'cat_a_high_confidence': cat_a,
            'cat_b_rule_only': cat_b,
            'cat_c_model_only': cat_c,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✓ data/label_errors_v2.json 저장")


if __name__ == '__main__':
    main()
