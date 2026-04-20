#!/usr/bin/env python3
"""v68 (fixed labels) vs v46 상세 비교
- 수정된 케이스 11건에서 v68이 실제로 개선됐는지 확인
- 나머지에서 regression 있는지 확인
"""
import os, sys, json, re, torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer


def load_model(ckpt_name):
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach()
    del sbert
    model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    ckpt = torch.load(f'checkpoints/cnn_multihead_{ckpt_name}.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state'])
    model.eval()
    return model


def predict(model, tok, text):
    tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
    with torch.no_grad():
        l = model(tk['input_ids'])
    p = {h: HEAD_I2L[h][l[h].argmax(1).item()] for h in HEAD_NAMES}
    if p['param_direction'] in ('open', 'close', 'stop'): p['param_type'] = 'none'
    if p['judge'] != 'none': p['param_type'] = 'none'
    if p['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'): p['param_type'] = 'none'
    conf = F.softmax(l['fn'], dim=1).max().item()
    return p, conf


def main():
    tok = AutoTokenizer.from_pretrained('tokenizer/')
    m46 = load_model('v46')
    m68 = load_model('v68')
    suite = json.load(open('data/test_suite.json'))

    # 1. 수정된 11건에서의 성능
    fixed_utts = [
        '커턴 닫아', '난방꺼줘', '환기꺼줘', '주방 남방 꺼줘',
        '주차 등록해줘', '방문 주차 등록해줘',
        '승강기 불러줘', '리프트 호출', '승강기 호출해줘',
        '승강기불러', '승강기 호출해줘봐',
    ]

    print(f"=== 수정된 11건에 대한 v46 vs v68 ===")
    v46_fix_ok = 0; v68_fix_ok = 0
    for t in suite:
        if t['utterance'] not in fixed_utts: continue
        p46, _ = predict(m46, tok, t['utterance'])
        p68, _ = predict(m68, tok, t['utterance'])

        ok46 = p46['fn']==t['fn'] and p46['exec_type']==t['exec'] and p46['param_direction']==t['dir']
        ok68 = p68['fn']==t['fn'] and p68['exec_type']==t['exec'] and p68['param_direction']==t['dir']
        if ok46: v46_fix_ok += 1
        if ok68: v68_fix_ok += 1

        m46_sym = '✓' if ok46 else '✗'
        m68_sym = '✓' if ok68 else '✗'
        print(f"  v46={m46_sym} v68={m68_sym} \"{t['utterance']}\"")
        if not ok68:
            print(f"    (expected fn={t['fn']}, exec={t['exec']}, dir={t['dir']})")
            print(f"    (v68 got fn={p68['fn']}, exec={p68['exec_type']}, dir={p68['param_direction']})")

    print(f"\n  수정된 케이스 정확도: v46 {v46_fix_ok}/11, v68 {v68_fix_ok}/11")

    # 2. 전체 정확도 비교
    print(f"\n=== 전체 3043 케이스 비교 ===")
    v46_correct = set(); v68_correct = set()
    for i, t in enumerate(suite):
        p46, _ = predict(m46, tok, t['utterance'])
        p68, _ = predict(m68, tok, t['utterance'])
        if p46['fn']==t['fn'] and p46['exec_type']==t['exec'] and p46['param_direction']==t['dir']:
            v46_correct.add(i)
        if p68['fn']==t['fn'] and p68['exec_type']==t['exec'] and p68['param_direction']==t['dir']:
            v68_correct.add(i)

    both = v46_correct & v68_correct
    v46_only = v46_correct - v68_correct
    v68_only = v68_correct - v46_correct
    neither = set(range(len(suite))) - v46_correct - v68_correct

    print(f"  둘 다 맞음: {len(both)}")
    print(f"  v46만 맞음: {len(v46_only)}")
    print(f"  v68만 맞음: {len(v68_only)}")
    print(f"  둘 다 틀림: {len(neither)}")
    print(f"  v46 combo: {len(v46_correct)/len(suite)*100:.2f}%")
    print(f"  v68 combo: {len(v68_correct)/len(suite)*100:.2f}%")

    # Sample regression
    if v46_only:
        print(f"\n  v46→v68 regression 예시 (최대 10개):")
        for i in sorted(v46_only)[:10]:
            t = suite[i]
            p68, _ = predict(m68, tok, t['utterance'])
            print(f"    \"{t['utterance']}\" (exp fn={t['fn']} exec={t['exec']} dir={t['dir']})")
            print(f"      v68: fn={p68['fn']} exec={p68['exec_type']} dir={p68['param_direction']}")


if __name__ == '__main__':
    main()
